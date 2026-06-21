"""B2-GC T1 coverage contract: retired_at set/clear at every tombstone path.

For each tombstone-creating operation, asserts that the row's ``retired_at``
column is non-NULL (and approximately current epoch) after the operation.
For each restore operation, asserts that ``retired_at`` is NULL.

``retired_at`` is storage-internal — not on the domain model — so it is read
via a direct ``SELECT retired_at FROM <table> WHERE <pk>=?``.
"""

import time

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    LineageContext,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FUZZ_SECONDS = 5  # retired_at must be within this many seconds of now


def _now() -> int:
    return int(time.time())


def _store(tmp_path, org_id: str) -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _get_retired_at_profile(s: SQLiteStorage, profile_id: str) -> int | None:
    row = s.conn.execute(
        "SELECT retired_at FROM profiles WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    assert row is not None, f"profile {profile_id!r} not found"
    return row["retired_at"]


def _get_retired_at_user_playbook(s: SQLiteStorage, upid: int) -> int | None:
    row = s.conn.execute(
        "SELECT retired_at FROM user_playbooks WHERE user_playbook_id = ?", (upid,)
    ).fetchone()
    assert row is not None, f"user_playbook {upid} not found"
    return row["retired_at"]


def _get_retired_at_agent_playbook(s: SQLiteStorage, apid: int) -> int | None:
    row = s.conn.execute(
        "SELECT retired_at FROM agent_playbooks WHERE agent_playbook_id = ?", (apid,)
    ).fetchone()
    assert row is not None, f"agent_playbook {apid} not found"
    return row["retired_at"]


def _assert_retired_now(retired_at: int | None) -> None:
    """Assert retired_at is non-NULL and within _FUZZ_SECONDS of now."""
    assert retired_at is not None, "retired_at must be non-NULL after tombstone op"
    assert abs(_now() - retired_at) <= _FUZZ_SECONDS, (
        f"retired_at {retired_at} not close to now {_now()}"
    )


def _assert_retired_null(retired_at: int | None) -> None:
    assert retired_at is None, (
        f"retired_at must be NULL after restore; got {retired_at!r}"
    )


def _make_profile(
    profile_id: str = "p1",
    user_id: str = "u1",
) -> UserProfile:
    return UserProfile(
        profile_id=profile_id,
        user_id=user_id,
        content="content",
        last_modified_timestamp=_now(),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _make_user_playbook(user_id: str = "u1") -> UserPlaybook:
    return UserPlaybook(
        user_id=user_id, agent_version="v1", request_id="r1", content="c"
    )


def _make_agent_playbook(playbook_name: str = "pb") -> AgentPlaybook:
    return AgentPlaybook(playbook_name=playbook_name, agent_version="v1", content="c")


# ---------------------------------------------------------------------------
# merge_records — profiles, user_playbooks, agent_playbooks
# ---------------------------------------------------------------------------


def test_merge_records_profile_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-merge-profile")
    source = _make_profile("src", "u1")
    survivor = _make_profile("surv", "u1")
    s.add_user_profile("u1", [source, survivor])

    s.merge_records(
        entity_type="profile",
        survivor_id=survivor.profile_id,
        source_ids=[source.profile_id],
        context=LineageContext(op_kind="merge", actor="test", request_id="r-merge-p"),
    )

    _assert_retired_now(_get_retired_at_profile(s, source.profile_id))
    # Survivor must not be retired
    assert _get_retired_at_profile(s, survivor.profile_id) is None


def test_merge_records_user_playbook_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-merge-up")
    source = _make_user_playbook("u1")
    survivor = _make_user_playbook("u1")
    s.save_user_playbooks([source, survivor])

    s.merge_records(
        entity_type="user_playbook",
        survivor_id=str(survivor.user_playbook_id),
        source_ids=[str(source.user_playbook_id)],
        context=LineageContext(op_kind="merge", actor="test", request_id="r-merge-up"),
    )

    _assert_retired_now(_get_retired_at_user_playbook(s, source.user_playbook_id))
    assert _get_retired_at_user_playbook(s, survivor.user_playbook_id) is None


def test_merge_records_agent_playbook_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-merge-ap")
    source = _make_agent_playbook("pb")
    survivor = _make_agent_playbook("pb2")
    s.save_agent_playbooks([source, survivor])

    s.merge_records(
        entity_type="agent_playbook",
        survivor_id=str(survivor.agent_playbook_id),
        source_ids=[str(source.agent_playbook_id)],
        context=LineageContext(op_kind="merge", actor="test", request_id="r-merge-ap"),
    )

    _assert_retired_now(_get_retired_at_agent_playbook(s, source.agent_playbook_id))
    assert _get_retired_at_agent_playbook(s, survivor.agent_playbook_id) is None


# ---------------------------------------------------------------------------
# supersede_record
# ---------------------------------------------------------------------------


def test_supersede_record_profile_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-super-record-p")
    incumbent = _make_profile("inc", "u1")
    successor = _make_profile("succ", "u1")
    s.add_user_profile("u1", [incumbent, successor])

    result = s.supersede_record(
        entity_type="profile",
        incumbent_id=incumbent.profile_id,
        successor_id=successor.profile_id,
        context=LineageContext(op_kind="revise", actor="test", request_id="r-super-p"),
    )

    assert result is True
    _assert_retired_now(_get_retired_at_profile(s, incumbent.profile_id))
    assert _get_retired_at_profile(s, successor.profile_id) is None


def test_supersede_record_user_playbook_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-super-record-up")
    incumbent = _make_user_playbook("u1")
    successor = _make_user_playbook("u1")
    s.save_user_playbooks([incumbent, successor])

    result = s.supersede_record(
        entity_type="user_playbook",
        incumbent_id=str(incumbent.user_playbook_id),
        successor_id=str(successor.user_playbook_id),
        context=LineageContext(op_kind="revise", actor="test", request_id="r-super-up"),
    )

    assert result is True
    _assert_retired_now(_get_retired_at_user_playbook(s, incumbent.user_playbook_id))
    assert _get_retired_at_user_playbook(s, successor.user_playbook_id) is None


# ---------------------------------------------------------------------------
# supersede_profiles_by_ids
# ---------------------------------------------------------------------------


def test_supersede_profiles_by_ids_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-super-profiles")
    p1 = _make_profile("sp1", "u1")
    p2 = _make_profile("sp2", "u1")
    s.add_user_profile("u1", [p1, p2])

    count = s.supersede_profiles_by_ids(
        "u1", [p1.profile_id, p2.profile_id], "req-super"
    )

    assert count == 2
    _assert_retired_now(_get_retired_at_profile(s, p1.profile_id))
    _assert_retired_now(_get_retired_at_profile(s, p2.profile_id))


# ---------------------------------------------------------------------------
# archive_agent_playbooks_by_ids
# ---------------------------------------------------------------------------


def test_archive_agent_playbooks_by_ids_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-archive-ap-ids")
    ap = _make_agent_playbook("pb")
    s.save_agent_playbooks([ap])

    s.archive_agent_playbooks_by_ids([ap.agent_playbook_id])

    _assert_retired_now(_get_retired_at_agent_playbook(s, ap.agent_playbook_id))


# ---------------------------------------------------------------------------
# archive_agent_playbooks_by_playbook_name
# ---------------------------------------------------------------------------


def test_archive_agent_playbooks_by_playbook_name_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-archive-ap-name")
    ap = _make_agent_playbook("mybook")
    s.save_agent_playbooks([ap])

    s.archive_agent_playbooks_by_playbook_name("mybook")

    _assert_retired_now(_get_retired_at_agent_playbook(s, ap.agent_playbook_id))


# ---------------------------------------------------------------------------
# supersede_agent_playbooks_by_ids
# ---------------------------------------------------------------------------


def test_supersede_agent_playbooks_by_ids_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-super-ap-ids")
    ap = _make_agent_playbook("pb")
    s.save_agent_playbooks([ap])

    count = s.supersede_agent_playbooks_by_ids([ap.agent_playbook_id], "req-super-ap")

    assert count == 1
    _assert_retired_now(_get_retired_at_agent_playbook(s, ap.agent_playbook_id))


# ---------------------------------------------------------------------------
# supersede_agent_playbooks_by_playbook_name
# ---------------------------------------------------------------------------


def test_supersede_agent_playbooks_by_playbook_name_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-super-ap-name")
    ap = _make_agent_playbook("archbook")
    s.save_agent_playbooks([ap])
    # Must be archived first (supersede_by_name only targets archived rows)
    s.archive_agent_playbooks_by_ids([ap.agent_playbook_id])

    count = s.supersede_agent_playbooks_by_playbook_name(
        "archbook", None, "req-super-apname"
    )

    assert count == 1
    _assert_retired_now(_get_retired_at_agent_playbook(s, ap.agent_playbook_id))


# ---------------------------------------------------------------------------
# update_all_profiles_status (tombstone → NULL restore)
# ---------------------------------------------------------------------------


def test_update_all_profiles_status_tombstone_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-upd-profiles-tomb")
    p = _make_profile("up1", "u1")
    s.add_user_profile("u1", [p])

    count = s.update_all_profiles_status(None, Status.SUPERSEDED)

    assert count >= 1
    _assert_retired_now(_get_retired_at_profile(s, p.profile_id))


def test_update_all_profiles_status_to_null_clears_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-upd-profiles-restore")
    p = _make_profile("up2", "u1")
    s.add_user_profile("u1", [p])
    # First tombstone it
    s.update_all_profiles_status(None, Status.SUPERSEDED)
    _assert_retired_now(_get_retired_at_profile(s, p.profile_id))

    # Now restore
    s.update_all_profiles_status(Status.SUPERSEDED, None)

    _assert_retired_null(_get_retired_at_profile(s, p.profile_id))


# ---------------------------------------------------------------------------
# update_all_user_playbooks_status (tombstone → NULL restore)
# ---------------------------------------------------------------------------


def test_update_all_user_playbooks_status_tombstone_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-upd-up-tomb")
    up = _make_user_playbook("u1")
    s.save_user_playbooks([up])

    count = s.update_all_user_playbooks_status(None, Status.MERGED)

    assert count >= 1
    _assert_retired_now(_get_retired_at_user_playbook(s, up.user_playbook_id))


def test_update_all_user_playbooks_status_to_null_clears_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-upd-up-restore")
    up = _make_user_playbook("u1")
    s.save_user_playbooks([up])
    s.update_all_user_playbooks_status(None, Status.MERGED)
    _assert_retired_now(_get_retired_at_user_playbook(s, up.user_playbook_id))

    s.update_all_user_playbooks_status(Status.MERGED, None)

    _assert_retired_null(_get_retired_at_user_playbook(s, up.user_playbook_id))


# ---------------------------------------------------------------------------
# restore_archived_agent_playbooks_by_playbook_name
# ---------------------------------------------------------------------------


def test_restore_archived_by_playbook_name_clears_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-restore-ap-name")
    ap = _make_agent_playbook("restorebook")
    s.save_agent_playbooks([ap])
    s.archive_agent_playbooks_by_ids([ap.agent_playbook_id])
    _assert_retired_now(_get_retired_at_agent_playbook(s, ap.agent_playbook_id))

    s.restore_archived_agent_playbooks_by_playbook_name("restorebook")

    _assert_retired_null(_get_retired_at_agent_playbook(s, ap.agent_playbook_id))


# ---------------------------------------------------------------------------
# restore_archived_agent_playbooks_by_ids
# ---------------------------------------------------------------------------


def test_restore_archived_by_ids_clears_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-restore-ap-ids")
    ap = _make_agent_playbook("restorebook2")
    s.save_agent_playbooks([ap])
    s.archive_agent_playbooks_by_ids([ap.agent_playbook_id])
    _assert_retired_now(_get_retired_at_agent_playbook(s, ap.agent_playbook_id))

    s.restore_archived_agent_playbooks_by_ids([ap.agent_playbook_id])

    _assert_retired_null(_get_retired_at_agent_playbook(s, ap.agent_playbook_id))


# ---------------------------------------------------------------------------
# F3 — missing contract cases: archive_user_playbook_by_id,
#       archive_profile_by_id, supersede_record(agent_playbook)
# ---------------------------------------------------------------------------


def test_archive_user_playbook_by_id_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-archive-up-by-id")
    up = _make_user_playbook()
    s.save_user_playbooks([up])

    result = s.archive_user_playbook_by_id("u1", up.user_playbook_id)

    assert result is True
    _assert_retired_now(_get_retired_at_user_playbook(s, up.user_playbook_id))


def test_archive_profile_by_id_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-archive-profile-by-id")
    p = _make_profile("ap1", "u1")
    s.add_user_profile("u1", [p])

    result = s.archive_profile_by_id("u1", p.profile_id)

    assert result is True
    _assert_retired_now(_get_retired_at_profile(s, p.profile_id))


def test_supersede_record_agent_playbook_sets_retired_at(tmp_path) -> None:
    s = _store(tmp_path, "org-super-record-ap")
    incumbent = _make_agent_playbook("pb-inc")
    successor = _make_agent_playbook("pb-succ")
    s.save_agent_playbooks([incumbent, successor])

    result = s.supersede_record(
        entity_type="agent_playbook",
        incumbent_id=str(incumbent.agent_playbook_id),
        successor_id=str(successor.agent_playbook_id),
        context=LineageContext(op_kind="revise", actor="test", request_id="r-super-ap"),
    )

    assert result is True
    _assert_retired_now(_get_retired_at_agent_playbook(s, incumbent.agent_playbook_id))
    assert _get_retired_at_agent_playbook(s, successor.agent_playbook_id) is None


# ---------------------------------------------------------------------------
# F4 — merge-on-archived: archived source must NOT be re-tombstoned to MERGED
# ---------------------------------------------------------------------------


def test_merge_does_not_re_tombstone_archived_source(tmp_path) -> None:
    """Archived source must keep status=ARCHIVED and retired_at=T0 after merge."""
    s = _store(tmp_path, "org-merge-archived-guard")
    source = _make_profile("src-arch", "u1")
    survivor = _make_profile("surv-arch", "u1")
    s.add_user_profile("u1", [source, survivor])

    # Archive the source first — records its retired_at=T0.
    s.archive_profile_by_id("u1", source.profile_id)
    retired_at_t0 = _get_retired_at_profile(s, source.profile_id)
    assert retired_at_t0 is not None

    # Now merge with the archived source among source_ids.
    s.merge_records(
        entity_type="profile",
        survivor_id=survivor.profile_id,
        source_ids=[source.profile_id],
        context=LineageContext(
            op_kind="merge", actor="test", request_id="r-merge-arch"
        ),
    )

    # Status must remain ARCHIVED, not flipped to MERGED.
    row = s.conn.execute(
        "SELECT status, retired_at FROM profiles WHERE profile_id = ?",
        (source.profile_id,),
    ).fetchone()
    assert row["status"] == "archived", (
        f"Expected status=archived, got {row['status']!r}"
    )
    # retired_at must be unchanged — not re-set by the merge.
    assert row["retired_at"] == retired_at_t0, (
        f"retired_at changed from {retired_at_t0} to {row['retired_at']} after merge"
    )


# ---------------------------------------------------------------------------
# F5 — GC legal-hold forward-progress: a held oldest row must not starve
#       eligible non-held rows in the same pass
# ---------------------------------------------------------------------------


def test_gc_legal_hold_does_not_starve_non_held_eligible_rows(tmp_path) -> None:
    """Hold the oldest tombstone; assert the next eligible row is still GC'd."""
    from unittest.mock import patch

    s = _store(tmp_path, "org-gc-hold-progress")

    # Seed two user_playbook tombstones: older_up (smaller retired_at) and newer_up.
    older_up = _make_user_playbook()
    newer_up = _make_user_playbook()
    s.save_user_playbooks([older_up, newer_up])

    old_epoch = _now() - 200  # 200 s ago — clearly past any cutoff
    recent_eligible_epoch = _now() - 100  # 100 s ago

    s.conn.execute(
        "UPDATE user_playbooks SET status = 'merged', retired_at = ? "
        "WHERE user_playbook_id = ?",
        (old_epoch, older_up.user_playbook_id),
    )
    s.conn.execute(
        "UPDATE user_playbooks SET status = 'merged', retired_at = ? "
        "WHERE user_playbook_id = ?",
        (recent_eligible_epoch, newer_up.user_playbook_id),
    )
    s.conn.commit()

    cutoff = _now() - 50  # both tombstones are older than 50 s

    held_id = str(older_up.user_playbook_id)

    def _hold_oldest(org_id: str, entity_type: str, entity_id: str) -> bool:  # noqa: ARG001
        return entity_id == held_id

    with patch.object(s.__class__, "_is_on_legal_hold", side_effect=_hold_oldest):
        deleted = s.gc_expired_tombstones(
            entity_type="user_playbook",
            older_than_epoch=cutoff,
            limit=1,  # small limit: only 1 non-held row can be GC'd per pass
        )

    assert deleted == 1, f"Expected 1 deletion (the non-held row), got {deleted}"
    # Held (older) row must still exist.
    still_there = s.conn.execute(
        "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?",
        (older_up.user_playbook_id,),
    ).fetchone()
    assert still_there is not None, "Held row was incorrectly deleted"
    # Non-held (newer) row must be gone.
    gone = s.conn.execute(
        "SELECT 1 FROM user_playbooks WHERE user_playbook_id = ?",
        (newer_up.user_playbook_id,),
    ).fetchone()
    assert gone is None, "Non-held eligible row was NOT deleted"


# ---------------------------------------------------------------------------
# Schema guard: retired_at column exists on all three tables
# ---------------------------------------------------------------------------


def test_retired_at_column_exists_on_all_tombstone_tables(tmp_path) -> None:
    s = _store(tmp_path, "org-schema-guard")
    for table in ("profiles", "user_playbooks", "agent_playbooks"):
        cols = {
            row["name"]
            for row in s.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        assert "retired_at" in cols, f"retired_at column missing from {table}"
