"""Integration tests: reconstruct_profile_change_log — read-side ProfileChangeLog.

Phase B3 / Task 2: Rebuild the ProfileChangeLog view on demand from the
content-free lineage_event log joined to live rows + tombstone frozen content.

Tested scenarios:
  - Supersede (revise event) produces added_profiles / removed_profiles matching
    what the legacy add_profile_change_log would record.
  - status_change event carries structured from_status/to_status (not freetext).
  - Purged tombstone (profile missing after GC) is handled gracefully — no crash.
  - limit bounds the number of reconstructed entries.
"""

from datetime import UTC, datetime

import pytest

from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.models.api_schema.domain.entities import LineageContext, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _store(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    return s


def _make_profile(
    user_id: str = "u1",
    profile_id: str = "p1",
    content: str = "hello world",
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


# --------------------------------------------------------------------------
# Helper: seed a supersede exactly as the reflection service would
# --------------------------------------------------------------------------


def _seed_supersede(
    s: SQLiteStorage,
    *,
    user_id: str,
    old_id: str,
    new_id: str,
    request_id: str,
    old_content: str = "old content",
    new_content: str = "new content",
) -> tuple[UserProfile, UserProfile]:
    """Add old and new profiles, then call supersede_record.

    Returns (old_profile, new_profile) as they were constructed.
    """
    old = _make_profile(user_id=user_id, profile_id=old_id, content=old_content)
    new = _make_profile(user_id=user_id, profile_id=new_id, content=new_content)
    s.add_user_profile(user_id, [old])
    s.add_user_profile(user_id, [new])
    ctx = LineageContext(op_kind="revise", actor="reflection", request_id=request_id)
    s.supersede_record(
        entity_type="profile",
        incumbent_id=old_id,
        successor_id=new_id,
        context=ctx,
    )
    return old, new


# --------------------------------------------------------------------------
# Core parity test: supersede -> reconstruct matches legacy shape
# --------------------------------------------------------------------------


def test_supersede_produces_one_changelog_row(tmp_path):
    """A single supersede produces exactly one reconstructed change-log entry."""
    s = _store(tmp_path)
    _seed_supersede(s, user_id="u1", old_id="p-9", new_id="p-17", request_id="req-abc")
    result = reconstruct_profile_change_log(s)
    assert result.success
    assert len(result.profile_change_logs) == 1


def test_supersede_added_profiles_is_successor(tmp_path):
    """added_profiles contains the successor (new) profile."""
    s = _store(tmp_path)
    old_p, new_p = _seed_supersede(
        s,
        user_id="u1",
        old_id="p-9",
        new_id="p-17",
        request_id="req-abc",
        new_content="the new content",
    )
    result = reconstruct_profile_change_log(s)
    row = result.profile_change_logs[0]
    assert len(row.added_profiles) == 1
    assert row.added_profiles[0].profile_id == "p-17"
    assert row.added_profiles[0].content == "the new content"


def test_supersede_removed_profiles_is_incumbent_tombstone(tmp_path):
    """removed_profiles contains the incumbent (old, tombstoned) profile."""
    s = _store(tmp_path)
    old_p, new_p = _seed_supersede(
        s,
        user_id="u1",
        old_id="p-9",
        new_id="p-17",
        request_id="req-abc",
        old_content="the old content",
    )
    result = reconstruct_profile_change_log(s)
    row = result.profile_change_logs[0]
    assert len(row.removed_profiles) == 1
    assert row.removed_profiles[0].profile_id == "p-9"
    assert row.removed_profiles[0].content == "the old content"


def test_mentioned_profiles_is_always_empty(tmp_path):
    """mentioned_profiles must be [] — Stage-1 keeps field present but empty."""
    s = _store(tmp_path)
    _seed_supersede(s, user_id="u1", old_id="p-9", new_id="p-17", request_id="req-abc")
    result = reconstruct_profile_change_log(s)
    row = result.profile_change_logs[0]
    assert row.mentioned_profiles == []


def test_request_id_matches_context(tmp_path):
    """request_id on the reconstructed row must equal the supersede's request_id."""
    s = _store(tmp_path)
    _seed_supersede(s, user_id="u1", old_id="p-9", new_id="p-17", request_id="req-xyz")
    result = reconstruct_profile_change_log(s)
    row = result.profile_change_logs[0]
    assert row.request_id == "req-xyz"


def test_parity_with_legacy_shape(tmp_path):
    """Reconstructed row matches what legacy add_profile_change_log would record.

    The legacy caller sets:
      added_profiles=[new_profile], removed_profiles=[old_profile], mentioned_profiles=[]
    This test asserts field-by-field parity on content, user_id, profile_id.
    """
    s = _store(tmp_path)
    old_p, new_p = _seed_supersede(
        s,
        user_id="u1",
        old_id="p-9",
        new_id="p-17",
        request_id="req-parity",
        old_content="old facts",
        new_content="new facts",
    )
    result = reconstruct_profile_change_log(s)
    row = result.profile_change_logs[0]

    # added
    assert len(row.added_profiles) == 1
    a = row.added_profiles[0]
    assert a.profile_id == new_p.profile_id
    assert a.user_id == new_p.user_id
    assert a.content == new_p.content

    # removed
    assert len(row.removed_profiles) == 1
    r = row.removed_profiles[0]
    assert r.profile_id == old_p.profile_id
    assert r.user_id == old_p.user_id
    assert r.content == old_p.content

    # mentioned always empty
    assert row.mentioned_profiles == []


# --------------------------------------------------------------------------
# status_change: reconstructs structured columns, NOT freetext reason
# --------------------------------------------------------------------------


def test_status_change_event_reads_structured_columns(tmp_path):
    """status_change events expose from_status/to_status from structured columns.

    The reconstruct function must NOT parse the freetext ``reason`` field.
    We verify by asserting the event's from_status/to_status match what was stored.
    """
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="psc-1")
    s.add_user_profile("u1", [profile])
    s.archive_profile_by_id("u1", "psc-1")

    # get_lineage_events returns structured columns
    events = s.get_lineage_events(entity_id="psc-1")
    sc_events = [e for e in events if e.op == "status_change"]
    assert sc_events, "expected a status_change event"
    evt = sc_events[0]
    # Must have structured fields, not just freetext
    assert evt.from_status is None
    assert evt.to_status == "archived"
    assert evt.status_namespace == "lifecycle_status"


def test_status_change_only_no_supersede_produces_no_changelog_row(tmp_path):
    """A status_change with no accompanying supersede produces no change-log row.

    Legacy semantics: add_profile_change_log was called only when
    all_new_profiles or superseded_profiles, not on pure status flips.
    """
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="psc-only")
    s.add_user_profile("u1", [profile])
    s.archive_profile_by_id("u1", "psc-only")

    result = reconstruct_profile_change_log(s)
    # No revise/merge events exist, so the change-log must be empty.
    assert result.profile_change_logs == []


# --------------------------------------------------------------------------
# Purged tombstone: graceful handling
# --------------------------------------------------------------------------


def test_purged_tombstone_no_crash(tmp_path):
    """When the incumbent tombstone has been GC'd, reconstruct must not crash.

    After GC the profile row is physically deleted. get_profile_by_id returns
    None. reconstruct must handle this gracefully and omit the removed profile
    (or include an empty list) rather than raising.
    """
    s = _store(tmp_path)
    old_p, new_p = _seed_supersede(
        s,
        user_id="u1",
        old_id="p-old-gc",
        new_id="p-new-gc",
        request_id="req-gc",
    )
    # Hard-delete the tombstone by simulating GC (direct SQL delete after supersede set status=SUPERSEDED)
    s.conn.execute("DELETE FROM profiles WHERE profile_id = ?", ("p-old-gc",))
    s.conn.commit()

    # Must not raise; removed_profiles is empty (GDPR blank) for the missing tombstone.
    result = reconstruct_profile_change_log(s)
    assert result.success
    assert len(result.profile_change_logs) == 1
    row = result.profile_change_logs[0]
    # removed_profiles is empty when the tombstone body was purged
    assert row.removed_profiles == []
    # added_profiles still has the survivor
    assert len(row.added_profiles) == 1
    assert row.added_profiles[0].profile_id == "p-new-gc"


# --------------------------------------------------------------------------
# limit: bounds the number of reconstructed entries
# --------------------------------------------------------------------------


def test_limit_bounds_reconstructed_entries(tmp_path):
    """limit=N returns at most N reconstructed change-log entries."""
    s = _store(tmp_path)
    for i in range(5):
        _seed_supersede(
            s,
            user_id="u1",
            old_id=f"p-old-{i}",
            new_id=f"p-new-{i}",
            request_id=f"req-limit-{i}",
        )
    result = reconstruct_profile_change_log(s, limit=3)
    assert result.success
    assert len(result.profile_change_logs) == 3


def test_limit_default_100(tmp_path):
    """Default limit is 100 — 2 supersedes return 2 rows."""
    s = _store(tmp_path)
    for i in range(2):
        _seed_supersede(
            s,
            user_id="u1",
            old_id=f"p-old-d{i}",
            new_id=f"p-new-d{i}",
            request_id=f"req-def-{i}",
        )
    result = reconstruct_profile_change_log(s)
    assert len(result.profile_change_logs) == 2


def test_limit_zero_returns_empty(tmp_path):
    """limit=0 returns an empty list."""
    s = _store(tmp_path)
    _seed_supersede(s, user_id="u1", old_id="p-9", new_id="p-17", request_id="req-abc")
    result = reconstruct_profile_change_log(s, limit=0)
    assert result.success
    assert result.profile_change_logs == []


# --------------------------------------------------------------------------
# Most-recent-first ordering
# --------------------------------------------------------------------------


def test_most_recent_first_ordering(tmp_path):
    """Reconstructed rows are ordered most-recent first (by created_at of event group)."""
    s = _store(tmp_path)
    _seed_supersede(
        s,
        user_id="u1",
        old_id="p-old-0",
        new_id="p-new-0",
        request_id="req-first",
    )
    _seed_supersede(
        s,
        user_id="u1",
        old_id="p-old-1",
        new_id="p-new-1",
        request_id="req-second",
    )
    result = reconstruct_profile_change_log(s)
    assert len(result.profile_change_logs) == 2
    # Most recent event group first
    assert result.profile_change_logs[0].request_id == "req-second"
    assert result.profile_change_logs[1].request_id == "req-first"


# --------------------------------------------------------------------------
# Empty storage
# --------------------------------------------------------------------------


def test_no_events_returns_empty(tmp_path):
    """With no lineage events, returns an empty list."""
    s = _store(tmp_path)
    result = reconstruct_profile_change_log(s)
    assert result.success
    assert result.profile_change_logs == []
