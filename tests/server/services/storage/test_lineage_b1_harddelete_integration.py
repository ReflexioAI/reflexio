"""Integration tests: hard_delete lineage events on all remaining physical-delete methods.

Phase A already covered delete_user_playbooks_by_ids / delete_agent_playbooks_by_ids.
This file covers the net-new methods added in Phase B1 / Task 2:
  - Single-row: delete_user_playbook, delete_agent_playbook, delete_user_profile
  - Bulk by-ids: delete_profiles_by_ids (+ emit_hard_delete=False skip)
  - Bulk GDPR/org-wipe: delete_all_profiles_for_user, delete_all_profiles,
      delete_all_user_playbooks, delete_all_agent_playbooks,
      delete_all_user_playbooks_by_playbook_name, delete_all_agent_playbooks_by_playbook_name,
      delete_archived_agent_playbooks_by_playbook_name, delete_all_profiles_by_status
  - Carve-out: delete_all_user_playbooks_by_status(PENDING) emits NO events
"""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.service_schemas import DeleteUserProfileRequest
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _store(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    return s


def _make_profile(
    user_id: str = "u1", profile_id: str = "p1", content: str = "c"
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=f"req_{profile_id}",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _make_agent_playbook(
    playbook_name: str = "pb", agent_version: str = "v1"
) -> AgentPlaybook:
    return AgentPlaybook(
        playbook_name=playbook_name, agent_version=agent_version, content="c"
    )


# --------------------------------------------------------------------------
# Single-row deletes
# --------------------------------------------------------------------------


def test_delete_user_playbook_emits_hard_delete(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.delete_user_playbook(pb.user_playbook_id)
    assert (
        s.get_user_playbook_by_id(pb.user_playbook_id, include_tombstones=True) is None
    )
    assert any(
        e.op == "hard_delete"
        for e in s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    )


def test_delete_agent_playbook_emits_hard_delete(tmp_path):
    s = _store(tmp_path)
    ap = _make_agent_playbook()
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.delete_agent_playbook(apid)
    assert s.get_agent_playbook_by_id(apid, include_tombstones=True) is None
    assert any(e.op == "hard_delete" for e in s.get_lineage_events(entity_id=str(apid)))


def test_delete_user_profile_emits_hard_delete(tmp_path):
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="pr1")
    s.add_user_profile("u1", [profile])
    s.delete_user_profile(DeleteUserProfileRequest(user_id="u1", profile_id="pr1"))
    assert any(e.op == "hard_delete" for e in s.get_lineage_events(entity_id="pr1"))


# --------------------------------------------------------------------------
# Bulk by-ids: delete_profiles_by_ids
# --------------------------------------------------------------------------


def test_delete_profiles_by_ids_emits_hard_delete(tmp_path):
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="pr2")
    s.add_user_profile("u1", [profile])
    s.delete_profiles_by_ids(["pr2"])
    assert any(e.op == "hard_delete" for e in s.get_lineage_events(entity_id="pr2"))


def test_delete_profiles_by_ids_multiple_emits_one_event_per_id(tmp_path):
    s = _store(tmp_path)
    p1 = _make_profile(user_id="u1", profile_id="pa")
    p2 = _make_profile(user_id="u1", profile_id="pb")
    s.add_user_profile("u1", [p1, p2])
    s.delete_profiles_by_ids(["pa", "pb"])
    events_a = [
        e for e in s.get_lineage_events(entity_id="pa") if e.op == "hard_delete"
    ]
    events_b = [
        e for e in s.get_lineage_events(entity_id="pb") if e.op == "hard_delete"
    ]
    assert len(events_a) == 1
    assert len(events_b) == 1


def test_delete_profiles_by_ids_emit_false_skips_event(tmp_path):
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="pr3")
    s.add_user_profile("u1", [profile])
    s.delete_profiles_by_ids(["pr3"], emit_hard_delete=False)
    assert not any(e.op == "hard_delete" for e in s.get_lineage_events(entity_id="pr3"))


# --------------------------------------------------------------------------
# Bulk GDPR/org-wipe paths
# --------------------------------------------------------------------------


def test_delete_all_profiles_for_user_emits_hard_delete_per_id(tmp_path):
    s = _store(tmp_path)
    p1 = _make_profile(user_id="u2", profile_id="u2p1")
    p2 = _make_profile(user_id="u2", profile_id="u2p2")
    s.add_user_profile("u2", [p1, p2])
    s.delete_all_profiles_for_user("u2")
    for pid in ["u2p1", "u2p2"]:
        assert any(
            e.op == "hard_delete" for e in s.get_lineage_events(entity_id=pid)
        ), f"no hard_delete for {pid}"


def test_delete_all_profiles_emits_hard_delete_per_id(tmp_path):
    s = _store(tmp_path)
    p1 = _make_profile(user_id="u1", profile_id="all1")
    p2 = _make_profile(user_id="u2", profile_id="all2")
    s.add_user_profile("u1", [p1])
    s.add_user_profile("u2", [p2])
    s.delete_all_profiles()
    for pid in ["all1", "all2"]:
        assert any(
            e.op == "hard_delete" for e in s.get_lineage_events(entity_id=pid)
        ), f"no hard_delete for {pid}"


def test_delete_all_profiles_by_status_emits_hard_delete_per_id(tmp_path):
    s = _store(tmp_path)
    p1 = _make_profile(user_id="u1", profile_id="arc1")
    p2 = _make_profile(user_id="u1", profile_id="arc2")
    p1.status = Status.ARCHIVED
    p2.status = Status.ARCHIVED
    s.add_user_profile("u1", [p1, p2])
    s.delete_all_profiles_by_status(Status.ARCHIVED)
    events_1 = [
        e for e in s.get_lineage_events(entity_id="arc1") if e.op == "hard_delete"
    ]
    events_2 = [
        e for e in s.get_lineage_events(entity_id="arc2") if e.op == "hard_delete"
    ]
    assert len(events_1) == 1
    assert len(events_2) == 1


def test_delete_all_user_playbooks_emits_hard_delete_per_id(tmp_path):
    s = _store(tmp_path)
    pb1 = UserPlaybook(user_id="u", agent_version="v", request_id="r1", content="c")
    pb2 = UserPlaybook(user_id="u", agent_version="v", request_id="r2", content="d")
    s.save_user_playbooks([pb1, pb2])
    s.delete_all_user_playbooks()
    for pbid in [pb1.user_playbook_id, pb2.user_playbook_id]:
        assert any(
            e.op == "hard_delete" for e in s.get_lineage_events(entity_id=str(pbid))
        ), f"no hard_delete for {pbid}"


def test_delete_all_agent_playbooks_emits_hard_delete_per_id(tmp_path):
    s = _store(tmp_path)
    ap1 = _make_agent_playbook(playbook_name="ap1")
    ap2 = _make_agent_playbook(playbook_name="ap2")
    saved1 = s.save_agent_playbooks([ap1])
    saved2 = s.save_agent_playbooks([ap2])
    apid1 = saved1[0].agent_playbook_id
    apid2 = saved2[0].agent_playbook_id
    s.delete_all_agent_playbooks()
    for apid in [apid1, apid2]:
        assert any(
            e.op == "hard_delete" for e in s.get_lineage_events(entity_id=str(apid))
        ), f"no hard_delete for {apid}"


def test_delete_all_user_playbooks_by_playbook_name_emits_hard_delete(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="c",
        playbook_name="mybook",
    )
    s.save_user_playbooks([pb])
    s.delete_all_user_playbooks_by_playbook_name("mybook")
    assert any(
        e.op == "hard_delete"
        for e in s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    )


def test_delete_all_agent_playbooks_by_playbook_name_emits_hard_delete(tmp_path):
    s = _store(tmp_path)
    ap = _make_agent_playbook(playbook_name="agentbook")
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.delete_all_agent_playbooks_by_playbook_name("agentbook")
    assert any(e.op == "hard_delete" for e in s.get_lineage_events(entity_id=str(apid)))


def test_delete_archived_agent_playbooks_by_playbook_name_emits_hard_delete(tmp_path):
    s = _store(tmp_path)
    ap = AgentPlaybook(
        playbook_name="archbook",
        agent_version="v1",
        content="c",
        status=Status.ARCHIVED,
    )
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.delete_archived_agent_playbooks_by_playbook_name("archbook")
    assert any(e.op == "hard_delete" for e in s.get_lineage_events(entity_id=str(apid)))


# --------------------------------------------------------------------------
# Carve-out: PENDING purge emits NO events
# --------------------------------------------------------------------------


def test_delete_all_user_playbooks_by_status_pending_no_events(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="c",
        status=Status.PENDING,
    )
    s.save_user_playbooks([pb])
    s.delete_all_user_playbooks_by_status(Status.PENDING)
    # PENDING purge must NOT emit hard_delete events (ephemeral scratch carve-out)
    events = s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    assert not any(e.op == "hard_delete" for e in events)


# --------------------------------------------------------------------------
# F002 negative: deleting nonexistent id emits NO hard_delete event
# --------------------------------------------------------------------------


def test_delete_user_playbook_nonexistent_no_event(tmp_path):
    s = _store(tmp_path)
    s.delete_user_playbook(99999)
    events = s.get_lineage_events(entity_id="99999")
    assert not any(e.op == "hard_delete" for e in events)


def test_delete_agent_playbook_nonexistent_no_event(tmp_path):
    s = _store(tmp_path)
    s.delete_agent_playbook(99999)
    events = s.get_lineage_events(entity_id="99999")
    assert not any(e.op == "hard_delete" for e in events)


def test_delete_user_profile_cross_user_no_event_and_no_delete(tmp_path):
    """delete_user_profile for a profile_id owned by a DIFFERENT user emits NO event."""
    s = _store(tmp_path)
    # Insert profile owned by user "owner"
    profile = _make_profile(user_id="owner", profile_id="cross-pr1")
    s.add_user_profile("owner", [profile])
    # Attempt delete as a different user
    s.delete_user_profile(
        DeleteUserProfileRequest(user_id="attacker", profile_id="cross-pr1")
    )
    # Profile must still exist
    assert s.get_profile_by_id("cross-pr1") is not None
    # No hard_delete event must have been emitted
    assert not any(
        e.op == "hard_delete" for e in s.get_lineage_events(entity_id="cross-pr1")
    )


# --------------------------------------------------------------------------
# F008: delete_all_user_playbooks_by_status deletes any status, emits NO events
# (parity with Supabase backend; the upgrade flow deletes old ARCHIVED entries)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [Status.ARCHIVED, Status.MERGED])
def test_delete_all_user_playbooks_by_status_non_pending_deletes_no_events(
    tmp_path, status
):
    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="c",
        status=status,
    )
    s.save_user_playbooks([pb])
    deleted = s.delete_all_user_playbooks_by_status(status)
    # Row is physically deleted (the upgrade flow relies on this for old ARCHIVED).
    assert deleted == 1
    assert not s.get_user_playbooks(status_filter=[status])
    # Bulk delete-by-status emits no hard_delete events (Supabase parity carve-out).
    events = s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    assert not any(e.op == "hard_delete" for e in events)


# --------------------------------------------------------------------------
# F012: by-ids delete methods emit actor="system"
# --------------------------------------------------------------------------


def test_delete_user_playbooks_by_ids_actor_is_system(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.delete_user_playbooks_by_ids([pb.user_playbook_id])
    events = [
        e
        for e in s.get_lineage_events(entity_id=str(pb.user_playbook_id))
        if e.op == "hard_delete"
    ]
    assert len(events) == 1
    assert events[0].actor == "system"


def test_delete_agent_playbooks_by_ids_actor_is_system(tmp_path):
    s = _store(tmp_path)
    ap = _make_agent_playbook()
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.delete_agent_playbooks_by_ids([apid])
    events = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "hard_delete"
    ]
    assert len(events) == 1
    assert events[0].actor == "system"


def test_delete_profiles_by_ids_actor_is_system(tmp_path):
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="sys-pr1")
    s.add_user_profile("u1", [profile])
    s.delete_profiles_by_ids(["sys-pr1"])
    events = [
        e for e in s.get_lineage_events(entity_id="sys-pr1") if e.op == "hard_delete"
    ]
    assert len(events) == 1
    assert events[0].actor == "system"


# --------------------------------------------------------------------------
# Nonexistent-id bulk deletes emit NO event (filter-to-existing guard)
# --------------------------------------------------------------------------


def test_delete_profiles_by_ids_nonexistent_no_event_and_returns_zero(tmp_path):
    s = _store(tmp_path)
    deleted = s.delete_profiles_by_ids(["ghost-1", "ghost-2"])
    assert deleted == 0
    assert not any(
        e.op == "hard_delete" for e in s.get_lineage_events(entity_id="ghost-1")
    )
    assert not any(
        e.op == "hard_delete" for e in s.get_lineage_events(entity_id="ghost-2")
    )


def test_delete_profiles_by_ids_partial_only_emits_for_existing(tmp_path):
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="real-1")
    s.add_user_profile("u1", [profile])
    deleted = s.delete_profiles_by_ids(["real-1", "ghost-3"])
    assert deleted == 1
    real_events = [
        e for e in s.get_lineage_events(entity_id="real-1") if e.op == "hard_delete"
    ]
    assert len(real_events) == 1
    assert not any(
        e.op == "hard_delete" for e in s.get_lineage_events(entity_id="ghost-3")
    )


def test_delete_user_playbooks_by_ids_nonexistent_no_event(tmp_path):
    """Calling delete_user_playbooks_by_ids with a non-existent id must not emit hard_delete."""
    s = _store(tmp_path)
    s.delete_user_playbooks_by_ids([99999])
    assert not any(
        e.op == "hard_delete"
        for e in s.get_lineage_events(entity_id="99999", entity_type="user_playbook")
    )


def test_delete_user_playbooks_by_ids_partial_only_emits_for_existing(tmp_path):
    """delete_user_playbooks_by_ids emits hard_delete only for ids that actually existed."""
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    real_id = pb.user_playbook_id
    s.delete_user_playbooks_by_ids([real_id, 99999])
    real_events = [
        e
        for e in s.get_lineage_events(
            entity_id=str(real_id), entity_type="user_playbook"
        )
        if e.op == "hard_delete"
    ]
    assert len(real_events) == 1
    assert not any(
        e.op == "hard_delete"
        for e in s.get_lineage_events(entity_id="99999", entity_type="user_playbook")
    )


def test_delete_agent_playbooks_by_ids_nonexistent_no_event(tmp_path):
    """Calling delete_agent_playbooks_by_ids with a non-existent id must not emit hard_delete."""
    s = _store(tmp_path)
    s.delete_agent_playbooks_by_ids([99999])
    assert not any(
        e.op == "hard_delete"
        for e in s.get_lineage_events(entity_id="99999", entity_type="agent_playbook")
    )


def test_delete_agent_playbooks_by_ids_partial_only_emits_for_existing(tmp_path):
    """delete_agent_playbooks_by_ids emits hard_delete only for ids that actually existed."""
    s = _store(tmp_path)
    ap = _make_agent_playbook()
    saved = s.save_agent_playbooks([ap])
    real_id = saved[0].agent_playbook_id
    s.delete_agent_playbooks_by_ids([real_id, 99999])
    real_events = [
        e
        for e in s.get_lineage_events(
            entity_id=str(real_id), entity_type="agent_playbook"
        )
        if e.op == "hard_delete"
    ]
    assert len(real_events) == 1
    assert not any(
        e.op == "hard_delete"
        for e in s.get_lineage_events(entity_id="99999", entity_type="agent_playbook")
    )


# --------------------------------------------------------------------------
# Bulk deletes clean up the vec table (when sqlite-vec is loaded)
# --------------------------------------------------------------------------


def _vec_rowids(s, table: str) -> set[int]:
    return {
        row["rowid"]
        for row in s.conn.execute(f"SELECT rowid FROM {table}").fetchall()  # noqa: S608
    }


def test_delete_profiles_by_ids_cleans_vec(tmp_path):
    s = _store(tmp_path)
    if not s._has_sqlite_vec:
        pytest.skip("sqlite-vec extension not loaded")
    profile = _make_profile(user_id="u1", profile_id="vec-pr1")
    s.add_user_profile("u1", [profile])
    rowid_row = s.conn.execute(
        "SELECT rowid FROM profiles WHERE profile_id = ?", ("vec-pr1",)
    ).fetchone()
    # Seed a vec row keyed on the profile's implicit sqlite rowid.
    s._vec_upsert("profiles_vec", rowid_row["rowid"], [0.1] * s.embedding_dimensions)
    assert rowid_row["rowid"] in _vec_rowids(s, "profiles_vec")
    s.delete_profiles_by_ids(["vec-pr1"])
    assert rowid_row["rowid"] not in _vec_rowids(s, "profiles_vec")


def test_delete_all_user_playbooks_cleans_vec(tmp_path):
    s = _store(tmp_path)
    if not s._has_sqlite_vec:
        pytest.skip("sqlite-vec extension not loaded")
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s._vec_upsert(
        "user_playbooks_vec", pb.user_playbook_id, [0.2] * s.embedding_dimensions
    )
    assert pb.user_playbook_id in _vec_rowids(s, "user_playbooks_vec")
    s.delete_all_user_playbooks()
    assert pb.user_playbook_id not in _vec_rowids(s, "user_playbooks_vec")


def test_delete_agent_playbooks_by_ids_cleans_vec(tmp_path):
    s = _store(tmp_path)
    if not s._has_sqlite_vec:
        pytest.skip("sqlite-vec extension not loaded")
    saved = s.save_agent_playbooks([_make_agent_playbook()])
    apid = saved[0].agent_playbook_id
    s._vec_upsert("agent_playbooks_vec", apid, [0.3] * s.embedding_dimensions)
    assert apid in _vec_rowids(s, "agent_playbooks_vec")
    s.delete_agent_playbooks_by_ids([apid])
    assert apid not in _vec_rowids(s, "agent_playbooks_vec")


# --------------------------------------------------------------------------
# No-phantom-event: delete_all_agent_playbooks
# Regression test for Finding 1: emit must happen AFTER DELETE (same commit),
# not before. We verify that:
#   - events emitted == rows that actually existed
#   - calling on an empty table emits zero events
#   - rows are gone after the call
# --------------------------------------------------------------------------


def test_delete_all_agent_playbooks_no_phantom_events_empty_table(tmp_path):
    """delete_all_agent_playbooks on empty table emits zero hard_delete events."""
    s = _store(tmp_path)
    s.delete_all_agent_playbooks()
    # No rows existed; no events should have been written.
    all_events = s.get_lineage_events()
    hard_deletes = [e for e in all_events if e.op == "hard_delete"]
    assert hard_deletes == []


def test_delete_all_agent_playbooks_emits_exactly_for_existing_rows(tmp_path):
    """delete_all_agent_playbooks emits one hard_delete per existing row, no more."""
    s = _store(tmp_path)
    ap1 = _make_agent_playbook(playbook_name="x")
    ap2 = _make_agent_playbook(playbook_name="y")
    saved1 = s.save_agent_playbooks([ap1])
    saved2 = s.save_agent_playbooks([ap2])
    apid1 = saved1[0].agent_playbook_id
    apid2 = saved2[0].agent_playbook_id

    s.delete_all_agent_playbooks()

    # Rows are gone.
    assert s.get_agent_playbook_by_id(apid1, include_tombstones=True) is None
    assert s.get_agent_playbook_by_id(apid2, include_tombstones=True) is None

    # Exactly one hard_delete event per id — no phantom events for extra ids.
    for apid in [apid1, apid2]:
        events = [
            e
            for e in s.get_lineage_events(entity_id=str(apid))
            if e.op == "hard_delete"
        ]
        assert len(events) == 1, f"expected 1 hard_delete for {apid}, got {len(events)}"

    # Total hard_delete count matches exactly the two rows that existed.
    all_hard_deletes = [e for e in s.get_lineage_events() if e.op == "hard_delete"]
    assert len(all_hard_deletes) == 2
