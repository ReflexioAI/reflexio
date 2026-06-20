"""Integration tests: status_change lineage events on archive + bulk status-flip methods.

Phase B1 / Task 3: emit op=status_change per affected id, atomically with the UPDATE,
for the following methods:
  - archive_agent_playbooks_by_ids
  - archive_agent_playbooks_by_playbook_name
  - update_all_user_playbooks_status
  - update_all_profiles_status
  - archive_profile_by_id
"""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.service_schemas import PlaybookStatus
from reflexio.server.services.storage.error import StorageError
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
# archive_agent_playbooks_by_ids
# --------------------------------------------------------------------------


def test_archive_agent_playbooks_by_ids_emits_status_change(tmp_path):
    s = _store(tmp_path)
    ap = _make_agent_playbook()
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_ids([apid])
    events = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert len(events) == 1


def test_archive_agent_playbooks_by_ids_emits_one_event_per_id(tmp_path):
    s = _store(tmp_path)
    ap1 = _make_agent_playbook(playbook_name="pb1")
    ap2 = _make_agent_playbook(playbook_name="pb2")
    saved1 = s.save_agent_playbooks([ap1])
    saved2 = s.save_agent_playbooks([ap2])
    id1 = saved1[0].agent_playbook_id
    id2 = saved2[0].agent_playbook_id
    s.archive_agent_playbooks_by_ids([id1, id2])
    for apid in [id1, id2]:
        events = [
            e
            for e in s.get_lineage_events(entity_id=str(apid))
            if e.op == "status_change"
        ]
        assert len(events) == 1, (
            f"expected 1 status_change for {apid}, got {len(events)}"
        )


def test_archive_agent_playbooks_by_ids_skips_approved(tmp_path):
    """APPROVED playbooks must not be archived (existing guard) — no status_change event emitted."""
    s = _store(tmp_path)
    ap = AgentPlaybook(
        playbook_name="approved_pb",
        agent_version="v1",
        content="c",
        playbook_status=PlaybookStatus.APPROVED,
    )
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_ids([apid])
    events = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert len(events) == 0, "APPROVED playbooks must not emit status_change on archive"


def test_archive_agent_playbooks_by_ids_reason_contains_transition(tmp_path):
    s = _store(tmp_path)
    ap = _make_agent_playbook()
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_ids([apid])
    events = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert events, "expected a status_change event"
    assert "archived" in events[0].reason


def test_archive_agent_playbooks_by_ids_per_row_reason(tmp_path):
    """Reason must reflect actual prior status: pending->archived or None->archived."""
    s = _store(tmp_path)
    # NULL-status playbook
    ap_null = _make_agent_playbook(playbook_name="null_pb")
    saved_null = s.save_agent_playbooks([ap_null])
    id_null = saved_null[0].agent_playbook_id

    # pending-status playbook (status column, not playbook_status)
    ap_pending = AgentPlaybook(
        playbook_name="pending_pb",
        agent_version="v1",
        content="c",
        status=Status.PENDING,
    )
    saved_pending = s.save_agent_playbooks([ap_pending])
    id_pending = saved_pending[0].agent_playbook_id

    s.archive_agent_playbooks_by_ids([id_null, id_pending])

    evts_null = [
        e
        for e in s.get_lineage_events(entity_id=str(id_null))
        if e.op == "status_change"
    ]
    assert len(evts_null) == 1
    assert evts_null[0].reason == "None->archived"

    evts_pending = [
        e
        for e in s.get_lineage_events(entity_id=str(id_pending))
        if e.op == "status_change"
    ]
    assert len(evts_pending) == 1
    assert evts_pending[0].reason == "pending->archived"


# --------------------------------------------------------------------------
# archive_agent_playbooks_by_playbook_name
# --------------------------------------------------------------------------


def test_archive_agent_playbooks_by_playbook_name_emits_status_change(tmp_path):
    s = _store(tmp_path)
    ap = _make_agent_playbook(playbook_name="mybook")
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_playbook_name("mybook")
    events = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert len(events) == 1


def test_archive_agent_playbooks_by_playbook_name_emits_one_event_per_id(tmp_path):
    s = _store(tmp_path)
    ap1 = _make_agent_playbook(playbook_name="shared_name", agent_version="v1")
    ap2 = _make_agent_playbook(playbook_name="shared_name", agent_version="v2")
    saved1 = s.save_agent_playbooks([ap1])
    saved2 = s.save_agent_playbooks([ap2])
    id1 = saved1[0].agent_playbook_id
    id2 = saved2[0].agent_playbook_id
    s.archive_agent_playbooks_by_playbook_name("shared_name")
    for apid in [id1, id2]:
        events = [
            e
            for e in s.get_lineage_events(entity_id=str(apid))
            if e.op == "status_change"
        ]
        assert len(events) == 1, (
            f"expected 1 status_change for {apid}, got {len(events)}"
        )


def test_archive_agent_playbooks_by_playbook_name_skips_approved(tmp_path):
    """APPROVED playbooks must not be archived via by-name method — zero status_change events."""
    s = _store(tmp_path)
    ap = AgentPlaybook(
        playbook_name="approved_pb",
        agent_version="v1",
        content="c",
        playbook_status=PlaybookStatus.APPROVED,
    )
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_playbook_name("approved_pb")
    events = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert len(events) == 0, "APPROVED playbooks must not emit status_change on archive"


def test_archive_agent_playbooks_by_playbook_name_per_row_reason(tmp_path):
    """Reason must reflect actual prior status per row."""
    s = _store(tmp_path)
    ap_null = _make_agent_playbook(playbook_name="reason_book")
    saved_null = s.save_agent_playbooks([ap_null])
    id_null = saved_null[0].agent_playbook_id

    ap_pending = AgentPlaybook(
        playbook_name="reason_book",
        agent_version="v2",
        content="c",
        status=Status.PENDING,
    )
    saved_pending = s.save_agent_playbooks([ap_pending])
    id_pending = saved_pending[0].agent_playbook_id

    s.archive_agent_playbooks_by_playbook_name("reason_book")

    evts_null = [
        e
        for e in s.get_lineage_events(entity_id=str(id_null))
        if e.op == "status_change"
    ]
    assert len(evts_null) == 1
    assert evts_null[0].reason == "None->archived"

    evts_pending = [
        e
        for e in s.get_lineage_events(entity_id=str(id_pending))
        if e.op == "status_change"
    ]
    assert len(evts_pending) == 1
    assert evts_pending[0].reason == "pending->archived"


# --------------------------------------------------------------------------
# update_all_user_playbooks_status
# --------------------------------------------------------------------------


def test_update_all_user_playbooks_status_emits_status_change(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="c",
        status=Status.PENDING,
    )
    s.save_user_playbooks([pb])
    s.update_all_user_playbooks_status(old_status=Status.PENDING, new_status=None)
    events = [
        e
        for e in s.get_lineage_events(entity_id=str(pb.user_playbook_id))
        if e.op == "status_change"
    ]
    assert len(events) == 1


def test_update_all_user_playbooks_status_emits_one_per_id(tmp_path):
    s = _store(tmp_path)
    pb1 = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r1",
        content="c",
        status=Status.PENDING,
    )
    pb2 = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r2",
        content="d",
        status=Status.PENDING,
    )
    s.save_user_playbooks([pb1, pb2])
    s.update_all_user_playbooks_status(old_status=Status.PENDING, new_status=None)
    for pb in [pb1, pb2]:
        evts = [
            e
            for e in s.get_lineage_events(entity_id=str(pb.user_playbook_id))
            if e.op == "status_change"
        ]
        assert len(evts) == 1, (
            f"expected 1 status_change for {pb.user_playbook_id}, got {len(evts)}"
        )


def test_update_all_user_playbooks_status_reason_contains_transition(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="c",
        status=Status.PENDING,
    )
    s.save_user_playbooks([pb])
    s.update_all_user_playbooks_status(old_status=Status.PENDING, new_status=None)
    evts = [
        e
        for e in s.get_lineage_events(entity_id=str(pb.user_playbook_id))
        if e.op == "status_change"
    ]
    assert evts, "expected a status_change event"
    assert "pending" in evts[0].reason.lower() or "None" in evts[0].reason


def test_update_all_user_playbooks_status_no_match_no_event(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    # playbook status is None (current), but we ask to flip PENDING -> None
    s.update_all_user_playbooks_status(old_status=Status.PENDING, new_status=None)
    events = s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    assert not any(e.op == "status_change" for e in events)


# --------------------------------------------------------------------------
# update_all_profiles_status
# --------------------------------------------------------------------------


def test_update_all_profiles_status_emits_status_change(tmp_path):
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="psc1")
    s.add_user_profile("u1", [profile])
    s.update_all_profiles_status(old_status=None, new_status=Status.ARCHIVED)
    events = s.get_lineage_events(entity_id="psc1")
    assert any(e.op == "status_change" for e in events)


def test_update_all_profiles_status_emits_one_per_id(tmp_path):
    s = _store(tmp_path)
    p1 = _make_profile(user_id="u1", profile_id="psc2")
    p2 = _make_profile(user_id="u1", profile_id="psc3")
    s.add_user_profile("u1", [p1, p2])
    s.update_all_profiles_status(old_status=None, new_status=Status.ARCHIVED)
    for pid in ["psc2", "psc3"]:
        evts = [
            e for e in s.get_lineage_events(entity_id=pid) if e.op == "status_change"
        ]
        assert len(evts) == 1, f"expected 1 status_change for {pid}, got {len(evts)}"


def test_update_all_profiles_status_with_user_id_filter(tmp_path):
    s = _store(tmp_path)
    p_u1 = _make_profile(user_id="ua", profile_id="pua")
    p_u2 = _make_profile(user_id="ub", profile_id="pub")
    s.add_user_profile("ua", [p_u1])
    s.add_user_profile("ub", [p_u2])
    s.update_all_profiles_status(
        old_status=None, new_status=Status.ARCHIVED, user_ids=["ua"]
    )
    assert any(e.op == "status_change" for e in s.get_lineage_events(entity_id="pua"))
    assert not any(
        e.op == "status_change" for e in s.get_lineage_events(entity_id="pub")
    )


# --------------------------------------------------------------------------
# archive_profile_by_id
# --------------------------------------------------------------------------


def test_archive_profile_by_id_emits_status_change(tmp_path):
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="arc_p1")
    s.add_user_profile("u1", [profile])
    result = s.archive_profile_by_id("u1", "arc_p1")
    assert result is True
    events = s.get_lineage_events(entity_id="arc_p1")
    assert any(e.op == "status_change" for e in events)


def test_archive_profile_by_id_already_archived_no_event(tmp_path):
    """Already-archived (status != NULL) profiles must not emit a second event."""
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="arc_p2")
    profile.status = Status.ARCHIVED
    s.add_user_profile("u1", [profile])
    # Guard in method: status IS NULL required — so this returns False and emits nothing.
    result = s.archive_profile_by_id("u1", "arc_p2")
    assert result is False
    events = [
        e for e in s.get_lineage_events(entity_id="arc_p2") if e.op == "status_change"
    ]
    assert len(events) == 0


def test_archive_profile_by_id_reason_contains_transition(tmp_path):
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="arc_p3")
    s.add_user_profile("u1", [profile])
    s.archive_profile_by_id("u1", "arc_p3")
    evts = [
        e for e in s.get_lineage_events(entity_id="arc_p3") if e.op == "status_change"
    ]
    assert evts
    assert "archived" in evts[0].reason.lower()


# --------------------------------------------------------------------------
# F003 negative: update_agent_playbook_status on nonexistent id emits NO event
# --------------------------------------------------------------------------


def test_update_agent_playbook_status_nonexistent_raises_no_event(tmp_path):
    """A nonexistent agent_playbook_id must raise ValueError and emit no status_change."""
    s = _store(tmp_path)
    with pytest.raises(StorageError):
        s.update_agent_playbook_status(99999, PlaybookStatus.APPROVED)
    events = s.get_lineage_events(entity_id="99999")
    assert not any(e.op == "status_change" for e in events)


# --------------------------------------------------------------------------
# F017: update_agent_playbook_status reason records actual transition
# --------------------------------------------------------------------------


def test_update_agent_playbook_status_reason_is_transition(tmp_path):
    """reason field must record old->new playbook_status, not a generic string."""
    s = _store(tmp_path)
    ap = _make_agent_playbook()
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.update_agent_playbook_status(apid, PlaybookStatus.APPROVED)
    evts = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert evts
    # Should be something like "pending->approved" or "None->approved"
    assert "approved" in evts[0].reason.lower()
    assert "->" in evts[0].reason


# --------------------------------------------------------------------------
# Structured status fields: from_status / to_status / status_namespace
# --------------------------------------------------------------------------


def test_archive_agent_playbooks_by_ids_structured_fields_null_prior(tmp_path):
    """NULL-status agent_playbook archive yields from_status=None, to_status='archived', ns='lifecycle_status'."""
    s = _store(tmp_path)
    ap = _make_agent_playbook()
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_ids([apid])
    evts = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert len(evts) == 1
    assert evts[0].from_status is None
    assert evts[0].to_status == "archived"
    assert evts[0].status_namespace == "lifecycle_status"


def test_archive_agent_playbooks_by_ids_structured_fields_pending_prior(tmp_path):
    """PENDING-status agent_playbook archive yields from_status='pending', to_status='archived'."""
    s = _store(tmp_path)
    ap = AgentPlaybook(
        playbook_name="pb_pending",
        agent_version="v1",
        content="c",
        status=Status.PENDING,
    )
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_ids([apid])
    evts = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert len(evts) == 1
    assert evts[0].from_status == "pending"
    assert evts[0].to_status == "archived"
    assert evts[0].status_namespace == "lifecycle_status"


def test_archive_agent_playbooks_by_playbook_name_structured_fields(tmp_path):
    """archive_agent_playbooks_by_playbook_name carries structured status fields."""
    s = _store(tmp_path)
    ap = _make_agent_playbook(playbook_name="struct_book")
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_playbook_name("struct_book")
    evts = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert len(evts) == 1
    assert evts[0].from_status is None
    assert evts[0].to_status == "archived"
    assert evts[0].status_namespace == "lifecycle_status"


def test_update_all_user_playbooks_status_structured_fields(tmp_path):
    """update_all_user_playbooks_status carries from_status/to_status/status_namespace."""
    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="c",
        status=Status.PENDING,
    )
    s.save_user_playbooks([pb])
    s.update_all_user_playbooks_status(old_status=Status.PENDING, new_status=None)
    evts = [
        e
        for e in s.get_lineage_events(entity_id=str(pb.user_playbook_id))
        if e.op == "status_change"
    ]
    assert len(evts) == 1
    assert evts[0].from_status == "pending"
    assert evts[0].to_status is None
    assert evts[0].status_namespace == "lifecycle_status"


def test_update_all_user_playbooks_status_null_prior_structured_fields(tmp_path):
    """update_all_user_playbooks_status with NULL->ARCHIVED carries from_status=None."""
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.update_all_user_playbooks_status(old_status=None, new_status=Status.ARCHIVED)
    evts = [
        e
        for e in s.get_lineage_events(entity_id=str(pb.user_playbook_id))
        if e.op == "status_change"
    ]
    assert len(evts) == 1
    assert evts[0].from_status is None
    assert evts[0].to_status == "archived"
    assert evts[0].status_namespace == "lifecycle_status"


def test_update_all_profiles_status_structured_fields(tmp_path):
    """update_all_profiles_status carries from_status/to_status/status_namespace."""
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="struct_p1")
    s.add_user_profile("u1", [profile])
    s.update_all_profiles_status(old_status=None, new_status=Status.ARCHIVED)
    evts = [
        e for e in s.get_lineage_events(entity_id="struct_p1") if e.op == "status_change"
    ]
    assert len(evts) == 1
    assert evts[0].from_status is None
    assert evts[0].to_status == "archived"
    assert evts[0].status_namespace == "lifecycle_status"


def test_archive_profile_by_id_structured_fields(tmp_path):
    """archive_profile_by_id always has NULL prior (guard is status IS NULL)."""
    s = _store(tmp_path)
    profile = _make_profile(user_id="u1", profile_id="struct_p2")
    s.add_user_profile("u1", [profile])
    s.archive_profile_by_id("u1", "struct_p2")
    evts = [
        e for e in s.get_lineage_events(entity_id="struct_p2") if e.op == "status_change"
    ]
    assert len(evts) == 1
    assert evts[0].from_status is None
    assert evts[0].to_status == "archived"
    assert evts[0].status_namespace == "lifecycle_status"


def test_update_agent_playbook_status_structured_fields_pending_to_approved(tmp_path):
    """update_agent_playbook_status: pending->approved yields structured fields with playbook_status ns."""
    s = _store(tmp_path)
    ap = _make_agent_playbook()
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    # Default playbook_status is 'pending'
    s.update_agent_playbook_status(apid, PlaybookStatus.APPROVED)
    evts = [
        e for e in s.get_lineage_events(entity_id=str(apid)) if e.op == "status_change"
    ]
    assert len(evts) == 1
    assert evts[0].from_status == "pending"
    assert evts[0].to_status == "approved"
    assert evts[0].status_namespace == "playbook_status"
