"""Integration tests: in-place update_* methods emit lineage events atomically.

Phase B1 — Task 1: verifies that each update_* method emits exactly one
lineage event (op=revise when content changes, op=status_change otherwise).

Also covers Task 9 (structured status fields on in-place update_* status_change path).
"""

from pathlib import Path

import pytest

import reflexio.server.services.playbook.service as playbook_generation_service
from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    LineageContext,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import PlaybookStatus
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _store(tmp_path: Path) -> SQLiteStorage:
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    return s


# ---------------------------------------------------------------------------
# update_user_playbook
# ---------------------------------------------------------------------------


def test_update_user_playbook_content_emits_revise(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="old")
    s.save_user_playbooks([pb])
    s.update_user_playbook(pb.user_playbook_id, content="new guidance")
    ev = s.get_lineage_events(
        entity_id=str(pb.user_playbook_id), entity_type="user_playbook"
    )
    assert [e.op for e in ev] == ["revise"]
    assert s.get_user_playbook_by_id(pb.user_playbook_id).content == "new guidance"


def test_update_user_playbook_metadata_only_emits_status_change(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.update_user_playbook(pb.user_playbook_id, playbook_name="renamed")  # no content
    ev = s.get_lineage_events(
        entity_id=str(pb.user_playbook_id), entity_type="user_playbook"
    )
    assert [e.op for e in ev] == ["status_change"]


def test_update_user_playbook_multiple_edits_each_produce_event(tmp_path):
    """Each call should produce a distinct event (not collapsed by idempotency key)."""
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.update_user_playbook(pb.user_playbook_id, content="v2")
    s.update_user_playbook(pb.user_playbook_id, content="v3")
    ev = s.get_lineage_events(
        entity_id=str(pb.user_playbook_id), entity_type="user_playbook"
    )
    assert [e.op for e in ev] == ["revise", "revise"]


def test_update_user_playbook_trigger_change_emits_revise(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.update_user_playbook(pb.user_playbook_id, trigger="when the user asks twice")
    ev = s.get_lineage_events(
        entity_id=str(pb.user_playbook_id), entity_type="user_playbook"
    )
    assert [e.op for e in ev] == ["revise"]


def test_read_user_playbook_as_of_for_learning_rejects_post_serve_revise(tmp_path):
    helper = getattr(
        playbook_generation_service,
        "read_user_playbook_as_of_for_learning",
        None,
    )
    assert helper is not None

    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        created_at=100,
        content="v1",
    )
    s.save_user_playbooks([pb])

    s.update_user_playbook(pb.user_playbook_id, content="v2")

    got = helper(s, user_playbook_id=pb.user_playbook_id, served_at=150)
    assert got is None


def test_read_user_playbook_as_of_for_learning_rejects_same_second_revise(tmp_path):
    helper = getattr(
        playbook_generation_service,
        "read_user_playbook_as_of_for_learning",
        None,
    )
    assert helper is not None

    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        created_at=100,
        content="v1",
    )
    s.save_user_playbooks([pb])

    s.update_user_playbook(pb.user_playbook_id, content="v2")
    events = s.get_lineage_events(
        entity_type="user_playbook",
        entity_id=str(pb.user_playbook_id),
    )
    assert events[-1].created_at >= 100

    got = helper(
        s,
        user_playbook_id=pb.user_playbook_id,
        served_at=events[-1].created_at,
    )
    assert got is None


def test_read_user_playbook_as_of_for_learning_allows_metadata_only_edit(tmp_path):
    helper = getattr(
        playbook_generation_service,
        "read_user_playbook_as_of_for_learning",
        None,
    )
    assert helper is not None

    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        created_at=100,
        playbook_name="original",
        content="keep this",
        trigger="when asked about X",
        rationale="because of prior confusion",
    )
    s.save_user_playbooks([pb])

    s.update_user_playbook(pb.user_playbook_id, playbook_name="renamed")

    got = helper(s, user_playbook_id=pb.user_playbook_id, served_at=150)
    assert got is not None
    assert got.user_playbook_id == pb.user_playbook_id
    assert got.content == "keep this"


def test_read_user_playbook_as_of_for_learning_rejects_blank_purged_content(tmp_path):
    helper = getattr(
        playbook_generation_service,
        "read_user_playbook_as_of_for_learning",
        None,
    )
    assert helper is not None

    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        created_at=100,
        content="sensitive guidance",
    )
    s.save_user_playbooks([pb])
    assert s.purge_content(
        entity_type="user_playbook", entity_id=str(pb.user_playbook_id)
    )

    got = helper(s, user_playbook_id=pb.user_playbook_id, served_at=150)
    assert got is None


def test_read_user_playbook_as_of_for_learning_rejects_future_created_row(tmp_path):
    helper = getattr(
        playbook_generation_service,
        "read_user_playbook_as_of_for_learning",
        None,
    )
    assert helper is not None

    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        created_at=200,
        content="future guidance",
    )
    s.save_user_playbooks([pb])

    got = helper(s, user_playbook_id=pb.user_playbook_id, served_at=150)
    assert got is None


def test_read_user_playbook_as_of_for_learning_rejects_same_second_created_row(
    tmp_path,
):
    helper = getattr(
        playbook_generation_service,
        "read_user_playbook_as_of_for_learning",
        None,
    )
    assert helper is not None

    s = _store(tmp_path)
    pb = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        created_at=150,
        content="same-second guidance",
    )
    s.save_user_playbooks([pb])

    got = helper(s, user_playbook_id=pb.user_playbook_id, served_at=150)
    assert got is None


def test_read_user_playbook_as_of_for_learning_does_not_resolve_to_current_survivor(
    tmp_path,
):
    helper = getattr(
        playbook_generation_service,
        "read_user_playbook_as_of_for_learning",
        None,
    )
    assert helper is not None

    s = _store(tmp_path)
    incumbent = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r-old",
        created_at=100,
        content="old exact content",
    )
    successor = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r-new",
        created_at=120,
        content="new current content",
    )
    s.save_user_playbooks([incumbent, successor])
    s.supersede_record(
        entity_type="user_playbook",
        incumbent_id=str(incumbent.user_playbook_id),
        successor_id=str(successor.user_playbook_id),
        context=LineageContext(
            op_kind="revise",
            actor="test",
            request_id="req-supersede",
        ),
    )

    got = helper(s, user_playbook_id=incumbent.user_playbook_id, served_at=150)
    assert got is not None
    assert got.user_playbook_id == incumbent.user_playbook_id
    assert got.content == "old exact content"


# ---------------------------------------------------------------------------
# update_agent_playbook
# ---------------------------------------------------------------------------


def test_update_agent_playbook_content_emits_revise(tmp_path):
    s = _store(tmp_path)
    ap = AgentPlaybook(agent_version="v", content="old")
    saved = s.save_agent_playbooks([ap])
    s.update_agent_playbook(saved[0].agent_playbook_id, content="new")
    ev = s.get_lineage_events(
        entity_id=str(saved[0].agent_playbook_id), entity_type="agent_playbook"
    )
    assert [e.op for e in ev] == ["revise"]


def test_update_agent_playbook_metadata_only_emits_status_change(tmp_path):
    s = _store(tmp_path)
    ap = AgentPlaybook(agent_version="v", content="c")
    saved = s.save_agent_playbooks([ap])
    s.update_agent_playbook(saved[0].agent_playbook_id, playbook_name="renamed")
    ev = s.get_lineage_events(
        entity_id=str(saved[0].agent_playbook_id), entity_type="agent_playbook"
    )
    assert [e.op for e in ev] == ["status_change"]


# ---------------------------------------------------------------------------
# update_agent_playbook_status
# ---------------------------------------------------------------------------


def test_update_agent_playbook_status_always_emits_status_change(tmp_path):
    s = _store(tmp_path)
    ap = AgentPlaybook(agent_version="v", content="c")
    saved = s.save_agent_playbooks([ap])
    s.update_agent_playbook_status(saved[0].agent_playbook_id, PlaybookStatus.APPROVED)
    ev = s.get_lineage_events(
        entity_id=str(saved[0].agent_playbook_id), entity_type="agent_playbook"
    )
    assert [e.op for e in ev] == ["status_change"]


# ---------------------------------------------------------------------------
# update_user_profile_by_id
# ---------------------------------------------------------------------------


def test_update_user_profile_emits_revise(tmp_path):
    s = _store(tmp_path)
    profile = UserProfile(
        profile_id="prof-1",
        user_id="u",
        content="original content",
        last_modified_timestamp=0,
        generated_from_request_id="r",
    )
    s.add_user_profile("u", [profile])
    updated = profile.model_copy(update={"content": "updated content"})
    s.update_user_profile_by_id("u", str(profile.profile_id), updated)
    ev = s.get_lineage_events(entity_id=str(profile.profile_id), entity_type="profile")
    assert [e.op for e in ev] == ["revise"]
    fetched = s.get_profile_by_id(str(profile.profile_id))
    assert fetched is not None
    assert fetched.content == "updated content"


def test_update_user_profile_nonexistent_emits_no_event(tmp_path):
    """Updating a profile that no longer exists must emit no revise event."""
    s = _store(tmp_path)
    ghost = UserProfile(
        profile_id="ghost-prof",
        user_id="u",
        content="content",
        last_modified_timestamp=0,
        generated_from_request_id="r",
    )
    # Never persisted — the pre-check returns early, but assert the contract.
    s.update_user_profile_by_id("u", "ghost-prof", ghost)
    ev = s.get_lineage_events(entity_id="ghost-prof", entity_type="profile")
    assert not any(e.op == "revise" for e in ev)


# ---------------------------------------------------------------------------
# archive_agent_playbooks_* — re-archiving an archived row emits no event
# ---------------------------------------------------------------------------


def test_archive_agent_playbooks_by_ids_already_archived_no_event(tmp_path):
    s = _store(tmp_path)
    ap = AgentPlaybook(agent_version="v", content="c")
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    # First archive emits one status_change event.
    s.archive_agent_playbooks_by_ids([apid])
    first = s.get_lineage_events(entity_id=str(apid), entity_type="agent_playbook")
    assert [e.op for e in first] == ["status_change"]
    # Re-archiving the already-archived row must emit no further event.
    s.archive_agent_playbooks_by_ids([apid])
    second = s.get_lineage_events(entity_id=str(apid), entity_type="agent_playbook")
    assert [e.op for e in second] == ["status_change"]


def test_archive_agent_playbooks_by_playbook_name_already_archived_no_event(tmp_path):
    s = _store(tmp_path)
    ap = AgentPlaybook(playbook_name="arch-pb", agent_version="v", content="c")
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.archive_agent_playbooks_by_playbook_name("arch-pb")
    first = s.get_lineage_events(entity_id=str(apid), entity_type="agent_playbook")
    assert [e.op for e in first] == ["status_change"]
    s.archive_agent_playbooks_by_playbook_name("arch-pb")
    second = s.get_lineage_events(entity_id=str(apid), entity_type="agent_playbook")
    assert [e.op for e in second] == ["status_change"]


# ---------------------------------------------------------------------------
# Structured status fields: update_agent_playbook playbook_status path
# ---------------------------------------------------------------------------


def test_update_agent_playbook_playbook_status_populates_structured_fields(tmp_path):
    """update_agent_playbook(playbook_status=X) emits status_change with structured fields populated."""
    s = _store(tmp_path)
    # Default playbook_status on save is 'pending'
    ap = AgentPlaybook(agent_version="v", content="c")
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.update_agent_playbook(apid, playbook_status=PlaybookStatus.APPROVED)
    ev = s.get_lineage_events(entity_id=str(apid), entity_type="agent_playbook")
    sc = [e for e in ev if e.op == "status_change"]
    assert len(sc) == 1
    assert sc[0].to_status == "approved"
    assert sc[0].from_status == "pending"
    assert sc[0].status_namespace == "playbook_status"


def test_update_agent_playbook_metadata_only_leaves_structured_fields_null(tmp_path):
    """update_agent_playbook(playbook_name=...) (no status, no content) emits status_change with all 3 fields NULL."""
    s = _store(tmp_path)
    ap = AgentPlaybook(agent_version="v", content="c")
    saved = s.save_agent_playbooks([ap])
    apid = saved[0].agent_playbook_id
    s.update_agent_playbook(apid, playbook_name="renamed")
    ev = s.get_lineage_events(entity_id=str(apid), entity_type="agent_playbook")
    sc = [e for e in ev if e.op == "status_change"]
    assert len(sc) == 1
    assert sc[0].from_status is None
    assert sc[0].to_status is None
    assert sc[0].status_namespace is None


def test_update_user_playbook_metadata_only_leaves_structured_fields_null(tmp_path):
    """update_user_playbook(playbook_name=...) (no status, no content) emits status_change with all 3 fields NULL."""
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.update_user_playbook(pb.user_playbook_id, playbook_name="renamed")
    ev = s.get_lineage_events(
        entity_id=str(pb.user_playbook_id), entity_type="user_playbook"
    )
    sc = [e for e in ev if e.op == "status_change"]
    assert len(sc) == 1
    assert sc[0].from_status is None
    assert sc[0].to_status is None
    assert sc[0].status_namespace is None
