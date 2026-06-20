"""Integration tests: in-place update_* methods emit lineage events atomically.

Phase B1 — Task 1: verifies that each update_* method emits exactly one
lineage event (op=revise when content changes, op=status_change otherwise).
"""

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import PlaybookStatus
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _store(tmp_path: pytest.TempPathFactory) -> SQLiteStorage:
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
    ev = s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    assert [e.op for e in ev] == ["revise"]
    assert s.get_user_playbook_by_id(pb.user_playbook_id).content == "new guidance"


def test_update_user_playbook_metadata_only_emits_status_change(tmp_path):
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.update_user_playbook(pb.user_playbook_id, playbook_name="renamed")  # no content
    ev = s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    assert [e.op for e in ev] == ["status_change"]


def test_update_user_playbook_multiple_edits_each_produce_event(tmp_path):
    """Each call should produce a distinct event (not collapsed by idempotency key)."""
    s = _store(tmp_path)
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.update_user_playbook(pb.user_playbook_id, content="v2")
    s.update_user_playbook(pb.user_playbook_id, content="v3")
    ev = s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    assert [e.op for e in ev] == ["revise", "revise"]


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
