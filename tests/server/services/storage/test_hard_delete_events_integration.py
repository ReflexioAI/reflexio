import pytest
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.models.api_schema.domain.entities import UserPlaybook

pytestmark = pytest.mark.integration


def test_hard_delete_emits_event_then_removes_row(tmp_path):
    s = SQLiteStorage(org_id="org-test", db_path=str(tmp_path / "t.db"))
    s.migrate()
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    s.delete_user_playbooks_by_ids([pb.user_playbook_id])
    assert s.get_user_playbook_by_id(pb.user_playbook_id, include_tombstones=True) is None
    events = s.get_lineage_events(entity_id=str(pb.user_playbook_id))
    assert any(e.op == "hard_delete" for e in events)
