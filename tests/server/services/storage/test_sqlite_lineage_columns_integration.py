import pytest
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.models.api_schema.domain.entities import UserPlaybook

pytestmark = pytest.mark.integration


def test_user_playbook_pointers_roundtrip(tmp_path):
    s = SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    pb = UserPlaybook(user_id="u1", agent_version="v1", request_id="r1",
                      content="c", merged_into=42, superseded_by=None)
    s.save_user_playbooks([pb])
    got = s.get_user_playbook_by_id(pb.user_playbook_id)
    assert got is not None and got.merged_into == 42 and got.superseded_by is None
