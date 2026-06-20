import pytest

from reflexio.models.api_schema.domain.entities import LineageContext, UserPlaybook
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def test_merge_records_tombstones_sources_sets_pointer_and_logs(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    survivor = UserPlaybook(user_id="u1", agent_version="v1", request_id="r1", content="merged")
    src = UserPlaybook(user_id="u1", agent_version="v1", request_id="r1", content="old")
    s.save_user_playbooks([survivor, src])
    ctx = LineageContext(op_kind="merge", actor="consolidator", source_ids=[str(src.user_playbook_id)],
                         reason="dup", request_id="r1")
    s.merge_records(entity_type="user_playbook", survivor_id=str(survivor.user_playbook_id),
                    source_ids=[str(src.user_playbook_id)], context=ctx)
    tomb = s.get_user_playbook_by_id(src.user_playbook_id, include_tombstones=True)
    assert tomb.status is Status.MERGED and tomb.merged_into == survivor.user_playbook_id
    events = s.get_lineage_events(entity_id=str(survivor.user_playbook_id))
    assert any(e.op == "merge" for e in events)
    # survivor still current
    assert s.get_user_playbook_by_id(survivor.user_playbook_id) is not None


def test_supersede_returns_false_when_incumbent_not_current(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    inc = UserPlaybook(user_id="u1", agent_version="v1", request_id="r1", content="v1",
                       status=Status.ARCHIVED)
    succ = UserPlaybook(user_id="u1", agent_version="v1", request_id="r1", content="v2")
    s.save_user_playbooks([inc, succ])
    ctx = LineageContext(op_kind="revise", actor="optimizer", request_id="r1")
    ok = s.supersede_record(entity_type="user_playbook", incumbent_id=str(inc.user_playbook_id),
                            successor_id=str(succ.user_playbook_id), context=ctx)
    assert ok is False  # incumbent wasn't CURRENT — atomic guard rejects
