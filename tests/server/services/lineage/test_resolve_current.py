import pytest

from reflexio.models.api_schema.domain.entities import LineageContext, UserPlaybook
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.lineage.resolve import resolve_current
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def test_current_resolves_to_self(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    pb = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="c")
    s.save_user_playbooks([pb])
    ref = resolve_current(s, "user_playbook", pb.user_playbook_id)
    assert (
        ref is not None
        and str(ref.id) == str(pb.user_playbook_id)
        and ref.is_purged is False
    )


def test_merged_resolves_to_survivor(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    surv = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="m")
    src = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="o")
    s.save_user_playbooks([surv, src])
    s.merge_records(
        entity_type="user_playbook",
        survivor_id=str(surv.user_playbook_id),
        source_ids=[str(src.user_playbook_id)],
        context=LineageContext(op_kind="merge", actor="t", request_id="r"),
    )
    ref = resolve_current(s, "user_playbook", src.user_playbook_id)
    assert str(ref.id) == str(surv.user_playbook_id)


def test_purged_survivor_flagged(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    pb = UserPlaybook(
        user_id="u", agent_version="v", request_id="r", content=""
    )  # blanked body
    pb.status = (
        Status.MERGED
    )  # purged tombstone that points nowhere current -> itself, is_purged
    s.save_user_playbooks([pb])
    ref = resolve_current(s, "user_playbook", pb.user_playbook_id)
    assert ref is not None and ref.is_purged is True


def test_superseded_resolves_to_successor(tmp_path):
    # F005: follow a superseded_by pointer set by supersede_record.
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    incumbent = UserPlaybook(
        user_id="u", agent_version="v", request_id="r", content="old"
    )
    successor = UserPlaybook(
        user_id="u", agent_version="v", request_id="r", content="new"
    )
    s.save_user_playbooks([incumbent, successor])
    ok = s.supersede_record(
        entity_type="user_playbook",
        incumbent_id=str(incumbent.user_playbook_id),
        successor_id=str(successor.user_playbook_id),
        context=LineageContext(op_kind="revise", actor="t", request_id="r"),
    )
    assert ok is True
    ref = resolve_current(s, "user_playbook", incumbent.user_playbook_id)
    assert ref is not None and str(ref.id) == str(successor.user_playbook_id)


def test_cycle_returns_none(tmp_path):
    s = SQLiteStorage(org_id="org-1", db_path=str(tmp_path / "t.db"))
    s.migrate()
    a = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="a",
        status=Status.MERGED,
    )
    b = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="b",
        status=Status.MERGED,
    )
    s.save_user_playbooks([a, b])
    # craft a cycle a->b->a via include-tombstone updates
    s.conn.execute(
        "UPDATE user_playbooks SET merged_into=? WHERE user_playbook_id=?",
        (b.user_playbook_id, a.user_playbook_id),
    )
    s.conn.execute(
        "UPDATE user_playbooks SET merged_into=? WHERE user_playbook_id=?",
        (a.user_playbook_id, b.user_playbook_id),
    )
    s.conn.commit()
    assert resolve_current(s, "user_playbook", a.user_playbook_id) is None
