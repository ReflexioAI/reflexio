"""Integration tests for apply_playbook_edit() — atomic supersede path."""

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.playbook.playbook_edit_apply import apply_playbook_edit
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def test_apply_supersedes_incumbent_and_links(tmp_path):
    s = SQLiteStorage(org_id="test_org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    inc = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="v1")
    s.save_user_playbooks([inc])
    new = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="v2")
    new_id = apply_playbook_edit(
        s,
        incumbent_id=inc.user_playbook_id,
        new_playbook=new,
        source="offline_optimizer",
        request_id="run-test-1",
    )
    assert new_id > 0
    tomb = s.get_user_playbook_by_id(inc.user_playbook_id, include_tombstones=True)
    assert tomb.status is Status.SUPERSEDED and tomb.superseded_by == new_id


def test_apply_no_orphan_when_incumbent_already_gone(tmp_path):
    s = SQLiteStorage(org_id="test_org", db_path=str(tmp_path / "t.db"))
    s.migrate()
    inc = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="v1",
        status=Status.ARCHIVED,
    )
    s.save_user_playbooks([inc])
    new = UserPlaybook(user_id="u", agent_version="v", request_id="r", content="v2")
    rc = apply_playbook_edit(
        s,
        incumbent_id=inc.user_playbook_id,
        new_playbook=new,
        source="offline_optimizer",
        request_id="run-test-2",
    )
    assert rc == -1
    # no orphan CURRENT row left behind
    currents = list(s.get_user_playbooks(user_id="u"))
    assert all(p.content != "v2" for p in currents)
