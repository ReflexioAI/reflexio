from __future__ import annotations

import pytest

from reflexio.models.api_schema.domain.entities import Request, UserPlaybook
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _store(tmp_path) -> SQLiteStorage:
    s = SQLiteStorage(org_id="org-char", db_path=str(tmp_path / "org-char.db"))
    s.migrate()
    return s


def _make_request(
    *, request_id: str, user_id: str, session_id: str = "session-1"
) -> Request:
    return Request(request_id=request_id, user_id=user_id, session_id=session_id)


def _make_user_playbook(
    *, user_id: str, request_id: str, content: str
) -> UserPlaybook:
    return UserPlaybook(
        user_id=user_id,
        agent_version="v1",
        request_id=request_id,
        playbook_name="deployment",
        content=content,
    )


def test_search_user_playbooks_user_id_includes_synthetic_generation_request_id(
    tmp_path,
):
    store = _store(tmp_path)
    store.add_request(_make_request(request_id="req-real", user_id="user-1"))
    store.save_user_playbooks(
        [
            _make_user_playbook(
                user_id="user-1",
                request_id="req-real",
                content="real request playbook",
            ),
            _make_user_playbook(
                user_id="user-1",
                request_id="manual_ab12cd34",
                content="manual generation playbook",
            ),
            _make_user_playbook(
                user_id="user-2",
                request_id="rerun_playbook_cd34ef56",
                content="other user playbook",
            ),
        ]
    )

    rows = store.search_user_playbooks(
        SearchUserPlaybookRequest(user_id="user-1", top_k=10)
    )

    assert {row.request_id for row in rows} == {"req-real", "manual_ab12cd34"}
