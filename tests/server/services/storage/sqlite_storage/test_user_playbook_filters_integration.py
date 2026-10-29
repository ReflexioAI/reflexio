"""SQLite user-playbook CRUD filters scope exactly the rows they name.

Each test seeds playbooks that differ on one axis (agent version, playbook
name, user, request, tag) and asserts that a filtered read, delete, status
change or search touches only the matching rows.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.common import BlockingIssue, BlockingIssueKind
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.api_schema.service_schemas import Status, UserPlaybook
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def store(tmp_path, worker_id) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.1] * 512):
        storage = SQLiteStorage(
            org_id=f"up-filters-{worker_id}", db_path=str(tmp_path / "up.db")
        )
    storage._get_embedding = Mock(return_value=[0.1] * 512)  # noqa: SLF001
    return storage


def _playbook(
    user: str,
    *,
    name: str = "refund",
    version: str = "v1",
    request: str = "req-1",
    content: str = "issue refunds politely",
    tags: list[str] | None = None,
    status: Status | None = None,
) -> UserPlaybook:
    return UserPlaybook(
        user_id=user,
        agent_version=version,
        request_id=request,
        playbook_name=name,
        content=content,
        trigger=f"when {content}",
        tags=tags or [],
        status=status,
    )


def _ids(rows) -> set[str]:
    return {f"{row.user_id}/{row.playbook_name}/{row.agent_version}" for row in rows}


def test_get_filters_by_request_and_tags(store):
    store.save_user_playbooks(
        [
            _playbook("u1", request="req-a", tags=["billing"]),
            _playbook("u2", request="req-b", tags=["shipping"]),
            _playbook("u3", request="req-a", tags=["shipping"]),
        ]
    )
    by_request = store.get_user_playbooks(request_id="req-a")
    assert _ids(by_request) == {"u1/refund/v1", "u3/refund/v1"}
    by_tag = store.get_user_playbooks(tags=["shipping"])
    assert _ids(by_tag) == {"u2/refund/v1", "u3/refund/v1"}
    both = store.get_user_playbooks(request_id="req-a", tags=["shipping"])
    assert _ids(both) == {"u3/refund/v1"}


def test_delete_by_playbook_name_respects_agent_version(store):
    store.save_user_playbooks(
        [
            _playbook("u1", name="refund", version="v1"),
            _playbook("u2", name="refund", version="v2"),
            _playbook("u3", name="shipping", version="v1"),
        ]
    )
    store.delete_all_user_playbooks_by_playbook_name("refund", agent_version="v2")
    assert _ids(store.get_user_playbooks()) == {"u1/refund/v1", "u3/shipping/v1"}
    # Nothing left to match: a second call is a quiet no-op.
    store.delete_all_user_playbooks_by_playbook_name("refund", agent_version="v2")
    store.delete_all_user_playbooks_by_playbook_name("refund")
    assert _ids(store.get_user_playbooks()) == {"u3/shipping/v1"}


def test_delete_by_ids_ignores_an_empty_list(store):
    store.save_user_playbooks([_playbook("u1")])
    assert store.delete_user_playbooks_by_ids([]) == 0
    assert len(store.get_user_playbooks()) == 1


def test_bulk_status_change_is_scoped_by_version_and_name(store):
    store.save_user_playbooks(
        [
            _playbook("u1", name="refund", version="v1"),
            _playbook("u2", name="refund", version="v2"),
            _playbook("u3", name="shipping", version="v2"),
        ]
    )
    changed = store.update_all_user_playbooks_status(
        None, Status.ARCHIVED, agent_version="v2", playbook_name="refund"
    )
    assert changed == 1
    archived = store.get_user_playbooks(status_filter=[Status.ARCHIVED])
    assert _ids(archived) == {"u2/refund/v2"}
    current = store.get_user_playbooks(status_filter=[None])
    assert _ids(current) == {"u1/refund/v1", "u3/shipping/v2"}


def test_delete_by_status_is_scoped_by_user_version_and_name(store):
    store.save_user_playbooks(
        [
            _playbook("u1", name="refund", version="v1", status=Status.ARCHIVED),
            _playbook("u1", name="shipping", version="v1", status=Status.ARCHIVED),
            _playbook("u2", name="refund", version="v1", status=Status.ARCHIVED),
            _playbook("u1", name="refund", version="v2", status=Status.ARCHIVED),
        ]
    )
    deleted = store.delete_all_user_playbooks_by_status(
        Status.ARCHIVED, agent_version="v1", playbook_name="refund", user_id="u1"
    )
    assert deleted == 1
    remaining = store.get_user_playbooks(status_filter=[Status.ARCHIVED])
    assert _ids(remaining) == {
        "u1/shipping/v1",
        "u2/refund/v1",
        "u1/refund/v2",
    }
    assert (
        store.delete_all_user_playbooks_by_status(
            Status.ARCHIVED, user_id="nobody", agent_version="v1"
        )
        == 0
    )


def test_search_filters_by_playbook_name_and_tags(store):
    store.save_user_playbooks(
        [
            _playbook("u1", name="refund", content="refund escalation", tags=["vip"]),
            _playbook("u2", name="shipping", content="refund escalation", tags=["vip"]),
            _playbook("u3", name="refund", content="refund escalation", tags=["std"]),
        ]
    )

    def search(**kwargs):
        return store.search_user_playbooks(
            SearchUserPlaybookRequest(
                query="refund escalation", search_mode=SearchMode.FTS, **kwargs
            )
        )

    assert _ids(search(playbook_name="refund")) == {"u1/refund/v1", "u3/refund/v1"}
    assert _ids(search(tags=["vip"])) == {"u1/refund/v1", "u2/shipping/v1"}
    assert _ids(search(playbook_name="refund", tags=["vip"])) == {"u1/refund/v1"}


def test_update_writes_rationale_and_blocking_issue_and_skips_missing_rows(store):
    store.save_user_playbooks([_playbook("u1")])
    [saved] = store.get_user_playbooks()
    issue = BlockingIssue(
        kind=BlockingIssueKind.MISSING_TOOL, details="no refund API available"
    )
    store.update_user_playbook(
        saved.user_playbook_id, rationale="customers expect it", blocking_issue=issue
    )
    [updated] = store.get_user_playbooks()
    assert updated.rationale == "customers expect it"
    assert updated.blocking_issue == issue
    assert updated.content == saved.content

    # A row that no longer exists is skipped without error or side effect.
    store.update_user_playbook(987_654, rationale="ghost")
    assert store.get_user_playbook_by_id(987_654) is None
    # No fields means no write at all.
    store.update_user_playbook(saved.user_playbook_id)
    [unchanged] = store.get_user_playbooks()
    assert unchanged.rationale == "customers expect it"


def test_update_inside_scope_commits_with_the_scope(store):
    store.save_user_playbooks([_playbook("u1")])
    [saved] = store.get_user_playbooks()
    with pytest.raises(RuntimeError, match="abort"), store.commit_scope():
        store.update_user_playbook(saved.user_playbook_id, rationale="draft")
        raise RuntimeError("abort")
    assert store.get_user_playbooks()[0].rationale == saved.rationale
    with store.commit_scope():
        store.update_user_playbook(saved.user_playbook_id, rationale="final")
    assert store.get_user_playbooks()[0].rationale == "final"
