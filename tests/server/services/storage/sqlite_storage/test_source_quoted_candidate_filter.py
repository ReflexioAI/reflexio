"""Consolidation eligibility precedes every storage candidate cap."""

import json
from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.config_schema import SearchMode, SearchOptions
from reflexio.server.services.playbook.source_quoted_overflow import POLICY
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration
VECTOR = [1.0] + [0.0] * 511


@pytest.fixture
def corpus(tmp_path):
    store = SQLiteStorage(org_id="quoted-filter", db_path=str(tmp_path / "test.db"))
    store.migrate()
    ordinary = [
        UserPlaybook(
            user_id="u",
            agent_version="v",
            request_id=f"ordinary-{i}",
            content="approval 日本語 " + "additional context " * 20,
            trigger="approval 日本語",
            embedding=[1.0, 0.1] + [0.0] * 510,
            created_at=100 + i,
            notes="Original narrative",
        )
        for i in range(3)
    ]
    protected = [
        UserPlaybook(
            user_id="u",
            agent_version="v",
            request_id=f"protected-{i}",
            content="approval 日本語",
            trigger="approval 日本語",
            embedding=VECTOR,
            created_at=200 + i,
            notes=json.dumps({"policy": POLICY}),
        )
        for i in range(125)
    ]
    unrelated = [
        ordinary[0].model_copy(update={"user_id": "other", "created_at": 500}),
        ordinary[0].model_copy(update={"agent_version": "other", "created_at": 500}),
    ]
    store.save_user_playbooks([*ordinary, *protected, *unrelated], skip_embedding=True)
    yield store, ordinary, protected
    store.conn.close()


@pytest.mark.parametrize(
    ("query", "mode", "embedding"),
    [
        (None, SearchMode.HYBRID, None),
        (None, SearchMode.HYBRID, VECTOR),
        ("approval", SearchMode.VECTOR, VECTOR),
        ("approval", SearchMode.FTS, None),
        ("approval", SearchMode.HYBRID, VECTOR),
        ("日本語", SearchMode.FTS, None),
        ("日本語", SearchMode.HYBRID, VECTOR),
    ],
)
def test_exclusion_returns_eligible_top_k_despite_125_protected_rows(
    corpus, query, mode, embedding
):
    store, ordinary, protected = corpus
    request = SearchUserPlaybookRequest(
        user_id="u",
        agent_version="v",
        query=query,
        top_k=2,
        threshold=0.1,
        search_mode=mode,
    )
    options = SearchOptions(query_embedding=embedding)
    default_rows = store.search_user_playbooks(request, options)
    assert len(default_rows) == 2
    assert {row.user_playbook_id for row in default_rows} <= {
        row.user_playbook_id for row in protected
    }
    rows = store.search_user_playbooks(
        request, SearchOptions(query_embedding=embedding, exclude_source_quoted=True)
    )
    assert len(rows) == 2
    assert {row.user_playbook_id for row in rows} <= {
        row.user_playbook_id for row in ordinary
    }


@pytest.mark.parametrize(
    "notes",
    [
        None,
        "not json",
        "{broken",
        "null",
        "42",
        '"text"',
        "[]",
        json.dumps([{"policy": POLICY}]),
        '{"policy": false}',
        '{"policy": "another-policy"}',
        '{"policy": "source-quoted-playbooks-v1", "policy": "ordinary"}',
        '{"policy": "source-quoted-playbooks-v1", "policy": false}',
    ],
)
def test_safe_notes_filter_and_empty_query_time_bounds(tmp_path, notes):
    store = SQLiteStorage(org_id="notes", db_path=str(tmp_path / "test.db"))
    store.migrate()
    rows = [
        UserPlaybook(
            user_id="u",
            agent_version="v",
            request_id=str(t),
            content="approval",
            notes=notes,
            created_at=t,
        )
        for t in (99, 101, 103)
    ]
    try:
        store.save_user_playbooks(rows, skip_embedding=True)
        found = store.search_user_playbooks(
            SearchUserPlaybookRequest(
                user_id="u",
                agent_version="v",
                top_k=1,
                start_time=datetime.fromtimestamp(100, UTC),
                end_time=datetime.fromtimestamp(102, UTC),
            ),
            SearchOptions(exclude_source_quoted=True),
        )
        assert [row.user_playbook_id for row in found] == [rows[1].user_playbook_id]
    finally:
        store.conn.close()


@pytest.mark.parametrize(
    "later", ['"source-quoted-playbooks-v1"', '"ordinary"', "false"]
)
def test_duplicate_policy_keys_follow_canonical_last_value(tmp_path, later):
    from reflexio.server.services.playbook.source_quoted_overflow import (
        is_source_quoted_playbook,
    )

    store = SQLiteStorage(org_id="duplicate-policy", db_path=str(tmp_path / "test.db"))
    store.migrate()
    row = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r",
        content="approval",
        notes='{"policy": "ordinary", "policy": ' + later + "}",
    )
    try:
        store.save_user_playbooks([row], skip_embedding=True)
        found = store.search_user_playbooks(
            SearchUserPlaybookRequest(user_id="u", top_k=1),
            SearchOptions(exclude_source_quoted=True),
        )
        assert bool(found) == (not is_source_quoted_playbook(row))
    finally:
        store.conn.close()
