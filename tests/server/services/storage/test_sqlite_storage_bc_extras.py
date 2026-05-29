"""SQLite-storage implementations of Plan B + Plan C storage methods.

Exercises real SQLite tables + queries (no mocks), verifying:
- `count_sessions_with_shadow_content` joins interactions ↔ requests correctly.
- `get_interactions_by_session` returns interactions filtered by session_id.
- BraintrustConnection upsert/get/delete roundtrip.
- ImportedScore upsert + idempotent re-sync (UNIQUE constraint).
- `get_imported_scores` filters by org_id + time window.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from collections.abc import Generator

import pytest

from reflexio.models.api_schema.braintrust_schema import (
    BraintrustConnection,
    ImportedScore,
)
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[SQLiteStorage]:
    """Per-test SQLite store in a temp dir."""
    with tempfile.TemporaryDirectory() as d:
        org_id = f"org_{uuid.uuid4().hex[:8]}"
        store = SQLiteStorage(org_id=org_id, db_path=f"{d}/reflexio.db")
        yield store


def _now() -> int:
    return int(time.time())


def _add_request_with_interactions(
    store: SQLiteStorage,
    *,
    session_id: str,
    user_id: str = "user_t",
    request_id: str | None = None,
    shadow_content_for_assistant: str = "",
    created_at: int | None = None,
) -> str:
    """Create a request + 2 interactions (1 user, 1 assistant); return request_id."""
    request_id = request_id or f"req_{uuid.uuid4().hex[:8]}"
    ts = _now() if created_at is None else created_at
    store.add_request(
        Request(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            agent_version="v_t",
            source="",
            created_at=ts,
        )
    )
    store.add_user_interactions_bulk(
        user_id=user_id,
        interactions=[
            Interaction(
                interaction_id=0,
                user_id=user_id,
                request_id=request_id,
                role="User",
                content="hello",
                created_at=ts,
            ),
            Interaction(
                interaction_id=0,
                user_id=user_id,
                request_id=request_id,
                role="Assistant",
                content="hi back",
                shadow_content=shadow_content_for_assistant,
                created_at=ts + 1,
            ),
        ],
    )
    return request_id


# ---------------------------------------------------------------------------
# count_sessions_with_shadow_content
# ---------------------------------------------------------------------------


def test_count_sessions_with_shadow_content_zero_when_no_shadow(
    storage: SQLiteStorage,
) -> None:
    _add_request_with_interactions(storage, session_id="s1")
    assert storage.count_sessions_with_shadow_content(0, _now() + 60) == 0


def test_count_sessions_with_shadow_content_counts_distinct_sessions(
    storage: SQLiteStorage,
) -> None:
    _add_request_with_interactions(
        storage, session_id="s_shadow_1", shadow_content_for_assistant="shadow-a"
    )
    _add_request_with_interactions(
        storage, session_id="s_shadow_1", shadow_content_for_assistant="shadow-b"
    )  # same session, two requests — still 1 session
    _add_request_with_interactions(
        storage, session_id="s_shadow_2", shadow_content_for_assistant="shadow-c"
    )
    _add_request_with_interactions(
        storage, session_id="s_no_shadow"
    )  # no shadow → not counted
    assert storage.count_sessions_with_shadow_content(0, _now() + 60) == 2


def test_count_sessions_with_shadow_content_filters_window(
    storage: SQLiteStorage,
) -> None:
    now = _now()
    _add_request_with_interactions(
        storage,
        session_id="s_old",
        shadow_content_for_assistant="shadow",
        created_at=now - 10_000,
    )
    _add_request_with_interactions(
        storage,
        session_id="s_recent",
        shadow_content_for_assistant="shadow",
        created_at=now - 60,
    )
    # Window includes only "recent"
    assert storage.count_sessions_with_shadow_content(now - 600, now + 60) == 1


# ---------------------------------------------------------------------------
# get_interactions_by_session
# ---------------------------------------------------------------------------


def test_get_interactions_by_session_returns_only_that_session(
    storage: SQLiteStorage,
) -> None:
    _add_request_with_interactions(storage, session_id="target")
    _add_request_with_interactions(storage, session_id="other")
    out = storage.get_interactions_by_session("target")
    assert len(out) == 2
    assert all(i.role in ("User", "Assistant") for i in out)


def test_get_interactions_by_session_empty_string_returns_empty(
    storage: SQLiteStorage,
) -> None:
    _add_request_with_interactions(storage, session_id="s")
    assert storage.get_interactions_by_session("") == []


def test_get_interactions_by_session_unknown_session_returns_empty(
    storage: SQLiteStorage,
) -> None:
    assert storage.get_interactions_by_session("never_existed") == []


# ---------------------------------------------------------------------------
# BraintrustConnection storage
# ---------------------------------------------------------------------------


def test_braintrust_connection_roundtrip(storage: SQLiteStorage) -> None:
    conn = BraintrustConnection(
        org_id="org_x",
        api_key_enc="enc-value",
        workspace_id="ws_1",
        workspace_name="My WS",
        project_ids=["p_a", "p_b"],
        last_sync_ts=1700000000,
        last_error=None,
    )
    storage.save_braintrust_connection(conn)
    fetched = storage.get_braintrust_connection("org_x")
    assert fetched is not None
    assert fetched.api_key_enc == "enc-value"
    assert fetched.project_ids == ["p_a", "p_b"]
    assert fetched.workspace_name == "My WS"
    assert fetched.last_sync_ts == 1700000000


def test_braintrust_connection_upsert_overwrites(storage: SQLiteStorage) -> None:
    conn1 = BraintrustConnection(
        org_id="org_y", api_key_enc="k1", workspace_id="ws_1", project_ids=["p1"]
    )
    storage.save_braintrust_connection(conn1)
    conn2 = BraintrustConnection(
        org_id="org_y", api_key_enc="k2", workspace_id="ws_2", project_ids=["p2", "p3"]
    )
    storage.save_braintrust_connection(conn2)
    fetched = storage.get_braintrust_connection("org_y")
    assert fetched is not None
    assert fetched.api_key_enc == "k2"
    assert fetched.workspace_id == "ws_2"
    assert fetched.project_ids == ["p2", "p3"]


def test_braintrust_connection_get_unknown_returns_none(
    storage: SQLiteStorage,
) -> None:
    assert storage.get_braintrust_connection("never") is None


def test_braintrust_connection_delete_is_idempotent(storage: SQLiteStorage) -> None:
    conn = BraintrustConnection(org_id="org_z", api_key_enc="k", workspace_id="ws")
    storage.save_braintrust_connection(conn)
    storage.delete_braintrust_connection("org_z")
    assert storage.get_braintrust_connection("org_z") is None
    # Delete again — no error.
    storage.delete_braintrust_connection("org_z")
    storage.delete_braintrust_connection("never_existed")


# ---------------------------------------------------------------------------
# ImportedScore storage
# ---------------------------------------------------------------------------


def _score(
    org_id: str = "org_x",
    *,
    source_run_id: str = "span_1",
    scorer_name: str = "hallucination",
    value: float = 0.1,
    session_id: str | None = None,
    ts: int = 1700000000,
) -> ImportedScore:
    return ImportedScore(
        org_id=org_id,
        source_run_id=source_run_id,
        scorer_name=scorer_name,
        value=value,
        session_id=session_id,
        ts=ts,
    )


def test_imported_scores_save_and_query_window(storage: SQLiteStorage) -> None:
    storage.save_imported_scores(
        [
            _score(source_run_id="s1", scorer_name="halu", value=0.1, ts=100),
            _score(source_run_id="s1", scorer_name="fact", value=0.9, ts=100),
            _score(source_run_id="s2", scorer_name="halu", value=0.3, ts=200),
        ]
    )
    out = storage.get_imported_scores("org_x", 0, 1000)
    assert len(out) == 3
    assert {s.scorer_name for s in out} == {"halu", "fact"}


def test_imported_scores_idempotent_on_re_sync(storage: SQLiteStorage) -> None:
    """Same span_id + scorer_name resyncs without duplicating rows."""
    storage.save_imported_scores(
        [_score(source_run_id="s1", scorer_name="halu", value=0.1)]
    )
    storage.save_imported_scores(
        [_score(source_run_id="s1", scorer_name="halu", value=0.2)]
    )
    out = storage.get_imported_scores("org_x", 0, 9999999999)
    assert len(out) == 1
    assert out[0].value == 0.2  # value updated to latest


def test_imported_scores_window_filter(storage: SQLiteStorage) -> None:
    storage.save_imported_scores(
        [
            _score(source_run_id="s_old", scorer_name="halu", value=0.5, ts=100),
            _score(source_run_id="s_new", scorer_name="halu", value=0.2, ts=2000),
        ]
    )
    # Window excludes the old score
    out = storage.get_imported_scores("org_x", 500, 9999)
    assert len(out) == 1
    assert out[0].source_run_id == "s_new"


def test_imported_scores_filter_by_org_id(storage: SQLiteStorage) -> None:
    storage.save_imported_scores(
        [
            _score(org_id="org_a", source_run_id="s1"),
            _score(org_id="org_b", source_run_id="s2"),
        ]
    )
    out_a = storage.get_imported_scores("org_a", 0, 9999999999)
    out_b = storage.get_imported_scores("org_b", 0, 9999999999)
    assert len(out_a) == 1 and out_a[0].source_run_id == "s1"
    assert len(out_b) == 1 and out_b[0].source_run_id == "s2"


def test_save_empty_imported_scores_is_noop(storage: SQLiteStorage) -> None:
    storage.save_imported_scores([])
    assert storage.get_imported_scores("org_x", 0, 9999999999) == []
