"""Contract tests for get_session_ids_in_window against SQLite storage."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.internal_schema import (
    RequestInteractionDataModel,
    SessionDescriptor,
)
from reflexio.models.api_schema.service_schemas import (
    AgentSuccessEvaluationResult,
    Request,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._requests import RequestMixin

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[BaseStorage]:
    """Yield a fresh, isolated SQLite storage instance."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(
            org_id="contract_test_session_window",
            db_path=f"{temp_dir}/reflexio.db",
        )


def _seed_request(storage: BaseStorage, user_id: str, session_id: str, ts: int) -> str:
    """Insert one Request at ``ts`` and return its request_id."""
    req = Request(
        request_id=f"req_{session_id}_{ts}",
        user_id=user_id,
        created_at=ts,
        source="test",
        agent_version="v1",
        session_id=session_id,
    )
    storage.add_request(req)
    return req.request_id


def _seed_request_with_source(
    storage: BaseStorage,
    *,
    user_id: str,
    session_id: str,
    ts: int,
    source: str,
    request_id: str,
) -> str:
    req = Request(
        request_id=request_id,
        user_id=user_id,
        created_at=ts,
        source=source,
        agent_version="v1",
        session_id=session_id,
    )
    storage.add_request(req)
    return req.request_id


def test_returns_distinct_sessions_in_window(storage: BaseStorage) -> None:
    _seed_request(storage, "u1", "s1", ts=1000)
    _seed_request(storage, "u1", "s1", ts=1005)
    _seed_request(storage, "u2", "s2", ts=2000)
    _seed_request(storage, "u3", "s3", ts=9999)

    out = storage.get_session_ids_in_window(from_ts=500, to_ts=5000)

    assert isinstance(out, list)
    assert all(isinstance(d, SessionDescriptor) for d in out)
    session_ids = {d.session_id for d in out}
    assert session_ids == {"s1", "s2"}
    assert sum(1 for d in out if d.session_id == "s1") == 1

    # Verify full descriptor identity for a seeded row, not just session_id.
    expected_s1 = SessionDescriptor(
        user_id="u1", session_id="s1", agent_version="v1", source="test"
    )
    assert expected_s1 in out


def test_empty_window_returns_empty_list(storage: BaseStorage) -> None:
    out = storage.get_session_ids_in_window(from_ts=0, to_ts=1)
    assert out == []


def test_window_boundaries_are_inclusive(storage: BaseStorage) -> None:
    _seed_request(storage, "u1", "edge_low", ts=100)
    _seed_request(storage, "u1", "edge_high", ts=200)
    out = storage.get_session_ids_in_window(from_ts=100, to_ts=200)
    assert {d.session_id for d in out} == {"edge_low", "edge_high"}


def test_null_session_rejected_by_request_model() -> None:
    """Requests must include a non-empty session_id."""
    with pytest.raises(ValidationError):
        Request(
            request_id="req_null",
            user_id="u1",
            created_at=1000,
            source="test",
            agent_version="v1",
            session_id=cast(Any, None),
        )


def test_sqlite_migration_backfills_blank_session_ids_per_request() -> None:
    """Old nullable request rows get distinct legacy sessions during migration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = f"{temp_dir}/reflexio.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                agent_version TEXT NOT NULL DEFAULT '',
                session_id TEXT
            );
            INSERT INTO requests
                (request_id, user_id, created_at, source, agent_version, session_id)
            VALUES
                ('r_null', 'u1', '1970-01-01T00:16:40+00:00', 'test', 'v1', NULL),
                ('r_blank', 'u1', '1970-01-01T00:16:41+00:00', 'test', 'v1', ''),
                ('r_ok', 'u1', '1970-01-01T00:16:42+00:00', 'test', 'v1', 'existing');
            """
        )
        conn.commit()
        conn.close()

        storage = SQLiteStorage(
            org_id="contract_test_session_migration",
            db_path=db_path,
        )

        rows = {
            row["request_id"]: row["session_id"]
            for row in storage.conn.execute(
                "SELECT request_id, session_id FROM requests"
            ).fetchall()
        }
        assert rows["r_ok"] == "existing"
        assert rows["r_null"].startswith("legacy-")
        assert rows["r_blank"].startswith("legacy-")
        assert rows["r_null"] != rows["r_blank"]

        with pytest.raises(sqlite3.IntegrityError):
            storage.conn.execute(
                "INSERT INTO requests "
                "(request_id, user_id, created_at, source, agent_version, session_id) "
                "VALUES ('r_bad', 'u1', '1970-01-01T00:16:43+00:00', 'test', 'v1', '')"
            )


def test_distinct_agent_versions_split_into_separate_descriptors(
    storage: BaseStorage,
) -> None:
    """Two requests with same (user_id, session_id) but different agent_version produce two descriptors."""
    storage.add_request(
        Request(
            request_id="r1",
            user_id="u1",
            created_at=1000,
            source="test",
            agent_version="v1",
            session_id="s1",
        )
    )
    storage.add_request(
        Request(
            request_id="r2",
            user_id="u1",
            created_at=1100,
            source="test",
            agent_version="v2",
            session_id="s1",
        )
    )

    out = storage.get_session_ids_in_window(from_ts=0, to_ts=5000)
    versions = {d.agent_version for d in out if d.session_id == "s1"}
    assert versions == {"v1", "v2"}


def _seed_eval_result(
    storage: BaseStorage,
    session_id: str,
    evaluation_name: str,
    agent_version: str = "v1",
    created_at: int = 1000,
) -> None:
    """Save one ``AgentSuccessEvaluationResult`` row with minimal fields set."""
    result = AgentSuccessEvaluationResult(
        session_id=session_id,
        agent_version=agent_version,
        evaluation_name=evaluation_name,
        is_success=True,
        failure_type=None,
        failure_reason=None,
        regular_vs_shadow=None,
        number_of_correction_per_session=0,
        user_turns_to_resolution=None,
        is_escalated=False,
        embedding=[],
        created_at=created_at,
    )
    storage.save_agent_success_evaluation_results([result])


def test_first_requests_by_session_ids_returns_earliest_request(
    storage: BaseStorage,
) -> None:
    _seed_request_with_source(
        storage,
        user_id="u1",
        session_id="s1",
        ts=2000,
        source="later",
        request_id="req_s1_later",
    )
    _seed_request_with_source(
        storage,
        user_id="u1",
        session_id="s1",
        ts=1000,
        source="first",
        request_id="req_s1_first",
    )
    _seed_request_with_source(
        storage,
        user_id="u2",
        session_id="s2",
        ts=1500,
        source="only",
        request_id="req_s2_only",
    )

    out = storage.get_first_requests_by_session_ids(["s2", "s1", "missing", "s1"])

    assert set(out) == {"s1", "s2"}
    assert out["s1"].user_id == "u1"
    assert out["s1"].source == "first"
    assert out["s1"].created_at == 1000
    assert out["s2"].user_id == "u2"
    assert out["s2"].source == "only"


def test_base_first_request_fallback_paginates_all_session_rows() -> None:
    class PaginatedSessionStorage:
        def __init__(self) -> None:
            self.calls: list[tuple[int | None, int]] = []
            self.requests = [
                Request(
                    request_id=f"req-{i}",
                    user_id="u1",
                    created_at=1_700_000_000 + i,
                    source=f"source-{i}",
                    agent_version="v1",
                    session_id="big-session",
                )
                for i in range(1001)
            ]

        def get_sessions(
            self,
            user_id: str | None = None,
            request_id: str | None = None,
            session_id: str | None = None,
            start_time: int | None = None,
            end_time: int | None = None,
            top_k: int | None = 30,
            offset: int = 0,
        ) -> dict[str, list[RequestInteractionDataModel]]:
            del user_id, request_id, start_time, end_time
            self.calls.append((top_k, offset))
            matching = [r for r in self.requests if r.session_id == session_id]
            rows = sorted(matching, key=lambda r: r.created_at, reverse=True)
            page = rows[offset : offset + (top_k or 0)]
            return {
                session_id or "": [
                    RequestInteractionDataModel(
                        session_id=r.session_id,
                        request=r,
                        interactions=[],
                    )
                    for r in page
                ]
            }

    storage = PaginatedSessionStorage()

    out = RequestMixin.get_first_requests_by_session_ids(
        cast(Any, storage), ["big-session"]
    )

    assert out["big-session"].created_at == 1_700_000_000
    assert out["big-session"].source == "source-0"
    assert storage.calls == [(1000, 0), (1000, 1000)]


def test_eval_result_window_read_filters_in_storage_contract(
    storage: BaseStorage,
) -> None:
    _seed_eval_result(storage, "old", "overall_success", created_at=100)
    _seed_eval_result(storage, "inside", "overall_success", created_at=200)
    _seed_eval_result(storage, "new", "overall_success", created_at=300)
    _seed_eval_result(
        storage, "inside_v2", "overall_success", agent_version="v2", created_at=250
    )

    all_inside = storage.get_agent_success_evaluation_results_in_window(150, 260)
    v1_inside = storage.get_agent_success_evaluation_results_in_window(
        150, 260, agent_version="v1"
    )

    assert {r.session_id for r in all_inside} == {"inside", "inside_v2"}
    assert {r.session_id for r in v1_inside} == {"inside"}


def test_targeted_eval_result_id_lookup(
    storage: BaseStorage,
) -> None:
    _seed_eval_result(storage, "s1", "overall_success", agent_version="v1")
    _seed_eval_result(storage, "s1", "safety", agent_version="v1")
    _seed_eval_result(storage, "s1", "overall_success", agent_version="v2")
    _seed_eval_result(storage, "s2", "overall_success", agent_version="v1")

    ids = storage.get_agent_success_evaluation_result_ids(
        session_id="s1",
        evaluation_name="overall_success",
        agent_version="v1",
    )

    assert len(ids) == 1
    row = [
        r
        for r in storage.get_agent_success_evaluation_results(limit=100)
        if r.result_id == ids[0]
    ][0]
    assert row.session_id == "s1"
    assert row.evaluation_name == "overall_success"
    assert row.agent_version == "v1"


def test_delete_scoped_to_session_and_name(storage: BaseStorage) -> None:
    _seed_eval_result(storage, "s1", "overall_success")
    _seed_eval_result(storage, "s1", "safety")
    _seed_eval_result(storage, "s2", "overall_success")

    n = storage.delete_agent_success_evaluation_results_for_session(
        session_id="s1", evaluation_name="overall_success", agent_version="v1"
    )

    assert n == 1
    remaining = storage.get_agent_success_evaluation_results(limit=100)
    sessions_and_names = {(r.session_id, r.evaluation_name) for r in remaining}
    assert sessions_and_names == {("s1", "safety"), ("s2", "overall_success")}


def test_delete_unknown_session_returns_zero(storage: BaseStorage) -> None:
    n = storage.delete_agent_success_evaluation_results_for_session(
        session_id="does_not_exist",
        evaluation_name="overall_success",
        agent_version="v1",
    )
    assert n == 0


def test_delete_respects_agent_version_scope(storage: BaseStorage) -> None:
    _seed_eval_result(storage, "s1", "overall_success", agent_version="v1")
    _seed_eval_result(storage, "s1", "overall_success", agent_version="v2")
    n = storage.delete_agent_success_evaluation_results_for_session(
        session_id="s1", evaluation_name="overall_success", agent_version="v1"
    )
    assert n == 1
    remaining = storage.get_agent_success_evaluation_results(limit=100)
    versions = {r.agent_version for r in remaining if r.session_id == "s1"}
    assert versions == {"v2"}


def test_delete_by_ids_empty_list_is_noop(storage: BaseStorage) -> None:
    """An empty result_ids list must short-circuit and return 0 without erroring."""
    _seed_eval_result(storage, "s1", "overall_success")
    n = storage.delete_agent_success_evaluation_results_by_ids([])
    assert n == 0
    # Existing row untouched.
    remaining = storage.get_agent_success_evaluation_results(limit=100)
    assert len(remaining) == 1


def test_delete_by_ids_removes_only_targeted_rows(storage: BaseStorage) -> None:
    """Pass a subset of stored result_ids — only those rows are deleted."""
    _seed_eval_result(storage, "s1", "overall_success")
    _seed_eval_result(storage, "s1", "safety")
    _seed_eval_result(storage, "s2", "overall_success")

    all_rows = storage.get_agent_success_evaluation_results(limit=100)
    # Pick the result_id for the s1/overall_success row.
    target_ids = [
        r.result_id
        for r in all_rows
        if r.session_id == "s1" and r.evaluation_name == "overall_success"
    ]
    assert len(target_ids) == 1
    assert target_ids[0] != 0  # storage layer assigned a real id

    n = storage.delete_agent_success_evaluation_results_by_ids(target_ids)
    assert n == 1

    remaining = storage.get_agent_success_evaluation_results(limit=100)
    pairs = {(r.session_id, r.evaluation_name) for r in remaining}
    assert pairs == {("s1", "safety"), ("s2", "overall_success")}


def test_delete_by_ids_silently_ignores_unknown_ids(storage: BaseStorage) -> None:
    """Non-existent result_ids must NOT raise; return count of rows actually removed."""
    _seed_eval_result(storage, "s1", "overall_success")
    all_rows = storage.get_agent_success_evaluation_results(limit=100)
    real_id = all_rows[0].result_id

    # Mix a real id with two unknown ids.
    n = storage.delete_agent_success_evaluation_results_by_ids(
        [real_id, 99_998, 99_999]
    )
    assert n == 1

    remaining = storage.get_agent_success_evaluation_results(limit=100)
    assert remaining == []


def test_delete_by_ids_all_unknown_returns_zero(storage: BaseStorage) -> None:
    """Calling with only unknown ids returns 0 and does not raise."""
    _seed_eval_result(storage, "s1", "overall_success")
    n = storage.delete_agent_success_evaluation_results_by_ids([12_345, 67_890])
    assert n == 0
    remaining = storage.get_agent_success_evaluation_results(limit=100)
    assert len(remaining) == 1
