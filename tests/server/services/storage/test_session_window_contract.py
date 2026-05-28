"""Contract tests for get_session_ids_in_window across storage backends.

This file defines its own parametrized ``storage`` fixture (shadowing the
conftest one) so the new method is exercised against BOTH SQLite and
Disk backends without enrolling pre-existing contract tests against the
Disk backend (which currently has unrelated failures in retention and
stall_state).
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.internal_schema import SessionDescriptor
from reflexio.models.api_schema.service_schemas import (
    AgentSuccessEvaluationResult,
    Request,
)
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "disk"])
def storage(request: pytest.FixtureRequest) -> Generator[BaseStorage]:
    """Yield a fresh, isolated storage instance for each backend."""
    backend = request.param

    with tempfile.TemporaryDirectory() as temp_dir:
        if backend == "sqlite":
            from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

            with patch.object(
                SQLiteStorage, "_get_embedding", return_value=[0.0] * 512
            ):
                yield SQLiteStorage(
                    org_id="contract_test_session_window",
                    db_path=f"{temp_dir}/reflexio.db",
                )
        elif backend == "disk":
            from reflexio.server.services.storage.disk_storage import DiskStorage

            yield DiskStorage(org_id="contract_test_session_window", base_dir=temp_dir)


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


def test_null_session_excluded(storage: BaseStorage) -> None:
    """Requests with session_id=None must be excluded from the result."""
    _seed_request(storage, "u1", "s1", ts=1000)
    req = Request(
        request_id="req_null",
        user_id="u1",
        created_at=1000,
        source="test",
        agent_version="v1",
        session_id=None,
    )
    storage.add_request(req)

    out = storage.get_session_ids_in_window(from_ts=0, to_ts=5000)
    assert {d.session_id for d in out} == {"s1"}


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
        created_at=1000,
    )
    storage.save_agent_success_evaluation_results([result])


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
