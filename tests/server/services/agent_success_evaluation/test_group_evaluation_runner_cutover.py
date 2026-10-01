"""Runner behaviour with and without a trajectory cutover."""

from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.agent_success_evaluation import _eval_health
from reflexio.server.services.agent_success_evaluation._eval_health import SkipReason
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    AgentSuccessEvaluationRequest,
)
from reflexio.server.services.agent_success_evaluation.runner import (
    run_group_evaluation,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_SERVICE = (
    "reflexio.server.services.agent_success_evaluation.runner"
    ".AgentSuccessEvaluationService"
)
USER = "u1"
SESSION = "s1"


def _now() -> int:
    return int(datetime.now(UTC).timestamp())


def _request(request_id: str, created_at: int) -> Request:
    return Request(
        request_id=request_id,
        user_id=USER,
        session_id=SESSION,
        source="published",
        created_at=created_at,
    )


def _interaction(interaction_id: int, request_id: str, created_at: int) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id=USER,
        request_id=request_id,
        content=f"turn {interaction_id}",
        role="user",
        created_at=created_at,
    )


def _run(storage, **kwargs) -> tuple[object, list[AgentSuccessEvaluationRequest]]:
    """Run the runner with the LLM-backed service replaced by a recorder."""
    request_context = MagicMock()
    request_context.storage = storage
    captured: list[AgentSuccessEvaluationRequest] = []
    with patch(_SERVICE) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service.run.side_effect = captured.append
        service_cls.return_value = service
        outcome = run_group_evaluation(
            org_id="org",
            user_id=USER,
            session_id=SESSION,
            agent_version="v1",
            source="published",
            request_context=request_context,
            llm_client=MagicMock(),
            run_retrieved_learning=False,
            **kwargs,
        )
    return outcome, captured


def _live_mock_storage(*, marked: bool) -> MagicMock:
    """A session whose last request is seconds old, optionally already marked."""
    now = _now()
    storage = MagicMock()
    storage.get_operation_state.return_value = (
        {"operation_state": {"evaluated": True}} if marked else None
    )
    storage.get_requests_by_session.return_value = [_request("r1", now - 5)]
    storage.get_interactions_by_request_ids.return_value = [
        _interaction(1, "r1", now - 4)
    ]
    return storage


def test_unset_cutover_keeps_idle_gate_and_legacy_storage_call() -> None:
    storage = _live_mock_storage(marked=False)

    outcome, captured = _run(storage)

    assert outcome.agent_success_status == "skipped"  # type: ignore[attr-defined]
    assert captured == []
    # No keyword: a backend that predates it must keep working.
    storage.get_requests_by_session.assert_called_once_with(USER, SESSION)


def test_unset_cutover_reads_and_writes_the_marker() -> None:
    storage = _live_mock_storage(marked=True)
    storage.get_requests_by_session.return_value = [_request("r1", _now() - 10_000)]

    outcome, captured = _run(storage)

    assert outcome.agent_success_status == "skipped"  # type: ignore[attr-defined]
    assert captured == []
    storage.get_operation_state.assert_called_once()

    storage = _live_mock_storage(marked=False)
    storage.get_requests_by_session.return_value = [_request("r1", _now() - 10_000)]
    outcome, _ = _run(storage)
    assert outcome.agent_success_status == "complete"  # type: ignore[attr-defined]
    storage.upsert_operation_state.assert_called_once()


def test_cutover_bypasses_idle_gate_and_marker_and_writes_no_marker() -> None:
    storage = _live_mock_storage(marked=True)

    outcome, captured = _run(storage, through_request_id="r1")

    assert outcome.agent_success_status == "complete"  # type: ignore[attr-defined]
    assert len(captured) == 1
    storage.get_requests_by_session.assert_called_once_with(
        USER, SESSION, through_request_id="r1"
    )
    storage.get_operation_state.assert_not_called()
    storage.upsert_operation_state.assert_not_called()


def test_missing_cutover_is_not_applicable() -> None:
    storage = _live_mock_storage(marked=False)
    storage.get_requests_by_session.return_value = []

    with patch.object(_eval_health, "record_skip") as record_skip:
        outcome, captured = _run(storage, through_request_id="gone")

    assert outcome.agent_success_status == "not_applicable"  # type: ignore[attr-defined]
    assert captured == []
    record_skip.assert_called_once_with(SkipReason.CUTOVER_NOT_FOUND)
    storage.get_interactions_by_request_ids.assert_not_called()


def test_whole_session_stamp_does_not_trust_storage_order_on_a_tie() -> None:
    old = _now() - 10_000
    storage = _live_mock_storage(marked=False)
    # A backend may return a same-second tie in any order.
    storage.get_requests_by_session.return_value = [
        _request("b", old + 5),
        _request("r1", old),
        _request("a", old + 5),
    ]
    storage.get_interactions_by_request_ids.return_value = [
        _interaction(1, "r1", old + 1),
        _interaction(2, "b", old + 6),
        _interaction(3, "a", old + 6),
    ]

    _, [judged] = _run(storage)

    assert [m.request.request_id for m in judged.request_interaction_data_models] == [
        "r1",
        "a",
        "b",
    ]
    assert judged.trajectory_through_request_id == "b"
    assert judged.trajectory_interaction_count == 3


# ---------------------------------------------------------------------------
# Real SQLite storage: the prefix is selected by SQL, not by the runner.
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_storage(tmp_path) -> Generator[SQLiteStorage]:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="runner-cutover", db_path=str(tmp_path / "r.db"))


def _seed(storage: SQLiteStorage, rows: list[tuple[str, int]]) -> None:
    for index, (request_id, created_at) in enumerate(rows, start=1):
        storage.add_request(_request(request_id, created_at))
        storage.add_user_interactions_bulk(
            USER,
            [_interaction(index, request_id, created_at + 1)],
            embeddings_prepared=True,
        )


def test_cutover_judge_ignores_later_requests(sqlite_storage: SQLiteStorage) -> None:
    now = _now()
    _seed(sqlite_storage, [("r1", now - 30), ("r2", now - 20), ("r3", now - 10)])

    outcome, [judged] = _run(sqlite_storage, through_request_id="r2")

    assert outcome.agent_success_status == "complete"  # type: ignore[attr-defined]
    assert [m.request.request_id for m in judged.request_interaction_data_models] == [
        "r1",
        "r2",
    ]
    assert judged.trajectory_through_request_id == "r2"
    assert judged.trajectory_interaction_count == 2


def test_cutover_tie_member_selects_only_through_itself(
    sqlite_storage: SQLiteStorage,
) -> None:
    now = _now()
    # Same second; inserted "b" before "a". request_id decides.
    _seed(sqlite_storage, [("r1", now - 30), ("b", now - 20), ("a", now - 20)])

    _, [judged] = _run(sqlite_storage, through_request_id="a")

    assert [m.request.request_id for m in judged.request_interaction_data_models] == [
        "r1",
        "a",
    ]


def test_whole_session_stamp_is_last_request_with_same_second_tie(
    sqlite_storage: SQLiteStorage,
) -> None:
    old = _now() - 10_000
    # "b" is inserted first and storage order is not trusted: (created_at,
    # request_id) puts "b" last.
    _seed(sqlite_storage, [("r1", old), ("b", old + 5), ("a", old + 5)])

    _, [judged] = _run(sqlite_storage)

    assert [m.request.request_id for m in judged.request_interaction_data_models] == [
        "r1",
        "a",
        "b",
    ]
    assert judged.trajectory_through_request_id == "b"
    assert judged.trajectory_interaction_count == 3
