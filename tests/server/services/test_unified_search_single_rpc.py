"""Phase B single-RPC routing: combined storage call, fallback, kill switch."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from typing import Any, cast

import pytest

from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.server.services import unified_search_service as uss
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.tracing import configure_tracer


class _RecordingSpan:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    def set_data(self, key: str, value: Any) -> None:
        self.record[key] = value


class _RecordingTracer:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    @contextmanager
    def span(self, name: str, **data: Any):
        record = {"name": name, **data}
        try:
            yield _RecordingSpan(record)
        finally:
            self.records.append(record)


def _agent_playbook(playbook_id: int, content: str) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=playbook_id, agent_version="v1", content=content
    )


class _CombinedStorage:
    """Fake storage advertising the combined Phase B capability."""

    embedding_model_name = "local/minilm-l6-v2"
    supports_embedding = True
    supports_unified_hybrid_search = True

    def __init__(
        self,
        result: tuple[list[Any], list[Any], list[Any]] = ([], [], []),
        raise_on_combined: bool = False,
    ) -> None:
        self.result = result
        self.raise_on_combined = raise_on_combined
        self.combined_calls: list[dict[str, Any]] = []
        self.scored_calls: list[dict[str, Any]] = []
        self.fanout_calls: list[str] = []

    def unified_hybrid_search(self, **kwargs: Any):
        self.combined_calls.append(kwargs)
        if self.raise_on_combined:
            raise RuntimeError("function public.unified_hybrid_search does not exist")
        return self.result

    # Selected instead of ``unified_hybrid_search`` when recency is on, since
    # recency needs the per-row ``combined_score`` sidecars.
    def unified_hybrid_search_scored(self, **kwargs: Any):
        self.scored_calls.append(kwargs)
        if self.raise_on_combined:
            raise RuntimeError("function public.unified_hybrid_search does not exist")
        return self.result

    # Per-arm methods used by the fan-out fallback path.
    def search_user_profile(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("profiles")
        return []

    def search_agent_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("agent_playbooks")
        return []

    def search_user_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("user_playbooks")
        return []


class _MissingCombinedMethodStorage:
    embedding_model_name = "local/minilm-l6-v2"
    supports_embedding = True
    supports_unified_hybrid_search = True

    def __init__(self) -> None:
        self.fanout_calls: list[str] = []

    def search_user_profile(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("profiles")
        return []

    def search_agent_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("agent_playbooks")
        return []

    def search_user_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.fanout_calls.append("user_playbooks")
        return []


def _run_phase_b(
    storage: _CombinedStorage,
    *,
    user_id: str | None = "u",
    tags: list[str] | None = None,
    recency_on: bool = False,
):
    return uss._run_phase_b(
        request=UnifiedSearchRequest(query="q", user_id=user_id, tags=tags, top_k=5),
        org_id="o",
        storage=cast(BaseStorage, storage),
        embedding=[0.1, 0.2],
        query="q",
        top_k=5,
        threshold=0.3,
        recency_on=recency_on,
    )


def test_single_rpc_used_when_supported(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    playbooks = [
        _agent_playbook(1, "a"),
        _agent_playbook(1, "dup"),
        _agent_playbook(2, "b"),
    ]
    user_playbooks = [UserPlaybook(agent_version="v1", request_id="r1", content="up")]
    storage = _CombinedStorage(result=([], playbooks, user_playbooks))

    profiles, agent_playbooks, returned_user_playbooks = _run_phase_b(storage)

    assert len(storage.combined_calls) == 1
    assert storage.fanout_calls == []
    call = storage.combined_calls[0]
    assert call["include_profiles"] is True
    assert call["include_agent_playbooks"] is True
    assert call["include_user_playbooks"] is True
    assert call["agent_playbook_statuses"] == [
        PlaybookStatus.APPROVED,
        PlaybookStatus.PENDING,
    ]
    # Duplicate agent playbook ids are deduped, mirroring the fan-out path.
    assert agent_playbooks is not None
    assert [p.agent_playbook_id for p in agent_playbooks] == [1, 2]
    assert profiles == []
    assert returned_user_playbooks == user_playbooks


def test_single_rpc_skips_profiles_without_user_id(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _CombinedStorage()

    _run_phase_b(storage, user_id=None)

    assert storage.combined_calls[0]["include_profiles"] is False


def test_single_rpc_passes_tag_filter(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _CombinedStorage()

    _run_phase_b(storage, tags=["billing", "support"])

    assert storage.combined_calls[0]["tags"] == ["billing", "support"]


def test_single_rpc_passes_tag_filter_on_scored_path(monkeypatch):
    """Recency routes to ``unified_hybrid_search_scored``; tags must survive."""
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _CombinedStorage()

    _run_phase_b(storage, tags=["billing", "support"], recency_on=True)

    assert storage.combined_calls == []
    assert storage.scored_calls[0]["tags"] == ["billing", "support"]


def test_single_rpc_failure_falls_back_to_fanout(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _CombinedStorage(raise_on_combined=True)

    profiles, agent_playbooks, user_playbooks = _run_phase_b(storage)

    assert len(storage.combined_calls) == 1
    assert set(storage.fanout_calls) == {
        "profiles",
        "agent_playbooks",
        "user_playbooks",
    }
    assert (profiles, agent_playbooks, user_playbooks) == ([], [], [])


def test_single_rpc_and_fallback_share_one_deadline(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _CombinedStorage(raise_on_combined=True)
    observed_deadlines: list[float] = []
    original_remaining = uss._remaining_deadline_seconds

    def record_remaining(deadline_monotonic: float) -> float:
        observed_deadlines.append(deadline_monotonic)
        return original_remaining(deadline_monotonic)

    monkeypatch.setattr(uss, "_remaining_deadline_seconds", record_remaining)

    assert _run_phase_b(storage) == ([], [], [])
    assert len(observed_deadlines) >= 4
    assert len(set(observed_deadlines)) == 1
    assert (
        storage.combined_calls[0]["_retry_deadline_monotonic"] == observed_deadlines[0]
    )


def test_single_rpc_missing_callable_falls_back_to_fanout(monkeypatch):
    monkeypatch.delenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", raising=False)
    storage = _MissingCombinedMethodStorage()

    profiles, agent_playbooks, user_playbooks = uss._run_phase_b(
        request=UnifiedSearchRequest(query="q", user_id="u", top_k=5),
        org_id="o",
        storage=cast(BaseStorage, storage),
        embedding=[0.1, 0.2],
        query="q",
        top_k=5,
        threshold=0.3,
    )

    assert set(storage.fanout_calls) == {
        "profiles",
        "agent_playbooks",
        "user_playbooks",
    }
    assert (profiles, agent_playbooks, user_playbooks) == ([], [], [])


def test_single_rpc_kill_switch_disables_combined_path(monkeypatch):
    monkeypatch.setenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", "0")
    storage = _CombinedStorage()

    _run_phase_b(storage)

    assert storage.combined_calls == []
    assert set(storage.fanout_calls) == {
        "profiles",
        "agent_playbooks",
        "user_playbooks",
    }


def test_cancels_queued_future_but_reports_running_future():
    executor = ThreadPoolExecutor(max_workers=1)
    running_started = threading.Event()
    release_running = threading.Event()
    queued_called = threading.Event()

    def blocking_task() -> list[Any]:
        running_started.set()
        assert release_running.wait(timeout=2)
        return []

    def queued_task() -> list[Any]:
        queued_called.set()
        return []

    deadline = time.monotonic() + 5
    running = uss._submit_with_current_context(
        executor, "running", deadline, blocking_task
    )
    assert running_started.wait(timeout=1)
    queued = uss._submit_with_current_context(executor, "queued", deadline, queued_task)

    try:
        counts = uss._cancel_unfinished_futures([running, queued])

        assert counts == {
            "cancelled_queued_count": 1,
            "running_count": 1,
            "completed_count": 0,
        }
        assert queued.future.cancelled()
        assert not queued_called.is_set()
    finally:
        release_running.set()
        running.future.result(timeout=1)
        executor.shutdown(wait=True)


def test_executor_span_separates_queue_wait_from_execution_time():
    records: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=1)
    running_started = threading.Event()
    release_running = threading.Event()

    def blocking_task() -> None:
        running_started.set()
        assert release_running.wait(timeout=2)

    configure_tracer(_RecordingTracer(records))
    deadline = time.monotonic() + 5
    try:
        running = uss._submit_with_current_context(
            executor, "running", deadline, blocking_task
        )
        assert running_started.wait(timeout=1)
        queued = uss._submit_with_current_context(
            executor, "queued", deadline, lambda: "done"
        )
        time.sleep(0.05)
        release_running.set()
        assert queued.future.result(timeout=1) == "done"
        running.future.result(timeout=1)
    finally:
        release_running.set()
        executor.shutdown(wait=True)
        configure_tracer(None)

    queued_record = next(
        record
        for record in records
        if record["name"] == "search.executor.task" and record["task"] == "queued"
    )
    assert queued_record["queue_wait_ms"] >= 10
    assert queued_record["execution_ms"] >= 0
    assert queued_record["deadline_remaining_ms_at_start"] > 0
    assert queued_record["outcome"] == "success"


@pytest.mark.parametrize(
    ("exc", "expected_outcome"),
    [(RuntimeError("failed"), "failure"), (FuturesTimeoutError(), "timeout")],
)
def test_executor_span_records_failure_outcomes(
    exc: BaseException, expected_outcome: str
):
    records: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=1)

    def fail() -> None:
        raise exc

    configure_tracer(_RecordingTracer(records))
    try:
        tracked = uss._submit_with_current_context(
            executor, "failing", time.monotonic() + 5, fail
        )
        with pytest.raises(type(exc)):
            tracked.future.result(timeout=1)
    finally:
        executor.shutdown(wait=True)
        configure_tracer(None)

    record = next(
        record for record in records if record["name"] == "search.executor.task"
    )
    assert record["outcome"] == expected_outcome
    assert record["execution_ms"] >= 0


def test_executor_task_does_not_call_storage_after_deadline():
    records: list[dict[str, Any]] = []
    called = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    configure_tracer(_RecordingTracer(records))
    try:
        tracked = uss._submit_with_current_context(
            executor,
            "expired",
            time.monotonic() - 1,
            called.set,
        )
        with pytest.raises(FuturesTimeoutError):
            tracked.future.result(timeout=1)
    finally:
        executor.shutdown(wait=True)
        configure_tracer(None)

    assert not called.is_set()
    record = next(
        record for record in records if record["name"] == "search.executor.task"
    )
    assert record["deadline_remaining_ms_at_start"] == 0
    assert record["outcome"] == "deadline_expired"


def test_cancelled_queued_task_records_cancellation_outcome():
    records: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=1)
    running_started = threading.Event()
    release_running = threading.Event()

    def blocking_task() -> None:
        running_started.set()
        assert release_running.wait(timeout=2)

    configure_tracer(_RecordingTracer(records))
    deadline = time.monotonic() + 5
    try:
        running = uss._submit_with_current_context(
            executor, "running", deadline, blocking_task
        )
        assert running_started.wait(timeout=1)
        queued = uss._submit_with_current_context(
            executor, "queued", deadline, lambda: None
        )

        counts = uss._cancel_unfinished_futures([running, queued])

        assert counts["cancelled_queued_count"] == 1
    finally:
        release_running.set()
        executor.shutdown(wait=True)
        configure_tracer(None)

    record = next(
        record
        for record in records
        if record["name"] == "search.executor.task" and record["task"] == "queued"
    )
    assert record["deadline_remaining_ms_at_start"] is None
    assert record["execution_ms"] == 0
    assert record["outcome"] == "cancelled_queued"


def test_phase_b_timeout_records_parent_cancellation_counts(monkeypatch):
    records: list[dict[str, Any]] = []

    class TimeoutStorage(_CombinedStorage):
        def search_agent_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            raise FuturesTimeoutError("storage timeout")

    monkeypatch.setenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", "0")
    configure_tracer(_RecordingTracer(records))
    try:
        assert _run_phase_b(TimeoutStorage()) == (None, None, None)
    finally:
        configure_tracer(None)

    phase_record = next(
        record for record in records if record["name"] == "search.phase_b"
    )
    assert phase_record["timed_out"] is True
    assert (
        phase_record["cancelled_queued_count"]
        + phase_record["running_count"]
        + phase_record["completed_count"]
        == 3
    )


def test_phase_b_failure_cleans_up_active_futures(monkeypatch):
    cleaned: list[list[uss._TrackedSearchFuture]] = []

    class FailingStorage(_CombinedStorage):
        def search_agent_playbooks(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            raise RuntimeError("storage failed")

    monkeypatch.setenv("REFLEXIO_UNIFIED_SEARCH_SINGLE_RPC", "0")
    monkeypatch.setattr(
        uss, "_cancel_unfinished_futures", lambda futures: cleaned.append(futures)
    )

    assert _run_phase_b(FailingStorage()) == (None, None, None)
    assert len(cleaned) == 1
    assert {tracked.task for tracked in cleaned[0]} == {
        "profiles",
        "agent_playbooks",
        "user_playbooks",
    }
