from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

import reflexio.server.routes._metering as metering
import reflexio.server.services.search_metering_worker as worker_module
from reflexio.server.services.search_metering_worker import (
    SearchMeteringJob,
    SearchMeteringWorker,
    enqueue_search_metering,
    reset_search_metering_worker_for_tests,
)
from reflexio.server.tracing import configure_tracer


@pytest.fixture(autouse=True)
def _reset_worker_and_tracer() -> Iterator[None]:
    reset_search_metering_worker_for_tests()
    configure_tracer(None)
    yield
    reset_search_metering_worker_for_tests()
    configure_tracer(None)


def _job(*, surfaced_count: int = 1) -> SearchMeteringJob:
    return SearchMeteringJob(
        org_id="org-1",
        surfaced_count=surfaced_count,
        record_search_request=True,
        request_id="request-1",
        trace_context={"sentry-trace": "trace-parent"},
    )


def test_worker_records_both_metrics_and_linked_background_trace(monkeypatch) -> None:
    calls: list[tuple[str, int | None]] = []

    def record_search(**_kwargs) -> bool:
        calls.append(("search_request", None))
        return True

    def record_applied(*, surfaced_count: int, **_kwargs) -> bool:
        calls.append(("learning_applied", surfaced_count))
        return True

    monkeypatch.setattr(metering, "_meter_search_request", record_search)
    monkeypatch.setattr(metering, "_meter_applied_learnings", record_applied)

    class RecordingSpan:
        def __init__(self) -> None:
            self.data: dict[str, object] = {}

        def set_data(self, key: str, value: object) -> None:
            self.data[key] = value

    class RecordingTracer:
        def __init__(self) -> None:
            self.transactions: list[tuple[str, str, dict[str, str], dict]] = []
            self.spans: list[tuple[str, RecordingSpan]] = []

        def capture_context(self) -> dict[str, str]:
            return {"sentry-trace": "trace-parent"}

        @contextmanager
        def transaction(
            self,
            name: str,
            *,
            op: str,
            trace_context: dict[str, str],
            **data,
        ):
            self.transactions.append((name, op, dict(trace_context), data))
            yield RecordingSpan()

        @contextmanager
        def span(self, name: str, **_data):
            span = RecordingSpan()
            self.spans.append((name, span))
            yield span

    tracer = RecordingTracer()
    configure_tracer(tracer)
    worker = SearchMeteringWorker(worker_count=1)
    assert worker.enqueue(_job(surfaced_count=3))
    assert worker.wait_until_idle(timeout=1.0)
    assert worker.stop() == 0

    assert calls == [("search_request", None), ("learning_applied", 3)]
    assert tracer.transactions[0][:3] == (
        "search.metering",
        "queue.process",
        {"sentry-trace": "trace-parent"},
    )
    assert tracer.transactions[0][3]["surfaced_count"] == 3
    assert [name for name, _span in tracer.spans] == [
        "search.metering.search_request",
        "search.metering.learning_applied",
    ]
    assert all(span.data["emitted"] is True for _name, span in tracer.spans)


def test_enqueue_returns_while_database_work_is_blocked(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_process(_job: SearchMeteringJob) -> None:
        started.set()
        assert release.wait(timeout=2.0)

    worker = SearchMeteringWorker(worker_count=1)
    monkeypatch.setattr(worker, "_process_job", blocked_process)

    assert worker.enqueue(_job())
    assert started.wait(timeout=1.0)
    assert not worker.wait_until_idle(timeout=0.01)
    release.set()
    assert worker.wait_until_idle(timeout=1.0)
    assert worker.stop() == 0


def test_full_queue_drops_without_backpressure_and_reports(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()
    anomalies: list[tuple[str, dict]] = []

    def blocked_process(_job: SearchMeteringJob) -> None:
        started.set()
        assert release.wait(timeout=2.0)

    monkeypatch.setattr(
        worker_module,
        "capture_anomaly",
        lambda name, **data: anomalies.append((name, data)),
    )
    worker = SearchMeteringWorker(queue_capacity=1, worker_count=1)
    monkeypatch.setattr(worker, "_process_job", blocked_process)

    assert worker.enqueue(_job())
    assert started.wait(timeout=1.0)
    assert worker.enqueue(_job())
    assert worker.enqueue(_job()) is False
    assert anomalies[-1][0] == "search.metering.job_dropped"
    assert anomalies[-1][1]["reason"] == "queue_full"

    release.set()
    assert worker.wait_until_idle(timeout=1.0)
    assert worker.stop() == 0


def test_shutdown_abandons_only_queued_tail_after_timeout(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocked_process(_job: SearchMeteringJob) -> None:
        started.set()
        release.wait(timeout=2.0)

    monkeypatch.setattr(worker_module, "capture_anomaly", lambda *_a, **_kw: None)
    worker = SearchMeteringWorker(queue_capacity=2, worker_count=1)
    monkeypatch.setattr(worker, "_process_job", blocked_process)

    assert worker.enqueue(_job())
    assert started.wait(timeout=1.0)
    assert worker.enqueue(_job())
    assert worker.stop(timeout=0.0) == 2
    release.set()
    for thread in worker._threads:
        thread.join(timeout=1.0)


def test_non_production_and_empty_applied_only_jobs_do_not_start_worker(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        worker_module,
        "get_search_metering_worker",
        lambda: pytest.fail("worker must not start"),
    )
    assert (
        enqueue_search_metering(
            org_id="org",
            caller_type="dashboard",
            surfaced_count=2,
            record_search_request=True,
        )
        is False
    )
    assert (
        enqueue_search_metering(
            org_id="org",
            caller_type="production_agent",
            surfaced_count=0,
            record_search_request=False,
        )
        is False
    )
