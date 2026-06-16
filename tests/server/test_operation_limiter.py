from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest
from fastapi import status

from reflexio.server.operation_limiter import (
    limiter_http_exception,
    operation_limit,
    reset_operation_limiters_for_tests,
)
from reflexio.server.tracing import configure_tracer
from reflexio.server.usage_metrics import UsageEvent, configure_usage_event_recorder


@pytest.fixture(autouse=True)
def _reset_limiters_and_metrics():
    reset_operation_limiters_for_tests()
    configure_tracer(None)
    configure_usage_event_recorder(None)
    yield
    reset_operation_limiters_for_tests()
    configure_tracer(None)
    configure_usage_event_recorder(None)


class _RecordingSpan:
    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def set_data(self, key: str, value: object) -> None:
        self.data[key] = value


class _RecordingTracer:
    def __init__(self) -> None:
        self.started: list[tuple[str, dict[str, object]]] = []
        self.spans: list[_RecordingSpan] = []

    @contextmanager
    def span(self, name: str, **data: object):
        span = _RecordingSpan()
        self.started.append((name, data))
        self.spans.append(span)
        yield span


def test_operation_limiter_records_success_and_timeout(monkeypatch):
    monkeypatch.setenv("REFLEXIO_SEARCH_CONCURRENCY_LIMIT", "1")
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    ready = threading.Event()
    release = threading.Event()

    def _hold_limit() -> None:
        with operation_limit("org_1", "search", timeout_seconds=0.1):
            ready.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=_hold_limit)
    holder.start()
    try:
        assert ready.wait(timeout=5)
        with (
            pytest.raises(TimeoutError),
            operation_limit("org_1", "search", timeout_seconds=0.01),
        ):
            pass
    finally:
        release.set()
        holder.join(timeout=5)

    event_names = [event.event_name for event in events]
    assert "limiter_acquired" in event_names
    assert "limiter_acquire_timeout" in event_names
    timeout_event = next(e for e in events if e.event_name == "limiter_acquire_timeout")
    assert timeout_event.event_category == "limiter"
    assert timeout_event.pipeline == "search"
    assert timeout_event.metadata["operation"] == "search"


def test_operation_limiter_emits_acquire_span(monkeypatch):
    monkeypatch.setenv("REFLEXIO_SEARCH_CONCURRENCY_LIMIT", "3")
    tracer = _RecordingTracer()
    configure_tracer(tracer)

    with operation_limit("org_1", "search", timeout_seconds=0.1):
        pass

    assert tracer.started == [
        (
            "search.operation_limit.acquire",
            {"limit": 3, "waiting": 1, "wait_forever": False},
        )
    ]
    assert tracer.spans[0].data["acquired"] is True
    assert isinstance(tracer.spans[0].data["wait_ms"], int)


def test_operation_limiter_wait_forever_queues_until_slot_available(monkeypatch):
    monkeypatch.setenv("REFLEXIO_PUBLISH_CONCURRENCY_LIMIT", "1")
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    ready = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _hold_limit() -> None:
        with operation_limit("org_1", "publish", timeout_seconds=0.1):
            ready.set()
            release.wait(timeout=5)

    def _wait_for_limit() -> None:
        with operation_limit("org_1", "publish", wait_forever=True):
            finished.set()

    holder = threading.Thread(target=_hold_limit)
    waiter = threading.Thread(target=_wait_for_limit)
    holder.start()
    try:
        assert ready.wait(timeout=5)
        waiter.start()
        assert not finished.wait(timeout=0.05)
    finally:
        release.set()
        holder.join(timeout=5)
        waiter.join(timeout=5)

    assert finished.is_set()
    assert [event.event_name for event in events].count("limiter_acquired") == 2


def test_limiter_http_exception_maps_search_to_429():
    exc = limiter_http_exception("search")

    assert exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert exc.headers == {"Retry-After": "2"}
