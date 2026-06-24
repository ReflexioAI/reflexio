from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

import pytest
from fastapi import status

import reflexio.server.operation_limiter as operation_limiter_module
from reflexio.server.operation_limiter import (
    limiter_http_exception,
    operation_limit,
    publish_hardware_capacity_snapshot,
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
    assert all(
        event.metadata["active"] == 1
        for event in events
        if event.event_name == "limiter_acquired"
    )


def test_publish_hardware_capacity_snapshot_reports_hardware_guidance(monkeypatch):
    monkeypatch.setenv("REFLEXIO_PUBLISH_CONCURRENCY_LIMIT", "7")
    monkeypatch.setenv("REFLEXIO_PUBLISH_CONCURRENCY_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setattr(operation_limiter_module.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(operation_limiter_module, "_cgroup_cpu_count", lambda: None)
    monkeypatch.setattr(
        operation_limiter_module, "_cgroup_memory_limit_bytes", lambda: None
    )
    monkeypatch.setattr(
        operation_limiter_module, "_cgroup_memory_current_bytes", lambda: None
    )
    monkeypatch.setattr(operation_limiter_module, "_cgroup_pids_limit", lambda: None)
    monkeypatch.setattr(operation_limiter_module, "_cgroup_pids_current", lambda: None)
    monkeypatch.setattr(
        operation_limiter_module, "_process_os_thread_count", lambda: 10
    )
    monkeypatch.setattr(operation_limiter_module, "_resource_limit_nproc", lambda: 100)
    monkeypatch.setattr(operation_limiter_module, "_linux_threads_max", lambda: 1000)
    monkeypatch.setattr(operation_limiter_module, "_available_memory_bytes", lambda: 42)
    monkeypatch.setattr(
        operation_limiter_module,
        "_anyio_threadpool_stats",
        lambda: {
            "anyio_threadpool_total": 40,
            "anyio_threadpool_borrowed": 3,
            "anyio_threadpool_waiting": 2,
        },
    )

    snapshot = publish_hardware_capacity_snapshot()

    assert snapshot["configured_publish_limit"] == 7
    assert snapshot["publish_timeout_seconds"] == 2.5
    assert snapshot["estimated_threads_per_publish"] == 3
    assert snapshot["hardware_guidance_publish_limit"] == 30
    assert snapshot["effective_cpu_count"] == 64
    assert snapshot["logical_cpu_count"] == 64
    assert snapshot["cgroup_cpu_count"] is None
    assert snapshot["process_os_threads"] == 10
    assert snapshot["rlimit_nproc_soft"] == 100
    assert snapshot["linux_threads_max"] == 1000
    assert snapshot["cgroup_pids_limit"] is None
    assert snapshot["cgroup_pids_current"] is None
    assert snapshot["cgroup_memory_limit_bytes"] is None
    assert snapshot["cgroup_memory_current_bytes"] is None
    assert snapshot["available_memory_bytes"] == 42
    assert snapshot["anyio_threadpool_total"] == 40
    assert snapshot["anyio_threadpool_borrowed"] == 3
    assert snapshot["anyio_threadpool_waiting"] == 2


def test_publish_hardware_capacity_snapshot_prefers_cgroup_v2(monkeypatch, tmp_path):
    (tmp_path / "cpu.max").write_text("200000 100000")
    (tmp_path / "memory.max").write_text("1000")
    (tmp_path / "memory.current").write_text("600")
    (tmp_path / "pids.max").write_text("20")
    (tmp_path / "pids.current").write_text("5")
    monkeypatch.setattr(operation_limiter_module, "_CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(
        operation_limiter_module, "_PROC_SELF_CGROUP", tmp_path / "missing"
    )
    monkeypatch.setattr(operation_limiter_module.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(
        operation_limiter_module, "_process_os_thread_count", lambda: 10
    )
    monkeypatch.setattr(operation_limiter_module, "_resource_limit_nproc", lambda: 100)
    monkeypatch.setattr(operation_limiter_module, "_linux_threads_max", lambda: 1000)
    monkeypatch.setattr(
        operation_limiter_module,
        "_anyio_threadpool_stats",
        lambda: {
            "anyio_threadpool_total": 40,
            "anyio_threadpool_borrowed": 3,
            "anyio_threadpool_waiting": 2,
        },
    )

    snapshot = publish_hardware_capacity_snapshot()

    assert snapshot["effective_cpu_count"] == 2.0
    assert snapshot["cgroup_cpu_count"] == 2.0
    assert snapshot["cgroup_pids_limit"] == 20
    assert snapshot["cgroup_pids_current"] == 5
    assert snapshot["cgroup_memory_limit_bytes"] == 1000
    assert snapshot["cgroup_memory_current_bytes"] == 600
    assert snapshot["available_memory_bytes"] == 400
    assert snapshot["hardware_guidance_publish_limit"] == 4


def test_publish_hardware_capacity_snapshot_reads_cgroup_v1(monkeypatch, tmp_path):
    (tmp_path / "self.cgroup").write_text(
        """2:cpu,cpuacct:/ecs/task
3:memory:/ecs/task
4:pids:/ecs/task"""
    )
    cpu_dir = tmp_path / "cpu" / "ecs" / "task"
    memory_dir = tmp_path / "memory" / "ecs" / "task"
    pids_dir = tmp_path / "pids" / "ecs" / "task"
    cpu_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    pids_dir.mkdir(parents=True)
    (cpu_dir / "cpu.cfs_quota_us").write_text("150000")
    (cpu_dir / "cpu.cfs_period_us").write_text("100000")
    (memory_dir / "memory.limit_in_bytes").write_text("1000")
    (memory_dir / "memory.usage_in_bytes").write_text("250")
    (pids_dir / "pids.max").write_text("13")
    (pids_dir / "pids.current").write_text("4")
    monkeypatch.setattr(operation_limiter_module, "_CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(
        operation_limiter_module, "_PROC_SELF_CGROUP", tmp_path / "self.cgroup"
    )
    monkeypatch.setattr(operation_limiter_module.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(
        operation_limiter_module, "_process_os_thread_count", lambda: 10
    )
    monkeypatch.setattr(operation_limiter_module, "_resource_limit_nproc", lambda: 100)
    monkeypatch.setattr(operation_limiter_module, "_linux_threads_max", lambda: 1000)
    monkeypatch.setattr(
        operation_limiter_module,
        "_anyio_threadpool_stats",
        lambda: {
            "anyio_threadpool_total": 40,
            "anyio_threadpool_borrowed": 3,
            "anyio_threadpool_waiting": 2,
        },
    )

    snapshot = publish_hardware_capacity_snapshot()

    assert snapshot["effective_cpu_count"] == 1.5
    assert snapshot["cgroup_cpu_count"] == 1.5
    assert snapshot["cgroup_pids_limit"] == 13
    assert snapshot["cgroup_pids_current"] == 4
    assert snapshot["cgroup_memory_limit_bytes"] == 1000
    assert snapshot["cgroup_memory_current_bytes"] == 250
    assert snapshot["available_memory_bytes"] == 750
    assert snapshot["hardware_guidance_publish_limit"] == 3


def test_publish_limiter_timeout_logs_hardware_pressure(monkeypatch, caplog):
    monkeypatch.setenv("REFLEXIO_PUBLISH_CONCURRENCY_LIMIT", "1")
    monkeypatch.setattr(
        operation_limiter_module,
        "publish_hardware_capacity_snapshot",
        lambda: {"hardware_guidance_publish_limit": 10},
    )
    ready = threading.Event()
    release = threading.Event()

    def _hold_limit() -> None:
        with operation_limit("org_1", "publish", timeout_seconds=0.1):
            ready.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=_hold_limit)
    holder.start()
    try:
        assert ready.wait(timeout=5)
        with (
            caplog.at_level(logging.WARNING),
            pytest.raises(TimeoutError),
            operation_limit("org_1", "publish", timeout_seconds=0.01),
        ):
            pass
    finally:
        release.set()
        holder.join(timeout=5)

    assert "publish_limiter_timeout org_id=org_1 limit=1 active=1" in caplog.text
    assert "hardware_guidance_publish_limit" in caplog.text


def test_limiter_http_exception_maps_search_to_429():
    exc = limiter_http_exception("search")

    assert exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert exc.headers == {"Retry-After": "2"}
