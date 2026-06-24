"""Process-local concurrency limiters for latency-sensitive operations."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException, status

from reflexio.server.tracing import profile_step
from reflexio.server.usage_metrics import record_usage_event

OperationName = Literal["search", "publish", "aggregation"]

logger = logging.getLogger(__name__)

_DEFAULT_LIMITS: dict[OperationName, int] = {
    "search": 8,
    "publish": 4,
    "aggregation": 1,
}
_DEFAULT_TIMEOUT_SECONDS: dict[OperationName, float] = {
    "search": 2.0,
    "publish": 5.0,
    "aggregation": 1.0,
}
_ESTIMATED_THREADS_PER_PUBLISH = 3
_PUBLISH_WAIT_LOG_THRESHOLD_MS = 250
_PUBLISH_PRESSURE_LOG_INTERVAL_SECONDS = 60.0
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_last_publish_pressure_log_by_org: dict[str, float] = {}


@dataclass
class _LimiterState:
    semaphore: threading.BoundedSemaphore
    limit: int
    waiting: int = 0
    active: int = 0


_limiters: dict[tuple[str, OperationName], _LimiterState] = {}
_limiters_lock = threading.Lock()


def _env_key(operation: OperationName, suffix: str) -> str:
    return f"REFLEXIO_{operation.upper()}_CONCURRENCY_{suffix}"


def _limit_for(operation: OperationName) -> int:
    raw = os.getenv(_env_key(operation, "LIMIT"), "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return _DEFAULT_LIMITS[operation]


def _timeout_for(operation: OperationName) -> float:
    raw = os.getenv(_env_key(operation, "TIMEOUT_SECONDS"), "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return _DEFAULT_TIMEOUT_SECONDS[operation]


def _resource_limit_nproc() -> int | str | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - platform-specific
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    except (OSError, ValueError, AttributeError):
        return None
    if soft == resource.RLIM_INFINITY:
        return "unlimited"
    return int(soft)


def _linux_threads_max() -> int | None:
    try:
        with Path("/proc/sys/kernel/threads-max").open() as threads_max_file:
            return int(threads_max_file.read().strip())
    except (OSError, ValueError):
        return None


def _process_os_thread_count() -> int | None:
    try:
        with Path("/proc/self/status").open() as status_file:
            for line in status_file:
                if line.startswith("Threads:"):
                    return int(line.split(":", maxsplit=1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


def _available_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(page_size, int) and isinstance(available_pages, int):
        return page_size * available_pages
    return None


def _read_path_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _positive_int_from_text(text: str | None) -> int | None:
    if text is None or text == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _cgroup_candidate_paths(controller: str, filename: str) -> list[Path]:
    candidates = [
        _CGROUP_ROOT / filename,
        _CGROUP_ROOT / controller / filename,
    ]
    cgroup_text = _read_path_text(_PROC_SELF_CGROUP)
    if not cgroup_text:
        return candidates

    for line in cgroup_text.splitlines():
        parts = line.split(":", maxsplit=2)
        if len(parts) != 3:
            continue
        controllers_text = parts[1]
        relative = parts[2].strip().lstrip("/")
        relative_path = Path(relative) if relative else Path()
        controllers = [item for item in controllers_text.split(",") if item]
        if controllers_text == "":
            candidates.append(_CGROUP_ROOT / relative_path / filename)
            continue
        if controller not in controllers:
            continue
        candidates.append(_CGROUP_ROOT / controller / relative_path / filename)
        candidates.append(
            _CGROUP_ROOT / ",".join(controllers) / relative_path / filename
        )
        candidates.append(_CGROUP_ROOT / relative_path / filename)
    return candidates


def _read_first_cgroup_value(controller: str, filename: str) -> str | None:
    seen: set[Path] = set()
    for path in _cgroup_candidate_paths(controller, filename):
        if path in seen:
            continue
        seen.add(path)
        value = _read_path_text(path)
        if value:
            return value
    return None


def _cgroup_v2_cpu_count() -> float | None:
    raw = _read_first_cgroup_value("unified", "cpu.max")
    if raw is None:
        return None
    parts = raw.split()
    if len(parts) < 2 or parts[0] == "max":
        return None
    try:
        quota = int(parts[0])
        period = int(parts[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _cgroup_v1_cpu_count() -> float | None:
    quota = _positive_int_from_text(_read_first_cgroup_value("cpu", "cpu.cfs_quota_us"))
    period = _positive_int_from_text(
        _read_first_cgroup_value("cpu", "cpu.cfs_period_us")
    )
    if quota is None or period is None:
        return None
    return quota / period


def _cgroup_cpu_count() -> float | None:
    return _cgroup_v2_cpu_count() or _cgroup_v1_cpu_count()


def _cgroup_memory_limit_bytes() -> int | None:
    v2_limit = _positive_int_from_text(
        _read_first_cgroup_value("unified", "memory.max")
    )
    if v2_limit is not None:
        return v2_limit
    return _positive_int_from_text(
        _read_first_cgroup_value("memory", "memory.limit_in_bytes")
    )


def _cgroup_memory_current_bytes() -> int | None:
    v2_current = _positive_int_from_text(
        _read_first_cgroup_value("unified", "memory.current")
    )
    if v2_current is not None:
        return v2_current
    return _positive_int_from_text(
        _read_first_cgroup_value("memory", "memory.usage_in_bytes")
    )


def _cgroup_pids_limit() -> int | None:
    v2_limit = _positive_int_from_text(_read_first_cgroup_value("unified", "pids.max"))
    if v2_limit is not None:
        return v2_limit
    return _positive_int_from_text(_read_first_cgroup_value("pids", "pids.max"))


def _cgroup_pids_current() -> int | None:
    v2_current = _positive_int_from_text(
        _read_first_cgroup_value("unified", "pids.current")
    )
    if v2_current is not None:
        return v2_current
    return _positive_int_from_text(_read_first_cgroup_value("pids", "pids.current"))


def _anyio_threadpool_stats() -> dict[str, int | None]:
    try:
        from anyio.to_thread import current_default_thread_limiter

        stats = current_default_thread_limiter().statistics()
        return {
            "anyio_threadpool_total": int(stats.total_tokens),
            "anyio_threadpool_borrowed": int(stats.borrowed_tokens),
            "anyio_threadpool_waiting": int(stats.tasks_waiting),
        }
    except Exception:  # noqa: BLE001 - best-effort diagnostics only
        return {
            "anyio_threadpool_total": None,
            "anyio_threadpool_borrowed": None,
            "anyio_threadpool_waiting": None,
        }


def publish_hardware_capacity_snapshot() -> dict[str, int | str | float | None]:
    """Return hardware/threadpool diagnostics for publish concurrency tuning."""
    host_cpu_count = os.cpu_count()
    cgroup_cpu_count = _cgroup_cpu_count()
    effective_cpu_count = cgroup_cpu_count or host_cpu_count
    process_os_threads = _process_os_thread_count()
    nproc_limit = _resource_limit_nproc()
    threads_max = _linux_threads_max()
    cgroup_pids_limit = _cgroup_pids_limit()
    cgroup_pids_current = _cgroup_pids_current()
    cgroup_memory_limit = _cgroup_memory_limit_bytes()
    cgroup_memory_current = _cgroup_memory_current_bytes()
    cgroup_available_memory = (
        max(0, cgroup_memory_limit - cgroup_memory_current)
        if cgroup_memory_limit is not None and cgroup_memory_current is not None
        else None
    )
    anyio_stats = _anyio_threadpool_stats()
    host_thread_budget_candidates = [
        value - process_os_threads
        for value in (nproc_limit, threads_max)
        if isinstance(value, int) and process_os_threads is not None
    ]
    cgroup_pid_budget = (
        cgroup_pids_limit - cgroup_pids_current
        if cgroup_pids_limit is not None and cgroup_pids_current is not None
        else None
    )
    thread_budget_candidates = [
        value
        for value in (cgroup_pid_budget, *host_thread_budget_candidates)
        if isinstance(value, int)
    ]
    max_by_os_threads = (
        max(1, min(thread_budget_candidates) // _ESTIMATED_THREADS_PER_PUBLISH)
        if thread_budget_candidates
        else None
    )
    guidance_candidates = [
        value
        for value in (
            anyio_stats["anyio_threadpool_total"],
            max_by_os_threads,
            max(1, int(effective_cpu_count * 2))
            if effective_cpu_count is not None
            else None,
        )
        if isinstance(value, int) and value > 0
    ]
    hardware_guidance = min(guidance_candidates) if guidance_candidates else None
    return {
        "configured_publish_limit": _limit_for("publish"),
        "publish_timeout_seconds": _timeout_for("publish"),
        "hardware_guidance_publish_limit": hardware_guidance,
        "estimated_threads_per_publish": _ESTIMATED_THREADS_PER_PUBLISH,
        "effective_cpu_count": effective_cpu_count,
        "logical_cpu_count": host_cpu_count,
        "cgroup_cpu_count": cgroup_cpu_count,
        "python_active_threads": threading.active_count(),
        "process_os_threads": process_os_threads,
        "rlimit_nproc_soft": nproc_limit,
        "linux_threads_max": threads_max,
        "cgroup_pids_limit": cgroup_pids_limit,
        "cgroup_pids_current": cgroup_pids_current,
        "cgroup_memory_limit_bytes": cgroup_memory_limit,
        "cgroup_memory_current_bytes": cgroup_memory_current,
        "available_memory_bytes": cgroup_available_memory
        if cgroup_available_memory is not None
        else _available_memory_bytes(),
        **anyio_stats,
    }


def log_publish_hardware_capacity() -> None:
    """Log one startup snapshot for publish concurrency hardware tuning."""
    logger.info(
        "publish_concurrency_hardware_capacity %s",
        publish_hardware_capacity_snapshot(),
    )


def _state_for(org_id: str, operation: OperationName) -> _LimiterState:
    key = (str(org_id), operation)
    desired_limit = _limit_for(operation)
    with _limiters_lock:
        state = _limiters.get(key)
        if state is None or state.limit != desired_limit:
            state = _LimiterState(
                semaphore=threading.BoundedSemaphore(desired_limit),
                limit=desired_limit,
            )
            _limiters[key] = state
        return state


def _record_limiter_event(
    *,
    org_id: str,
    operation: OperationName,
    event_name: str,
    outcome: str,
    wait_ms: int,
    waiting: int,
    limit: int,
    active: int,
) -> None:
    record_usage_event(
        org_id=org_id,
        event_name=event_name,
        event_category="limiter",
        pipeline=operation,
        outcome=outcome,
        duration_ms=wait_ms,
        metadata={
            "operation": operation,
            "waiting": waiting,
            "limit": limit,
            "active": active,
        },
    )


def _should_log_publish_pressure(org_id: str, now: float) -> bool:
    with _limiters_lock:
        last_logged = _last_publish_pressure_log_by_org.get(org_id)
        if (
            last_logged is not None
            and now - last_logged < _PUBLISH_PRESSURE_LOG_INTERVAL_SECONDS
        ):
            return False
        _last_publish_pressure_log_by_org[org_id] = now
        return True


@contextmanager
def operation_limit(
    org_id: str,
    operation: OperationName,
    *,
    timeout_seconds: float | None = None,
    wait_forever: bool = False,
) -> Any:
    """Acquire the per-org limiter for an operation, recording wait metrics."""
    state = _state_for(org_id, operation)
    timeout = (
        None
        if wait_forever
        else (_timeout_for(operation) if timeout_seconds is None else timeout_seconds)
    )
    start = time.perf_counter()
    with _limiters_lock:
        state.waiting += 1
        waiting = state.waiting
    acquired = False
    try:
        with profile_step(
            f"{operation}.operation_limit.acquire",
            limit=state.limit,
            waiting=waiting,
            wait_forever=wait_forever,
        ) as span:
            acquired = (
                state.semaphore.acquire()
                if timeout is None
                else state.semaphore.acquire(timeout=timeout)
            )
            wait_ms = int((time.perf_counter() - start) * 1000)
            span.set_data("wait_ms", wait_ms)
            span.set_data("acquired", acquired)
        with _limiters_lock:
            state.waiting -= 1
            waiting_after = state.waiting
            if acquired:
                state.active += 1
            active = state.active
        if not acquired:
            _record_limiter_event(
                org_id=org_id,
                operation=operation,
                event_name="limiter_acquire_timeout",
                outcome="timeout",
                wait_ms=wait_ms,
                waiting=waiting_after,
                limit=state.limit,
                active=active,
            )
            if operation == "publish":
                logger.warning(
                    "publish_limiter_timeout org_id=%s limit=%s active=%s "
                    "waiting=%s wait_ms=%s timeout_seconds=%s hardware=%s",
                    org_id,
                    state.limit,
                    active,
                    waiting_after,
                    wait_ms,
                    timeout,
                    publish_hardware_capacity_snapshot(),
                )
            raise TimeoutError(
                f"{operation} concurrency limit reached for org {org_id}"
            )
        if (
            operation == "publish"
            and wait_ms >= _PUBLISH_WAIT_LOG_THRESHOLD_MS
            and _should_log_publish_pressure(str(org_id), time.monotonic())
        ):
            logger.info(
                "publish_limiter_wait org_id=%s limit=%s active=%s waiting=%s "
                "wait_ms=%s timeout_seconds=%s hardware=%s",
                org_id,
                state.limit,
                active,
                waiting_after,
                wait_ms,
                timeout,
                publish_hardware_capacity_snapshot(),
            )
        _record_limiter_event(
            org_id=org_id,
            operation=operation,
            event_name="limiter_acquired",
            outcome="acquired",
            wait_ms=wait_ms,
            waiting=waiting,
            limit=state.limit,
            active=active,
        )
        yield
    finally:
        if acquired:
            with _limiters_lock:
                state.active = max(0, state.active - 1)
            state.semaphore.release()


def run_with_operation_limit[T](
    *,
    org_id: str,
    operation: OperationName,
    fn: Callable[[], T],
    timeout_seconds: float | None = None,
    wait_forever: bool = False,
) -> T:
    with operation_limit(
        org_id,
        operation,
        timeout_seconds=timeout_seconds,
        wait_forever=wait_forever,
    ):
        return fn()


def limiter_http_exception(operation: OperationName) -> HTTPException:
    """Return the API error used when an operation limiter is saturated."""
    retry_after = str(max(1, int(_timeout_for(operation))))
    status_code = (
        status.HTTP_429_TOO_MANY_REQUESTS
        if operation == "search"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return HTTPException(
        status_code=status_code,
        detail=f"{operation} concurrency limit reached",
        headers={"Retry-After": retry_after},
    )


def reset_operation_limiters_for_tests() -> None:
    with _limiters_lock:
        _limiters.clear()
        _last_publish_pressure_log_by_org.clear()
