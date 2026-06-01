"""Process-local concurrency limiters for latency-sensitive operations."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import HTTPException, status

from reflexio.server.usage_metrics import record_usage_event

OperationName = Literal["search", "publish", "aggregation"]

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


@dataclass
class _LimiterState:
    semaphore: threading.BoundedSemaphore
    limit: int
    waiting: int = 0


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
        },
    )


@contextmanager
def operation_limit(
    org_id: str,
    operation: OperationName,
    *,
    timeout_seconds: float | None = None,
) -> Any:
    """Acquire the per-org limiter for an operation, recording wait metrics."""
    state = _state_for(org_id, operation)
    timeout = _timeout_for(operation) if timeout_seconds is None else timeout_seconds
    start = time.perf_counter()
    with _limiters_lock:
        state.waiting += 1
        waiting = state.waiting
    acquired = False
    try:
        acquired = state.semaphore.acquire(timeout=timeout)
        wait_ms = int((time.perf_counter() - start) * 1000)
        with _limiters_lock:
            state.waiting -= 1
            waiting_after = state.waiting
        if not acquired:
            _record_limiter_event(
                org_id=org_id,
                operation=operation,
                event_name="limiter_acquire_timeout",
                outcome="timeout",
                wait_ms=wait_ms,
                waiting=waiting_after,
                limit=state.limit,
            )
            raise TimeoutError(
                f"{operation} concurrency limit reached for org {org_id}"
            )
        _record_limiter_event(
            org_id=org_id,
            operation=operation,
            event_name="limiter_acquired",
            outcome="acquired",
            wait_ms=wait_ms,
            waiting=waiting,
            limit=state.limit,
        )
        yield
    finally:
        if acquired:
            state.semaphore.release()


def run_with_operation_limit[T](
    *,
    org_id: str,
    operation: OperationName,
    fn: Callable[[], T],
    timeout_seconds: float | None = None,
) -> T:
    with operation_limit(org_id, operation, timeout_seconds=timeout_seconds):
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
