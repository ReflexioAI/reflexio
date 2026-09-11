"""Request-scoped waiting, separate from ingestion and extraction capacity."""

import asyncio
import threading
import time
from contextvars import ContextVar
from typing import Any

from reflexio.server.operation_limiter import operation_limit_value

admission_deadline: ContextVar[float | None] = ContextVar(
    "publish_admission_deadline", default=None
)
_lock = threading.Lock()
_ingesting: dict[str, int] = {}
_waiting: dict[str, int] = {}
_total_waiting = 0


def coverage_stalled(status: dict[str, Any]) -> bool:
    """Report whether coverage can no longer advance without further input.

    ``extraction_status`` emits ``waiting_for_window`` only when *every* still
    pending cursor both lacks a window and cannot select one -- a full window's
    worth of eligible interactions has not arrived yet. A caller that is holding
    no new input can never see that resolve, so waiting on it burns the whole
    publish deadline and then reports ``wait_timeout``, which misdescribes a
    perfectly healthy stream.

    Returning promptly is the documented contract: "A partial window waits for
    more input or an eligible ``force_extraction`` request" (README). Callers
    should surface ``waiting_for_window`` as the reason so the distinction from
    a real timeout survives into the response.

    Args:
        status (dict[str, Any]): A ``BaseStorage.extraction_status`` result.

    Returns:
        bool: True when no eligible cursor can progress without more input.
    """
    return status["status"] == "pending" and status["reason"] == "waiting_for_window"


def check_admission_deadline() -> None:
    deadline = admission_deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Publish admission deadline exceeded")


async def acquire_ingestion(org_id: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        with _lock:
            if _ingesting.get(org_id, 0) < operation_limit_value("publish"):
                _ingesting[org_id] = _ingesting.get(org_id, 0) + 1
                return True
        await asyncio.sleep(0.02)
    return False


def release_ingestion(org_id: str) -> None:
    with _lock:
        _ingesting[org_id] -= 1
        if not _ingesting[org_id]:
            del _ingesting[org_id]


def acquire_waiter(org_id: str) -> bool:
    global _total_waiting
    with _lock:
        if _total_waiting >= 64 or _waiting.get(org_id, 0) >= 8:
            return False
        _total_waiting += 1
        _waiting[org_id] = _waiting.get(org_id, 0) + 1
        return True


def release_waiter(org_id: str) -> None:
    global _total_waiting
    with _lock:
        _total_waiting -= 1
        _waiting[org_id] -= 1
        if not _waiting[org_id]:
            del _waiting[org_id]
