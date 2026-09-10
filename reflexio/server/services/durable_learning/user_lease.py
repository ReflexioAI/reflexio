"""Shared extraction admission for explicit manual and resumed operations."""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from reflexio.server.services.durable_learning.worker import _release, _reserve
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._extraction_stream import (
    LeaseLostError,
)

_active_lease: ContextVar[tuple[str, str] | None] = ContextVar(
    "explicit_extraction_lease", default=None
)


def fence_explicit_extraction(storage: BaseStorage) -> None:
    lease = _active_lease.get()
    if lease is not None:
        storage.fence_user_extraction(*lease)


@contextmanager
def user_extraction_lease(
    storage: BaseStorage, user_id: str, *, max_wait_seconds: float | None = None
) -> Iterator[None]:
    deadline = None if max_wait_seconds is None else time.monotonic() + max_wait_seconds
    token = None
    while token is None:
        if deadline is not None and time.monotonic() >= deadline:
            raise UserLeaseBusyError("Extraction capacity or user lease is busy")
        if _reserve():
            try:
                token = storage.claim_user_extraction(user_id, "explicit", 300)
            except Exception:
                _release()
                raise
            if token is None:
                _release()
        if token is None:
            time.sleep(0.1)
    stop = threading.Event()
    lost = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(60):
            try:
                if not storage.renew_extraction(user_id, token, 300):
                    lost.set()
                    return
            except Exception:
                lost.set()
                return

    thread = threading.Thread(
        target=heartbeat, daemon=True, name="reflexio-explicit-extraction-heartbeat"
    )
    thread.start()
    binding = _active_lease.set((user_id, token))
    try:
        yield
        if lost.is_set():
            raise LeaseLostError("Explicit extraction lease expired")
    finally:
        _active_lease.reset(binding)
        stop.set()
        try:
            storage.release_user_extraction(user_id, token)
        finally:
            _release()
        thread.join(timeout=1)


class UserLeaseBusyError(RuntimeError):
    """A bounded caller could not acquire capacity and a user lease in time."""
