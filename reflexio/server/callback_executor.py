"""Process-wide bounded executor for deferred fire-and-forget callbacks.

Replaces thread-per-fire in the debounce schedulers (tagging, group
evaluation, playbook optimization) so thread count is O(1) instead of
O(active keys) (spec section 6.2). Sizes are module constants, not env vars.

Overflow policy: drop-oldest (a dropped fire is a stale debounced trigger —
the next event for that key re-creates it) with one warning log per drop and
a throttled anomaly when drops exceed ``_DROP_RATE_PER_MINUTE`` in a rolling
minute (the "this is routine, resize it" signal).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from reflexio.server.tracing import capture_anomaly

logger = logging.getLogger(__name__)

_WORKERS = 16
_QUEUE_SIZE = 256
_DROP_RATE_PER_MINUTE = 10
_DROP_ANOMALY_THROTTLE_SECONDS = 3600.0


class BoundedCallbackExecutor:
    """Fixed worker pool + bounded FIFO queue with drop-oldest overflow.

    Args:
        workers: Number of daemon worker threads.
        queue_size: Max queued callbacks before drop-oldest applies.
    """

    def __init__(
        self, *, workers: int = _WORKERS, queue_size: int = _QUEUE_SIZE
    ) -> None:
        self._queue: deque[tuple[str, Callable[[], None]]] = deque()
        self._queue_size = queue_size
        self._cond = threading.Condition()
        self._drop_times: deque[float] = deque()
        self._last_drop_anomaly = 0.0
        for i in range(workers):
            threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"callback-exec-{i}",
            ).start()

    def submit(self, name: str, fn: Callable[[], None]) -> None:
        """Enqueue a callback; on a full queue, evict the oldest queued fire.

        Args:
            name: Short label for logs (e.g. the scheduler key).
            fn: Zero-arg callback to run on a worker.
        """
        with self._cond:
            if len(self._queue) >= self._queue_size:
                dropped_name, _ = self._queue.popleft()
                self._record_drop_locked(dropped_name)
            self._queue.append((name, fn))
            self._cond.notify()

    def _record_drop_locked(self, dropped_name: str) -> None:
        """Log one drop and escalate a throttled anomaly on sustained overflow."""
        now = time.monotonic()
        logger.warning("event=callback_executor_drop_oldest dropped=%s", dropped_name)
        self._drop_times.append(now)
        while self._drop_times and now - self._drop_times[0] > 60.0:
            self._drop_times.popleft()
        if (
            len(self._drop_times) > _DROP_RATE_PER_MINUTE
            and now - self._last_drop_anomaly > _DROP_ANOMALY_THROTTLE_SECONDS
        ):
            self._last_drop_anomaly = now
            capture_anomaly(
                "callback_executor.drop_rate_exceeded",
                drops_last_minute=len(self._drop_times),
            )

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                name, fn = self._queue.popleft()
            try:
                fn()
            except Exception:  # noqa: BLE001 — one callback never kills a worker
                logger.exception(
                    "event=callback_executor_callback_failed name=%s", name
                )


_executor: BoundedCallbackExecutor | None = None
_executor_lock = threading.Lock()


def submit_callback(name: str, fn: Callable[[], None]) -> None:
    """Submit to the lazily-created process-wide executor.

    Args:
        name: Short label for logs.
        fn: Zero-arg callback.
    """
    global _executor  # noqa: PLW0603
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = BoundedCallbackExecutor()
    _executor.submit(name, fn)
