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
from typing import NamedTuple

from reflexio.server.tracing import capture_anomaly

logger = logging.getLogger(__name__)

_WORKERS = 16
_QUEUE_SIZE = 256
_DROP_RATE_PER_MINUTE = 10
_DROP_ANOMALY_THROTTLE_SECONDS = 3600.0


class _DropFacts(NamedTuple):
    """Facts about one drop, collected while ``_cond`` is held.

    Carries everything :meth:`BoundedCallbackExecutor._emit_drop` needs to log
    + escalate AFTER the lock is released, so the contended lock is never held
    across a logging/anomaly call.
    """

    dropped_name: str
    drops_last_minute: int
    fire_anomaly: bool


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
        drop_facts: _DropFacts | None = None
        with self._cond:
            if len(self._queue) >= self._queue_size:
                dropped_name, _ = self._queue.popleft()
                drop_facts = self._record_drop_locked(dropped_name)
            self._queue.append((name, fn))
            self._cond.notify()
        if drop_facts is not None:
            self._emit_drop(drop_facts)

    def _record_drop_locked(self, dropped_name: str) -> _DropFacts:
        """Collect the facts of one drop while ``self._cond`` is held.

        Only bookkeeping happens under the lock (all 16 workers + all
        producers contend on it): pruning the rolling drop-time window and
        deciding — and, if firing, committing — the throttle state
        (``_last_drop_anomaly``) so the decision stays race-free. Logging and
        anomaly emission are the caller's job, done AFTER the lock is
        released (see :meth:`_emit_drop`).
        """
        now = time.monotonic()
        self._drop_times.append(now)
        while self._drop_times and now - self._drop_times[0] > 60.0:
            self._drop_times.popleft()
        drops_last_minute = len(self._drop_times)
        fire_anomaly = (
            drops_last_minute > _DROP_RATE_PER_MINUTE
            and now - self._last_drop_anomaly > _DROP_ANOMALY_THROTTLE_SECONDS
        )
        if fire_anomaly:
            self._last_drop_anomaly = now
        return _DropFacts(
            dropped_name=dropped_name,
            drops_last_minute=drops_last_minute,
            fire_anomaly=fire_anomaly,
        )

    def _emit_drop(self, facts: _DropFacts) -> None:
        """Log one drop and (if decided under the lock) escalate the anomaly.

        Called with ``self._cond`` NOT held — logging and ``capture_anomaly``
        never happen while the contended lock is owned. References the
        module-level ``capture_anomaly`` name (not a bound attribute) so
        tests that ``monkeypatch.setattr(module, "capture_anomaly", ...)``
        still intercept the call.
        """
        logger.warning(
            "event=callback_executor_drop_oldest dropped=%s", facts.dropped_name
        )
        if facts.fire_anomaly:
            capture_anomaly(
                "callback_executor.drop_rate_exceeded",
                drops_last_minute=facts.drops_last_minute,
            )

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                name, fn = self._queue.popleft()
            try:
                fn()
            except BaseException:  # noqa: BLE001 — daemon worker threads must
                # survive anything a callback raises (including a callback
                # that itself raises SystemExit); KeyboardInterrupt is only
                # ever delivered to the main thread, so swallowing it here is
                # safe. The fixed 16-worker pool must never silently shrink.
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
