"""Process-wide bounded executor for deferred fire-and-forget callbacks.

Replaces thread-per-fire in the debounce schedulers (tagging, group
evaluation, playbook optimization) so thread count is O(1) instead of
O(active keys) (spec section 6.2). Sizes are module constants, not env vars.

Optional keys coalesce queued work and serialize callbacks for the same key.
Ready keys can proceed past a blocked successor. Overflow remains best-effort
drop-oldest; another trigger can recreate discarded work. Emit a warning per drop and
a throttled anomaly when drops exceed ``_DROP_RATE_PER_MINUTE`` in a rolling
minute (the "this is routine, resize it" signal).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import NamedTuple

from reflexio.server.background_work import background_work
from reflexio.server.error_reporting import capture_anomaly
from reflexio.server.work_scope import WorkScopeError

logger = logging.getLogger(__name__)

_WORKERS = 16
_QUEUE_SIZE = 256
_DROP_RATE_PER_MINUTE = 10
_DROP_ANOMALY_THROTTLE_SECONDS = 3600.0


@dataclass
class _Callback:
    name: str
    fn: Callable[[], None]
    coalesce_key: Hashable | None


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
    """Fixed worker pool + bounded queue, FIFO among ready callbacks.

    Args:
        workers: Number of daemon worker threads.
        queue_size: Max queued callbacks before drop-oldest applies.
    """

    def __init__(
        self, *, workers: int = _WORKERS, queue_size: int = _QUEUE_SIZE
    ) -> None:
        self._queue: deque[_Callback] = deque()
        self._pending_by_key: dict[Hashable, _Callback] = {}
        self._active_keys: set[Hashable] = set()
        self._queue_size = queue_size
        self._cond = threading.Condition()
        self._active = 0
        self._drop_times: deque[float] = deque()
        self._last_drop_anomaly: float | None = None
        for i in range(workers):
            threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"callback-exec-{i}",
            ).start()

    def submit(
        self,
        name: str,
        fn: Callable[[], None],
        *,
        coalesce_key: Hashable | None = None,
    ) -> None:
        """Enqueue a callback; on a full queue, evict the oldest queued fire.

        Args:
            name: Short label for logs (e.g. the scheduler key).
            fn: Zero-arg callback to run on a worker.
            coalesce_key: Optional identity with one active and one queued callback.
                Queued replacements keep their position; unrelated keys can proceed.
        """
        # A key owns at most one queued successor as well as one active fire.
        # Replacement keeps its queue position; running callbacks are immutable.
        drop_facts: _DropFacts | None = None
        with self._cond:
            if coalesce_key is not None:
                pending = self._pending_by_key.get(coalesce_key)
                if pending is not None:
                    pending.name, pending.fn = name, fn
                    return
            if len(self._queue) >= self._queue_size:
                dropped = self._queue.popleft()
                if dropped.coalesce_key is not None:
                    del self._pending_by_key[dropped.coalesce_key]
                drop_facts = self._record_drop_locked(dropped.name)
            callback = _Callback(name, fn, coalesce_key)
            self._queue.append(callback)
            if coalesce_key is not None:
                self._pending_by_key[coalesce_key] = callback
            self._cond.notify_all()
        if drop_facts is not None:
            self._emit_drop(drop_facts)

    def drain(self, *, timeout_seconds: float = 5.0) -> bool:
        """Wait until queued and in-flight callbacks finish.

        This is primarily used by tests and graceful harnesses that must not let
        daemon callback logs outlive their capture/lifecycle boundary.
        """
        deadline = time.monotonic() + timeout_seconds
        with self._cond:
            while self._queue or self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            return True

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
        fire_anomaly = drops_last_minute > _DROP_RATE_PER_MINUTE and (
            self._last_drop_anomaly is None
            or now - self._last_drop_anomaly > _DROP_ANOMALY_THROTTLE_SECONDS
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

    @background_work()
    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                # Skip successors of active keys, preserving FIFO among ready
                # work. The scan is bounded by the existing queue capacity.
                while True:
                    ready = next(
                        (
                            index
                            for index, callback in enumerate(self._queue)
                            if callback.coalesce_key is None
                            or callback.coalesce_key not in self._active_keys
                        ),
                        None,
                    )
                    if ready is not None:
                        callback = self._queue[ready]
                        del self._queue[ready]
                        break
                    self._cond.wait()
                name, fn, key = callback.name, callback.fn, callback.coalesce_key
                if key is not None:
                    del self._pending_by_key[key]
                    self._active_keys.add(key)
                self._active += 1
            try:
                fn()
            except WorkScopeError:
                # A scope/attribution failure is NOT an operational error, so it
                # does not belong in the blanket branch below where it would be
                # one more indistinguishable log line. It gets its own event and
                # is escalated through the error-reporting seam.
                #
                # Escalated, NOT propagated: this is the top frame of a daemon
                # worker: an exception let out here kills the thread, and after
                # 16 of them the pool is gone and ALL deferred work stops
                # silently. Reporting is strictly more observable than dying.
                logger.exception(
                    "event=callback_executor_work_scope_failed name=%s", name
                )
                capture_anomaly(
                    "callback_executor.work_scope_failed",
                    level="error",
                    callback=name,
                )
            except BaseException:  # noqa: BLE001 — daemon worker threads must
                # survive anything a callback raises (including a callback
                # that itself raises SystemExit); KeyboardInterrupt is only
                # ever delivered to the main thread, so swallowing it here is
                # safe. The fixed 16-worker pool must never silently shrink.
                logger.exception(
                    "event=callback_executor_callback_failed name=%s", name
                )
            finally:
                with self._cond:
                    self._active -= 1
                    if key is not None:
                        self._active_keys.remove(key)
                    self._cond.notify_all()


_executor: BoundedCallbackExecutor | None = None
_executor_lock = threading.Lock()


def submit_callback(
    name: str,
    fn: Callable[[], None],
    *,
    coalesce_key: Hashable | None = None,
) -> None:
    """Submit to the lazily-created process-wide executor.

    Args:
        name: Short label for logs.
        fn: Zero-arg callback.
        coalesce_key: Optional identity for serialized, coalesced dispatch.
    """
    global _executor  # noqa: PLW0603
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = BoundedCallbackExecutor()
    _executor.submit(name, fn, coalesce_key=coalesce_key)


def drain_callbacks(*, timeout_seconds: float = 5.0) -> bool:
    """Wait for the process-global callback executor to become idle."""
    if _executor is None:
        return True
    return _executor.drain(timeout_seconds=timeout_seconds)
