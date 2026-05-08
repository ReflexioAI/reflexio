from __future__ import annotations

import heapq
import logging
import threading
import time
from collections.abc import Callable

from .optimizer import PlaybookOptimizationRunStatus, PlaybookOptimizationTarget

logger = logging.getLogger(__name__)

ScheduleKey = tuple[str, str, int]
ScheduledCallback = Callable[[], PlaybookOptimizationRunStatus | None]


class PlaybookOptimizationScheduler:
    """Process-local scheduler that debounces optimization runs.

    Behaviour:

    - **Debounce** by ``(org_id, target.kind, target.target_id)`` — repeated
      enqueues for the same playbook collapse into one fire.
    - **Jitter** the fire time by up to ``jitter_seconds`` to spread load
      across simultaneous saves.
    - **Cooldown** when a callback returns ``"aborted"`` (the assistant
      backend repeatedly failed). After ``abort_cooldown_threshold``
      consecutive aborts, further enqueues for the same key are silently
      dropped for ``cooldown_after_aborts_seconds``. A ``completed`` or
      ``skipped`` outcome resets the counter.

    The scheduler is a singleton with a daemon thread; multiple processes
    schedule independently, so debounce is per-process.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> PlaybookOptimizationScheduler:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._scheduled: dict[
            ScheduleKey, tuple[float, ScheduledCallback, int, int]
        ] = {}
        self._heap: list[tuple[float, ScheduleKey]] = []
        self._mutex = threading.Lock()
        self._wake_event = threading.Event()
        self._abort_counts: dict[ScheduleKey, tuple[int, float]] = {}
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="playbook-optimizer-scheduler",
        )
        self._thread.start()

    def enqueue(
        self,
        *,
        org_id: str,
        target: PlaybookOptimizationTarget,
        callback: ScheduledCallback,
        jitter_seconds: float = 1.0,
        abort_cooldown_threshold: int = 2,
        cooldown_after_aborts_seconds: int = 3600,
    ) -> None:
        key: ScheduleKey = (org_id, target.kind, target.target_id)
        now = time.monotonic()
        jitter = (time.monotonic() % 1.0) * jitter_seconds
        fire_time = now + jitter
        with self._mutex:
            if self._cooldown_remaining_locked(key, now) > 0:
                logger.info(
                    "Skipping playbook optimization enqueue during cooldown: %s", key
                )
                return
            self._scheduled[key] = (
                fire_time,
                callback,
                abort_cooldown_threshold,
                cooldown_after_aborts_seconds,
            )
            heapq.heappush(self._heap, (fire_time, key))
        self._wake_event.set()

    def _scheduler_loop(self) -> None:
        while True:
            try:
                with self._mutex:
                    next_fire_time = self._heap[0][0] if self._heap else None
                if next_fire_time is None:
                    self._wake_event.wait()
                    self._wake_event.clear()
                    continue
                wait_seconds = next_fire_time - time.monotonic()
                if wait_seconds > 0:
                    self._wake_event.wait(timeout=wait_seconds)
                    self._wake_event.clear()
                    continue
                with self._mutex:
                    while self._heap and self._heap[0][0] <= time.monotonic():
                        fire_time, key = heapq.heappop(self._heap)
                        current = self._scheduled.get(key)
                        if current is None or abs(current[0] - fire_time) > 0.001:
                            continue
                        _, callback, abort_threshold, cooldown_seconds = current
                        del self._scheduled[key]
                        threading.Thread(
                            target=self._run_callback,
                            args=(key, callback, abort_threshold, cooldown_seconds),
                            daemon=True,
                            name=f"playbook-opt-{key[1]}-{key[2]}",
                        ).start()
            except Exception:
                logger.exception("Playbook optimization scheduler loop failed")
                time.sleep(1)

    def _run_callback(
        self,
        key: ScheduleKey,
        callback: ScheduledCallback,
        abort_threshold: int,
        cooldown_seconds: int,
    ) -> None:
        try:
            status = callback()
            if status == "aborted":
                self._record_abort(key, abort_threshold, cooldown_seconds)
            elif status in {"completed", "skipped"}:
                self._clear_abort_state(key)
        except Exception:
            logger.exception("Playbook optimization callback failed for key=%s", key)
            self._record_abort(key, abort_threshold, cooldown_seconds)

    def _cooldown_remaining_locked(self, key: ScheduleKey, now: float) -> float:
        state = self._abort_counts.get(key)
        if state is None:
            return 0.0
        count, cooldown_until = state
        if cooldown_until <= 0:
            return 0.0
        remaining = cooldown_until - now
        if remaining > 0:
            return remaining
        del self._abort_counts[key]
        return 0.0

    def _record_abort(
        self, key: ScheduleKey, abort_threshold: int, cooldown_seconds: int
    ) -> None:
        now = time.monotonic()
        with self._mutex:
            self._cooldown_remaining_locked(key, now)
            count, _ = self._abort_counts.get(key, (0, 0.0))
            count += 1
            if count >= abort_threshold:
                cooldown_until = now + cooldown_seconds
                self._abort_counts[key] = (0, cooldown_until)
                logger.warning(
                    "Playbook optimization entered abort cooldown for key=%s seconds=%s",
                    key,
                    cooldown_seconds,
                )
            else:
                self._abort_counts[key] = (count, 0.0)

    def _clear_abort_state(self, key: ScheduleKey) -> None:
        with self._mutex:
            self._abort_counts.pop(key, None)
