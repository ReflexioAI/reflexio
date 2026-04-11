"""Singleton scheduler for delayed session evaluation.

Uses a single daemon thread with a min-heap priority queue.
Each new request upserts the fire time for its group.
When the fire time arrives, a daemon thread runs the evaluation callback.
"""

import heapq
import logging
import os
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Delay in seconds before evaluating a group after the last request
GROUP_EVALUATION_DELAY_SECONDS = 600  # 10 minutes

IS_TEST_ENV = os.environ.get("IS_TEST_ENV", "false").strip() == "true"
_EFFECTIVE_DELAY_SECONDS = 30 if IS_TEST_ENV else GROUP_EVALUATION_DELAY_SECONDS

# Type alias for the scheduling key
GroupKey = tuple[str, str, str]  # (org_id, user_id, session_id)


class GroupEvaluationScheduler:
    """Singleton scheduler that fires group evaluations after a period of inactivity.

    Uses one daemon thread with a min-heap. Each new request upserts the fire time
    for its group. Handles hundreds of concurrent groups efficiently.
    """

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> GroupEvaluationScheduler:
        """Get or create the singleton scheduler instance.

        Returns:
            GroupEvaluationScheduler: The singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._scheduled: dict[GroupKey, tuple[float, Callable]] = {}
        self._heap: list[tuple[float, GroupKey]] = []
        self._mutex = threading.Lock()
        self._wake_event = threading.Event()
        self._thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="group-eval-scheduler"
        )
        self._thread.start()
        logger.info("GroupEvaluationScheduler started")

    def schedule(self, key: GroupKey, callback: Callable) -> None:
        """Schedule or reschedule a group evaluation.

        If the group already has a pending evaluation, its fire time is updated
        (slid forward). The callback will be invoked after GROUP_EVALUATION_DELAY_SECONDS
        of inactivity.

        Args:
            key: Tuple of (org_id, user_id, session_id)
            callback: Zero-argument callable to run when the timer fires
        """
        fire_time = time.monotonic() + _EFFECTIVE_DELAY_SECONDS
        with self._mutex:
            self._scheduled[key] = (fire_time, callback)
            heapq.heappush(self._heap, (fire_time, key))
        self._wake_event.set()
        logger.info(
            "Scheduled group evaluation for key=%s fire_time=%.1f", key, fire_time
        )

    def _scheduler_loop(self) -> None:
        """Main loop for the scheduler thread.

        Pops due items from the heap, verifies they are still current
        (not superseded by a newer schedule), and spawns daemon threads
        to run the callback.
        """
        while True:
            try:
                with self._mutex:
                    next_fire_time = self._heap[0][0] if self._heap else None

                if next_fire_time is None:
                    # Nothing scheduled, wait for a wake signal
                    self._wake_event.wait()
                    self._wake_event.clear()
                    continue

                now = time.monotonic()
                wait_seconds = next_fire_time - now

                if wait_seconds > 0:
                    # Wait until the next fire time or a wake signal
                    self._wake_event.wait(timeout=wait_seconds)
                    self._wake_event.clear()
                    continue

                # Process due items
                with self._mutex:
                    while self._heap and self._heap[0][0] <= time.monotonic():
                        fire_time, key = heapq.heappop(self._heap)

                        # Check if this entry is still current (not superseded)
                        current = self._scheduled.get(key)
                        if current is None:
                            continue
                        current_fire_time, callback = current
                        if abs(current_fire_time - fire_time) > 0.001:
                            # This entry was superseded by a newer schedule
                            continue

                        # Remove from scheduled map and fire
                        del self._scheduled[key]

                        # Spawn daemon thread for the callback
                        t = threading.Thread(
                            target=self._run_callback,
                            args=(key, callback),
                            daemon=True,
                            name=f"group-eval-{key[2][:20]}",
                        )
                        t.start()

            except Exception:
                logger.exception("Error in group evaluation scheduler loop")
                # Brief sleep to avoid tight error loops
                time.sleep(1)

    @staticmethod
    def _run_callback(key: GroupKey, callback: Callable) -> None:
        """Run the evaluation callback, catching any exceptions.

        Args:
            key: The group key for logging
            callback: The evaluation callback to run
        """
        try:
            logger.info("Firing group evaluation for key=%s", key)
            callback()
        except Exception:
            logger.exception("Group evaluation callback failed for key=%s", key)
