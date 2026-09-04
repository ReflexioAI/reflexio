"""Singleton scheduler for delayed session evaluation.

Uses a single daemon thread with a min-heap priority queue.
Each new request upserts the fire time for its group.
When the fire time arrives, a daemon thread runs the evaluation callback.
"""

from __future__ import annotations

import heapq
import logging
import os
import threading
import time
from collections.abc import Callable
from functools import partial

from reflexio.server.callback_executor import submit_callback
from reflexio.server.error_reporting import capture_anomaly
from reflexio.server.services.agent_success_evaluation import _eval_health
from reflexio.server.work_scope import WorkScope, WorkScopeError, bind_work_scope

logger = logging.getLogger(__name__)

# Default delay in seconds before evaluating a group after the last request.
_DEFAULT_DELAY_SECONDS = 600  # 10 minutes

# Inactivity delay before a session is evaluated. Override the default via the
# GROUP_EVALUATION_DELAY_SECONDS environment variable (in seconds).
GROUP_EVALUATION_DELAY_SECONDS = int(
    os.environ.get("GROUP_EVALUATION_DELAY_SECONDS", str(_DEFAULT_DELAY_SECONDS))
)

# Kept as a patch point for tests. Production should use the full inactivity
# delay unless a test deliberately lowers this constant.
IS_TEST_ENV = os.environ.get("IS_TEST_ENV", "false").strip() == "true"
_EFFECTIVE_DELAY_SECONDS = 30 if IS_TEST_ENV else GROUP_EVALUATION_DELAY_SECONDS

# Type alias for the scheduling key.
# (org_id, project_id, user_id, session_id)
#
# ``project_id`` is part of the DEBOUNCE IDENTITY, not decoration. Two projects
# in one org evaluating the same user/session inside one inactivity window would
# otherwise collapse into a single evaluation attributed to whichever request
# won the race. ``None`` in OSS, where projects do not exist.
GroupKey = tuple[str, str | None, str, str]


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
                _eval_health.record_tick()
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

                        submit_callback(
                            # key[3] is session_id — key[1] is the nullable project.
                            f"group-eval-{key[3][:20]}",
                            partial(self._run_callback, key, callback),
                        )

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
            with bind_work_scope(WorkScope(org_id=key[0], project_id=key[1])):
                callback()
        except WorkScopeError:
            # NOT an operational failure: the evaluation could not be attributed
            # to a project, so tolerating it would write it under the wrong one.
            # Escalated rather than propagated — see WorkScopeError.
            logger.exception("Group evaluation scope binding failed for key=%s", key)
            capture_anomaly(
                "agent_success_evaluation.work_scope_failed",
                level="error",
                org_id=key[0],
                project_id=key[1],
                user_id=key[2],
            )
        except Exception:
            logger.exception("Group evaluation callback failed for key=%s", key)
