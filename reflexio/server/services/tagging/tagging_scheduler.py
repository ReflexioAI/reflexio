"""Singleton scheduler for deferred, off-publish-path entity tagging.

Tagging runs an LLM call per newly generated profile/playbook, so it must not
block the publish request. This scheduler mirrors
:class:`GroupEvaluationScheduler`: a single daemon thread with a min-heap, where
each publish upserts the fire time for its ``(org_id, user_id, agent_version)``
key. Rapid successive publishes for the same key debounce into a single tagging
pass; when the timer fires, the tagging callback runs on its own daemon thread.

Tagging is idempotent (already-tagged entities are skipped), so a deferred pass
naturally handles both newly generated entities and the one-time backfill that
happens when a tagging definition is first configured.
"""

from __future__ import annotations

import heapq
import logging
import os
import threading
import time
from collections.abc import Callable

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.tagging.tagging_service import TaggingService

logger = logging.getLogger(__name__)

# Delay before a tagging pass fires after the last publish for a key. Long enough
# to debounce a burst of publishes into one pass, short enough that tags appear
# promptly. Kept as a patch point for tests.
TAGGING_DELAY_SECONDS = 15
IS_TEST_ENV = os.environ.get("IS_TEST_ENV", "false").strip().lower() == "true"
_EFFECTIVE_DELAY_SECONDS = 1 if IS_TEST_ENV else TAGGING_DELAY_SECONDS

# (org_id, user_id, agent_version)
TaggingKey = tuple[str, str, str]


class TaggingScheduler:
    """Singleton scheduler that fires entity tagging after a short debounce."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> TaggingScheduler:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._scheduled: dict[TaggingKey, tuple[float, Callable]] = {}
        self._heap: list[tuple[float, TaggingKey]] = []
        self._mutex = threading.Lock()
        self._wake_event = threading.Event()
        self._thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name="tagging-scheduler"
        )
        self._thread.start()
        logger.info("TaggingScheduler started")

    def schedule(self, key: TaggingKey, callback: Callable) -> None:
        """Schedule or reschedule a tagging pass for ``key`` (slides the fire time forward)."""
        fire_time = time.monotonic() + _EFFECTIVE_DELAY_SECONDS
        with self._mutex:
            self._scheduled[key] = (fire_time, callback)
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
                        if current is None:
                            continue
                        current_fire_time, callback = current
                        if abs(current_fire_time - fire_time) > 0.001:
                            # Superseded by a newer schedule for the same key.
                            continue

                        del self._scheduled[key]
                        t = threading.Thread(
                            target=self._run_callback,
                            args=(key, callback),
                            daemon=True,
                            name=f"tagging-{key[1][:20]}",
                        )
                        t.start()
            except Exception:
                logger.exception("Error in tagging scheduler loop")
                time.sleep(1)

    @staticmethod
    def _run_callback(key: TaggingKey, callback: Callable) -> None:
        try:
            logger.info("Firing tagging for key=%s", key)
            callback()
            logger.info("Completed tagging for key=%s", key)
        except Exception:
            logger.exception("Tagging callback failed for key=%s", key)


def schedule_tagging(
    *,
    org_id: str,
    user_id: str,
    agent_version: str,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
) -> None:
    """Enqueue a deferred tagging pass for the given user/agent off the publish path."""
    if not user_id:
        return

    key: TaggingKey = (org_id, user_id, agent_version)
    storage_base_dir = getattr(request_context, "storage_base_dir", None)

    def callback() -> None:
        fresh_context = RequestContext(
            org_id=org_id,
            storage_base_dir=storage_base_dir,
        )
        TaggingService(llm_client=llm_client, request_context=fresh_context).run(
            user_id=user_id, agent_version=agent_version
        )

    TaggingScheduler.get_instance().schedule(key, callback)
