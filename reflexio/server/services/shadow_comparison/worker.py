"""Bounded background worker for publish-time shadow comparison."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.server.services.shadow_comparison.dispatcher import (
    dispatch_shadow_comparison_judge,
)
from reflexio.server.usage_metrics import record_usage_event

logger = logging.getLogger(__name__)

_DEFAULT_WORKER_COUNT = 2
_DEFAULT_QUEUE_SIZE = 1000


@dataclass(frozen=True)
class ShadowComparisonJob:
    """Metadata needed to judge one publish request's shadow-bearing turns.

    Carries only ``org_id`` plus request-scoped data — never the live
    ``storage`` / ``request_context`` / ``llm_client`` objects. Those are
    re-resolved via ``get_reflexio(org_id)`` inside the worker so a job that
    waits in the queue across a config reload / cache eviction runs against
    the *current* per-org instance rather than a stale one.
    """

    org_id: str
    interactions: list[Interaction]
    session_id: str
    agent_version: str


class ShadowComparisonWorker:
    """Small process-local daemon worker pool for shadow comparison jobs."""

    def __init__(
        self,
        *,
        worker_count: int = _DEFAULT_WORKER_COUNT,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.worker_count = max(1, worker_count)
        self._queue: queue.Queue[ShadowComparisonJob] = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self._threads: list[threading.Thread] = []
        self._start_lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._threads = [
                threading.Thread(
                    target=self._worker_loop,
                    name=f"shadow-comparison-worker-{idx}",
                    daemon=True,
                )
                for idx in range(self.worker_count)
            ]
            for thread in self._threads:
                thread.start()
            self._started = True
            logger.info(
                "event=shadow_comparison_worker_started workers=%d queue_size=%d",
                self.worker_count,
                self._queue.maxsize,
            )

    def enqueue(self, job: ShadowComparisonJob) -> bool:
        self.start()
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            logger.warning(
                "event=shadow_comparison_queue_full org_id=%s session_id=%s queue_size=%d",
                job.org_id,
                job.session_id,
                self._queue.maxsize,
            )
            record_usage_event(
                org_id=job.org_id,
                user_id=None,
                request_id=None,
                session_id=job.session_id,
                source=None,
                agent_version=job.agent_version,
                event_name="shadow_comparison_dropped",
                event_category="shadow_comparison",
                outcome="dropped",
                metadata={"queue_size": self._queue.maxsize},
            )
            return False
        return True

    def _worker_loop(self) -> None:
        # Imported lazily to break the import cycle generation_service ->
        # shadow_comparison.worker -> reflexio_cache -> reflexio_lib ->
        # generation_service (the publish_learning_worker does the same).
        from reflexio.server.cache.reflexio_cache import get_reflexio

        while True:
            job = self._queue.get()
            try:
                reflexio = get_reflexio(org_id=job.org_id)
                request_context = reflexio.request_context
                dispatch_shadow_comparison_judge(
                    storage=request_context.storage,
                    interactions=job.interactions,
                    session_id=job.session_id,
                    agent_version=job.agent_version,
                    request_context=request_context,
                    llm_client=reflexio.llm_client,
                )
            except Exception:
                logger.exception(
                    "event=shadow_comparison_job_failed org_id=%s session_id=%s",
                    job.org_id,
                    job.session_id,
                )
            finally:
                self._queue.task_done()


_worker = ShadowComparisonWorker()


def enqueue_shadow_comparison(job: ShadowComparisonJob) -> bool:
    """Enqueue a shadow comparison job, dropping it when the local queue is full."""
    return _worker.enqueue(job)
