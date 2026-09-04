"""Process-local queue for learning work after async publish persistence."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

from reflexio.models.api_schema.common import sanitise_for_log
from reflexio.server.cache.reflexio_cache import get_reflexio
from reflexio.server.env_utils import env_str
from reflexio.server.error_reporting import capture_anomaly
from reflexio.server.operation_limiter import operation_limit, operation_limit_value
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.usage_metrics import record_usage_event
from reflexio.server.work_scope import WorkScope, WorkScopeError, bind_work_scope

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_WARN_SIZE = 1000


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = env_str(name, str(default))
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return max(minimum, default)


@dataclass(frozen=True)
class PublishLearningJob:
    """Metadata needed to run post-persist publish learning."""

    org_id: str
    user_id: str
    request_id: str
    session_id: str | None
    source: str | None
    agent_version: str
    force_extraction: bool
    skip_aggregation: bool
    # Owning project, carried on the payload because the worker runs long after
    # the enqueueing request returned. ``None`` in OSS, where projects do not
    # exist — an absent project is normal here, never an error.
    project_id: str | None = None
    enqueued_at: float = field(default_factory=time.monotonic)


class PublishLearningWorker:
    """Process-local worker pool for deferred publish learning."""

    def __init__(
        self,
        *,
        queue_warn_size: int | None = None,
        worker_count: int | None = None,
    ) -> None:
        default_workers = operation_limit_value("publish")
        self.queue_warn_size = (
            queue_warn_size
            if queue_warn_size is not None
            else _env_int(
                "REFLEXIO_PUBLISH_LEARNING_QUEUE_WARN_SIZE",
                _DEFAULT_QUEUE_WARN_SIZE,
            )
        )
        self.worker_count = (
            worker_count
            if worker_count is not None
            else _env_int("REFLEXIO_PUBLISH_LEARNING_WORKERS", default_workers)
        )
        self._queue: queue.Queue[PublishLearningJob] = queue.Queue()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._start_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._stop_event.clear()
            self._threads = [
                threading.Thread(
                    target=self._worker_loop,
                    name=f"publish-learning-worker-{idx}",
                    daemon=True,
                )
                for idx in range(self.worker_count)
            ]
            for thread in self._threads:
                thread.start()
            self._started = True
            logger.info(
                "event=publish_learning_worker_started workers=%d queue_warn_size=%d",
                self.worker_count,
                self.queue_warn_size,
            )

    def enqueue(self, job: PublishLearningJob) -> bool:
        self.start()
        self._queue.put_nowait(job)
        queue_depth = self._queue.qsize()
        if queue_depth >= self.queue_warn_size:
            record_usage_event(
                org_id=job.org_id,
                user_id=job.user_id,
                request_id=job.request_id,
                session_id=job.session_id,
                source=job.source,
                agent_version=job.agent_version,
                event_name="learning_queue_pressure",
                event_category="publish_learning",
                outcome="queued",
                metadata={
                    "queue_depth": queue_depth,
                    "queue_warn_size": self.queue_warn_size,
                },
            )
            logger.warning(
                "event=publish_learning_queue_pressure org_id=%s user_id=%s "
                "request_id=%s queue_depth=%d queue_warn_size=%d",
                job.org_id,
                job.user_id,
                job.request_id,
                queue_depth,
                self.queue_warn_size,
            )
        logger.info(
            "event=publish_learning_enqueued org_id=%s user_id=%s request_id=%s "
            "queue_depth=%d queue_warn_size=%d",
            job.org_id,
            job.user_id,
            job.request_id,
            queue_depth,
            self.queue_warn_size,
        )
        return True

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in list(self._threads):
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            thread.join(timeout=remaining)
        if all(not thread.is_alive() for thread in self._threads):
            self._threads = []
            self._started = False
        logger.info("event=publish_learning_worker_stopped")

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._process_job(job)
            finally:
                self._queue.task_done()

    def _process_job(self, job: PublishLearningJob) -> None:
        try:
            with (
                operation_limit(
                    job.org_id,
                    "publish",
                    wait_forever=False,
                    log_timeout=False,
                ),
                bind_work_scope(
                    WorkScope(org_id=job.org_id, project_id=job.project_id)
                ),
            ):
                reflexio = get_reflexio(org_id=job.org_id)
                GenerationService(
                    llm_client=reflexio.llm_client,
                    request_context=reflexio.request_context,
                ).run_deferred_learning(
                    user_id=job.user_id,
                    request_id=job.request_id,
                    session_id=job.session_id,
                    source=job.source,
                    agent_version=job.agent_version,
                    force_extraction=job.force_extraction,
                    skip_aggregation=job.skip_aggregation,
                )
        except TimeoutError:
            self._requeue_after_limiter_timeout(job)
            return
        except WorkScopeError as exc:
            # NOT an operational failure. The old blanket handler recorded this
            # as a routine `learning_failed` usage event and dropped the job,
            # which is indistinguishable from an LLM/storage hiccup — exactly
            # the "silent no-op reported as success if nothing checks" the
            # design warns about. Escalate it under its own event so the drop
            # is visible. Escalated rather than propagated: letting it out of
            # _process_job kills the worker thread (see WorkScopeError).
            record_usage_event(
                org_id=job.org_id,
                user_id=job.user_id,
                request_id=job.request_id,
                session_id=job.session_id,
                source=job.source,
                agent_version=job.agent_version,
                event_name="learning_scope_failed",
                event_category="publish_learning",
                outcome="failed",
                error_kind=type(exc).__name__,
            )
            capture_anomaly(
                "publish_learning.work_scope_failed",
                level="error",
                org_id=job.org_id,
                project_id=job.project_id,
                user_id=job.user_id,
                request_id=job.request_id,
            )
            logger.exception(
                "event=publish_learning_scope_failed org_id=%s user_id=%s request_id=%s",
                job.org_id,
                job.user_id,
                sanitise_for_log(job.request_id),
            )
            return
        except Exception as exc:
            record_usage_event(
                org_id=job.org_id,
                user_id=job.user_id,
                request_id=job.request_id,
                session_id=job.session_id,
                source=job.source,
                agent_version=job.agent_version,
                event_name="learning_failed",
                event_category="publish_learning",
                outcome="failed",
                error_kind=type(exc).__name__,
            )
            logger.exception(
                "event=publish_learning_failed org_id=%s user_id=%s request_id=%s",
                job.org_id,
                job.user_id,
                job.request_id,
            )
            return

        record_usage_event(
            org_id=job.org_id,
            user_id=job.user_id,
            request_id=job.request_id,
            session_id=job.session_id,
            source=job.source,
            agent_version=job.agent_version,
            event_name="learning_succeeded",
            event_category="publish_learning",
            outcome="success",
        )
        logger.info(
            "event=publish_learning_done org_id=%s user_id=%s request_id=%s "
            "queue_depth=%d",
            job.org_id,
            job.user_id,
            job.request_id,
            self._queue.qsize(),
        )

    def _requeue_after_limiter_timeout(self, job: PublishLearningJob) -> None:
        self._queue.put_nowait(job)

        record_usage_event(
            org_id=job.org_id,
            user_id=job.user_id,
            request_id=job.request_id,
            session_id=job.session_id,
            source=job.source,
            agent_version=job.agent_version,
            event_name="learning_requeued_limiter_busy",
            event_category="publish_learning",
            outcome="requeued",
            metadata={
                "age_seconds": self._job_age_seconds(job),
                "queue_depth": self._queue.qsize(),
            },
        )
        logger.info(
            "event=publish_learning_requeued_limiter_busy org_id=%s user_id=%s "
            "request_id=%s queue_depth=%d age_seconds=%.3f",
            job.org_id,
            job.user_id,
            job.request_id,
            self._queue.qsize(),
            self._job_age_seconds(job),
        )

    @staticmethod
    def _job_age_seconds(job: PublishLearningJob) -> float:
        return time.monotonic() - job.enqueued_at


_worker: PublishLearningWorker | None = None
_worker_lock = threading.Lock()


def get_publish_learning_worker() -> PublishLearningWorker:
    global _worker  # noqa: PLW0603
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = PublishLearningWorker()
    return _worker


def enqueue_publish_learning(job: PublishLearningJob) -> bool:
    return get_publish_learning_worker().enqueue(job)


def stop_publish_learning_worker(timeout: float | None = 5.0) -> None:
    global _worker  # noqa: PLW0603
    worker = _worker
    if worker is not None:
        worker.stop(timeout=timeout)


def reset_publish_learning_worker_for_tests() -> None:
    global _worker  # noqa: PLW0603
    worker = _worker
    if worker is not None:
        worker.stop(timeout=0.1)
    _worker = None
