"""Process-local queue for search metering outside the HTTP request lifecycle."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from reflexio.server.error_reporting import capture_anomaly
from reflexio.server.tracing import (
    capture_trace_context,
    profile_background_transaction,
    profile_step,
)

logger = logging.getLogger(__name__)

_QUEUE_CAPACITY = 1_000
_WORKER_COUNT = 4


@dataclass(frozen=True, slots=True)
class SearchMeteringJob:
    """The complete metering work produced by one customer-facing search."""

    org_id: str
    surfaced_count: int
    record_search_request: bool
    request_id: str | None = None
    session_id: str | None = None
    trace_context: Mapping[str, str] = field(default_factory=dict)
    enqueued_at: float = field(default_factory=time.monotonic)


class SearchMeteringWorker:
    """Bounded process-local worker pool for eventually consistent search usage."""

    def __init__(
        self,
        *,
        queue_capacity: int = _QUEUE_CAPACITY,
        worker_count: int = _WORKER_COUNT,
    ) -> None:
        self.queue_capacity = queue_capacity
        self.worker_count = worker_count
        self._queue: queue.Queue[SearchMeteringJob] = queue.Queue(
            maxsize=queue_capacity
        )
        self._stop_event = threading.Event()
        self._abort_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False
        self._accepting = True
        self._lifecycle_lock = threading.Lock()

    def start(self) -> bool:
        """Start worker threads once, returning false after shutdown begins."""
        with self._lifecycle_lock:
            if self._started:
                return self._accepting
            if not self._accepting:
                return False
            self._stop_event.clear()
            self._abort_event.clear()
            threads = [
                threading.Thread(
                    target=self._worker_loop,
                    name=f"search-metering-worker-{index}",
                    daemon=True,
                )
                for index in range(self.worker_count)
            ]
            started_threads: list[threading.Thread] = []
            try:
                for thread in threads:
                    thread.start()
                    started_threads.append(thread)
            except BaseException:
                self._stop_event.set()
                for thread in started_threads:
                    thread.join()
                self._stop_event.clear()
                self._threads = []
                raise
            self._threads = threads
            self._started = True
        logger.info(
            "event=search_metering_worker_started workers=%d queue_capacity=%d",
            self.worker_count,
            self.queue_capacity,
        )
        return True

    def enqueue(self, job: SearchMeteringJob) -> bool:
        """Enqueue without waiting; fail open when stopped or saturated."""
        if not self.start():
            self._report_drop(job, reason="stopping")
            return False
        with self._lifecycle_lock:
            if not self._accepting:
                self._report_drop(job, reason="stopping")
                return False
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                self._report_drop(job, reason="queue_full")
                return False
        return True

    def wait_until_idle(self, timeout: float) -> bool:
        """Wait for all accepted jobs, primarily for deterministic tests."""
        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(timeout=remaining)
        return True

    def stop(self, timeout: float = 5.0) -> int:
        """Drain briefly, then abandon queued work and return the dropped count."""
        deadline = time.monotonic() + max(timeout, 0.0)
        with self._lifecycle_lock:
            self._accepting = False
            if not self._started:
                return 0
            self._stop_event.set()

        drained = self.wait_until_idle(timeout=max(0.0, deadline - time.monotonic()))
        abandoned = 0
        if not drained:
            self._abort_event.set()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    abandoned += 1
                    self._queue.task_done()

            # Jobs already taken by a worker remain unfinished at the drain
            # deadline and are no longer guaranteed to persist before teardown.
            with self._queue.all_tasks_done:
                abandoned += self._queue.unfinished_tasks

        for thread in list(self._threads):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

        alive = sum(thread.is_alive() for thread in self._threads)
        if abandoned:
            logger.error(
                "event=search_metering_shutdown_abandoned jobs=%d alive_workers=%d",
                abandoned,
                alive,
            )
            capture_anomaly(
                "search.metering.shutdown_abandoned",
                level="error",
                abandoned_jobs=abandoned,
                alive_workers=alive,
            )
        else:
            logger.info("event=search_metering_worker_stopped alive_workers=%d", alive)
        return abandoned

    def _worker_loop(self) -> None:
        while not self._abort_event.is_set():
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop_event.is_set():
                    return
                continue
            try:
                self._process_job(job)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "event=search_metering_job_failed org_id=%s", job.org_id
                )
                capture_anomaly(
                    "search.metering.job_failed",
                    level="error",
                    org_id=job.org_id,
                )
            finally:
                self._queue.task_done()

    def _process_job(self, job: SearchMeteringJob) -> None:
        from reflexio.server.routes._metering import (
            _meter_applied_learnings,
            _meter_search_request,
        )

        queue_wait_ms = int((time.monotonic() - job.enqueued_at) * 1_000)
        with profile_background_transaction(
            "search.metering",
            op="queue.process",
            trace_context=job.trace_context,
            queue_wait_ms=queue_wait_ms,
            queue_depth=self._queue.qsize(),
            record_search_request=job.record_search_request,
            surfaced_count=job.surfaced_count,
        ):
            if job.record_search_request:
                with profile_step("search.metering.search_request") as span:
                    span.set_data(
                        "emitted",
                        _meter_search_request(
                            org_id=job.org_id,
                            caller_type="production_agent",
                            request_id=job.request_id,
                            session_id=job.session_id,
                        ),
                    )
            if job.surfaced_count > 0:
                with profile_step("search.metering.learning_applied") as span:
                    span.set_data(
                        "emitted",
                        _meter_applied_learnings(
                            org_id=job.org_id,
                            caller_type="production_agent",
                            surfaced_count=job.surfaced_count,
                            request_id=job.request_id,
                            session_id=job.session_id,
                        ),
                    )

    def _report_drop(self, job: SearchMeteringJob, *, reason: str) -> None:
        queue_depth = self._queue.qsize()
        logger.error(
            "event=search_metering_job_dropped reason=%s org_id=%s queue_depth=%d",
            reason,
            job.org_id,
            queue_depth,
        )
        capture_anomaly(
            "search.metering.job_dropped",
            level="error",
            reason=reason,
            org_id=job.org_id,
            queue_depth=queue_depth,
        )


_worker: SearchMeteringWorker | None = None
_worker_lock = threading.Lock()


def get_search_metering_worker() -> SearchMeteringWorker:
    """Return the process-local worker singleton."""
    global _worker  # noqa: PLW0603
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = SearchMeteringWorker()
    return _worker


def enqueue_search_metering(
    *,
    org_id: str,
    caller_type: str,
    surfaced_count: int,
    record_search_request: bool,
    request_id: str | None = None,
    session_id: str | None = None,
) -> bool:
    """Queue search usage without performing database work in the request."""
    if caller_type != "production_agent":
        return False
    if not record_search_request and surfaced_count <= 0:
        return False
    return get_search_metering_worker().enqueue(
        SearchMeteringJob(
            org_id=org_id,
            surfaced_count=max(surfaced_count, 0),
            record_search_request=record_search_request,
            request_id=request_id,
            session_id=session_id,
            trace_context=capture_trace_context(),
        )
    )


def start_search_metering_worker() -> None:
    """Start worker threads before the first customer search."""
    get_search_metering_worker().start()


def stop_search_metering_worker(timeout: float = 5.0) -> int:
    """Stop the process-local worker, returning queued jobs abandoned."""
    global _worker  # noqa: PLW0603
    with _worker_lock:
        worker = _worker
    if worker is None:
        return 0
    abandoned = worker.stop(timeout=timeout)
    with _worker_lock:
        if _worker is worker:
            _worker = None
    return abandoned


def reset_search_metering_worker_for_tests() -> None:
    """Reset singleton state between tests."""
    global _worker  # noqa: PLW0603
    worker = _worker
    if worker is not None:
        worker.stop(timeout=0.1)
    _worker = None
