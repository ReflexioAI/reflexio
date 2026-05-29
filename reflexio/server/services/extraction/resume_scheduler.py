"""Process-local scheduler for resumable extraction follow-up work.

The scheduler is intentionally multi-tenant: each tick it discovers every org
that has actionable resumable-extraction work (a run ready to resume, a run
awaiting finalization retry, or a pending tool call due to expire) and drives a
per-org :class:`ExtractionResumeWorker` for each. Worker claims are org-scoped,
so a worker only ever resumes runs belonging to the org context it was built
with — never another tenant's runs.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.extraction.resumable_agent import (
    is_resumable_extraction_enabled,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 5.0


class ExtractionResumeScheduler:
    """Small polling wrapper that drives :class:`ExtractionResumeWorker` per org."""

    def __init__(
        self,
        *,
        request_context_factory: Callable[[str], RequestContext],
        bootstrap_org_id: str,
        max_runs_per_tick: int = 10,
    ) -> None:
        self.request_context_factory = request_context_factory
        self.bootstrap_org_id = bootstrap_org_id
        self.max_runs_per_tick = max_runs_per_tick
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="reflexio-extraction-resume-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("event=extraction_resume_scheduler_started")

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
        self._thread = None
        logger.info("event=extraction_resume_scheduler_stopped")

    def _discover_org_ids(self, bootstrap_ctx: RequestContext) -> list[str]:
        """Return every org with actionable work, always including the bootstrap org."""
        org_ids: list[str] = []
        storage = getattr(bootstrap_ctx, "storage", None)
        if storage is not None:
            try:
                org_ids = storage.list_resumable_work_org_ids(now=datetime.now(UTC))
            except NotImplementedError:
                org_ids = []
        # Always sweep the bootstrap org so the maintenance loop runs even when
        # the cross-org discovery query is empty or unsupported.
        if bootstrap_ctx.org_id not in org_ids:
            org_ids = [bootstrap_ctx.org_id, *org_ids]
        return org_ids

    def _expire_pending_tool_calls(self, bootstrap_ctx: RequestContext) -> None:
        storage = getattr(bootstrap_ctx, "storage", None)
        if storage is None:
            return
        # ``expire_pending_tool_calls`` is not org-scoped, so a single call
        # sweeps every tenant's overdue pending rows for this tick.
        try:
            expired = storage.expire_pending_tool_calls(now=datetime.now(UTC))
        except NotImplementedError:
            return
        if expired:
            logger.info("event=pending_tool_calls_expired expired=%d", expired)

    def _drain_org(self, org_id: str) -> None:
        try:
            ctx = self.request_context_factory(org_id)
            if not is_resumable_extraction_enabled(ctx):
                return
            resumed = ExtractionResumeWorker(request_context=ctx).drain(
                max_runs=self.max_runs_per_tick
            )
            if resumed:
                logger.info(
                    "event=extraction_resume_scheduler_tick org_id=%s resumed=%d",
                    org_id,
                    resumed,
                )
        except Exception as exc:
            logger.exception(
                "event=extraction_resume_scheduler_org_failed org_id=%s "
                "error_type=%s error=%s",
                org_id,
                type(exc).__name__,
                exc,
            )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            poll_interval = _DEFAULT_POLL_INTERVAL_SECONDS
            try:
                bootstrap_ctx = self.request_context_factory(self.bootstrap_org_id)
                config = bootstrap_ctx.configurator.get_config()
                poll_interval = (
                    config.pending_tool_call_config.resume_poll_interval_seconds
                )
                self._expire_pending_tool_calls(bootstrap_ctx)
                for org_id in self._discover_org_ids(bootstrap_ctx):
                    if self._stop_event.is_set():
                        break
                    self._drain_org(org_id)
            except Exception as exc:
                logger.exception(
                    "event=extraction_resume_scheduler_tick_failed "
                    "error_type=%s error=%s",
                    type(exc).__name__,
                    exc,
                )
            self._stop_event.wait(poll_interval)


def maybe_start_resume_scheduler(
    request_context_factory: Callable[[str], RequestContext],
    *,
    bootstrap_org_id: str,
) -> ExtractionResumeScheduler | None:
    """Start the scheduler only when the bootstrap-org config enables the feature.

    Args:
        request_context_factory: Builds an org-scoped :class:`RequestContext`.
        bootstrap_org_id: Org used to read config and to seed cross-org discovery.
    """
    try:
        ctx = request_context_factory(bootstrap_org_id)
        if not is_resumable_extraction_enabled(ctx):
            return None
    except Exception as exc:
        logger.warning(
            "event=extraction_resume_scheduler_start_skipped error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return None

    scheduler = ExtractionResumeScheduler(
        request_context_factory=request_context_factory,
        bootstrap_org_id=bootstrap_org_id,
    )
    scheduler.start()
    return scheduler
