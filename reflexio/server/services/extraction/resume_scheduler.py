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
from collections.abc import Callable
from datetime import UTC, datetime

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.auth import DEFAULT_ORG_ID
from reflexio.server.error_reporting import error_tags
from reflexio.server.scheduling import ThreadedScheduler
from reflexio.server.services.extraction.resumable_agent import (
    pending_tool_calls_enabled,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 5.0


class ExtractionResumeScheduler(ThreadedScheduler):
    """Small polling wrapper that drives :class:`ExtractionResumeWorker` per org."""

    def __init__(
        self,
        *,
        request_context_factory: Callable[[str], RequestContext],
        bootstrap_org_id: str,
        org_id_provider: Callable[[], list[str]] | None = None,
        max_runs_per_tick: int = 10,
    ) -> None:
        super().__init__(thread_name="reflexio-extraction-resume-scheduler")
        self.request_context_factory = request_context_factory
        self.bootstrap_org_id = bootstrap_org_id
        self.org_id_provider = org_id_provider
        self.max_runs_per_tick = max_runs_per_tick

    def _on_started(self) -> None:
        logger.info("event=extraction_resume_scheduler_started")

    def _on_stopped(self) -> None:
        logger.info("event=extraction_resume_scheduler_stopped")

    def _discover_local_org_ids(self, bootstrap_ctx: RequestContext) -> list[str]:
        """Return local actionable orgs plus the bootstrap org."""
        org_ids: list[str] = []
        storage = getattr(bootstrap_ctx, "storage", None)
        if storage is not None:
            try:
                org_ids = storage.list_resumable_work_org_ids(now=datetime.now(UTC))
            except NotImplementedError:
                org_ids = []
        if bootstrap_ctx.org_id not in org_ids:
            org_ids = [bootstrap_ctx.org_id, *org_ids]
        return list(dict.fromkeys(org_ids))

    def _discover_provider_org_ids(self) -> list[str] | None:
        """Return the provider's authoritative list, or ``None`` on failure."""
        if self.org_id_provider is None:
            return None
        try:
            return list(
                dict.fromkeys(
                    org_id
                    for org_id in self.org_id_provider()
                    if org_id != DEFAULT_ORG_ID
                )
            )
        except Exception as exc:
            with error_tags(
                subsystem="extraction",
                op="scheduler_org_discovery",
                error_type=type(exc).__name__,
            ):
                logger.exception("event=extraction_resume_scheduler_provider_failed")
            return None

    def _expire_pending_tool_calls(self, ctx: RequestContext) -> None:
        storage = getattr(ctx, "storage", None)
        if storage is None:
            return
        # One storage ref can contain several tenants, but another ref is an
        # independent queue. Sweep every discovered ref before draining it.
        try:
            expired = storage.expire_pending_tool_calls(now=datetime.now(UTC))
        except NotImplementedError:
            return
        if expired:
            logger.info("event=pending_tool_calls_expired expired=%d", expired)

    def _drain_org(self, org_id: str) -> None:
        try:
            ctx = self.request_context_factory(org_id)
            if not pending_tool_calls_enabled(ctx):
                return
            self._expire_pending_tool_calls(ctx)
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
            with error_tags(
                subsystem="extraction",
                op="scheduler_org_drain",
                org_id=org_id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=extraction_resume_scheduler_org_failed org_id=%s",
                    org_id,
                )

    def _run_once(self) -> float:
        poll_interval = _DEFAULT_POLL_INTERVAL_SECONDS
        try:
            provider_org_ids = self._discover_provider_org_ids()
            if self.org_id_provider is not None and provider_org_ids is not None:
                if not provider_org_ids:
                    return poll_interval
                # Resolve config through an org that the provider proved is
                # actionable on this tick. The previous bootstrap may have
                # been deleted or moved and must not gate future discovery.
                self.bootstrap_org_id = provider_org_ids[0]
            bootstrap_ctx = self.request_context_factory(self.bootstrap_org_id)
            config = bootstrap_ctx.configurator.get_config()
            poll_interval = config.pending_tool_call_config.resume_poll_interval_seconds
            if provider_org_ids is not None:
                org_ids = provider_org_ids
            elif self.org_id_provider is not None:
                # A raised provider cannot authoritatively replace the list;
                # preserve the last known bootstrap as a one-org fallback.
                org_ids = [bootstrap_ctx.org_id]
            else:
                org_ids = self._discover_local_org_ids(bootstrap_ctx)
            for org_id in org_ids:
                if self._stop_event.is_set():
                    break
                self._drain_org(org_id)
        except Exception as exc:
            with error_tags(
                subsystem="extraction",
                op="scheduler_tick",
                error_type=type(exc).__name__,
            ):
                logger.exception("event=extraction_resume_scheduler_tick_failed")
        return poll_interval


def maybe_start_resume_scheduler(
    request_context_factory: Callable[[str], RequestContext],
    *,
    bootstrap_org_id: str,
    org_id_provider: Callable[[], list[str]] | None = None,
) -> ExtractionResumeScheduler | None:
    """Start the scheduler only when the bootstrap-org config enables the feature.

    Args:
        request_context_factory: Builds an org-scoped :class:`RequestContext`.
        bootstrap_org_id: Org used to read config and to seed cross-org discovery.
    """
    try:
        ctx = request_context_factory(bootstrap_org_id)
        if not pending_tool_calls_enabled(ctx):
            return None
    except Exception as exc:
        if bootstrap_org_id == DEFAULT_ORG_ID and org_id_provider is not None:
            logger.info(
                "event=extraction_resume_scheduler_start_deferred "
                "reason=no_organizations"
            )
        else:
            logger.warning(
                "event=extraction_resume_scheduler_start_skipped error_type=%s error=%s",
                type(exc).__name__,
                exc,
            )
            return None

    scheduler = ExtractionResumeScheduler(
        request_context_factory=request_context_factory,
        bootstrap_org_id=bootstrap_org_id,
        org_id_provider=org_id_provider,
    )
    scheduler.start()
    return scheduler
