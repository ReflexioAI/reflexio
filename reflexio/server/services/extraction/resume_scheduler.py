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
    pending_tool_calls_disabled_reason,
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

    def _discover_local_org_ids(
        self, bootstrap_ctx: RequestContext
    ) -> tuple[list[str], set[str]]:
        """Return (orgs to drain, the subset KNOWN to hold resumable work).

        The two differ, and conflating them is a defect rather than a detail.
        The bootstrap org is prepended unconditionally so a single-tenant
        install still drains, which means it appears here even when storage
        reported no resumable work at all. Anything that wants to say "this
        org has work" must consult the second value, never the first.

        Args:
            bootstrap_ctx (RequestContext): Context of the bootstrap org.

        Returns:
            tuple[list[str], set[str]]: Orgs to visit, and those with work.
        """
        discovered: list[str] = []
        storage = getattr(bootstrap_ctx, "storage", None)
        if storage is not None:
            try:
                discovered = storage.list_resumable_work_org_ids(now=datetime.now(UTC))
            except NotImplementedError:
                discovered = []
        known_work = set(discovered)
        org_ids = list(discovered)
        if bootstrap_ctx.org_id not in org_ids:
            org_ids = [bootstrap_ctx.org_id, *org_ids]
        return list(dict.fromkeys(org_ids)), known_work

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

    def _drain_org(self, org_id: str, *, has_known_work: bool = False) -> None:
        try:
            ctx = self.request_context_factory(org_id)
            if not pending_tool_calls_enabled(ctx):
                # Behaviour is deliberate and specified -- a disabled org's
                # resume machinery stops. What was NOT deliberate is that it
                # stopped SILENTLY: the provider had just named this org as
                # having resumable work, and nothing said why none of it moved.
                #
                # Measured on prod 2026-09-20: 238 runs across three orgs sat
                # here, every one of them a finalization retry already holding
                # `committed_output`, none of them waiting on a pending-info
                # tool -- and the sweep logged `resumable_orgs` without a
                # single line explaining the stall. An operator reading the
                # logs could not tell "disabled on purpose" from "broken".
                #
                # Gated on `has_known_work`, and that gate is the whole
                # difference between a diagnostic and a noise generator. The
                # default OSS path has no provider and prepends the bootstrap
                # org whether or not storage found anything, so an idle
                # single-tenant install reaches here on EVERY tick. At the
                # 5s default poll interval an unconditional line would emit
                # >17,000 entries a day, each one claiming work exists that
                # does not. Say nothing unless discovery actually named it.
                if has_known_work:
                    logger.info(
                        "event=extraction_resume_drain_skipped_gate_disabled "
                        "org_id=%s reason=%s -- the org has resumable work and "
                        "this gate is shut, so none of it will be drained "
                        "until that changes",
                        org_id,
                        # Which gate, not a guess: `pending_tool_calls_enabled`
                        # is false for three different reasons, and naming the
                        # wrong one sends an operator to change a setting that
                        # is already correct.
                        pending_tool_calls_disabled_reason(ctx) or "gate closed",
                    )
                return
            self._expire_pending_tool_calls(ctx)
            inspected = ExtractionResumeWorker(request_context=ctx).drain(
                max_runs=self.max_runs_per_tick
            )
            if inspected:
                logger.info(
                    "event=extraction_resume_scheduler_tick org_id=%s claims_inspected=%d",
                    org_id,
                    inspected,
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
            # Orgs discovery PROVED hold resumable work, as opposed to orgs we
            # merely visit. Only the former may be described as stalled.
            known_work: set[str] = set()
            if provider_org_ids is not None:
                org_ids = provider_org_ids
                known_work = set(provider_org_ids)
            elif self.org_id_provider is not None:
                # A raised provider cannot authoritatively replace the list;
                # preserve the last known bootstrap as a one-org fallback.
                # It is a fallback, NOT a discovery result -- the provider
                # failed, so nothing here proves this org has work.
                org_ids = [bootstrap_ctx.org_id]
            else:
                org_ids, known_work = self._discover_local_org_ids(bootstrap_ctx)
            for org_id in org_ids:
                if self._stop_event.is_set():
                    break
                self._drain_org(org_id, has_known_work=org_id in known_work)
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
) -> ExtractionResumeScheduler:
    """Start discovery; gate each organization's execution on its current config.

    A disabled bootstrap org must not disable recovery for other organizations
    or for a feature enabled after startup. Provider discovery runs before the
    bootstrap lookup, so an empty platform can safely wait for its first org.
    """
    scheduler = ExtractionResumeScheduler(
        request_context_factory=request_context_factory,
        bootstrap_org_id=bootstrap_org_id,
        org_id_provider=org_id_provider,
    )
    scheduler.start()
    return scheduler
