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
import weakref
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from reflexio.pending_tool_policy import resolve_resume_timing
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.auth import DEFAULT_ORG_ID
from reflexio.server.background_work import (
    report_background_failure,
    report_background_success,
)
from reflexio.server.error_reporting import error_tags
from reflexio.server.scheduling import ThreadedScheduler
from reflexio.server.services.extraction.resumable_agent import (
    pending_tool_calls_disabled_reason,
    pending_tool_calls_enabled,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker

logger = logging.getLogger(__name__)

# Ceiling on the idle backoff. Fixed, not configurable: it bounds how long new
# work can wait for an idle scheduler that nothing woke, and 60s keeps that
# latency well inside the claim TTL and the expiry deadlines the sweep serves.
# An operator poll interval above it is honoured as-is (never shortened).
IDLE_BACKOFF_CAP_SECONDS = 60.0


class ResumeDiscoveryIncomplete(Exception):  # noqa: N818 -- a result, not an error
    """Raised by an ``org_id_provider`` whose discovery did not cover the fleet.

    A provider that swallows its own failures and returns ``[]`` makes an
    outage look like an idle fleet, and the scheduler would back off to the
    idle cap exactly when it should keep polling. Raising this instead says
    "the list is incomplete": the scheduler drains ``org_ids`` (or, when empty,
    falls back to the last known bootstrap org, as for any provider failure),
    never counts the tick as idle, and does NOT report it again -- the provider
    raising this has already reported the failure itself. Any other exception
    is an unreported provider failure and is reported by the scheduler.

    Args:
        org_ids (Iterable[str]): Orgs the partial discovery did find with work.
    """

    def __init__(self, org_ids: Iterable[str] = ()) -> None:
        self.org_ids = list(org_ids)
        super().__init__(f"resume org discovery incomplete ({len(self.org_ids)} found)")


_live_schedulers: weakref.WeakSet[ExtractionResumeScheduler] = weakref.WeakSet()
_live_lock = threading.Lock()


def wake_resume_schedulers() -> None:
    """Wake process-local consumers after a committed tool resolution."""
    with _live_lock:
        schedulers = list(_live_schedulers)
    for scheduler in schedulers:
        scheduler.wake()


class ExtractionResumeScheduler(ThreadedScheduler):
    """Small polling wrapper that drives :class:`ExtractionResumeWorker` per org."""

    def __init__(
        self,
        *,
        request_context_factory: Callable[[str], RequestContext],
        bootstrap_org_id: str,
        org_id_provider: Callable[[], list[str]] | None = None,
        max_runs_per_tick: int = 10,
        start_gate: Callable[[], bool] | None = None,
    ) -> None:
        self.timing = resolve_resume_timing()
        super().__init__(
            thread_name="reflexio-extraction-resume-scheduler", start_gate=start_gate
        )
        self.request_context_factory = request_context_factory
        self.bootstrap_org_id = bootstrap_org_id
        self.org_id_provider = org_id_provider
        self.max_runs_per_tick = max_runs_per_tick
        # Consecutive completed ticks that found no work. Owned by the loop
        # thread; reset on work, on any failure, and on stop/start.
        self._idle_ticks = 0

    def start(self) -> None:
        """Start the loop with a fresh idle backoff (first wait is the poll)."""
        if not self.is_running():
            self._idle_ticks = 0
        super().start()

    def _next_interval(self, *, idle: bool) -> float:
        """Return the wait before the next tick and advance the idle streak.

        Every tick is still a complete scan; only the gap between idle ticks
        grows. The first idle tick after work waits ``poll_seconds``; each
        further consecutive idle tick doubles it, up to
        ``max(poll_seconds, IDLE_BACKOFF_CAP_SECONDS)``. Work or a failure
        returns to ``poll_seconds``. ``wake()`` cuts any wait short.

        Args:
            idle (bool): True when this tick proved there was no work.

        Returns:
            float: Seconds to wait before the next tick.
        """
        poll = self.timing.poll_seconds
        if not idle:
            self._idle_ticks = 0
            return poll
        cap = max(poll, IDLE_BACKOFF_CAP_SECONDS)
        # Exponent clamp keeps the float finite however long the streak runs.
        interval = min(poll * 2 ** min(self._idle_ticks, 32), cap)
        self._idle_ticks += 1
        return interval

    def _on_started(self) -> None:
        with _live_lock:
            _live_schedulers.add(self)
        logger.info("event=extraction_resume_scheduler_started")

    def _on_stopped(self) -> None:
        self._idle_ticks = 0
        if not self.is_running():
            with _live_lock:
                _live_schedulers.discard(self)
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
        """Return the provider's org list, or ``None`` when it proves nothing.

        ``None`` means discovery failed with nothing to drain: the caller falls
        back to the bootstrap org and never treats the tick as idle. An empty
        list is returned only from a COMPLETE discovery, so it alone may be
        read as idle. A :class:`ResumeDiscoveryIncomplete` with orgs returns
        those orgs (drained, so never idle either).
        """
        if self.org_id_provider is None:
            return None
        complete = True
        try:
            raw = self.org_id_provider()
        except ResumeDiscoveryIncomplete as incomplete:
            # Reported by the provider; reporting it here too would double the
            # signal. Not a success either, so the failure streak stays open.
            raw = incomplete.org_ids
            complete = False
        except Exception as exc:
            with error_tags(
                subsystem="extraction",
                op="scheduler_org_discovery",
                error_type=type(exc).__name__,
            ):
                report_background_failure(
                    logger,
                    "extraction_resume_scheduler_provider_failed",
                    exc,
                    scope="extraction-resume-org-discovery",
                )
            return None
        discovered = list(
            dict.fromkeys(org_id for org_id in raw if org_id != DEFAULT_ORG_ID)
        )
        if not complete:
            return discovered or None
        report_background_success("extraction-resume-org-discovery")
        return discovered

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

    def _drain_org(self, org_id: str, *, has_known_work: bool = False) -> bool:
        """Drain one org; return True when the worker inspected any claim.

        A failed drain also returns True: a failure is never evidence of
        idleness, so it must not lengthen the next wait.
        """
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
                return False
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
                report_background_failure(
                    logger,
                    "extraction_resume_scheduler_org_failed",
                    exc,
                    scope=f"extraction-resume:{org_id}",
                    org_id=org_id,
                )
            return True
        report_background_success(f"extraction-resume:{org_id}")
        return bool(inspected)

    def _run_once(self) -> float:
        # Idle means this tick PROVED there was nothing to do: the provider
        # answered with no orgs, or (no provider) local discovery named no org
        # and no drain inspected a claim. The bootstrap org that local
        # discovery prepends unconditionally is not evidence of work, but its
        # drain result is -- a backend without discovery still reports work
        # through the drain. A provider failure (raised, or signalled as
        # ResumeDiscoveryIncomplete) or a tick failure is never idle, so a
        # broken discovery keeps polling at ``poll_seconds``.
        idle = False
        try:
            provider_org_ids = self._discover_provider_org_ids()
            if self.org_id_provider is not None and provider_org_ids is not None:
                if not provider_org_ids:
                    # Not a success: the context and config resolution this
                    # streak counts failures of was never attempted.
                    return self._next_interval(idle=True)
                # Remember a discovered org only for provider-failure fallback.
                # Resolve each context inside _drain_org so a broken first org
                # cannot prevent healthy orgs from being processed.
                self.bootstrap_org_id = provider_org_ids[0]
            # Orgs discovery PROVED hold resumable work, as opposed to orgs we
            # merely visit. Only the former may be described as stalled.
            known_work: set[str] = set()
            local_discovery = False
            if provider_org_ids is not None:
                org_ids = provider_org_ids
                known_work = set(provider_org_ids)
            elif self.org_id_provider is not None:
                # A raised provider cannot authoritatively replace the list;
                # preserve the last known bootstrap as a one-org fallback.
                # It is a fallback, NOT a discovery result -- the provider
                # failed, so nothing here proves this org has work.
                bootstrap_ctx = self.request_context_factory(self.bootstrap_org_id)
                org_ids = [bootstrap_ctx.org_id]
            else:
                bootstrap_ctx = self.request_context_factory(self.bootstrap_org_id)
                org_ids, known_work = self._discover_local_org_ids(bootstrap_ctx)
                local_discovery = True
            drained_work = False
            for org_id in org_ids:
                if self._stop_event.is_set():
                    break
                if self._drain_org(org_id, has_known_work=org_id in known_work):
                    drained_work = True
            idle = local_discovery and not known_work and not drained_work
        except Exception as exc:
            idle = False
            with error_tags(
                subsystem="extraction",
                op="scheduler_tick",
                error_type=type(exc).__name__,
            ):
                report_background_failure(
                    logger,
                    "extraction_resume_scheduler_tick_failed",
                    exc,
                    scope="extraction-resume-tick",
                )
        else:
            report_background_success("extraction-resume-tick")
        return self._next_interval(idle=idle)


def maybe_start_resume_scheduler(
    request_context_factory: Callable[[str], RequestContext],
    *,
    bootstrap_org_id: str,
    org_id_provider: Callable[[], list[str]] | None = None,
    start_gate: Callable[[], bool] | None = None,
) -> ExtractionResumeScheduler:
    """Start discovery; gate each organization's execution on its current config.

    A disabled bootstrap org must not disable recovery for other organizations
    or for a feature enabled after startup. Provider discovery runs before the
    bootstrap lookup, so an empty platform can safely wait for its first org.

    ``start_gate`` (default None: tick at once) holds the first tick until it
    returns True; see :class:`~reflexio.server.scheduling.ThreadedScheduler`.
    """
    scheduler = ExtractionResumeScheduler(
        request_context_factory=request_context_factory,
        bootstrap_org_id=bootstrap_org_id,
        org_id_provider=org_id_provider,
        start_gate=start_gate,
    )
    scheduler.start()
    return scheduler
