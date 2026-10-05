"""Durable, fleet-fenced incremental playbook aggregation scheduler."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.background_work import (
    background_work,
    report_background_failure,
    report_background_success,
)
from reflexio.server.env_utils import env_str
from reflexio.server.extensions import get_service
from reflexio.server.operation_limiter import run_with_operation_limit
from reflexio.server.scheduling import LeaderGate, ThreadedScheduler
from reflexio.server.services.playbook.aggregation_prompt_processing import (
    AGGREGATION_PROMPT_PROCESSOR,
)
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationClaim,
)
from reflexio.server.work_scope import current_project_id

logger = logging.getLogger("reflexio.server.services.playbook.aggregation_scheduler")

_POLL_SECONDS = 10.0
AGGREGATION_LEASE_SECONDS = 300
AGGREGATION_RETRY_SECONDS = 60
AGGREGATION_BACKLOG_RETRY_SECONDS = 1
AGGREGATION_INVALIDATION_BATCH_SIZE = 100
_REPAIR_INTERVAL_SECONDS = 300.0


# Result keys with which `PlaybookAggregator.run` reports that it returned
# without fully doing its work. Any of them truthy means the run made partial
# or no progress, so it is not evidence the run unit recovered:
# - skipped: legacy cluster adoption pending (its embedding errors swallowed),
#   operation already applied, no config, too few new playbooks, no changes;
# - retryable_failures: per-cluster generation failures, logged at ERROR;
# - cluster_fence_losses: clusters lost to a concurrent fence;
# - embedding_pending: members deferred for want of a vector.
_PARTIAL_RESULT_MARKERS = (
    "skipped",
    "retryable_failures",
    "cluster_fence_losses",
    "embedding_pending",
)


def _is_unqualified_success(result: dict[str, Any]) -> bool:
    return not any(result.get(marker) for marker in _PARTIAL_RESULT_MARKERS)


def aggregation_min_interval_seconds() -> int:
    raw = env_str("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", "3600")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "event=playbook_aggregation_invalid_interval value=%r default=3600", raw
        )
        return 3600


class _FinalizationReportedError(Exception):
    """Finalization failed and was already reported under its own scope."""


class AggregationLeaseHeartbeat:
    def __init__(
        self,
        storage: Any,
        claim: PlaybookAggregationClaim,
        *,
        org_id: str,
        project_id: str | None,
    ) -> None:
        self.storage = storage
        self.claim = claim
        # Captured here, on the caller's thread: the heartbeat thread does not
        # inherit contextvars, so `current_project_id()` there is unbound.
        # The failure scope must name this (org, project) lease, or every
        # tenant's heartbeat shares one streak and one tenant's renewals would
        # clear another's.
        self._failure_scope = (
            f"playbook-aggregation-heartbeat:{org_id}:{project_id}:"
            f"{claim.agent_version}"
        )
        self._stop = threading.Event()
        self._lost = threading.Event()
        # Why renewal failed, chained onto `require_live`'s error so the
        # failure policy sees a dropped connection rather than a bare RuntimeError.
        self._renewal_error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="reflexio-playbook-aggregation-heartbeat",
            daemon=True,
        )
        self._thread.start()

    @background_work()
    def _run(self) -> None:
        while not self._stop.wait(AGGREGATION_LEASE_SECONDS / 3):
            try:
                renewed = self.storage.renew_playbook_aggregation_claim(
                    self.claim, lease_seconds=AGGREGATION_LEASE_SECONDS
                )
            except Exception as exc:
                report_background_failure(
                    logger,
                    "playbook_aggregation_progress",
                    exc,
                    scope=self._failure_scope,
                    state="lease_lost",
                    agent_version=self.claim.agent_version,
                    fence=self.claim.fence,
                    reason="renewal_failed",
                )
                self._renewal_error = exc
                self._lost.set()
                return
            if renewed is None:
                self._lost.set()
                return
            report_background_success(self._failure_scope)
            self.claim = renewed

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def require_live(self) -> None:
        if self._lost.is_set():
            raise RuntimeError(
                "playbook aggregation lease was lost"
            ) from self._renewal_error


class PlaybookAggregationScheduler(ThreadedScheduler):
    """Poll durable per-version state and run one bounded unit per organization.

    Sparse context providers must supply ``scope_inventory_provider``: the
    complete live (organization, project) inventory, independent of due work.
    Return None or raise when that inventory is unavailable; throttle state
    is then retained. Without it, context iteration is a complete fleet sweep.
    """

    def __init__(
        self,
        *,
        context_provider: Callable[[], Iterable[RequestContext]],
        scope_inventory_provider: Callable[[], Iterable[tuple[str, str | None]] | None]
        | None = None,
        on_work_claimed: Callable[[RequestContext], None] | None = None,
        on_scope_deferred: Callable[[RequestContext, float], None] | None = None,
        poll_interval_seconds: float = _POLL_SECONDS,
        leader_gate: LeaderGate | None = None,
        worker_id: str | None = None,
    ) -> None:
        super().__init__(
            thread_name="reflexio-playbook-aggregation-scheduler",
            leader_gate=leader_gate,
        )
        self._context_provider = context_provider
        self._scope_inventory_provider = scope_inventory_provider
        self._on_work_claimed = on_work_claimed
        self._on_scope_deferred = on_scope_deferred
        self._poll_interval_seconds = poll_interval_seconds
        self._worker_id = worker_id or uuid.uuid4().hex
        # Keyed by (org_id, project_id) TUPLES, not by a joined string. Codex
        # on reflexio#510: with `f"{org}:{project}"`, an org id containing a
        # colon collides with a different (org, project) pair -- both
        # ("a:b", "c") and ("a", "b:c") render as "a:b:c" -- and the first
        # scope processed keeps refreshing the shared timestamp, permanently
        # starving the other. Both identifiers are unconstrained strings, so a
        # tuple is the only key that cannot be ambiguous.
        self._last_repair_at: dict[tuple[str, str | None], float] = {}
        self._retry_after: dict[tuple[str, str | None], float] = {}
        # Organization execution is sequential on the scheduler thread.
        self._active_stage = "configuration"

    @staticmethod
    def _repair_scope_key(context: RequestContext) -> tuple[str, str | None]:
        """Throttle key for one unit of work: (org, project).

        OSS has no project concept, so this is just the org id there. Under the
        enterprise context provider the same ``RequestContext`` is yielded once
        per project with a different project bound, so the bound project has to
        enter the key or all but one project share a single throttle slot.

        Read through the OSS ``work_scope`` seam, NOT by reaching for an
        attribute on the context. ``RequestContext`` has no ``work_scope``
        attribute -- a ``getattr(context, "work_scope", None)`` would return
        ``None`` forever and silently leave this keyed by org alone, which is
        the bug it is meant to fix. ``current_project_id()`` is inert in bare
        OSS (no provider registered) and returns the bound project under
        enterprise, which registers one.
        """
        return (str(context.org_id), current_project_id())

    def _run_context(self, context: RequestContext) -> None:
        """Run one bounded unit for ``context``.

        Three units, three streaks. The org unit -- configuration, repair and
        the claim -- reports success here as soon as the claim returns,
        whatever the claimed run or its finalization then do; its failures
        escape to ``_run_once``. The claimed run and its finalization report
        both outcomes here under their own scopes. A context with aggregation
        off or unsupported attempts nothing and reports nothing.

        Raises:
            _FinalizationReportedError: Finalization failed after being reported
                under its own scope; the caller backs off without reporting.
        """
        self._active_stage = "configuration"
        storage = context.storage
        playbook_config = getattr(
            context.configurator.get_config(), "user_playbook_extractor_config", None
        )
        if playbook_config is None or playbook_config.aggregation_config is None:
            return
        if storage is None:
            return
        if not getattr(storage, "supports_incremental_playbook_aggregation", False):
            blocked_reason = getattr(
                storage, "playbook_aggregation_blocked_reason", None
            )
            if blocked_reason:
                logger.warning(
                    "event=playbook_aggregation_progress state=blocked org_id=%s "
                    "reason=%s",
                    context.org_id,
                    blocked_reason,
                )
            return
        org_id, project_id = self._repair_scope_key(context)
        repair_now = time.monotonic()
        # Keyed by (org, work scope) rather than by org alone. The enterprise
        # context provider yields the SAME RequestContext once per project, so
        # an org-keyed throttle let only the FIRST project of an org reach
        # repair in each interval -- every other project's pending state was
        # never repaired at all. That was masked while repair's own SQL was
        # org-wide (the first project happened to repair everyone's); once that
        # SQL is project-scoped, an org-keyed throttle here silently starves
        # projects 2..N. The two must change together.
        repair_key = self._repair_scope_key(context)
        last_repair_at = self._last_repair_at.get(repair_key)
        if (
            last_repair_at is None
            or repair_now - last_repair_at >= _REPAIR_INTERVAL_SECONDS
        ):
            self._last_repair_at[repair_key] = repair_now
            self._active_stage = "repair"
            repaired = storage.repair_playbook_aggregation_pending_state()
            for agent_version in repaired:
                logger.info(
                    "event=playbook_aggregation_progress state=scheduled org_id=%s "
                    "agent_version=%s reason=discovery_repair pending=true",
                    context.org_id,
                    agent_version,
                )
        self._active_stage = "claim"
        claim = storage.claim_due_playbook_aggregation(
            owner=f"aggregation:{self._worker_id}:{context.org_id}",
            lease_seconds=AGGREGATION_LEASE_SECONDS,
        )
        # The org unit ran: configuration, repair (when due) and the claim
        # all returned, whether or not anything was due.
        report_background_success(f"playbook-aggregation-org:{org_id}:{project_id}")
        if claim is None:
            return
        started = time.perf_counter()
        logger.info(
            "event=playbook_aggregation_progress state=claimed org_id=%s "
            "agent_version=%s fence=%s pending=true",
            context.org_id,
            claim.agent_version,
            claim.fence,
        )
        self._active_stage = "aggregation"
        # Distinct from the org-failure scope in `_run_once`: this unit is the
        # claimed (org, project, agent_version) run, retried by the next claim
        # of that version; that one is everything else `_run_context` raises.
        run_scope = (
            f"playbook-aggregation-run:{org_id}:{project_id}:{claim.agent_version}"
        )
        heartbeat = AggregationLeaseHeartbeat(
            storage, claim, org_id=org_id, project_id=project_id
        )
        heartbeat.start()
        success = False
        aggregated = False
        result: dict[str, Any] = {}
        after = None
        try:
            if self._on_work_claimed is not None:
                try:
                    self._on_work_claimed(context)
                except Exception:
                    # Stays at ERROR: the notification is one-shot -- nothing
                    # re-invokes it for this claim -- so there is no retry for
                    # a WARNING to lean on.
                    logger.exception(
                        "event=playbook_aggregation_claim_notification_failed org_id=%s",
                        context.org_id,
                    )
            budget = _aggregation_budget()
            invalidation_page = storage.get_playbook_aggregation_invalidations(
                claim.agent_version, limit=AGGREGATION_INVALIDATION_BATCH_SIZE + 1
            )
            invalidations = invalidation_page[:AGGREGATION_INVALIDATION_BATCH_SIZE]
            if invalidations and not storage.apply_playbook_aggregation_invalidations(
                claim, [item.invalidation_id for item in invalidations]
            ):
                raise RuntimeError("playbook aggregation invalidation fence was lost")
            if len(invalidation_page) > AGGREGATION_INVALIDATION_BATCH_SIZE:
                logger.info(
                    "event=playbook_aggregation_progress "
                    "state=draining_invalidations org_id=%s agent_version=%s "
                    "fence=%s processed=%s",
                    context.org_id,
                    claim.agent_version,
                    claim.fence,
                    len(invalidations),
                )
                result = {"invalidations_processed": len(invalidations)}
            else:
                from reflexio.lib.generation_client import (
                    create_generation_litellm_client,
                )

                kwargs: dict[str, Any] = {}
                processor = get_service(AGGREGATION_PROMPT_PROCESSOR)
                if processor is not None:
                    kwargs["aggregation_prompt_processor"] = processor
                aggregator = PlaybookAggregator(
                    llm_client=create_generation_litellm_client(context),
                    request_context=context,
                    agent_version=claim.agent_version,
                    aggregation_claim=claim,
                    residual_batch_limit=budget,
                    **kwargs,
                )
                logger.info(
                    "event=playbook_aggregation_progress state=started org_id=%s "
                    "agent_version=%s fence=%s",
                    context.org_id,
                    claim.agent_version,
                    claim.fence,
                )
                result = run_with_operation_limit(
                    org_id=context.org_id,
                    operation="aggregation",
                    fn=lambda: aggregator.run(
                        PlaybookAggregatorRequest(agent_version=claim.agent_version)
                    ),
                )
                aggregated = True
            heartbeat.require_live()
            # `success` is set AFTER the backlog read, not before. Set before,
            # a failing read left success=True, so the `finally:` below called
            # finish(success=True, backlog=None) -- and finish recomputes the
            # backlog when it is None, re-running the same failing query and
            # raising a SECOND time, out of `finally:`. The lease was then
            # never released and `_run_once` backed off the whole ORG for
            # `_REPAIR_INTERVAL_SECONDS` rather than the one failing unit.
            # With success=False, finish takes its `if not success:` branch,
            # skips the recompute, releases the lease and schedules a retry.
            after = storage.get_playbook_aggregation_backlog(claim.agent_version)
            success = True
            # Only an aggregation that actually ran and fully completed ends
            # the run streak. An invalidation-only pass (more than one batch
            # pending) never ran the aggregator, and a run that returned with
            # any `_PARTIAL_RESULT_MARKERS` set skipped or deferred work --
            # possibly after swallowing the very failure this streak counts.
            if aggregated and _is_unqualified_success(result):
                report_background_success(run_scope)
        except TimeoutError:
            logger.warning(
                "event=playbook_aggregation_progress state=deferred org_id=%s "
                "agent_version=%s reason=limiter_saturated",
                context.org_id,
                claim.agent_version,
            )
        except Exception as exc:
            report_background_failure(
                logger,
                "playbook_aggregation_progress",
                exc,
                scope=run_scope,
                state="retryable_failed",
                org_id=context.org_id,
                agent_version=claim.agent_version,
                fence=claim.fence,
            )
        finally:
            self._active_stage = "finalization"
            heartbeat.stop()
            active_claim = heartbeat.claim
            # Finalization is its own unit: it runs (and is retried) whether
            # or not the run succeeded, so its streak must reset when IT
            # succeeds -- not when an unrelated run does.
            finalize_scope = (
                f"playbook-aggregation-finalize:{org_id}:{project_id}:"
                f"{claim.agent_version}"
            )
            try:
                finished = storage.finish_playbook_aggregation_claim(
                    active_claim,
                    success=success,
                    retry_after_seconds=AGGREGATION_RETRY_SECONDS,
                    backlog_retry_after_seconds=AGGREGATION_BACKLOG_RETRY_SECONDS,
                    min_interval_seconds=aggregation_min_interval_seconds(),
                    backlog=after if success else None,
                )
            except Exception as exc:
                report_background_failure(
                    logger,
                    "playbook_aggregation_scheduler_org_failed",
                    exc,
                    scope=finalize_scope,
                    org_id=org_id,
                    project_id=project_id,
                    stage="finalization",
                    retry_after_seconds=_REPAIR_INTERVAL_SECONDS,
                )
                raise _FinalizationReportedError from exc
            report_background_success(finalize_scope)
            if not finished:
                logger.warning(
                    "event=playbook_aggregation_progress state=lease_lost org_id=%s "
                    "agent_version=%s fence=%s",
                    context.org_id,
                    claim.agent_version,
                    claim.fence,
                )
            elif success and after is not None:
                logger.info(
                    "event=playbook_aggregation_progress state=succeeded org_id=%s "
                    "agent_version=%s fence=%s pending=%s undisposed=%s residual=%s "
                    "invalidations=%s oldest_residual_age_seconds=%s "
                    "dirty_repairs=%s duration_ms=%s creations=%s supersessions=%s "
                    "retryable_failures=%s embedding_pending=%s",
                    context.org_id,
                    claim.agent_version,
                    claim.fence,
                    str(after.pending).lower(),
                    after.undisposed,
                    after.residual,
                    after.invalidations,
                    after.oldest_residual_age_seconds,
                    after.dirty_repairs,
                    int((time.perf_counter() - started) * 1000),
                    result.get("playbooks_generated", 0),
                    result.get("supersessions", 0),
                    result.get("retryable_failures", 0),
                    result.get("embedding_pending", 0),
                )

    def _defer_scope(self, context: RequestContext, delay: float) -> None:
        if self._on_scope_deferred is None:
            return
        org_id, project_id = self._repair_scope_key(context)
        defer_scope = f"playbook-aggregation-defer:{org_id}:{project_id}"
        try:
            self._on_scope_deferred(context, delay)
        except Exception as exc:
            # A provider failure cannot mask the original storage error or
            # prevent independent scopes from making progress.
            report_background_failure(
                logger,
                "playbook_aggregation_scope_defer_failed",
                exc,
                scope=defer_scope,
                org_id=context.org_id,
            )
        else:
            report_background_success(defer_scope)

    def _run_once(self) -> float:
        seen: set[tuple[str, str | None]] = set()
        try:
            for context in self._context_provider():
                if self._stop_event.is_set():
                    break
                org_id = context.org_id
                # The FAILURE backoff carries the same (org, project) scope as
                # the repair throttle. Codex on reflexio#510, and it corrects a
                # call I got wrong: I had recorded `_retry_after` as "a backoff,
                # not a correctness gate" and left it org-keyed. It is a
                # correctness gate. Keyed by org, one project that fails
                # repeatedly skips EVERY later project of that org -- the exact
                # starvation the repair re-key exists to remove, arriving by a
                # different route.
                scope = self._repair_scope_key(context)
                seen.add(scope)
                retry_delay = self._retry_after.get(scope, 0) - time.monotonic()
                if retry_delay > 0:
                    self._defer_scope(context, retry_delay)
                    continue
                project_id = scope[1]
                org_scope = f"playbook-aggregation-org:{org_id}:{project_id}"
                try:
                    self._run_context(context)
                except _FinalizationReportedError:
                    # Reported under its own scope inside `_run_context`.
                    self._retry_after[scope] = (
                        time.monotonic() + _REPAIR_INTERVAL_SECONDS
                    )
                    self._defer_scope(context, _REPAIR_INTERVAL_SECONDS)
                except Exception as exc:
                    self._retry_after[scope] = (
                        time.monotonic() + _REPAIR_INTERVAL_SECONDS
                    )
                    self._defer_scope(context, _REPAIR_INTERVAL_SECONDS)
                    report_background_failure(
                        logger,
                        "playbook_aggregation_scheduler_org_failed",
                        exc,
                        scope=org_scope,
                        org_id=org_id,
                        project_id=project_id,
                        stage=self._active_stage,
                        retry_after_seconds=_REPAIR_INTERVAL_SECONDS,
                    )
                else:
                    self._retry_after.pop(scope, None)
        except Exception as exc:
            report_background_failure(
                logger,
                "playbook_aggregation_scheduler_tick_failed",
                exc,
                scope="playbook-aggregation-tick",
                stage="context_provider",
            )
        else:
            report_background_success("playbook-aggregation-tick")
            if not self._stop_event.is_set():
                try:
                    inventory = (
                        self._scope_inventory_provider()
                        if self._scope_inventory_provider is not None
                        else seen
                    )
                    # None means unavailable, not an empty fleet. Materialize
                    # before pruning so a partial iterator cannot erase state.
                    if inventory is not None:
                        live_scopes = set(inventory)
                        if not self._stop_event.is_set():
                            self._prune_scope_state(live_scopes)
                        # None ("unavailable") is not a success: the provider
                        # may have swallowed the very failure this streak counts.
                        report_background_success("playbook-aggregation-inventory")
                except Exception as exc:
                    report_background_failure(
                        logger,
                        "playbook_aggregation_scheduler_tick_failed",
                        exc,
                        scope="playbook-aggregation-inventory",
                        stage="scope_inventory",
                    )
        return self._poll_interval_seconds

    def _prune_scope_state(self, seen: set[tuple[str, str | None]]) -> None:
        """Drop state only for scopes absent from a complete live inventory.

        Sparse providers supply that inventory separately from due contexts.
        Legacy full-sweep providers use contexts observed in a completed pass.
        Failed or interrupted iterations must never prune partial inventories.
        """
        for state in (self._last_repair_at, self._retry_after):
            for key in [k for k in state if k not in seen]:
                del state[key]

    def _on_started(self) -> None:
        logger.info("event=playbook_aggregation_scheduler_started")

    def _on_stopped(self) -> None:
        if not self.is_running():
            with _LOCAL_SCHEDULERS_LOCK:
                stale_org_ids = [
                    org_id
                    for org_id, scheduler in _LOCAL_SCHEDULERS.items()
                    if scheduler is self
                ]
                for org_id in stale_org_ids:
                    _LOCAL_SCHEDULERS.pop(org_id, None)
        logger.info("event=playbook_aggregation_scheduler_stopped")


def _aggregation_budget() -> int:
    from reflexio.server.services.playbook.components.aggregator_clustering import (
        max_clustering_playbooks,
    )

    return max_clustering_playbooks()


_LOCAL_SCHEDULERS: dict[str, PlaybookAggregationScheduler] = {}
_LOCAL_SCHEDULERS_LOCK = threading.Lock()


def ensure_local_playbook_aggregation_scheduler(
    request_context: RequestContext,
) -> PlaybookAggregationScheduler | None:
    """Lazily provide the time-driven scheduler in OSS local mode."""
    if os.getenv("DEPLOYMENT_MODE", "").strip() in {"platform", "self_host"}:
        return None
    with _LOCAL_SCHEDULERS_LOCK:
        scheduler = _LOCAL_SCHEDULERS.get(request_context.org_id)
        if scheduler is None or not scheduler.is_running():
            scheduler = PlaybookAggregationScheduler(
                context_provider=lambda: [request_context]
            )
            _LOCAL_SCHEDULERS[request_context.org_id] = scheduler
            scheduler.start()
        return scheduler
