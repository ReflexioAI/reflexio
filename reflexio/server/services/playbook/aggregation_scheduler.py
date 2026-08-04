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

logger = logging.getLogger("reflexio.server.services.playbook.aggregation_scheduler")

_POLL_SECONDS = 10.0
AGGREGATION_LEASE_SECONDS = 300
AGGREGATION_RETRY_SECONDS = 60
AGGREGATION_BACKLOG_RETRY_SECONDS = 1
AGGREGATION_INVALIDATION_BATCH_SIZE = 100
_REPAIR_INTERVAL_SECONDS = 300.0


def aggregation_min_interval_seconds() -> int:
    raw = env_str("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", "3600")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "event=playbook_aggregation_invalid_interval value=%r default=3600", raw
        )
        return 3600


class AggregationLeaseHeartbeat:
    def __init__(self, storage: Any, claim: PlaybookAggregationClaim) -> None:
        self.storage = storage
        self.claim = claim
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="reflexio-playbook-aggregation-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(AGGREGATION_LEASE_SECONDS / 3):
            try:
                renewed = self.storage.renew_playbook_aggregation_claim(
                    self.claim, lease_seconds=AGGREGATION_LEASE_SECONDS
                )
            except Exception:
                logger.exception(
                    "event=playbook_aggregation_progress state=lease_lost "
                    "agent_version=%s fence=%s reason=renewal_failed",
                    self.claim.agent_version,
                    self.claim.fence,
                )
                self._lost.set()
                return
            if renewed is None:
                self._lost.set()
                return
            self.claim = renewed

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def require_live(self) -> None:
        if self._lost.is_set():
            raise RuntimeError("playbook aggregation lease was lost")


class PlaybookAggregationScheduler(ThreadedScheduler):
    """Poll durable per-version state and run one bounded unit per organization."""

    def __init__(
        self,
        *,
        context_provider: Callable[[], Iterable[RequestContext]],
        poll_interval_seconds: float = _POLL_SECONDS,
        leader_gate: LeaderGate | None = None,
        worker_id: str | None = None,
    ) -> None:
        super().__init__(
            thread_name="reflexio-playbook-aggregation-scheduler",
            leader_gate=leader_gate,
        )
        self._context_provider = context_provider
        self._poll_interval_seconds = poll_interval_seconds
        self._worker_id = worker_id or uuid.uuid4().hex
        self._last_repair_at: dict[str, float] = {}

    def _run_context(self, context: RequestContext) -> None:
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
        repair_now = time.monotonic()
        last_repair_at = self._last_repair_at.get(context.org_id)
        if (
            last_repair_at is None
            or repair_now - last_repair_at >= _REPAIR_INTERVAL_SECONDS
        ):
            repaired = storage.repair_playbook_aggregation_pending_state()
            self._last_repair_at[context.org_id] = repair_now
            for agent_version in repaired:
                logger.info(
                    "event=playbook_aggregation_progress state=scheduled org_id=%s "
                    "agent_version=%s reason=discovery_repair pending=true",
                    context.org_id,
                    agent_version,
                )
        claim = storage.claim_due_playbook_aggregation(
            owner=f"aggregation:{self._worker_id}:{context.org_id}",
            lease_seconds=AGGREGATION_LEASE_SECONDS,
        )
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
        heartbeat = AggregationLeaseHeartbeat(storage, claim)
        heartbeat.start()
        success = False
        result: dict[str, Any] = {}
        after = None
        try:
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
            heartbeat.require_live()
            success = True
            after = storage.get_playbook_aggregation_backlog(claim.agent_version)
        except TimeoutError:
            logger.warning(
                "event=playbook_aggregation_progress state=deferred org_id=%s "
                "agent_version=%s reason=limiter_saturated",
                context.org_id,
                claim.agent_version,
            )
        except Exception:
            logger.exception(
                "event=playbook_aggregation_progress state=retryable_failed "
                "org_id=%s agent_version=%s fence=%s",
                context.org_id,
                claim.agent_version,
                claim.fence,
            )
        finally:
            heartbeat.stop()
            active_claim = heartbeat.claim
            finished = storage.finish_playbook_aggregation_claim(
                active_claim,
                success=success,
                retry_after_seconds=AGGREGATION_RETRY_SECONDS,
                backlog_retry_after_seconds=AGGREGATION_BACKLOG_RETRY_SECONDS,
                min_interval_seconds=aggregation_min_interval_seconds(),
                backlog=after if success else None,
            )
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
                    "retryable_failures=%s",
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
                )

    def _run_once(self) -> float:
        try:
            for context in self._context_provider():
                if self._stop_event.is_set():
                    break
                self._run_context(context)
        except Exception:
            logger.exception("event=playbook_aggregation_scheduler_tick_failed")
        return self._poll_interval_seconds

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
