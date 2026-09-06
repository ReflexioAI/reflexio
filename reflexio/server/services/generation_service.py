from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reflexio.defaults import resolve_agent_version
from reflexio.models.api_schema.common import sanitise_for_log
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    PublishUserInteractionRequest,
    Request,
)
from reflexio.models.config_schema import Config
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.env_utils import env_str, env_truthy
from reflexio.server.error_reporting import error_tags
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.operation_limiter import operation_limit
from reflexio.server.services.agent_success_evaluation.runner import (
    run_group_evaluation,
)
from reflexio.server.services.agent_success_evaluation.sampling import (
    samples_agent_success,
    samples_retrieved_learning,
)
from reflexio.server.services.agent_success_evaluation.scheduler import (
    GroupEvaluationScheduler,
)
from reflexio.server.services.deferred_learning_plan import DeferredLearningPlan
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
)
from reflexio.server.services.profile.service import (
    ProfileGenerationService,
)
from reflexio.server.services.retrieval_experiment import (
    validate_retrieval_experiment_attribution,
)
from reflexio.server.services.shadow_comparison.worker import (
    ShadowComparisonJob,
    enqueue_shadow_comparison,
)
from reflexio.server.services.storage.retention import (
    delete_count_for_retention,
    get_row_retention_limits,
)
from reflexio.server.services.tagging.tagging_scheduler import schedule_tagging
from reflexio.server.usage_metrics import record_usage_event
from reflexio.server.work_scope import current_project_id

if TYPE_CHECKING:
    from reflexio.server.services.unified_search_service import UnifiedSearchService

logger = logging.getLogger(__name__)
# Stale lock timeout - if cleanup started > 10 min ago and still "in_progress", assume it crashed
CLEANUP_STALE_LOCK_SECONDS = 600
# Timeout for the outer generation service parallel execution
GENERATION_SERVICE_TIMEOUT_SECONDS = 600
_STALL_WARNING_PREFIX = "Reflexio learning is paused"


class _SharedDeadline:
    """One wall-clock budget shared by several sequentially awaited futures.

    Profile and playbook generation run concurrently but are awaited one after
    the other. Passing each ``future.result()`` the same constant timeout hands
    the second service a FRESH full budget on top of whatever the first already
    consumed — so a 600s setting really allowed up to 1200s, and the
    "timed out after 600s" log line misreported the budget it had enforced.

    Attributes:
        _deadline (float): Monotonic timestamp after which nothing may wait.
    """

    def __init__(self, timeout_seconds: float) -> None:
        self._deadline = time.monotonic() + timeout_seconds

    def remaining(self) -> float:
        """Return seconds left in the shared budget.

        Returns:
            float: Time until the deadline, clamped at ``0.0`` once spent so an
                exhausted budget fails the wait immediately instead of wrapping
                into a negative (which ``Future.result`` treats as "no wait" but
                which reads as a bug at the call site).
        """
        return max(0.0, self._deadline - time.monotonic())


def _completed_within_shared_budget(
    awaited: Sequence[tuple[Future[Any], str]],
    *,
    request_id: str,
    warnings: list[str],
    timeout_seconds: float = GENERATION_SERVICE_TIMEOUT_SECONDS,
) -> Iterator[tuple[str, Any]]:
    """Await futures in order under ONE shared budget, yielding what completed.

    Both generation call sites awaited their futures with a copy of this loop.
    Sharing it keeps the budget arithmetic and the timeout/failure reporting in
    a single place, so neither site can silently drift back to handing every
    future a fresh full timeout.

    Args:
        awaited (Sequence[tuple[Future[Any], str]]): Futures paired with the
            service name used in logs and warnings, awaited in this order.
        request_id (str): Correlation id for the log line.
        warnings (list[str]): Mutated in place with one entry per failed or
            timed-out service, mirroring the previous per-site behaviour.
        timeout_seconds (float): Total wall-clock budget for ALL of *awaited*.

    Yields:
        tuple[str, Any]: ``(service_name, value)`` for each future that
            completed inside the shared budget. Timed-out and failed services
            are reported and skipped, never yielded.
    """
    budget = _SharedDeadline(timeout_seconds)
    for future, service_name in awaited:
        remaining = budget.remaining()
        try:
            value = future.result(timeout=remaining)
        except FuturesTimeoutError:  # noqa: PERF203
            # Report the budget actually enforced. The second service inherits
            # only the leftover, so naming the full total here would misreport
            # the wait that just expired.
            msg = (
                f"{service_name} timed out after {remaining:.1f}s "
                f"of the shared {timeout_seconds:.0f}s budget"
            )
            with error_tags(
                subsystem="generation",
                service=service_name,
                request_id=request_id,
                error_type="timeout",
            ):
                logger.error("%s for request %s", msg, sanitise_for_log(request_id))
            warnings.append(msg)
            continue
        except Exception as e:
            msg = f"{service_name} failed: {e}"
            with error_tags(
                subsystem="generation",
                service=service_name,
                request_id=request_id,
                error_type=type(e).__name__,
            ):
                logger.exception(
                    "Generation service failed for request %s",
                    sanitise_for_log(request_id),
                )
            warnings.append(msg)
            continue
        yield service_name, value


# Retrieved-learning statuses that committed NOTHING and have no other retry
# trigger. These are the only ones worth re-running.
#
# "degraded" is deliberately EXCLUDED. It is an APPLIED, fingerprint-fenced
# commit: the rows are persisted and only the chunks whose judge failed carry
# NULL impact. Retrying it re-executes EVERY relevance + impact chunk for the
# session and delete/re-inserts rows that are already committed — and a
# deterministically degrading chunk (an over-length learning, a content-filter
# refusal) degrades again on every attempt, so the whole bill buys nothing.
# It is re-judged fresh (new generation) on the next scheduled run and
# self-heals to "complete" once the transient failure clears.
_RETRIABLE_RETRIEVED_STATUSES = frozenset({"failed", "pending"})
_MAX_RETRIEVED_LEARNING_RETRIES = 3

# ── Durable-learning same-user guard (F4) ──
# Service prefix for the per-user in-progress lock that serializes concurrent
# durable-learning jobs for one user (compute → persist across the fence).
_DURABLE_LEARNING_LOCK_SERVICE = "durable_learning"
# Stale timeout for the per-user durable lock. Deliberately set ABOVE the 300s
# durable-job claim lease (V4): a job's claim can be re-leased after 300s, so the
# lock must outlive one claim window and NOT expire in lockstep with the lease
# under compute-time throttling — 900s = 3x the lease (>= the required 2x).
_DURABLE_LOCK_STALE_SECONDS = 900
# Operation-state payload written when releasing the per-user lock (mirrors the
# clean-slate state OperationStateManager.clear_lock_if_owner writes).
_DURABLE_LOCK_CLEARED_STATE = {
    "in_progress": False,
    "current_request_id": None,
    "pending_request_id": None,
    "pending_request_queue": [],
}


def _retention_cleanup_interval_seconds() -> float:
    raw = os.getenv("REFLEXIO_RETENTION_CLEANUP_INTERVAL_SECONDS", "300") or "300"
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "Invalid REFLEXIO_RETENTION_CLEANUP_INTERVAL_SECONDS=%r; using 300",
            raw,
        )
        return 300.0


_RETENTION_CLEANUP_INTERVAL_SECONDS = _retention_cleanup_interval_seconds()
# Keyed ``(org_id, project_id, target_name)``. The project component is what
# keeps per-project retention alive: the throttle is consulted BEFORE the
# ``storage_table_cleanup`` lease is taken, so an org-wide key would let the
# first project to sweep silence every sibling project for a full interval --
# no error, no log, just a retention pass that never happens. ``project_id`` is
# ``None`` wherever projects do not exist (OSS) or the caller is unbound, which
# makes the key a 1:1 relabel of the old org-wide one for those installs.
_retention_cleanup_last_run: dict[tuple[str, str | None, str], float] = {}
_retention_cleanup_lock = threading.Lock()
# Soft cap on tracked keys. The key space now grows with project count as well
# as org count, so bound it -- but only by evicting entries whose interval has
# already elapsed. Such an entry would admit its next check anyway, so dropping
# it is semantics-preserving; the surviving working set is whatever actually
# published inside one interval, which real traffic already bounds.
_RETENTION_CLEANUP_TRACKED_KEYS_SOFT_CAP = 4096


def _prune_expired_retention_keys(now: float) -> None:
    """Drop throttle entries whose interval has elapsed.

    Callers must hold ``_retention_cleanup_lock``.

    Args:
        now (float): ``time.monotonic()`` reading of the current check.
    """
    expired = [
        key
        for key, last_run in _retention_cleanup_last_run.items()
        if now - last_run >= _RETENTION_CLEANUP_INTERVAL_SECONDS
    ]
    for key in expired:
        del _retention_cleanup_last_run[key]


def _org_in_durable_allowlist(org_id: str | None) -> bool:
    """Whether ``org_id`` may use the durable learning queue.

    Reads ``REFLEXIO_DURABLE_LEARNING_QUEUE_ORG_ALLOWLIST`` (comma-separated org
    IDs). An empty/whitespace-only value means the allowlist is unset — the
    default global behavior, so every org is eligible. The allowlist can only
    *narrow* the durable path; it never enables it on its own (the
    ``REFLEXIO_DURABLE_LEARNING_QUEUE`` flag still gates activation).
    """
    raw = env_str("REFLEXIO_DURABLE_LEARNING_QUEUE_ORG_ALLOWLIST", "")
    allowed = {s.strip() for s in raw.split(",") if s.strip()}
    if not allowed:
        return True
    return str(org_id) in allowed


@dataclass
class GenerationServiceResult:
    """Result of a GenerationService.run call.

    Exposes the internally generated request_id plus any warnings so callers
    (CLI, API) can report back to users where their publish landed.

    Attributes:
        request_id (str | None): The UUID assigned to this publish call, or
            None when ``run()`` returned early before generating one (e.g.
            empty request, missing ``user_id``, or no interactions).
        warnings (list[str]): Non-fatal warnings raised by individual
            generation services during the run.
    """

    request_id: str | None = None
    warnings: list[str] = field(default_factory=list)


class GenerationService:
    """
    Main service for orchestrating profile, playbook, and agent success evaluation generation.

    This service coordinates multiple generation services (profile, playbook, agent success)
    and manages the overall interaction processing workflow.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
    ) -> None:
        """
        Initialize the generation service.

        Args:
            llm_client: Pre-configured LLM client for making API calls.
            request_context: Request context with storage and configurator.
        """
        self.client = llm_client
        self.storage = request_context.storage
        self.org_id = request_context.org_id
        self.configurator = request_context.configurator
        self.request_context = request_context

    # ===============================
    # public methods
    # ===============================

    def run(
        self,
        publish_user_interaction_request: PublishUserInteractionRequest,
        *,
        use_publish_limiter: bool = True,
        publish_limiter_wait_forever: bool = True,
        defer_learning: bool = False,
    ) -> GenerationServiceResult:
        """
        Process a user interaction request by storing interactions and triggering generation services.

        Profile and playbook generation services run inline in parallel. Agent success
        evaluation is deferred via GroupEvaluationScheduler when a session_id is present,
        so the full session can be evaluated after a period of inactivity.

        Each generation service (profile, playbook) handles its own:
        - Data collection based on extractor-specific configs
        - Stride checking based on extractor-specific settings
        - Operation state tracking per extractor

        Args:
            publish_user_interaction_request: The incoming user interaction request
            use_publish_limiter: Whether this service should throttle the post-write
                learning pipeline. Server sync calls acquire the bounded API
                limiter before invoking this service, so they disable the inner
                limiter to avoid waiting after storage side effects.
            publish_limiter_wait_forever: Whether to queue indefinitely for the
                post-write learning limiter when ``use_publish_limiter`` is true.
            defer_learning: Whether to persist the publish and enqueue learning
                for a process-local worker instead of running it inline.

        Returns:
            GenerationServiceResult: The request_id assigned to this publish call
                and any non-fatal warnings raised by individual generation services.
        """
        result = GenerationServiceResult()

        if not publish_user_interaction_request:
            logger.error("Received None publish_user_interaction_request")
            return result

        user_id = publish_user_interaction_request.user_id
        if not user_id:
            logger.error("Received None user_id in publish_user_interaction_request")
            return result

        # Check if cleanup is needed before adding new interactions.
        self._cleanup_storage_tables_if_needed()

        publish_start = time.perf_counter()
        # Resolve agent_version: explicit > env var > default. Resolved here
        # (before the try) so success and failure telemetry share the same value.
        agent_version = resolve_agent_version(
            publish_user_interaction_request.agent_version
        )

        try:
            retrieval_experiment_id = getattr(
                publish_user_interaction_request,
                "retrieval_experiment_id",
                None,
            )
            retrieval_experiment_arm = getattr(
                publish_user_interaction_request,
                "retrieval_experiment_arm",
                None,
            )
            validate_retrieval_experiment_attribution(
                config=self.configurator.get_config(),
                org_id=self.org_id,
                user_id=user_id,
                experiment_id=retrieval_experiment_id,
                arm=retrieval_experiment_arm,
            )
            caller_request_id = publish_user_interaction_request.request_id
            request_id = (
                caller_request_id
                if caller_request_id is not None
                else str(uuid.uuid4())
            )
            result.request_id = request_id

            if (
                caller_request_id is not None
                and self.storage.get_request(request_id) is not None  # type: ignore[reportOptionalMemberAccess]
            ):
                raise ValueError(f"request_id {request_id!r} already exists")

            new_interactions: list[Interaction] = (
                GenerationService.get_interaction_from_publish_user_interaction_request(
                    publish_user_interaction_request, request_id
                )
            )

            if not new_interactions:
                logger.info(
                    "No interactions from the publish user interaction request: %s, get all interactions for the user: %s",
                    sanitise_for_log(request_id),
                    user_id,
                )
                return result

            record_usage_event(
                org_id=self.org_id,
                user_id=user_id,
                request_id=request_id,
                session_id=publish_user_interaction_request.session_id,
                source=publish_user_interaction_request.source,
                agent_version=agent_version,
                event_name="publish_request_received",
                event_category="publish",
                outcome="received",
                count_value=len(new_interactions),
            )

            # Store Request before adding interactions so downstream workers can
            # resolve the session's source and evaluation-only status.
            new_request = Request(
                request_id=request_id,
                user_id=user_id,
                source=publish_user_interaction_request.source,
                agent_version=agent_version,
                session_id=publish_user_interaction_request.session_id,
                evaluation_only=publish_user_interaction_request.evaluation_only,
                retrieval_experiment_id=retrieval_experiment_id,
                retrieval_experiment_arm=retrieval_experiment_arm,
            )

            # When the durable queue is enabled and this is a deferred publish, the
            # request + interactions + job row are written together inside a single
            # commit_scope (below).  Every other path persists here unconditionally.
            use_durable_queue = env_truthy(
                env_str("REFLEXIO_DURABLE_LEARNING_QUEUE", "false")
            )
            _durable_defer = (
                defer_learning
                and use_durable_queue
                and _org_in_durable_allowlist(self.org_id)
            )

            if not _durable_defer:
                self.storage.add_request(new_request)  # type: ignore[reportOptionalMemberAccess]
                # Add interactions to storage (bulk insert with batched embedding generation)
                self.storage.add_user_interactions_bulk(  # type: ignore[reportOptionalMemberAccess]
                    user_id=user_id, interactions=new_interactions
                )

            # Extract source (empty string treated as None)
            source = publish_user_interaction_request.source or None

            if publish_user_interaction_request.evaluation_only:
                if _durable_defer:
                    # evaluation_only returns early; ensure data lands even on durable path.
                    self.storage.add_request(new_request)  # type: ignore[reportOptionalMemberAccess]
                    self.storage.add_user_interactions_bulk(  # type: ignore[reportOptionalMemberAccess]
                        user_id=user_id, interactions=new_interactions
                    )
                self._schedule_post_publish_evaluations(
                    new_request=new_request,
                    interactions=new_interactions,
                    user_id=user_id,
                    agent_version=agent_version,
                    source=source,
                )
                self._emit_publish_success_events(
                    interactions=new_interactions,
                    user_id=user_id,
                    request_id=request_id,
                    session_id=new_request.session_id,
                    source=source,
                    agent_version=agent_version,
                    backend="evaluation_only",
                    duration_ms=int((time.perf_counter() - publish_start) * 1000),
                    metadata={
                        "evaluation_only": True,
                        "warning_count": len(result.warnings),
                    },
                )
                return result

            if (
                not publish_user_interaction_request.override_learning_stall
                and (stall_warning := self._active_learning_stall_warning()) is not None
            ):
                if _durable_defer:
                    # Stall skips learning but data must still land.
                    self.storage.add_request(new_request)  # type: ignore[reportOptionalMemberAccess]
                    self.storage.add_user_interactions_bulk(  # type: ignore[reportOptionalMemberAccess]
                        user_id=user_id, interactions=new_interactions
                    )
                result.warnings.append(stall_warning)
                logger.warning("%s; skipping automatic extraction", stall_warning)
                self._schedule_post_publish_evaluations(
                    new_request=new_request,
                    interactions=new_interactions,
                    user_id=user_id,
                    agent_version=agent_version,
                    source=source,
                )
                self._emit_publish_success_events(
                    interactions=new_interactions,
                    user_id=user_id,
                    request_id=request_id,
                    session_id=new_request.session_id,
                    source=source,
                    agent_version=agent_version,
                    duration_ms=int((time.perf_counter() - publish_start) * 1000),
                    metadata={"warning_count": len(result.warnings)},
                )
                return result

            if defer_learning:
                if _durable_defer:
                    # Durable atomic path: pre-generate embeddings OUTSIDE the
                    # transaction (network call), then persist request + interactions
                    # + job in one commit_scope.
                    self.storage.prepare_interaction_embeddings(new_interactions)  # type: ignore[reportOptionalMemberAccess]
                    covers_through = max(i.created_at for i in new_interactions)
                    with self.storage.commit_scope():  # type: ignore[reportOptionalMemberAccess]
                        self.storage.add_request(new_request)  # type: ignore[reportOptionalMemberAccess]
                        self.storage.add_user_interactions_bulk(  # type: ignore[reportOptionalMemberAccess]
                            user_id=user_id,
                            interactions=new_interactions,
                            embeddings_prepared=True,
                        )
                        self.storage.enqueue_learning_job(  # type: ignore[reportOptionalMemberAccess]
                            org_id=self.org_id,
                            user_id=user_id,
                            request_id=request_id,
                            covers_through=covers_through,
                            force_extraction=publish_user_interaction_request.force_extraction,
                            skip_aggregation=publish_user_interaction_request.skip_aggregation,
                            # Resolved HERE, on the request thread, not in the
                            # worker: the job runs long after this returns.
                            project_id=current_project_id(),
                        )
                    self._schedule_post_publish_evaluations(
                        new_request=new_request,
                        interactions=new_interactions,
                        user_id=user_id,
                        agent_version=agent_version,
                        source=source,
                    )
                    self._emit_publish_success_events(
                        interactions=new_interactions,
                        user_id=user_id,
                        request_id=request_id,
                        session_id=new_request.session_id,
                        source=source,
                        agent_version=agent_version,
                        backend="durable",
                        duration_ms=int((time.perf_counter() - publish_start) * 1000),
                        metadata={
                            "defer_learning": True,
                            "durable_queue": True,
                            "warning_count": len(result.warnings),
                        },
                    )
                    return result

                from reflexio.server.services.publish_learning_worker import (
                    PublishLearningJob,
                    enqueue_publish_learning,
                )

                self._schedule_post_publish_evaluations(
                    new_request=new_request,
                    interactions=new_interactions,
                    user_id=user_id,
                    agent_version=agent_version,
                    source=source,
                )
                learning_queued = enqueue_publish_learning(
                    PublishLearningJob(
                        org_id=self.org_id,
                        user_id=user_id,
                        request_id=request_id,
                        session_id=new_request.session_id,
                        source=source,
                        agent_version=agent_version,
                        force_extraction=publish_user_interaction_request.force_extraction,
                        skip_aggregation=publish_user_interaction_request.skip_aggregation,
                        project_id=current_project_id(),
                    )
                )
                self._emit_publish_success_events(
                    interactions=new_interactions,
                    user_id=user_id,
                    request_id=request_id,
                    session_id=new_request.session_id,
                    source=source,
                    agent_version=agent_version,
                    backend="classic",
                    duration_ms=int((time.perf_counter() - publish_start) * 1000),
                    metadata={
                        "defer_learning": True,
                        "learning_queued": learning_queued,
                        "warning_count": len(result.warnings),
                    },
                )
                return result

            # The durable write above must not sit behind the publish limiter:
            # async API callers have already received success=True. Throttle only
            # the post-write learning pipeline so a hung LLM cannot silently drop
            # later acknowledged publishes before they reach storage.
            learning_limit = (
                operation_limit(
                    self.org_id,
                    "publish",
                    wait_forever=publish_limiter_wait_forever,
                )
                if use_publish_limiter
                else nullcontext()
            )
            with learning_limit:
                self._run_learning_steps(
                    user_id=user_id,
                    request_id=request_id,
                    agent_version=agent_version,
                    force_extraction=publish_user_interaction_request.force_extraction,
                    skip_aggregation=publish_user_interaction_request.skip_aggregation,
                    source=source,
                    result=result,
                )

            self._schedule_post_publish_evaluations(
                new_request=new_request,
                interactions=new_interactions,
                user_id=user_id,
                agent_version=agent_version,
                source=source,
            )

            self._emit_publish_success_events(
                interactions=new_interactions,
                user_id=user_id,
                request_id=request_id,
                session_id=new_request.session_id,
                source=source,
                agent_version=agent_version,
                backend="classic",
                duration_ms=int((time.perf_counter() - publish_start) * 1000),
                metadata={"warning_count": len(result.warnings)},
            )
            return result

        except Exception as e:
            record_usage_event(
                org_id=self.org_id,
                user_id=user_id,
                request_id=result.request_id,
                session_id=publish_user_interaction_request.session_id,
                source=publish_user_interaction_request.source,
                agent_version=agent_version,
                event_name="publish_request_failed",
                event_category="publish",
                outcome="failed",
                duration_ms=int((time.perf_counter() - publish_start) * 1000),
                error_kind=type(e).__name__,
            )
            with error_tags(
                subsystem="generation",
                op="refresh_profile",
                org_id=self.org_id,
                user_id=user_id,
                request_id=result.request_id,
                error_type=type(e).__name__,
            ):
                logger.exception(
                    "Failed to refresh user profile for user id: %s",
                    user_id,
                )
            raise

    def _emit_publish_success_events(
        self,
        *,
        interactions: list[Interaction],
        user_id: str,
        request_id: str,
        session_id: str,
        source: str | None,
        agent_version: str,
        duration_ms: int,
        metadata: Mapping[str, Any],
        backend: str | None = None,
    ) -> None:
        """Emit one ``publish_request_succeeded`` event per interaction.

        Replaces the old single aggregate event (``count_value=len(interactions)``)
        with one entity-backed event per interaction (``count_value=1``,
        ``event_key=f"pub:{interaction_id}"``, ``entity_id=interaction_id``) so
        downstream dedup can key on the interaction id. The summed
        ``count_value`` across the emitted events equals the old aggregate
        count, so totals are unchanged.
        """
        for interaction in interactions:
            record_usage_event(
                org_id=self.org_id,
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                source=source,
                agent_version=agent_version,
                backend=backend,
                event_name="publish_request_succeeded",
                event_category="publish",
                outcome="success",
                count_value=1,
                event_key=f"pub:{interaction.interaction_id}",
                entity_id=str(interaction.interaction_id),
                duration_ms=duration_ms,
                metadata=metadata,
            )

    # ===============================
    # deferred learning
    # ===============================

    def run_deferred_learning(
        self,
        *,
        user_id: str,
        request_id: str,
        session_id: str | None,  # noqa: ARG002 - kept for worker telemetry symmetry
        source: str | None,
        agent_version: str,
        force_extraction: bool,
        skip_aggregation: bool,
    ) -> GenerationServiceResult:
        """Run the learning steps for a deferred job (synchronous / non-durable
        callers: ``PublishLearningWorker`` and manual reruns).

        Runs compute + persist + side-effects together, with profile and playbook
        generation in parallel — no external ``commit_scope`` is held on this
        path (the durable worker uses the compute/persist/emit split instead).
        """
        result = GenerationServiceResult(request_id=request_id)
        self._run_learning_steps(
            user_id=user_id,
            request_id=request_id,
            agent_version=agent_version,
            force_extraction=force_extraction,
            skip_aggregation=skip_aggregation,
            source=source,
            result=result,
            parallel=True,
        )
        return result

    # ===============================
    # deferred learning — compute / persist / emit split (gate b)
    # ===============================

    def compute_deferred_learning(
        self,
        *,
        user_id: str,
        request_id: str,
        session_id: str | None,  # noqa: ARG002 - kept for worker call symmetry
        source: str | None,
        agent_version: str,
        force_extraction: bool,
        skip_aggregation: bool,
    ) -> DeferredLearningPlan:
        """Compute half of one durable-learning job — NO ``commit_scope`` held.

        Runs the same-user guard (F4) first, then profile + playbook
        compute (LLM extraction, dedup, embeddings) entirely OUTSIDE any writer
        transaction, and assembles a :class:`DeferredLearningPlan` of the held
        ``(service, plan)`` pairs. Issues NO learning DB write (F3): the only
        writes are the extractor's own ``agent_run`` rows + the per-user
        coordination lock (``try_acquire_in_progress_lock``, not a learning
        terminal).

        F4 GUARD FIRST: acquires a DB-backed per-user in-progress lock BEFORE any
        LLM work. On contention returns ``DeferredLearningPlan(lock_acquired=
        False, profile/playbook=None)`` immediately (no compute) — the
        worker must then leave the job reclaimable (Task 9), NOT complete it.

        Profile + playbook ``compute_generation`` run in parallel (no
        ``commit_scope`` is held, so the old ``sequential=True`` RLock avoidance
        is unnecessary). Per-half failures are best-effort — captured into
        ``warnings`` and the half dropped — mirroring ``_run_learning_steps``.
        """
        warnings: list[str] = []

        # F4 GUARD — before any LLM. On contention, no compute runs.
        if not self._acquire_durable_learning_lock(
            user_id=user_id, request_id=request_id
        ):
            logger.info(
                "durable-learning same-user lock held; leaving job reclaimable "
                "(user_id=%s request_id=%s)",
                user_id,
                sanitise_for_log(request_id),
            )
            return DeferredLearningPlan(
                request_id=request_id,
                user_id=user_id,
                agent_version=agent_version,
                lock_acquired=False,
                profile=None,
                playbook=None,
                warnings=warnings,
            )

        # Profile + playbook compute in parallel — no scope held, so no RLock
        # contention (the pre-split sequential mode only existed to avoid the
        # SQLite commit_scope RLock, which compute never takes).
        profile_service = ProfileGenerationService(
            llm_client=self.client, request_context=self.request_context
        )
        profile_request = ProfileGenerationRequest(
            user_id=user_id,
            request_id=request_id,
            source=source,
            force_extraction=force_extraction,
        )
        playbook_service = PlaybookGenerationService(
            llm_client=self.client,
            request_context=self.request_context,
            skip_aggregation=skip_aggregation,
        )
        playbook_request = PlaybookGenerationRequest(
            request_id=request_id,
            agent_version=agent_version,
            user_id=user_id,
            source=source,
            force_extraction=force_extraction,
        )

        profile_pair: tuple = None  # type: ignore[assignment]
        playbook_pair: tuple = None  # type: ignore[assignment]
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            profile_future = executor.submit(
                contextvars.copy_context().run,
                profile_service.compute_generation,
                profile_request,
            )
            playbook_future = executor.submit(
                contextvars.copy_context().run,
                playbook_service.compute_generation,
                playbook_request,
            )
            for service_name, plan in _completed_within_shared_budget(
                (
                    (profile_future, "profile_generation"),
                    (playbook_future, "playbook_generation"),
                ),
                request_id=request_id,
                warnings=warnings,
            ):
                if plan is None:
                    continue
                if service_name == "profile_generation":
                    profile_pair = (profile_service, plan)
                else:
                    playbook_pair = (playbook_service, plan)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return DeferredLearningPlan(
            request_id=request_id,
            user_id=user_id,
            agent_version=agent_version,
            lock_acquired=True,
            profile=profile_pair,
            playbook=playbook_pair,
            warnings=warnings,
        )

    def persist_deferred_learning(self, plan: DeferredLearningPlan) -> None:
        """Persist half — apply the fence-critical writes for a computed job.

        This is the ONLY part the durable worker runs inside its fenced
        ``commit_scope``. For each present half it calls the held instance's
        persist (profile/playbook row writes + extractor bookmark advance).
        Issues NO side-effects (telemetry, billing, tagging, off-thread
        schedulers, lock release) — those are post-commit
        (``emit_deferred_learning_side_effects``) so a fence-lost job never
        fires them.
        """
        if plan.profile is not None:
            profile_service, profile_plan = plan.profile
            profile_service.persist_generation(profile_plan)
        if plan.playbook is not None:
            playbook_service, playbook_plan = plan.playbook
            playbook_service.persist_generation(playbook_plan)

    def emit_deferred_learning_side_effects(self, plan: DeferredLearningPlan) -> None:
        """Post-commit side-effects — telemetry, billing, tagging, lock release.

        Runs only for a fence-winning job (the durable worker calls it after the
        scope commits). Fires each held half's post-commit emit + schedules the
        deferred tagging pass, then ALWAYS releases the per-user F4 lock (even if
        an emit raised) so a completed job never strands the lock.

        A ``lock_acquired=False`` plan never reaches here (the worker skips emit
        on contention), so releasing here is safe — this instance owns the lock.

        Each emit half is isolated in its own try/except (mirroring how
        ``schedule_tagging`` already self-guards): the halves are independent
        post-commit side effects (billing / telemetry / off-thread schedulers),
        so a failure in an early half must NOT starve the later halves. The
        per-user lock release always runs in ``finally`` so a completed job never
        strands the lock.
        """
        try:
            if plan.profile is not None:
                profile_service, profile_plan = plan.profile
                try:
                    profile_service.emit_generation_side_effects(profile_plan)
                except Exception:
                    logger.exception(
                        "Failed to emit profile side effects for deferred "
                        "learning request %s",
                        sanitise_for_log(plan.request_id),
                    )
            if plan.playbook is not None:
                playbook_service, playbook_plan = plan.playbook
                try:
                    playbook_service.emit_generation_side_effects(playbook_plan)
                except Exception:
                    logger.exception(
                        "Failed to emit playbook side effects for deferred "
                        "learning request %s",
                        sanitise_for_log(plan.request_id),
                    )
            try:
                schedule_tagging(
                    org_id=self.org_id,
                    user_id=plan.user_id,
                    agent_version=plan.agent_version,
                    request_context=self.request_context,
                    llm_client=self.client,
                )
            except Exception:
                logger.exception(
                    "Failed to schedule tagging for deferred learning request %s",
                    sanitise_for_log(plan.request_id),
                )
        finally:
            self._release_durable_learning_lock(
                user_id=plan.user_id, request_id=plan.request_id
            )

    # ===============================
    # private methods
    # ===============================

    def _durable_learning_lock_key(self, user_id: str) -> str:
        """Per-user durable-learning in-progress lock key.

        Mirrors ``OperationStateManager._lock_key`` shape
        (``{service}::{org}::{scope}::lock``) but on a DEDICATED
        ``durable_learning`` service prefix, distinct from the
        profile/playbook generation locks and from any extractor bookmark row.
        """
        return f"{_DURABLE_LEARNING_LOCK_SERVICE}::{self.org_id}::{user_id}::lock"

    def _acquire_durable_learning_lock(self, *, user_id: str, request_id: str) -> bool:
        """Try to acquire the per-user durable-learning lock (F4).

        Uses the atomic DB-backed ``try_acquire_in_progress_lock`` (a distinct
        storage method, NOT one of the learning-write terminals the compute
        purity contract forbids), so the guard does not trip the compute
        write-tripwire. Returns ``True`` when acquired (proceed with compute),
        ``False`` on contention (another same-user durable job holds it).
        """
        if self.storage is None:
            return True
        result = self.storage.try_acquire_in_progress_lock(
            self._durable_learning_lock_key(user_id),
            request_id,
            stale_lock_seconds=_DURABLE_LOCK_STALE_SECONDS,
        )
        return bool(result.get("acquired", False))

    def _release_durable_learning_lock(self, *, user_id: str, request_id: str) -> None:
        """Release the per-user durable-learning lock iff we still own it.

        Uses ``clear_in_progress_lock_if_owner`` (CAS on the holder) so a job
        whose lock was already stolen after a stale-timeout does not clobber the
        new holder's lock.
        """
        if self.storage is None:
            return
        self.storage.clear_in_progress_lock_if_owner(
            self._durable_learning_lock_key(user_id),
            request_id,
            dict(_DURABLE_LOCK_CLEARED_STATE),
        )

    def _run_learning_steps(
        self,
        *,
        user_id: str,
        request_id: str,
        agent_version: str,
        force_extraction: bool,
        skip_aggregation: bool,
        source: str | None,
        result: GenerationServiceResult,
        parallel: bool = True,
    ) -> None:
        profile_generation_service = ProfileGenerationService(
            llm_client=self.client, request_context=self.request_context
        )
        source_request_id = request_id
        profile_generation_request = ProfileGenerationRequest(
            user_id=user_id,
            request_id=source_request_id,
            source=source,
            force_extraction=force_extraction,
        )

        playbook_generation_service = PlaybookGenerationService(
            llm_client=self.client,
            request_context=self.request_context,
            skip_aggregation=skip_aggregation,
        )
        playbook_generation_request = PlaybookGenerationRequest(
            request_id=source_request_id,
            agent_version=agent_version,
            user_id=user_id,
            source=source,
            force_extraction=force_extraction,
        )

        if not parallel:
            # Sequential mode: run both services on the calling thread.
            # Required when the caller holds a commit_scope lock — spawning
            # child threads inside that scope would deadlock the SQLite RLock.
            for svc_name, _run in (
                (
                    "profile_generation",
                    lambda: profile_generation_service.run(profile_generation_request),
                ),
                (
                    "playbook_generation",
                    lambda: playbook_generation_service.run(
                        playbook_generation_request
                    ),
                ),
            ):
                try:
                    _run()
                except Exception as e:
                    msg = f"{svc_name} failed: {e}"
                    with error_tags(
                        subsystem="generation",
                        service=svc_name,
                        request_id=request_id,
                        error_type=type(e).__name__,
                    ):
                        logger.exception(
                            "Generation service failed for request %s",
                            sanitise_for_log(request_id),
                        )
                    result.warnings.append(msg)
        else:
            executor = ThreadPoolExecutor(max_workers=2)
            try:
                futures = [
                    executor.submit(
                        contextvars.copy_context().run,
                        profile_generation_service.run,
                        profile_generation_request,
                    ),
                    executor.submit(
                        contextvars.copy_context().run,
                        playbook_generation_service.run,
                        playbook_generation_request,
                    ),
                ]

                service_names = ["profile_generation", "playbook_generation"]
                # Drained for its warnings side-effect; this path keeps no
                # per-service value (each service already persisted its own).
                for _ in _completed_within_shared_budget(
                    list(zip(futures, service_names, strict=True)),
                    request_id=request_id,
                    warnings=result.warnings,
                ):
                    pass
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        try:
            schedule_tagging(
                org_id=self.org_id,
                user_id=user_id,
                agent_version=agent_version,
                request_context=self.request_context,
                llm_client=self.client,
            )
        except Exception:
            logger.exception(
                "Failed to schedule tagging for publish request %s",
                sanitise_for_log(request_id),
            )

    def _schedule_post_publish_evaluations(
        self,
        *,
        new_request: Request,
        interactions: list[Interaction],
        user_id: str,
        agent_version: str,
        source: str | None,
    ) -> None:
        """Schedule all best-effort evaluation work after publish persistence."""
        if any(interaction.shadow_content for interaction in interactions):
            try:
                enqueue_shadow_comparison(
                    ShadowComparisonJob(
                        org_id=self.org_id,
                        interactions=interactions,
                        session_id=new_request.session_id,
                        agent_version=agent_version,
                        project_id=current_project_id(),
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to enqueue shadow comparison for session %s",
                    new_request.session_id,
                )

        self._schedule_group_evaluation_if_needed(
            new_request=new_request,
            user_id=user_id,
            agent_version=agent_version,
            source=source,
        )

    def _schedule_group_evaluation_if_needed(
        self,
        *,
        new_request: Request,
        user_id: str,
        agent_version: str,
        source: str | None,
    ) -> None:
        """Enqueue agent-success evaluation for this session.

        Must be called once per publish — from BOTH the classic and the agentic
        extraction code paths — so that ``AgentSuccessEvaluationResult`` records
        get produced regardless of which backend is in use. Skipping this for
        the agentic path was the silent root cause of empty /evaluations tiles.

        Args:
            new_request (Request): The just-stored request whose session is being
                published into.
            user_id (str): The user owning the session.
            agent_version (str): Agent version string carried into the evaluator.
            source (str | None): Optional source label.
        """
        session_id = new_request.session_id
        run_agent_success, run_retrieved_learning = self._sampled_evaluation_families(
            user_id=user_id,
            session_id=session_id,
            evaluation_only=new_request.evaluation_only,
        )
        if self.storage is not None:
            try:
                self.storage.record_retrieved_learning_sampling_decision(
                    user_id=user_id,
                    session_id=session_id,
                    request_id=new_request.request_id,
                    sampled=run_retrieved_learning,
                )
            except Exception:
                logger.exception(
                    "Failed to persist retrieved-learning sampling decision for "
                    "request=%s session=%s user=%s",
                    sanitise_for_log(new_request.request_id),
                    session_id,
                    user_id,
                )
        if not (run_agent_success or run_retrieved_learning):
            logger.info(
                "Skipping group evaluation scheduling for unsampled session=%s user=%s",
                session_id,
                user_id,
            )
            return

        scheduler = GroupEvaluationScheduler.get_instance()
        # Project resolved HERE, on the request thread — the callback fires
        # after an inactivity window that may span several requests.
        key = (self.org_id, current_project_id(), user_id, session_id)

        def make_callback(
            _org_id: str,
            _user_id: str,
            _sid: str,
            _av: str,
            _src: str | None,
            _rc: RequestContext,
            _llm: LiteLLMClient,
            _run_agent_success: bool,
            _run_retrieved_learning: bool,
            _attempt: int,
        ) -> Callable[[], None]:
            def callback() -> None:
                outcome = run_group_evaluation(
                    org_id=_org_id,
                    user_id=_user_id,
                    session_id=_sid,
                    agent_version=_av,
                    source=_src,
                    request_context=_rc,
                    llm_client=_llm,
                    run_agent_success=_run_agent_success,
                    run_retrieved_learning=_run_retrieved_learning,
                )
                # Retry sweep: a retrieved-learning run that ends non-terminal
                # (degraded / failed / pending) is NEVER retried on its own —
                # nothing re-triggers it unless the session receives more
                # traffic, so a session that degrades once would stay invisible
                # to downstream consumers forever. Re-arm the session on the
                # scheduler, bounded, and only for the agent-success family's
                # sibling: regen jobs and the on-demand grade route drive their
                # own retries.
                if (
                    _run_retrieved_learning
                    and outcome.retrieved_learning_status
                    in _RETRIABLE_RETRIEVED_STATUSES
                    and _attempt < _MAX_RETRIEVED_LEARNING_RETRIES
                ):
                    logger.info(
                        "event=retrieved_learning_retry_scheduled session=%s"
                        " status=%s attempt=%d",
                        _sid,
                        outcome.retrieved_learning_status,
                        _attempt + 1,
                    )
                    scheduler.schedule(
                        key,
                        make_callback(
                            _org_id,
                            _user_id,
                            _sid,
                            _av,
                            _src,
                            _rc,
                            _llm,
                            # The success judge already ran (or was skipped); do
                            # not pay for it again on a retrieved-only retry.
                            False,
                            True,
                            _attempt + 1,
                        ),
                    )

            return callback

        scheduler.schedule(
            key,
            make_callback(
                self.org_id,
                user_id,
                session_id,
                agent_version,
                source,
                self.request_context,
                self.client,
                run_agent_success,
                run_retrieved_learning,
                0,
            ),
        )

    def _sampled_evaluation_families(
        self, *, user_id: str, session_id: str, evaluation_only: bool
    ) -> tuple[bool, bool]:
        """Which judge families this session is sampled for.

        The two families sample independently (see
        ``agent_success_evaluation.sampling``). We schedule when EITHER admits
        the session and pass both flags to the runner, so a session sampled only
        for retrieved-learning never pays the session-success judge.

        Returns:
            tuple[bool, bool]: ``(run_agent_success, run_retrieved_learning)``.
        """
        config = self.configurator.get_config()
        agent_success_config = getattr(config, "agent_success_config", None)
        scope = {
            "org_id": self.org_id,
            "user_id": user_id,
            "session_id": session_id,
        }
        return (
            samples_agent_success(
                agent_success_config, evaluation_only=evaluation_only, **scope
            ),
            samples_retrieved_learning(agent_success_config, **scope),
        )

    def _cleanup_storage_tables_if_needed(self) -> None:
        """Best-effort publish-boundary cleanup for capped storage tables."""
        now = time.monotonic()
        # Resolved once, here on the request thread that is publishing. Under
        # per-project retention each project must be throttled independently;
        # see the note on ``_retention_cleanup_last_run``.
        project_id = current_project_id()
        limits = {
            target_name: limit
            for target_name, limit in get_row_retention_limits().items()
            if limit > 0
            and self._should_check_retention_target(target_name, project_id, now)
        }
        if not limits:
            return

        try:
            mgr = OperationStateManager(
                self.storage,  # type: ignore[reportArgumentType]
                self.org_id,
                "storage_table_cleanup",  # type: ignore[reportArgumentType]
            )
            if not mgr.acquire_simple_lock(stale_seconds=CLEANUP_STALE_LOCK_SECONDS):
                return

            try:
                for target_name, limit in limits.items():
                    # Isolate per-target failures so one bad table does not
                    # short-circuit cleanup for every subsequent target.
                    try:
                        self._cleanup_retention_target(target_name, limit)
                    except Exception as e:  # noqa: BLE001
                        with error_tags(
                            subsystem="generation",
                            op="cleanup_retention_target",
                            org_id=self.org_id,
                            target_name=target_name,
                            error_type=type(e).__name__,
                        ):
                            logger.exception(
                                "Failed to cleanup retention target %s",
                                target_name,
                            )
            finally:
                mgr.release_simple_lock()

        except Exception as e:
            with error_tags(
                subsystem="generation",
                op="cleanup_storage_tables",
                org_id=self.org_id,
                error_type=type(e).__name__,
            ):
                logger.exception("Failed to cleanup storage tables")
            # Don't raise - cleanup failure shouldn't block normal operation

    def _should_check_retention_target(
        self, target_name: str, project_id: str | None, now: float
    ) -> bool:
        """Whether this org/project/target is due for a retention sweep.

        Args:
            target_name (str): Capped storage table being considered.
            project_id (str | None): Project the publishing request is bound to,
                or ``None`` in OSS and for an unbound caller.
            now (float): ``time.monotonic()`` reading of the current check.

        Returns:
            bool: True when the target is due, stamping the throttle as a side
            effect; False while the interval is still open.
        """
        if _RETENTION_CLEANUP_INTERVAL_SECONDS <= 0:
            return True
        key = (self.org_id, project_id, target_name)
        with _retention_cleanup_lock:
            last_run = _retention_cleanup_last_run.get(key)
            if (
                last_run is not None
                and now - last_run < _RETENTION_CLEANUP_INTERVAL_SECONDS
            ):
                return False
            if (
                len(_retention_cleanup_last_run)
                >= _RETENTION_CLEANUP_TRACKED_KEYS_SOFT_CAP
            ):
                _prune_expired_retention_keys(now)
            _retention_cleanup_last_run[key] = now
            return True

    def _active_learning_stall_warning(self) -> str | None:
        """Return a warning when extraction should not auto-retry.

        Plugin publishes should still store raw interactions while the local
        LLM provider is blocked by auth or billing, but they
        must not keep invoking extraction on every publish. Only callers that
        pass ``override_learning_stall=True`` bypass this check so an explicit
        retry after reauth can clear the stall state on a successful provider
        call.
        """
        try:
            stall_state = self.storage.get_stall_state()  # type: ignore[reportOptionalMemberAccess]
        except (AttributeError, NotImplementedError):
            return None
        except Exception as exc:  # noqa: BLE001 - stall telemetry must not block publish.
            logger.debug("Failed to read stall_state before extraction: %s", exc)
            return None

        if not getattr(stall_state, "stalled", False):
            return None
        reason = getattr(stall_state, "reason", None) or "unknown"
        suffix = (
            "reauthenticate the active coding-agent provider, then run an explicit "
            "override retry to resume."
            if reason == "auth_error"
            else "wait for the limit/reset condition to clear, then run an explicit "
            "override retry to resume."
        )
        return f"{_STALL_WARNING_PREFIX} ({reason}); {suffix}"

    def _cleanup_retention_target(self, target_name: str, limit: int) -> None:
        total_count = self.storage.count_retention_target_rows(target_name)  # type: ignore[reportOptionalMemberAccess]
        if total_count < limit:
            return
        delete_count = delete_count_for_retention(total_count)
        deleted = self.storage.delete_oldest_retention_target_rows(  # type: ignore[reportOptionalMemberAccess]
            target_name,
            delete_count,
        )
        logger.info(
            "Cleaned up %d oldest %s row(s) (total was %d, limit %d)",
            deleted,
            target_name,
            total_count,
            limit,
        )

    # ===============================
    # static methods
    # ===============================

    @staticmethod
    def get_interaction_from_publish_user_interaction_request(
        publish_user_interaction_request: PublishUserInteractionRequest,
        request_id: str,
    ) -> list[Interaction]:
        """get interaction from publish user interaction request

        Args:
            publish_user_interaction_request (PublishUserInteractionRequest): The publish user interaction request
            request_id (str): The request ID generated by the service

        Returns:
            list[Interaction]: List of interactions created from the request
        """
        interaction_data_list = publish_user_interaction_request.interaction_data_list

        user_id = publish_user_interaction_request.user_id
        # Honor the client-provided ``created_at`` — InteractionData defaults
        # it to client-side ``now()`` on construction, so it's always populated.
        # Apps that publish backdated conversations (e.g., a benchmark replay
        # of 2023 chats run in 2026) need the wall-clock time preserved so the
        # extraction agent has a real temporal anchor for relative-time
        # references like "X weeks ago" / "yesterday". Stamping server-now here
        # would erase that anchor and force every event onto today's date.
        return [
            Interaction(
                # interaction_id is auto-generated by DB
                user_id=user_id,
                request_id=request_id,
                created_at=interaction_data.created_at,
                content=interaction_data.content,
                role=interaction_data.role,
                user_action=interaction_data.user_action,
                user_action_description=interaction_data.user_action_description,
                interacted_image_url=interaction_data.interacted_image_url,
                image_encoding=interaction_data.image_encoding,
                shadow_content=interaction_data.shadow_content,
                expert_content=interaction_data.expert_content,
                tools_used=interaction_data.tools_used,
                citations=interaction_data.citations,
                retrieved_learnings=interaction_data.retrieved_learnings,
            )
            for interaction_data in interaction_data_list
        ]


def build_extraction_service(
    config: Config,
    *,
    llm_client: LiteLLMClient,
    request_context: RequestContext,
) -> ProfileGenerationService:
    """Return the profile extraction service.

    Args:
        config (Config): Top-level ``Config`` (unused; kept for API consistency).
        llm_client (LiteLLMClient): Configured ``LiteLLMClient``.
        request_context (RequestContext): Current request context.

    Returns:
        ProfileGenerationService: Classic profile extraction service.
    """
    del config  # unused — agentic path bypasses this factory
    return ProfileGenerationService(
        llm_client=llm_client, request_context=request_context
    )


def build_search_service(
    config: Config,  # noqa: ARG001
    *,
    llm_client: LiteLLMClient,
    request_context: RequestContext,
) -> UnifiedSearchService:
    """Build the unified search service.

    Args:
        config (Config): Top-level ``Config`` (unused; kept for API consistency).
        llm_client (LiteLLMClient): Configured ``LiteLLMClient``.
        request_context (RequestContext): Current request context.

    Returns:
        A ``UnifiedSearchService`` holding ``llm_client`` and ``request_context``.
    """
    from reflexio.server.services.unified_search_service import UnifiedSearchService

    return UnifiedSearchService(llm_client=llm_client, request_context=request_context)
