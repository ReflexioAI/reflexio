from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
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
from reflexio.server import publish_timing
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.error_reporting import error_tags
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.tools import describe_tool_loop_error
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
from reflexio.server.services.base_generation_service import ExtractorExecutionError
from reflexio.server.services.external_admission import ExternalAdmissionParticipant
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
from reflexio.server.usage_metrics import record_usage_event
from reflexio.server.work_scope import current_project_id

if TYPE_CHECKING:
    from reflexio.server.services.storage.storage_base import BaseStorage
    from reflexio.server.services.unified_search_service import UnifiedSearchService

logger = logging.getLogger(__name__)
# Stale lock timeout - if cleanup started > 10 min ago and still "in_progress", assume it crashed
_STALL_WARNING_PREFIX = "Reflexio learning is paused"


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
# One bound for every re-arm of a scheduled session evaluation: a retriable
# retrieved-learning status above, or a transient provider failure of the
# agent-success judge (see ``_is_transient_evaluation_failure``).
_MAX_GROUP_EVALUATION_RETRIES = 3


def _is_transient_evaluation_failure(exc: BaseException) -> bool:
    """Whether an agent-success judge failure is an expected provider outage.

    Classified by type and status code only (``describe_tool_loop_error``),
    walking the ``__cause__`` / ``__context__`` chain from the
    ``ExtractorExecutionError`` wrapper down to the provider error, so a 429
    rate limit or a 529 overload is transient while a parse / semantic failure
    or a bug is not.

    Args:
        exc (BaseException): The failure ``run_group_evaluation`` raised.

    Returns:
        bool: True when some link of the chain is a transient LLM failure.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if describe_tool_loop_error(current)[1] == "transient":
            return True
        current = current.__cause__ or current.__context__
    return False


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


class PublishRequestIdConflictError(ValueError):
    """A caller-supplied ``request_id`` already names a different publish.

    Raised when the id is taken by another user, another payload, or a request
    this caller cannot see. An exact retry of a committed publish is replayed
    instead (see ``GenerationService._replay_existing_request``). This is a
    caller error, not a server fault: it is logged at WARNING.
    """

    def __init__(self, request_id: str) -> None:
        super().__init__(
            f"request_id {request_id!r} already exists for a different publish; "
            "use a new request_id"
        )
        self.request_id = request_id


class _RequestIdTakenError(Exception):
    """Internal: the unique request key refused the insert; roll back and re-check."""


# What makes a retry "the same publish". Server-assigned values are excluded:
# ``created_at`` on the request is server time, and an interaction's
# ``created_at`` defaults to the time the payload was parsed, so a retry that
# omits it differs on every attempt. ``agent_version`` is compared only when
# the caller states it: an omitted one is resolved from server configuration,
# which can change between an attempt and its retry. Everything else the
# caller states is compared, so success is never reported for a different
# payload.
_REPLAY_REQUEST_FIELDS = (
    "user_id",
    "session_id",
    "source",
    "agent_version",
    "evaluation_only",
    "retrieval_experiment_id",
    "retrieval_experiment_arm",
)
_REPLAY_INTERACTION_FIELDS = frozenset(
    {
        "role",
        "content",
        "user_action",
        "user_action_description",
        "interacted_image_url",
        "image_encoding",
        "shadow_content",
        "expert_content",
        "tools_used",
        "citations",
    }
)


def _replay_view(interaction: Interaction) -> dict[str, Any]:
    """An interaction as storage persists it, for exact-retry comparison.

    Retrieved learnings are compared in their persisted form: storage keeps an
    exposure receipt only as its digest, so the submitted token never
    round-trips.
    """
    view = interaction.model_dump(mode="json", include=set(_REPLAY_INTERACTION_FIELDS))
    # Storage reads an empty role back as the model default.
    view["role"] = view["role"] or "User"
    view["retrieved_learnings"] = [
        learning.persistence_payload() for learning in interaction.retrieved_learnings
    ]
    return view


def _is_same_publish(
    existing: Request,
    stored: list[Interaction],
    request: Request,
    interactions: list[Interaction],
    *,
    agent_version_stated: bool,
) -> bool:
    """Whether a committed request and its rows are exactly this publish."""
    if any(
        getattr(existing, name) != getattr(request, name)
        for name in _REPLAY_REQUEST_FIELDS
        if agent_version_stated or name != "agent_version"
    ):
        return False
    if len(stored) != len(interactions):
        return False
    return all(
        _replay_view(old) == _replay_view(new)
        for old, new in zip(stored, interactions, strict=True)
    )


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
        replayed (bool): True when this call matched an already committed
            publish with the same ``request_id`` and stored nothing new.
        interaction_ids (list[int]): Interaction ids admitted (or replayed).
    """

    request_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    replayed: bool = False
    interaction_ids: list[int] = field(default_factory=list)


def _warn_publish_followup(result: GenerationServiceResult, error: Exception) -> None:
    logger.warning(
        "event=publish_followup_failed request_id=%s error_type=%s",
        sanitise_for_log(result.request_id or ""),
        type(error).__name__,
        exc_info=True,
    )
    warning = (
        "Interactions were stored; follow-up processing was deferred. "
        "Check learning status before retrying this request."
    )
    if warning not in result.warnings:
        result.warnings.append(warning)


@contextmanager
def _publish_followup(result: GenerationServiceResult) -> Iterator[None]:
    """Contain independent best-effort work after durable admission."""
    try:
        yield
    except Exception as error:
        _warn_publish_followup(result, error)


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
        use_publish_limiter: bool = True,  # noqa: ARG002 -- retained library compatibility
        publish_limiter_wait_forever: bool = True,  # noqa: ARG002 -- retained library compatibility
        defer_learning: bool = False,
        admission_participant: ExternalAdmissionParticipant | None = None,
    ) -> GenerationServiceResult:
        """Atomically admit interactions to the durable extraction stream.

        Automatic extraction runs in bounded background workers. With
        ``defer_learning=False``, wait up to 240 seconds for this request's
        eligible profile/playbook coverage, outside ingestion capacity.
        The legacy limiter arguments remain accepted for call compatibility;
        HTTP ingestion and the shared extraction worker own their own budgets.
        Session evaluations retain their separate inactivity scheduler.
        """
        result = GenerationServiceResult()

        if not publish_user_interaction_request:
            logger.error("Received None publish_user_interaction_request")
            return result

        user_id = publish_user_interaction_request.user_id
        if not user_id:
            logger.error("Received None user_id in publish_user_interaction_request")
            return result

        publish_start = time.perf_counter()
        # Resolve agent_version: explicit > env var > default. Resolved here
        # (before the try) so success and failure telemetry share the same value.
        agent_version = resolve_agent_version(
            publish_user_interaction_request.agent_version
        )
        agent_version_stated = bool(publish_user_interaction_request.agent_version)
        admitted = False

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
            with publish_timing.phase("publish_config"):
                attribution_config = self.configurator.get_config()
            validate_retrieval_experiment_attribution(
                config=attribution_config,
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
            storage = self.storage
            if storage is None:
                raise RuntimeError("Publishing requires storage")

            from reflexio.server.services.durable_learning.waiting import (
                check_admission_deadline,
            )

            new_interactions: list[Interaction] = (
                GenerationService.get_interaction_from_publish_user_interaction_request(
                    publish_user_interaction_request, request_id
                )
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

            # An external receipt is replayed before any embedding work. An
            # ordinary publish takes no preflight read: the unique request key
            # in the admission transaction arbitrates, and a taken id is
            # replayed or refused after that rollback (below).
            if admission_participant is not None:
                replay = self._replay_existing_request(
                    admission_participant,
                    new_request,
                    new_interactions,
                    agent_version_stated=agent_version_stated,
                )
                if replay is not None:
                    return replay

            # Same phase key as the success events below, which accumulates.
            # This call reaches the same usage-event recorder and can wait on
            # the same advisory lock, so leaving it outside would report a
            # small `metering_ms` and leave the wait as an unexplained gap.
            #
            # Metering is NOT the prime suspect, whatever this module's history
            # says: `pg_stat_statements` on the metrics DB put the lock's mean
            # wait at 0.0086ms over a live 650s window (208 calls), against the
            # 119,833ms maximum `append.py`'s docstring still quotes. Measured
            # today it is ~0.24s of ~8s. The phase stays because it is free and
            # the tail can return, not because it is where the time went.
            with publish_timing.phase("metering"):
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

            from reflexio.server.services.durable_learning.admission import (
                extraction_admission,
            )
            from reflexio.server.services.durable_learning.local import (
                ensure_local_extraction,
            )

            source = publish_user_interaction_request.source or None
            stall_warning = self._refresh_publish_stall_warning(
                publish_user_interaction_request, result
            )
            with publish_timing.phase("publish_config"):
                admission_config = self.configurator.get_config()
            admission = extraction_admission(
                admission_config,
                publish_user_interaction_request,
                stalled=stall_warning is not None,
            )
            if not new_interactions:
                for kind in ("profile", "playbook"):
                    admission[kind]["eligible"] = False
            # Network preparation is outside the writer scope. The work row is
            # locked FIRST and its sequence allocation commits with every input.
            check_admission_deadline()
            publish_timing.record("n_interactions", len(new_interactions))
            with publish_timing.phase("embeddings"):
                replay = self._prepare_or_replay_publish(
                    storage,
                    admission_participant,
                    new_request,
                    new_interactions,
                    agent_version_stated=agent_version_stated,
                )
                if replay is not None:
                    return replay
            check_admission_deadline()
            from reflexio.server.services.storage.error import PostCommitStorageError

            try:
                with publish_timing.phase("commit_scope"), storage.commit_scope():
                    with publish_timing.phase("lock_stream"):
                        storage.lock_publish_stream(user_id, new_interactions)
                    check_admission_deadline()
                    if admission_participant is not None:
                        receipt = admission_participant.claim(request_id, user_id)
                        if receipt is not None:
                            result.request_id = receipt.request_id
                            result.interaction_ids = list(receipt.interaction_ids)
                            result.replayed = True
                            return result
                        # Preparation may have outlived a policy edit or billing stall.
                        stall_warning = self._refresh_publish_stall_warning(
                            publish_user_interaction_request, result, stall_warning
                        )
                        admission = extraction_admission(
                            self.configurator.get_config(),
                            publish_user_interaction_request,
                            stalled=stall_warning is not None,
                        )
                    # The unique request key arbitrates concurrent publishes, even
                    # across users whose extraction-stream locks are independent.
                    # Never upsert here: a duplicate must roll back the whole scope.
                    with publish_timing.phase("add_request"):
                        if not storage.add_request_if_absent(new_request):
                            raise _RequestIdTakenError
                    with publish_timing.phase("add_interactions"):
                        storage.add_user_interactions_bulk(
                            user_id, new_interactions, embeddings_prepared=True
                        )
                    with publish_timing.phase("admit_extraction"):
                        storage.admit_extraction(
                            user_id,
                            request_id,
                            [i.interaction_id for i in new_interactions],
                            admission,
                        )
                    if admission_participant is not None:
                        admission_participant.complete(
                            new_request, new_interactions, admission
                        )
                    result.interaction_ids = [
                        i.interaction_id for i in new_interactions
                    ]
            except PostCommitStorageError as error:
                _warn_publish_followup(result, error)
            except _RequestIdTakenError:
                # The id is already committed (a retry, a concurrent duplicate,
                # or a reuse). Nothing of ours committed: replay or refuse.
                replay = self._replay_existing_request(
                    admission_participant,
                    new_request,
                    new_interactions,
                    agent_version_stated=agent_version_stated,
                )
                if replay is None:
                    # Taken, but not visible to this caller (another project).
                    raise PublishRequestIdConflictError(request_id) from None
                return replay
            admitted = True
            extraction_ready = False
            with publish_timing.phase("post_publish"), _publish_followup(result):
                ensure_local_extraction(self.request_context)
                extraction_ready = True
            with publish_timing.phase("evaluation_schedule"), _publish_followup(result):
                self._schedule_post_publish_evaluations(
                    new_request=new_request,
                    interactions=new_interactions,
                    user_id=user_id,
                    agent_version=agent_version,
                    source=source,
                )
            with publish_timing.phase("metering"), _publish_followup(result):
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
                        "warning_count": len(result.warnings),
                    },
                )
            # No `emit` here, deliberately. This frame is too EARLY to report:
            # the caller's post-commit coverage reads
            # (`lib/_interactions.py::_safe_coverage`, two remote round trips
            # on every publish) run after it returns and are part of serving
            # the request. The worker's outermost frame,
            # `publisher_api.add_user_interaction`, reports instead -- which
            # also catches a failure that never reaches this method at all,
            # such as a cold `get_reflexio` construction.
            if not defer_learning and extraction_ready:
                self._wait_for_extraction_coverage(
                    storage, user_id, request_id, publish_start
                )
            return result

        except Exception as e:
            if admitted:
                # This request and its durable work are already committed.
                # A follow-up failure must not invite another admission/retry.
                _warn_publish_followup(result, e)
                return result
            with publish_timing.phase("metering"):
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
            if isinstance(e, PublishRequestIdConflictError):
                # The caller reused an id for a different publish; nothing on
                # the server is broken, so this is not an error report.
                logger.warning(
                    "event=publish_request_id_conflict request_id=%s user_id=%s",
                    sanitise_for_log(e.request_id),
                    sanitise_for_log(user_id),
                )
                raise
            self._log_publish_failure(e, user_id, result.request_id)
            raise

    def _prepare_or_replay_publish(
        self,
        storage: BaseStorage,
        participant: ExternalAdmissionParticipant | None,
        new_request: Request,
        new_interactions: list[Interaction],
        *,
        agent_version_stated: bool,
    ) -> GenerationServiceResult | None:
        """Prepare new vectors, or recover a committed replay while unready."""
        from reflexio.server.services.storage.error import ReadinessUnavailableError

        try:
            storage.prepare_interaction_embeddings(new_interactions)
        except ReadinessUnavailableError:
            # Ordinary retries discover committed duplicates after preparation.
            # During a readiness refusal, use the same fenced replay check:
            # committed work needs no new vector admission. Successful first
            # attempts still pay no extra replay read.
            replay = self._replay_existing_request(
                participant,
                new_request,
                new_interactions,
                agent_version_stated=agent_version_stated,
            )
            if replay is not None:
                return replay
            raise
        return None

    def _wait_for_extraction_coverage(
        self,
        storage: BaseStorage,
        user_id: str,
        request_id: str,
        publish_start: float,
    ) -> None:
        """Wait (bounded) for this committed request's extraction coverage."""
        from reflexio.server.services.durable_learning.waiting import (
            acquire_waiter,
            coverage_stalled,
            release_waiter,
        )

        # Blocking on a background worker is not time spent SERVING
        # this publish. Because the reporting frame sits further OUT
        # than this one, the wait is SUBTRACTED rather than the clock
        # stopped. Without this, a threshold low enough to be useful
        # reports every waited request as slow -- measured at 609ms for
        # a publish whose own phases summed to ~110ms.
        if acquire_waiter(self.org_id):
            try:
                with publish_timing.excluded():
                    deadline = publish_start + 240
                    while time.perf_counter() < deadline:
                        status = storage.extraction_status(user_id, request_id)
                        if status["status"] == "done":
                            break
                        # A partial window needs input this caller does
                        # not have, so the remaining deadline cannot
                        # change the answer. Without this the default
                        # library publish blocks the full 240s for any
                        # user below one window.
                        if coverage_stalled(status):
                            break
                        time.sleep(min(0.25, max(0, deadline - time.perf_counter())))
            except Exception:  # noqa: BLE001
                # Everything above this point has COMMITTED. The waiter
                # only observes how far extraction has got; it performs
                # no writes. Letting it raise would fall into the outer
                # handler, which records `publish_request_failed` and
                # re-raises -- the caller is then told their durable
                # publish failed, and the response carries no
                # `request_id`, so they cannot even poll for the work
                # that actually landed. Stop waiting instead: coverage
                # is reported separately and is allowed to be unknown.
                logger.warning(
                    "Publish for user %s committed, but waiting for "
                    "extraction coverage of request %s failed; "
                    "returning the admitted publish.",
                    sanitise_for_log(user_id),
                    sanitise_for_log(request_id),
                    exc_info=True,
                )
            finally:
                release_waiter(self.org_id)

    def _log_publish_failure(
        self, error: Exception, user_id: str, request_id: str | None
    ) -> None:
        """Log a publish that failed before admission, by what it means."""
        from reflexio.server.services.durable_learning.waiting import (
            AdmissionDeadlineExceededError,
        )
        from reflexio.server.services.storage.error import ReadinessUnavailableError
        from reflexio.server.work_scope import current_project_id

        if isinstance(error, ReadinessUnavailableError):
            # run() calls this only before admission committed. The host
            # returns a retryable 503, so this is a refusal rather than a
            # profile-refresh fault. It still blocks release verification
            # until the affected operation demonstrably recovers: retryable
            # does not mean the dependency is healthy. Never log provider
            # exception values, which can contain credentials.
            project_id = current_project_id()
            with error_tags(
                subsystem="generation",
                op="publish_interaction",
                org_id=self.org_id,
                project_id=project_id,
                user_id=user_id,
                request_id=request_id,
                release_blocking="true",
            ):
                logger.warning(
                    "event=publish_readiness_unavailable admitted=false retryable=true "
                    "release_blocking=true org_id=%s project_id=%s user_id=%s request_id=%s",
                    sanitise_for_log(self.org_id),
                    sanitise_for_log(project_id or ""),
                    sanitise_for_log(user_id),
                    sanitise_for_log(request_id or ""),
                )
            return

        if isinstance(error, AdmissionDeadlineExceededError):
            # Capacity, not a fault: the deadline passed before admission
            # committed, so nothing landed and the route answers with a
            # retryable 503. An ERROR here would report each busy publish
            # as a server fault.
            logger.warning(
                "event=publish_admission_deadline_exceeded user_id=%s request_id=%s",
                sanitise_for_log(user_id),
                sanitise_for_log(request_id or ""),
            )
            return
        with error_tags(
            subsystem="generation",
            op="refresh_profile",
            org_id=self.org_id,
            user_id=user_id,
            request_id=request_id,
            error_type=type(error).__name__,
        ):
            logger.exception(
                "Failed to refresh user profile for user id: %s",
                user_id,
            )

    def _replay_existing_request(
        self,
        participant: ExternalAdmissionParticipant | None,
        new_request: Request,
        new_interactions: list[Interaction],
        *,
        agent_version_stated: bool,
    ) -> GenerationServiceResult | None:
        """Answer a retry of an already committed publish without re-admitting it.

        Returns None when ``request_id`` is unused or invisible to this caller.
        With an external admission participant the replay is its fenced
        receipt. Otherwise the committed
        request and its interactions must equal this publish exactly
        (``_is_same_publish``); a match returns the original identifiers with
        ``replayed=True``. Anything else raises PublishRequestIdConflictError.
        """
        request_id, user_id = new_request.request_id, new_request.user_id
        storage = self.storage
        if storage is None:
            raise RuntimeError("Publishing requires storage")
        from reflexio.server.services.durable_learning.waiting import (
            check_admission_deadline,
        )

        # Read through the writer scope, but create/lock a work row only for a
        # replay; ordinary misses must leave no state behind before admission.
        with storage.commit_scope():
            existing = storage.get_request(request_id)
            if existing is None:
                return None
            if participant is not None:
                storage.lock_extraction_stream(user_id)
                check_admission_deadline()
                receipt = participant.claim(request_id, user_id)
                if receipt is None:
                    raise PublishRequestIdConflictError(request_id)
                return GenerationServiceResult(
                    request_id=receipt.request_id,
                    interaction_ids=list(receipt.interaction_ids),
                    replayed=True,
                )
            stored = sorted(
                storage.get_interactions_by_request_ids([request_id]),
                key=lambda interaction: interaction.interaction_id,
            )
        if not _is_same_publish(
            existing,
            stored,
            new_request,
            new_interactions,
            agent_version_stated=agent_version_stated,
        ):
            raise PublishRequestIdConflictError(request_id)
        return GenerationServiceResult(
            request_id=existing.request_id,
            interaction_ids=[interaction.interaction_id for interaction in stored],
            replayed=True,
        )

    def _refresh_publish_stall_warning(
        self,
        request: PublishUserInteractionRequest,
        result: GenerationServiceResult,
        previous_warning: str | None = None,
    ) -> str | None:
        """Keep the returned warning aligned with the admission's stall policy."""
        if previous_warning:
            result.warnings.remove(previous_warning)
        warning = (
            self._active_learning_stall_warning()
            if not request.override_learning_stall
            else None
        )
        if warning:
            result.warnings.append(warning)
        return warning

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
                try:
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
                        caller_reports_extractor_failure=True,
                    )
                except ExtractorExecutionError as exc:
                    # The agent-success judge failed before anything was saved
                    # or marked, and nothing else re-triggers this session
                    # unless it gets more traffic. This callback owns the one
                    # report (the runner logged it at WARNING): a transient
                    # provider failure is re-armed with the same families,
                    # bounded; anything else, or the last attempt, is an ERROR.
                    transient = _is_transient_evaluation_failure(exc)
                    if transient and _attempt < _MAX_GROUP_EVALUATION_RETRIES:
                        logger.warning(
                            "event=agent_success_retry_scheduled session=%s"
                            " attempt=%d/%d error_type=%s: %s",
                            _sid,
                            _attempt + 1,
                            _MAX_GROUP_EVALUATION_RETRIES,
                            type(exc.__cause__ or exc).__name__,
                            exc,
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
                                _run_agent_success,
                                _run_retrieved_learning,
                                _attempt + 1,
                            ),
                        )
                        return
                    logger.error(
                        "Agent-success evaluation failed for session=%s user=%s"
                        " after %d attempt(s) (transient=%s)",
                        _sid,
                        _user_id,
                        _attempt + 1,
                        transient,
                        exc_info=exc,
                    )
                    return
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
                    and _attempt < _MAX_GROUP_EVALUATION_RETRIES
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
        with publish_timing.phase("evaluation_config"):
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
