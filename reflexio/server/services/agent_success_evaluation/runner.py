"""Runner for group-level agent success evaluation.

Fetches all requests and interactions for a session,
checks completion status, runs evaluation, and marks the group as evaluated.

Also runs the retrieved-learning relevance/impact evaluation for the session
after agent-success work completes. The two completions are independent: the
existing ``agent_success_group_eval`` marker stays agent-success-only, and the
retrieved evaluation keeps its own generation/fingerprint-fenced state (see
``storage_base/retrieved_learning_state.py``).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation import _eval_health
from reflexio.server.services.agent_success_evaluation._eval_health import SkipReason
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    AgentSuccessEvaluationRequest,
)
from reflexio.server.services.agent_success_evaluation.components.retrieved_learning_evaluator import (
    RetrievedLearningEvaluator,
)
from reflexio.server.services.agent_success_evaluation.scheduler import (
    _EFFECTIVE_DELAY_SECONDS,
)
from reflexio.server.services.agent_success_evaluation.service import (
    AgentSuccessEvaluationService,
)
from reflexio.server.services.extractor_config_utils import get_extractor_name
from reflexio.server.services.storage.storage_base.evaluation_state_keys import (
    build_agent_success_marker_key,
)
from reflexio.server.services.storage.storage_base.retrieved_learning_state import (
    session_fingerprint,
)

logger = logging.getLogger(__name__)

# Key prefix for operation state tracking
OPERATION_STATE_KEY_PREFIX = "agent_success_group_eval"

type AgentSuccessInvocationStatus = Literal[
    "complete", "failed", "not_applicable", "skipped"
]
type RetrievedLearningInvocationStatus = Literal[
    "pending",
    "complete",
    "degraded",
    "failed",
    "not_applicable",
    "stale",
    "superseded",
    "skipped",
]


@dataclass
class GroupEvaluationOutcome:
    """What one runner invocation did, per evaluation family.

    ``retrieved_learning_fingerprint`` carries the session fingerprint at
    which a terminal/applied retrieved outcome linearized; ``None`` for
    nonterminal or skipped-without-state outcomes.
    """

    agent_success_status: AgentSuccessInvocationStatus
    retrieved_learning_status: RetrievedLearningInvocationStatus
    retrieved_learning_fingerprint: str | None = None


def _build_state_key(org_id: str, user_id: str, session_id: str) -> str:
    """Build the operation state key for a session.

    Delegates to the shared key builder so governance erasure scrubs the
    exact same keys this runner writes.

    Args:
        org_id: Organization ID
        user_id: User ID
        session_id: Session identifier

    Returns:
        str: The operation state key
    """
    return build_agent_success_marker_key(org_id, user_id, session_id)


def run_group_evaluation(
    org_id: str,
    user_id: str,
    session_id: str,
    agent_version: str,
    source: str | None,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
    *,
    force_regenerate: bool = False,
    run_agent_success: bool = True,
    run_retrieved_learning: bool = True,
) -> GroupEvaluationOutcome:
    """Run agent success evaluation for an entire session.

    Steps:
    1. Check if already evaluated via operation state (skipped when force_regenerate)
    2. Fetch all requests for the session
    3. Verify completion (latest request created_at >= delay ago; skipped when force_regenerate)
    4. Fetch interactions and build data models
    5. Capture prior result_ids (when regenerating) so they
       can be removed AFTER the new save lands
    6. Run evaluation service (which saves new rows)
    7. On success, delete the captured prior rows by id — the new rows have
       fresh auto-increment ids that do not overlap. A failure here leaves
       the session in a consistent pre-regen state instead of zero rows.
    8. Mark as evaluated in operation state

    Args:
        org_id: Organization ID
        user_id: User ID who owns the requests
        session_id: Session identifier
        agent_version: Agent version string
        source: Source of the interactions
        request_context: Request context with storage and configurator
        llm_client: LLM client for evaluation
        force_regenerate: When True, bypass the already-evaluated short-circuit
            and the completeness delay gate so the regenerate worker can
            re-evaluate sessions of any age regardless of prior state.
        run_agent_success: Whether this session was admitted for the
            session-success judge. The publish scheduler samples the two
            families independently (see ``sampling.py``) and passes the result;
            direct callers leave it True.
        run_retrieved_learning: Whether this session was admitted for the
            retrieved-learning relevance/impact judge.

    Returns:
        GroupEvaluationOutcome: Per-family statuses for this invocation.
    """
    storage = request_context.storage
    state_key = _build_state_key(org_id, user_id, session_id)

    # 1. Fetch all requests for the session
    requests = storage.get_requests_by_session(user_id, session_id)  # type: ignore[reportOptionalMemberAccess]
    if not requests:
        _eval_health.record_skip(SkipReason.NO_REQUESTS)
        logger.info("No requests found for session %s, skipping", session_id)
        return GroupEvaluationOutcome("not_applicable", "skipped")

    # 2. Verify completion: latest request must be >= delay ago — skipped in
    # force_regenerate mode so the operator can re-evaluate any session. The
    # liveness gate applies to both evaluation families.
    if not force_regenerate:
        latest_created_at = max(r.created_at for r in requests)
        now = int(datetime.now(UTC).timestamp())
        elapsed = now - latest_created_at
        if elapsed < _EFFECTIVE_DELAY_SECONDS:
            _eval_health.record_skip(SkipReason.NOT_YET_COMPLETE)
            logger.info(
                "Session %s not yet complete (latest request %ds ago, need %ds), skipping",
                session_id,
                elapsed,
                _EFFECTIVE_DELAY_SECONDS,
            )
            return GroupEvaluationOutcome("skipped", "skipped")

    # 3. Per-family admission. The scheduler samples the two families
    # independently and tells us which ones this session was admitted for, so a
    # session sampled only for retrieved-learning never pays the session-success
    # judge. Direct callers (regen jobs, the on-demand grade route) leave both
    # flags at their default and run both families, as before.
    if not run_agent_success:
        return _finish_with_retrieved_evaluation(
            "skipped",
            user_id=user_id,
            session_id=session_id,
            agent_version=agent_version,
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=force_regenerate,
            run_retrieved_learning=run_retrieved_learning,
        )

    # 4. Check if agent success is already evaluated — skipped in
    # force_regenerate mode so the regenerate worker can re-evaluate a session
    # that's already been marked. Retrieved-learning evaluation still runs
    # below: its completion is independent of the agent-success marker.
    agent_success_already_evaluated = False
    if not force_regenerate:
        existing_state = storage.get_operation_state(state_key)  # type: ignore[reportOptionalMemberAccess]
        if existing_state and isinstance(existing_state.get("operation_state"), dict):
            op_state = existing_state["operation_state"]
            if op_state.get("evaluated"):
                _eval_health.record_skip(SkipReason.ALREADY_EVALUATED)
                logger.info(
                    "Session %s already evaluated (agent success), skipping to"
                    " retrieved-learning evaluation",
                    session_id,
                )
                agent_success_already_evaluated = True

    if agent_success_already_evaluated:
        return _finish_with_retrieved_evaluation(
            "skipped",
            user_id=user_id,
            session_id=session_id,
            agent_version=agent_version,
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=force_regenerate,
            run_retrieved_learning=run_retrieved_learning,
        )

    # 4. Fetch interactions for all requests
    request_ids = [r.request_id for r in requests]
    all_interactions = storage.get_interactions_by_request_ids(request_ids)  # type: ignore[reportOptionalMemberAccess]
    if not all_interactions:
        _eval_health.record_skip(SkipReason.NO_INTERACTIONS)
        logger.info("No interactions found for session %s, skipping", session_id)
        return GroupEvaluationOutcome("not_applicable", "skipped")

    # Group interactions by request_id
    interactions_by_request: dict[str, list] = defaultdict(list)
    for interaction in all_interactions:
        interactions_by_request[interaction.request_id].append(interaction)

    # Build RequestInteractionDataModel list, sorted by request created_at
    requests_sorted = sorted(requests, key=lambda r: r.created_at)
    request_interaction_data_models = []
    for req in requests_sorted:
        req_interactions = interactions_by_request.get(req.request_id, [])
        if req_interactions:
            # Sort interactions by created_at within each request
            req_interactions.sort(key=lambda i: i.created_at)
            request_interaction_data_models.append(
                RequestInteractionDataModel(
                    session_id=session_id,
                    request=req,
                    interactions=req_interactions,
                )
            )

    if not request_interaction_data_models:
        _eval_health.record_skip(SkipReason.NO_DATA_MODELS)
        logger.info(
            "No request interaction data models built for session %s, skipping",
            session_id,
        )
        return GroupEvaluationOutcome("not_applicable", "skipped")

    # 5. When regenerating, capture the prior result_ids
    # so we can delete ONLY them AFTER the new rows have been saved. Doing
    # the delete before the LLM call risks wiping the session's rows if the
    # call fails (rate limit, network) and nothing replaces them. The new
    # rows always get fresh auto-increment ids, so deleting the captured set
    # afterwards cannot remove the new rows.
    old_result_ids: list[int] = []
    if force_regenerate:
        config = request_context.configurator.get_config()
        old_result_ids = storage.get_agent_success_evaluation_result_ids(  # type: ignore[reportOptionalMemberAccess]
            user_id=user_id,
            session_id=session_id,
            evaluation_name=get_extractor_name(config),
            agent_version=agent_version,
        )

    logger.info(
        "Running group evaluation for session=%s with %d requests and %d interactions"
        " (force_regenerate=%s, prior_result_ids=%d)",
        session_id,
        len(request_interaction_data_models),
        len(all_interactions),
        force_regenerate,
        len(old_result_ids),
    )

    evaluation_request = AgentSuccessEvaluationRequest(
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        source=source,
        request_interaction_data_models=request_interaction_data_models,
    )

    evaluation_service = AgentSuccessEvaluationService(
        llm_client=llm_client, request_context=request_context
    )
    evaluation_service.run(evaluation_request)

    if evaluation_service.has_run_failures():
        logger.warning(
            "Group evaluation for session=%s had failures (save_failed=%s);"
            " preserving %d prior result row(s) and skipping evaluated marker",
            session_id,
            evaluation_service.last_run_save_failed,
            len(old_result_ids),
        )
        return GroupEvaluationOutcome("failed", "skipped")

    if evaluation_service.last_run_saved_result_count == 0:
        logger.warning(
            "Group evaluation for session=%s saved no results;"
            " preserving %d prior result row(s) and skipping evaluated marker",
            session_id,
            len(old_result_ids),
        )
        return GroupEvaluationOutcome("failed", "skipped")

    # 6. New rows saved successfully — now safe to remove the captured prior
    # rows. New rows have fresh auto-increment result_ids that do not overlap
    # with old_result_ids, so this cannot delete the regenerated verdict.
    if old_result_ids:
        deleted = storage.delete_agent_success_evaluation_results_by_ids(  # type: ignore[reportOptionalMemberAccess]
            old_result_ids
        )
        logger.info(
            "Regenerate cleanup: deleted %d prior result row(s) for session=%s"
            " (expected %d)",
            deleted,
            session_id,
            len(old_result_ids),
        )

    # 7. Mark as evaluated
    evaluated_at = int(datetime.now(UTC).timestamp())
    storage.upsert_operation_state(  # type: ignore[reportOptionalMemberAccess]
        state_key,
        {"evaluated": True, "evaluated_at": evaluated_at},
    )
    logger.info("Marked session %s as evaluated at %d", session_id, evaluated_at)

    # 8. Retrieved-learning evaluation — independent completion; a failure
    # here can never be reported as complete nor force agent-success rows to
    # regenerate.
    return _finish_with_retrieved_evaluation(
        "complete",
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        request_context=request_context,
        llm_client=llm_client,
        force_regenerate=force_regenerate,
        run_retrieved_learning=run_retrieved_learning,
    )


def _finish_with_retrieved_evaluation(
    agent_success_status: AgentSuccessInvocationStatus,
    *,
    user_id: str,
    session_id: str,
    agent_version: str,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
    force_regenerate: bool,
    run_retrieved_learning: bool = True,
) -> GroupEvaluationOutcome:
    """Run the retrieved-learning phase best-effort and build the outcome."""
    if not run_retrieved_learning:
        return GroupEvaluationOutcome(agent_success_status, "skipped")
    try:
        retrieved_status, fingerprint = _run_retrieved_learning_evaluation(
            user_id=user_id,
            session_id=session_id,
            agent_version=agent_version,
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=force_regenerate,
        )
    except Exception:
        # Best-effort: never let the retrieved phase break the runner. The
        # generation-guarded state was not advanced to a terminal status, so
        # the next scheduled or forced run retries.
        logger.exception(
            "event=retrieved_learning_eval_failed session_id=%s reason=unexpected_error",
            session_id,
        )
        _eval_health.record_retrieved_outcome("failed")
        _eval_health.record_producer_failure()
        return GroupEvaluationOutcome(agent_success_status, "failed")
    return GroupEvaluationOutcome(agent_success_status, retrieved_status, fingerprint)


def _run_retrieved_learning_evaluation(
    *,
    user_id: str,
    session_id: str,
    agent_version: str,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
    force_regenerate: bool,
) -> tuple[RetrievedLearningInvocationStatus, str | None]:
    """Evaluate the session's retrieved learnings with fencing.

    Implements the generation + session-fingerprint protocol: allocate a
    generation, judge the post-allocation snapshot, and atomically replace
    the session's result set only while the fingerprint recomputed under the
    replacement lock still matches. One immediate retry on a stale snapshot,
    then ``pending`` for the next trigger.

    Returns:
        tuple: (invocation status, session fingerprint for terminal/applied
        outcomes else None).
    """
    storage = request_context.storage
    if storage is None:
        return "skipped", None

    snapshot = storage.load_bounded_retrieved_learning_snapshot(user_id, session_id)
    fingerprint = session_fingerprint(snapshot)

    # Fast path: terminal state at this exact fingerprint — nothing changed.
    if not force_regenerate:
        terminal = storage.get_matching_retrieved_learning_terminal_state(
            user_id, session_id, fingerprint
        )
        if terminal:
            status = terminal.get("status")
            if status in ("complete", "not_applicable"):
                return status, fingerprint

    config = request_context.configurator.get_config()
    agent_success = config.agent_success_config if config else None
    success_definition = (
        agent_success.success_definition_prompt.strip()
        if agent_success and agent_success.success_definition_prompt
        else ""
    )
    evaluator = RetrievedLearningEvaluator(
        request_context=request_context,
        llm_client=llm_client,
        agent_context=(config.agent_context_prompt or "") if config else "",
        success_definition=success_definition,
    )

    logger.info(
        "event=retrieved_learning_eval_started session_id=%s raw_attachments=%d",
        session_id,
        snapshot.raw_attachment_count,
    )

    generation = 0
    for _stale_attempt in range(2):
        generation = storage.begin_retrieved_learning_evaluation_run(
            user_id, session_id
        )
        # Judge the post-allocation snapshot, never the freshness-check one:
        # every mutation is either visible here or changes the fingerprint
        # that replacement recomputes under lock at commit time.
        snapshot = storage.load_bounded_retrieved_learning_snapshot(user_id, session_id)
        fingerprint = session_fingerprint(snapshot)
        run = evaluator.evaluate(user_id, session_id, agent_version, snapshot)
        if run.outcome == "failed":
            storage.finish_retrieved_learning_evaluation_run(
                user_id, session_id, generation, "failed", run.diagnostics
            )
            logger.warning(
                "event=retrieved_learning_eval_failed session_id=%s reason=%s",
                session_id,
                run.diagnostics.get("error_type", "unknown"),
            )
            _eval_health.record_retrieved_outcome("failed", diagnostics=run.diagnostics)
            _eval_health.record_producer_failure()
            return "failed", None
        commit = storage.replace_retrieved_learning_evaluation_results(
            user_id,
            session_id,
            generation,
            fingerprint,
            run.proposed_status,
            run.diagnostics,
            run.rows,
        )
        if commit.disposition == "applied":
            # An applied commit's authoritative status is always one of
            # complete/degraded/not_applicable (the storage layer validates
            # proposed_status and may only downgrade to not_applicable).
            final_status: RetrievedLearningInvocationStatus = (
                commit.status
                if commit.status in ("complete", "degraded", "not_applicable")
                else run.proposed_status
            )
            logger.info(
                "event=retrieved_learning_eval_%s session_id=%s candidates=%d"
                " committed=%d failed_relevance_chunks=%s failed_impact_chunks=%s",
                "completed"
                if final_status in ("complete", "not_applicable")
                else final_status,
                session_id,
                len(run.rows),
                commit.committed_count,
                run.diagnostics.get("failed_relevance_chunks", 0),
                run.diagnostics.get("failed_impact_chunks", 0),
            )
            _eval_health.record_retrieved_outcome(
                final_status, diagnostics=run.diagnostics
            )
            return final_status, fingerprint
        if commit.disposition == "superseded":
            logger.info(
                "event=retrieved_learning_eval_superseded session_id=%s generation=%d",
                session_id,
                generation,
            )
            return "superseded", None
        logger.info(
            "event=retrieved_learning_eval_stale session_id=%s generation=%d",
            session_id,
            generation,
        )

    # Two stale snapshots in a row: leave pending for the next trigger.
    storage.finish_retrieved_learning_evaluation_run(
        user_id, session_id, generation, "pending", {"error_type": "stale_snapshot"}
    )
    return "pending", None
