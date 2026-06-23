"""Runner for group-level agent success evaluation.

Fetches all requests and interactions for a session,
checks completion status, runs evaluation, and marks the group as evaluated.
"""

import logging
import random
from collections import defaultdict
from datetime import UTC, datetime

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation import _eval_health
from reflexio.server.services.agent_success_evaluation._eval_health import SkipReason
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_service import (
    AgentSuccessEvaluationService,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    AgentSuccessEvaluationRequest,
)
from reflexio.server.services.agent_success_evaluation.delayed_group_evaluator import (
    _EFFECTIVE_DELAY_SECONDS,
)
from reflexio.server.services.extractor_config_utils import get_extractor_name
from reflexio.server.services.shadow_comparison.judge import ShadowComparisonJudge

logger = logging.getLogger(__name__)

# Key prefix for operation state tracking
OPERATION_STATE_KEY_PREFIX = "agent_success_group_eval"


def _build_state_key(org_id: str, user_id: str, session_id: str) -> str:
    """Build the operation state key for a session.

    Args:
        org_id: Organization ID
        user_id: User ID
        session_id: Session identifier

    Returns:
        str: The operation state key
    """
    return f"{OPERATION_STATE_KEY_PREFIX}::{org_id}::{user_id}::{session_id}"


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
) -> None:
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
    """
    storage = request_context.storage
    state_key = _build_state_key(org_id, user_id, session_id)

    # 1. Check if already evaluated — skipped in force_regenerate mode so the
    # regenerate worker can re-evaluate a session that's already been marked.
    if not force_regenerate:
        existing_state = storage.get_operation_state(state_key)  # type: ignore[reportOptionalMemberAccess]
        if existing_state and isinstance(existing_state.get("operation_state"), dict):
            op_state = existing_state["operation_state"]
            if op_state.get("evaluated"):
                _eval_health.record_skip(SkipReason.ALREADY_EVALUATED)
                logger.info("Session %s already evaluated, skipping", session_id)
                return

    # 2. Fetch all requests for the session
    requests = storage.get_requests_by_session(user_id, session_id)  # type: ignore[reportOptionalMemberAccess]
    if not requests:
        _eval_health.record_skip(SkipReason.NO_REQUESTS)
        logger.info("No requests found for session %s, skipping", session_id)
        return

    # 3. Verify completion: latest request must be >= delay ago — skipped in
    # force_regenerate mode so the operator can re-evaluate any session.
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
            return

    # 4. Fetch interactions for all requests
    request_ids = [r.request_id for r in requests]
    all_interactions = storage.get_interactions_by_request_ids(request_ids)  # type: ignore[reportOptionalMemberAccess]
    if not all_interactions:
        _eval_health.record_skip(SkipReason.NO_INTERACTIONS)
        logger.info("No interactions found for session %s, skipping", session_id)
        return

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
        return

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
        return

    if evaluation_service.last_run_saved_result_count == 0:
        logger.warning(
            "Group evaluation for session=%s saved no results;"
            " preserving %d prior result row(s) and skipping evaluated marker",
            session_id,
            len(old_result_ids),
        )
        return

    # F1: per-turn shadow comparison. Dispatched only AFTER the regular
    # success eval succeeds — a session whose success grade is unreliable
    # would yield noisy verdicts that mislead the headline metric. The
    # dispatch loop swallows per-interaction failures so one judge call
    # cannot abort an entire batch.
    _dispatch_shadow_comparison_judge(
        storage=storage,
        interactions=all_interactions,
        session_id=session_id,
        agent_version=agent_version,
        request_context=request_context,
        llm_client=llm_client,
    )

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


def _dispatch_shadow_comparison_judge(
    *,
    storage,  # noqa: ANN001 — BaseStorage; imported lazily to avoid cycles
    interactions: list[Interaction],
    session_id: str,
    agent_version: str,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
) -> None:
    """F1: grade each shadow-bearing interaction with the per-turn judge.

    Iterates the session's interactions, skips any without ``shadow_content``,
    invokes :class:`ShadowComparisonJudge.judge_turn`, and persists each
    returned verdict via ``storage.save_shadow_comparison_verdict``. Per-
    interaction exceptions are logged and the loop continues — partial
    verdict sets are strictly better than nothing for the headline metric.

    Args:
        storage: The session storage. Must implement
            ``save_shadow_comparison_verdict`` (currently SQLite + Supabase
            + disk; backends without it surface ``NotImplementedError`` at
            save time and the loop logs+continues).
        interactions (list[Interaction]): Every interaction in the session,
            in chronological order. Only those with non-empty
            ``shadow_content`` are graded.
        session_id (str): Denormalized onto each verdict.
        agent_version (str): Denormalized onto each verdict.
        request_context (RequestContext): Provides the configurator (for
            the pinned ``shadow_comparison_judge_prompt_version``) and the
            shared ``prompt_manager``.
        llm_client (LiteLLMClient): The unified LLM client the judge uses
            for the structured-output call.

    Returns:
        None: Verdicts are persisted as a side effect; the caller does not
            need the count for control flow.
    """
    config = request_context.configurator.get_config()  # type: ignore[reportOptionalMemberAccess]
    judge = ShadowComparisonJudge(
        llm_client=llm_client,
        prompt_manager=request_context.prompt_manager,  # type: ignore[reportOptionalMemberAccess]
        prompt_version=config.shadow_comparison_judge_prompt_version,
    )
    rng = random.Random()  # noqa: S311 — position randomization, not crypto
    saved_count = 0

    for interaction in interactions:
        if not interaction.shadow_content:
            continue
        try:
            verdict = judge.judge_turn(
                interaction=interaction,
                session_id=session_id,
                agent_version=agent_version,
                rng=rng,
            )
        except Exception as exc:  # noqa: BLE001 — judge failure must not abort batch
            logger.warning(
                "F1 shadow_comparison dispatch failed for interaction %s: %s",
                interaction.interaction_id,
                exc,
            )
            continue
        if verdict is None:
            continue
        try:
            storage.save_shadow_comparison_verdict(verdict)
            saved_count += 1
        except Exception as exc:  # noqa: BLE001 — single-row save failure must not abort batch
            logger.warning(
                "F1 shadow_comparison verdict save failed for interaction %s: %s",
                interaction.interaction_id,
                exc,
            )

    if saved_count:
        logger.info(
            "F1: saved %d shadow_comparison verdict(s) for session=%s",
            saved_count,
            session_id,
        )
