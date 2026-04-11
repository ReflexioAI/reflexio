"""Runner for group-level agent success evaluation.

Fetches all requests and interactions for a session,
checks completion status, runs evaluation, and marks the group as evaluated.
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_service import (
    AgentSuccessEvaluationService,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    AgentSuccessEvaluationRequest,
)
from reflexio.server.services.agent_success_evaluation.delayed_group_evaluator import (
    _EFFECTIVE_DELAY_SECONDS,
)

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
) -> None:
    """Run agent success evaluation for an entire session.

    Steps:
    1. Check if already evaluated via operation state
    2. Fetch all requests for the session
    3. Verify completion (latest request created_at >= delay ago)
    4. Fetch interactions and build data models
    5. Run evaluation service
    6. Mark as evaluated in operation state

    Args:
        org_id: Organization ID
        user_id: User ID who owns the requests
        session_id: Session identifier
        agent_version: Agent version string
        source: Source of the interactions
        request_context: Request context with storage and configurator
        llm_client: LLM client for evaluation
    """
    storage = request_context.storage
    state_key = _build_state_key(org_id, user_id, session_id)

    # 1. Check if already evaluated
    existing_state = storage.get_operation_state(state_key)  # type: ignore[reportOptionalMemberAccess]
    if existing_state and isinstance(existing_state.get("operation_state"), dict):
        op_state = existing_state["operation_state"]
        if op_state.get("evaluated"):
            logger.info("Session %s already evaluated, skipping", session_id)
            return

    # 2. Fetch all requests for the session
    requests = storage.get_requests_by_session(user_id, session_id)  # type: ignore[reportOptionalMemberAccess]
    if not requests:
        logger.info("No requests found for session %s, skipping", session_id)
        return

    # 3. Verify completion: latest request must be >= delay ago
    latest_created_at = max(r.created_at for r in requests)
    now = int(datetime.now(UTC).timestamp())
    elapsed = now - latest_created_at
    if elapsed < _EFFECTIVE_DELAY_SECONDS:
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
        logger.info(
            "No request interaction data models built for session %s, skipping",
            session_id,
        )
        return

    # 5. Run evaluation
    logger.info(
        "Running group evaluation for session=%s with %d requests and %d interactions",
        session_id,
        len(request_interaction_data_models),
        len(all_interactions),
    )

    evaluation_request = AgentSuccessEvaluationRequest(
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
            "Group evaluation for session=%s had failures (save_failed=%s); skipping evaluated marker",
            session_id,
            evaluation_service.last_run_save_failed,
        )
        return

    if evaluation_service.last_run_saved_result_count == 0:
        logger.warning(
            "Group evaluation for session=%s saved no results; skipping evaluated marker",
            session_id,
        )
        return

    # 6. Mark as evaluated
    evaluated_at = int(datetime.now(UTC).timestamp())
    storage.upsert_operation_state(  # type: ignore[reportOptionalMemberAccess]
        state_key,
        {"evaluated": True, "evaluated_at": evaluated_at},
    )
    logger.info("Marked session %s as evaluated at %d", session_id, evaluated_at)
