"""Utility functions for agent success evaluation service"""

from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_constants import (
    AgentSuccessEvaluationConstants,
)
from reflexio.server.services.service_utils import (
    MessageConstructionConfig,
    PromptConfig,
    construct_messages_from_interactions,
    extract_interactions_from_request_interaction_data_models,
    format_sessions_to_history_string,
    slice_content_by_tokens,
)

# Whole-transcript ceiling for one agent-success evaluation prompt. Each
# interaction is already capped (DEFAULT_MAX_INTERACTION_CONTENT_TOKENS) but the
# SUM was not, and tool-call prefixes are not budgeted at all, so a long session
# built a prompt the model rejected outright -- "Prompt is too long" on a
# 41-interaction session (claude-smart#162). The sibling retrieved-learning judge
# in this service is bounded the same way (TRANSCRIPT_TOKEN_LIMIT).
#
# slice_content_by_tokens keeps the first and last halves with a visible marker,
# which suits a success judgement: the task is stated at the head and the
# outcome lands at the tail. 64k sits well inside a 128k-200k context window
# while leaving the template and reply room; a ~40 KB session that evaluates
# fine today is ~10k tokens and is untouched.
EVALUATION_TRANSCRIPT_TOKEN_LIMIT = 64_000


class AgentSuccessEvaluationRequest(BaseModel):
    """Request schema for agent success evaluation"""

    user_id: str = ""
    session_id: str
    agent_version: str
    request_interaction_data_models: list[RequestInteractionDataModel]
    source: str | None = None
    #: Last judged request in ``(created_at, request_id)`` order, and the number
    #: of interactions judged, stamped onto every verdict this run produces.
    trajectory_through_request_id: str | None = None
    trajectory_interaction_count: int | None = None
    #: PostgreSQL snapshot from the actual input read, never the result write.
    trajectory_visibility_snapshot: str | None = None


def construct_agent_success_evaluation_messages_from_sessions(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    agent_context_prompt: str,
    success_definition_prompt: str,
    tool_can_use: str,
) -> list[dict]:
    """
    Construct LLM messages for agent success evaluation from request interaction groups.

    This function uses the shared message construction interface to build messages
    with a final user prompt specific to agent success evaluation.

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_interaction_data_models: List of request interaction groups to evaluate
        agent_context_prompt: Context about the agent
        success_definition_prompt: Definition of what constitutes agent success
        tool_can_use: Description of tools available to the agent
    Returns:
        list[dict]: List of messages ready for agent success evaluation
    """
    # Configure final user message (after interactions)
    # Note: This evaluation doesn't use a system message, just interactions followed by evaluation prompt
    user_config = PromptConfig(
        prompt_id=AgentSuccessEvaluationConstants.AGENT_SUCCESS_EVALUATION_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "success_definition_prompt": success_definition_prompt,
            "tool_can_use": tool_can_use,
            "interactions": slice_content_by_tokens(
                format_sessions_to_history_string(request_interaction_data_models),
                EVALUATION_TRANSCRIPT_TOKEN_LIMIT,
            ),
        },
    )

    # Extract flat interactions for image attachment
    interactions = extract_interactions_from_request_interaction_data_models(
        request_interaction_data_models
    )

    # Use shared message construction (no system prompt for this use case)
    config = MessageConstructionConfig(
        prompt_manager=prompt_manager,
        system_prompt_config=None,  # No system message needed
        user_prompt_config=user_config,
    )

    return construct_messages_from_interactions(interactions, config)


# F1 cleanup: ``has_shadow_content``, ``format_interactions_for_request``, and
# ``construct_agent_success_evaluation_with_comparison_messages`` were retracted
# along with the session-level shadow comparison branch. Per-turn shadow
# comparison reads ``Interaction.shadow_content`` directly inside the dedicated
# ``services/shadow_comparison/`` judge.
