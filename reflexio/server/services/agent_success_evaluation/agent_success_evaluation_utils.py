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
)


class AgentSuccessEvaluationRequest(BaseModel):
    """Request schema for agent success evaluation"""

    user_id: str = ""
    session_id: str
    agent_version: str
    request_interaction_data_models: list[RequestInteractionDataModel]
    source: str | None = None


def construct_agent_success_evaluation_messages_from_sessions(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    agent_context_prompt: str,
    success_definition_prompt: str,
    tool_can_use: str,
    metadata_definition_prompt: str | None = None,
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
        metadata_definition_prompt: Optional additional metadata definition

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
            "metadata_definition_prompt": metadata_definition_prompt or "",
            "interactions": format_sessions_to_history_string(
                request_interaction_data_models
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
