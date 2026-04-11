"""Utility functions for agent success evaluation service"""

from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, UserActionType
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


def has_shadow_content(interactions: list[Interaction]) -> bool:
    """
    Check if any interaction in the list has shadow_content.

    Args:
        interactions: List of interactions to check

    Returns:
        True if any interaction has non-empty shadow_content, False otherwise
    """
    return any(i.shadow_content for i in interactions)


def format_interactions_for_request(
    interactions: list[Interaction], use_shadow: bool = False
) -> str:
    """
    Format interactions as a string, optionally using shadow_content instead of content.

    For each interaction:
    - If use_shadow=True and shadow_content exists, use shadow_content
    - Otherwise, use the regular content

    Args:
        interactions: List of interactions to format
        use_shadow: If True, use shadow_content when available

    Returns:
        Formatted string with each interaction on a new line
    """
    formatted = []
    for interaction in interactions:
        # Determine which content to use
        if use_shadow and interaction.shadow_content:
            content = interaction.shadow_content
        else:
            content = interaction.content

        # Add content if present
        if content:
            formatted.append(f"{interaction.role}: {content}")

        # Add user action if present
        if interaction.user_action != UserActionType.NONE:
            formatted.append(
                f"{interaction.role}: {interaction.user_action.value} {interaction.user_action_description}"
            )

    return "\n".join(formatted)


def construct_agent_success_evaluation_with_comparison_messages(
    prompt_manager: PromptManager,
    request_1_interactions: str,
    request_2_interactions: str,
    agent_context_prompt: str,
    success_definition_prompt: str,
    tool_can_use: str,
    metadata_definition_prompt: str | None = None,
    interactions_for_images: list[Interaction] | None = None,
) -> list[dict]:
    """
    Construct LLM messages for combined agent success evaluation with comparison.

    This function builds a prompt that:
    1. Evaluates Request 1 for success (is_success, failure_type, etc.)
    2. Compares Request 1 vs Request 2 to determine which is better

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_1_interactions: Formatted interactions for Request 1
        request_2_interactions: Formatted interactions for Request 2
        agent_context_prompt: Context about the agent
        success_definition_prompt: Definition of what constitutes agent success
        tool_can_use: Description of tools available to the agent
        metadata_definition_prompt: Optional additional metadata definition
        interactions_for_images: Optional list of interactions for image attachment

    Returns:
        list[dict]: List of messages ready for combined evaluation
    """
    user_config = PromptConfig(
        prompt_id=AgentSuccessEvaluationConstants.AGENT_SUCCESS_EVALUATION_WITH_COMPARISON_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "success_definition_prompt": success_definition_prompt,
            "request_1_interactions": request_1_interactions,
            "request_2_interactions": request_2_interactions,
            "tool_can_use": tool_can_use,
            "metadata_definition_prompt": metadata_definition_prompt or "",
        },
    )

    # Use shared message construction (no system prompt for this use case)
    config = MessageConstructionConfig(
        prompt_manager=prompt_manager,
        system_prompt_config=None,
        user_prompt_config=user_config,
    )

    return construct_messages_from_interactions(interactions_for_images or [], config)
