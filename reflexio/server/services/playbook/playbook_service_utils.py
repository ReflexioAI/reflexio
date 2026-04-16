from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    BlockingIssue,
    UserPlaybook,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.playbook_service_constants import (
    PlaybookServiceConstants,
)
from reflexio.server.services.service_utils import (
    MessageConstructionConfig,
    PromptConfig,
    construct_messages_from_interactions,
    extract_interactions_from_request_interaction_data_models,
    format_sessions_to_history_string,
)

logger = logging.getLogger(__name__)

# ===============================
# Pydantic classes for playbook_extraction_main prompt output schema
# ===============================


class StructuredPlaybookContent(BaseModel):
    """
    Structured representation of a single playbook entry from LLM output.

    Field order matters for autoregressive conditioning: rationale is generated
    first, then trigger, then content is synthesized last as a summary.
    """

    rationale: str | None = Field(
        default=None,
        description="The reasoning behind this playbook entry — generated first for autoregressive conditioning",
    )
    trigger: str | None = Field(
        default=None,
        description="The condition or context when this rule applies",
    )
    blocking_issue: BlockingIssue | None = Field(
        default=None,
        description="Present only when the agent could not complete the user's request due to a capability limitation",
    )
    content: str | None = Field(
        default=None,
        description="The main actionable content of the playbook entry — what to do or what to avoid",
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )

    @property
    def is_structured(self) -> bool:
        """Check if this playbook entry has both a trigger and content."""
        return bool(
            self.trigger
            and self.trigger.strip()
            and self.content
            and self.content.strip()
        )

    @property
    def has_content(self) -> bool:
        """Check if this output contains actual content."""
        return bool(self.content and self.content.strip())


class StructuredPlaybookList(BaseModel):
    """
    Wrapper schema for extracting zero or more playbook entries in a single LLM call.

    The canonical shape is ``{"playbooks": [...]}``. An empty list means the model
    found no valid SOPs in the window. This wrapper exists because OpenAI structured
    output requires a JSON object at the root, so ``list[StructuredPlaybookContent]``
    cannot be used directly as ``response_format``.
    """

    playbooks: list[StructuredPlaybookContent] = Field(
        default_factory=list,
        description="Extracted playbook entries — empty list when no valid SOP was found",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False},
    )


class PlaybookAggregationOutput(BaseModel):
    """
    Output schema for the playbook_aggregation prompt.

    Contains the consolidated playbook entry or null if no new entry should be generated
    (e.g., when it duplicates existing approved playbook).
    """

    playbook: StructuredPlaybookContent | None = Field(
        default=None,
        description="The consolidated playbook entry, or null if no new entry should be generated",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_wrapper_key(cls, data: Any) -> Any:
        """Accept both 'playbook' and legacy 'feedback' as the wrapper key."""
        if isinstance(data, dict) and "feedback" in data and "playbook" not in data:
            data["playbook"] = data.pop("feedback")
        return data

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


def format_structured_fields_for_display(
    structured: StructuredPlaybookContent,
) -> str:
    """
    Format structured metadata fields for display/debug purposes.

    This is NOT for producing content values. Use ensure_playbook_content()
    when you need to obtain the freeform content string.

    Args:
        structured (StructuredPlaybookContent): The structured playbook content

    Returns:
        str: Formatted structured fields string for display
    """
    lines = []

    if structured.trigger:
        lines.append(f'Trigger: "{structured.trigger}"')

    if structured.rationale:
        lines.append(f'Rationale: "{structured.rationale}"')

    if structured.blocking_issue:
        lines.append(
            f"Blocked by: [{structured.blocking_issue.kind.value}] {structured.blocking_issue.details}"
        )

    if not lines and structured.content:
        return structured.content

    return "\n".join(lines)


def ensure_playbook_content(
    playbook_content: str | None,
    structured: StructuredPlaybookContent,
) -> str:
    """
    Return playbook_content if present; legacy fallback from structured fields.

    Args:
        playbook_content (str | None): The freeform content from the LLM
        structured (StructuredPlaybookContent): Structured fields for fallback

    Returns:
        str: The freeform playbook_content, or a formatted fallback from structured fields.
    """
    if playbook_content and playbook_content.strip():
        return playbook_content
    return format_structured_fields_for_display(structured)


class PlaybookGenerationRequest(BaseModel):
    request_id: str
    agent_version: str
    user_id: str | None = None  # for per-user playbook extraction
    source: str | None = None
    rerun_start_time: int | None = None  # Unix timestamp for rerun flows
    rerun_end_time: int | None = None  # Unix timestamp for rerun flows
    playbook_name: str | None = None  # Filter to run only specific extractor
    auto_run: bool = (
        True  # True for regular flow (checks batch_interval), False for rerun/manual
    )
    force_extraction: bool = False  # when True, bypass batch_interval checks


class PlaybookAggregatorRequest(BaseModel):
    agent_version: str
    playbook_name: str
    rerun: bool = False


def construct_playbook_extraction_messages_from_sessions(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    agent_context_prompt: str,
    extraction_definition_prompt: str,
    tool_can_use: str | None = None,
) -> list[dict]:
    """
    Construct LLM messages for playbook extraction from sessions.

    This function uses the shared message construction interface to build messages
    with a system prompt and a final user prompt specific to playbook extraction.

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_interaction_data_models: List of request interaction groups to extract playbook entries from
        agent_context_prompt: Context about the agent for system message
        extraction_definition_prompt: Definition of what the playbook should contain
        tool_can_use: Optional formatted string of tools available to the agent

    Returns:
        list[dict]: List of messages ready for playbook extraction
    """
    # Configure system message (before interactions)
    # Stable content (instructions, examples, definitions) goes in system message for token caching
    system_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_CONTEXT_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "extraction_definition_prompt": extraction_definition_prompt,
            "tool_can_use": tool_can_use or "",
        },
    )

    # Configure final user message (after interactions)
    # Only dynamic per-call data goes in user message
    user_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_PROMPT_ID,
        variables={
            "interactions": format_sessions_to_history_string(
                request_interaction_data_models
            ),
        },
    )

    # Extract flat interactions for message construction
    interactions = extract_interactions_from_request_interaction_data_models(
        request_interaction_data_models
    )

    # Use shared message construction
    config = MessageConstructionConfig(
        prompt_manager=prompt_manager,
        system_prompt_config=system_config,
        user_prompt_config=user_config,
    )

    return construct_messages_from_interactions(interactions, config)


def construct_incremental_playbook_extraction_messages(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    agent_context_prompt: str,
    extraction_definition_prompt: str,
    previously_extracted: list[UserPlaybook] | None = None,
    tool_can_use: str | None = None,
) -> list[dict]:
    """
    Construct LLM messages for incremental playbook extraction.

    Uses incremental prompts that show what previous extractors already found,
    so this extractor focuses on finding additional policies not already covered.

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_interaction_data_models: List of request interaction groups to extract playbook entries from
        agent_context_prompt: Context about the agent for system message
        extraction_definition_prompt: Definition of what the playbook should contain
        previously_extracted: Flattened list of all UserPlaybook from previous extractors
        tool_can_use: Optional formatted string of tools available to the agent

    Returns:
        list[dict]: List of messages ready for incremental playbook extraction
    """
    # Configure system message with incremental prompt
    system_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_CONTEXT_INCREMENTAL_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "extraction_definition_prompt": extraction_definition_prompt,
            "tool_can_use": tool_can_use or "",
        },
    )

    # Format previously extracted entries
    formatted_previously_extracted = ""
    if previously_extracted:
        formatted_previously_extracted = "\n".join(
            [f"- {playbook.content}" for playbook in previously_extracted]
        )
    else:
        formatted_previously_extracted = "(None)"

    # Configure final user message with incremental prompt
    user_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_INCREMENTAL_PROMPT_ID,
        variables={
            "previously_extracted_playbooks": formatted_previously_extracted,
            "interactions": format_sessions_to_history_string(
                request_interaction_data_models
            ),
        },
    )

    # Extract flat interactions for message construction
    interactions = extract_interactions_from_request_interaction_data_models(
        request_interaction_data_models
    )

    # Use shared message construction
    config = MessageConstructionConfig(
        prompt_manager=prompt_manager,
        system_prompt_config=system_config,
        user_prompt_config=user_config,
    )

    return construct_messages_from_interactions(interactions, config)


# ===============================
# Expert content utilities
# ===============================


def has_expert_content(interactions: list[Interaction]) -> bool:
    """Check if any interaction has non-empty expert_content."""
    return any(i.expert_content for i in interactions)


def format_expert_comparison_pairs(
    request_interaction_data_models: list[RequestInteractionDataModel],
) -> str:
    """
    Format interactions with expert_content as agent-vs-expert comparison blocks.

    For each agent interaction that has expert_content, includes the preceding user
    question for context, the agent's actual response, and the expert's ideal response.

    Args:
        request_interaction_data_models: Session data models containing interactions

    Returns:
        str: Formatted comparison pairs string
    """
    interactions = extract_interactions_from_request_interaction_data_models(
        request_interaction_data_models
    )

    pairs: list[str] = []
    pair_num = 0
    for i, interaction in enumerate(interactions):
        if not interaction.expert_content:
            continue

        pair_num += 1
        # Find the preceding user question for context
        user_question = ""
        for j in range(i - 1, -1, -1):
            if interactions[j].role.lower() == "user":
                user_question = interactions[j].content
                break

        parts = [f"=== Comparison {pair_num} ==="]
        if user_question:
            parts.append(f"User Question: ```{user_question}```")
        parts.append(f"Agent Response: ```{interaction.content}```")
        parts.append(f"Expert Response: ```{interaction.expert_content}```")
        pairs.append("\n".join(parts))

    return "\n\n".join(pairs)


def construct_expert_playbook_extraction_messages(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    agent_context_prompt: str,
    extraction_definition_prompt: str,
) -> list[dict]:
    """
    Construct LLM messages for expert-content playbook extraction.

    Uses expert-specific prompts that compare agent responses against expert
    responses and extract playbook entries about alignment gaps.

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_interaction_data_models: Session data with expert_content interactions
        agent_context_prompt: Context about the agent for system message
        extraction_definition_prompt: Definition of what the playbook should contain

    Returns:
        list[dict]: List of messages ready for expert playbook extraction
    """
    system_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_CONTEXT_EXPERT_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "extraction_definition_prompt": extraction_definition_prompt,
        },
    )

    comparison_pairs = format_expert_comparison_pairs(request_interaction_data_models)

    user_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_EXPERT_PROMPT_ID,
        variables={
            "comparison_pairs": comparison_pairs,
        },
    )

    interactions = extract_interactions_from_request_interaction_data_models(
        request_interaction_data_models
    )

    config = MessageConstructionConfig(
        prompt_manager=prompt_manager,
        system_prompt_config=system_config,
        user_prompt_config=user_config,
    )

    return construct_messages_from_interactions(interactions, config)
