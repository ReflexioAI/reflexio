"""Utility functions for the profile generation service"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    ProfileTimeToLive,
    UserProfile,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.service_utils import (
    MessageConstructionConfig,
    PromptConfig,
    construct_messages_from_interactions,
    extract_interactions_from_request_interaction_data_models,
    format_sessions_to_history_string,
)


class ProfileUpdates(BaseModel):
    add_profiles: list[UserProfile] = []
    delete_profiles: list[UserProfile] = []
    mention_profiles: list[UserProfile] = []

    @field_validator(
        "add_profiles", "delete_profiles", "mention_profiles", mode="before"
    )
    @classmethod
    def coerce_none_to_list(cls, v: Any) -> list[UserProfile]:
        return v if v is not None else []


class ProfileGenerationRequest(BaseModel):
    user_id: str
    request_id: str
    source: str | None = None
    extractor_names: list[str] | None = None
    rerun_start_time: int | None = None  # Unix timestamp for rerun flows
    rerun_end_time: int | None = None  # Unix timestamp for rerun flows
    auto_run: bool = (
        True  # True for regular flow (checks batch_interval), False for rerun/manual
    )
    force_extraction: bool = False  # when True, bypass batch_interval checks


@dataclass(frozen=True)
class ProfileGenerationServiceConstants:
    PROFILE_EXTRACTORS_CONFIG_NAME = "profile_extractor_configs"
    # ===============================
    # prompt ids
    # ===============================
    PROFILE_SHOULD_GENERATE_PROMPT_ID = "profile_should_generate"
    PROFILE_SHOULD_GENERATE_OVERRIDE_PROMPT_ID = "profile_should_generate_override"
    PROFILE_UPDATE_MAIN_PROMPT_ID = "profile_update_main"
    PROFILE_UPDATE_INSTRUCTION_START_PROMPT_ID = "profile_update_instruction_start"
    PROFILE_UPDATE_INSTRUCTION_PROMPT_ID = "profile_update_instruction"
    PROFILE_UPDATE_INSTRUCTION_INCREMENTAL_PROMPT_ID = (
        "profile_update_instruction_incremental"
    )
    PROFILE_UPDATE_MAIN_INCREMENTAL_PROMPT_ID = "profile_update_main_incremental"


# ===============================
# Pydantic classes for profile_update_main prompt output schema
# ===============================


class ProfileAddItem(BaseModel):
    """
    Schema for a single profile item to be added.

    Attributes:
        content (str): The profile content based on content definition
        time_to_live (str): Time to live for the profile - one of: 'one_day', 'one_week', 'one_month', 'one_quarter', 'one_year', 'infinity'
        metadata (str, optional): Metadata extracted for the profile based on metadata definition
    """

    content: str = Field(description="Profile content based on content definition")
    time_to_live: Literal[
        "one_day", "one_week", "one_month", "one_quarter", "one_year", "infinity"
    ] = Field(
        description="Time to live for the profile - determines when the profile expires"
    )
    metadata: str | None = Field(
        default=None,
        description="Metadata extracted for the profile based on metadata definition",
    )

    # OpenAI structured output requires explicit schema constraints
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class ProfileUpdateOutput(BaseModel):
    """
    Legacy output schema for profile_update_main prompt (kept for backward compatibility).
    Represents the complete set of profile updates including additions, deletions, and mentions.

    Attributes:
        add (list[ProfileAddItem], optional): List of new profiles to be added
        delete (list[str], optional): List of existing profile contents to be deleted
        mention (list[str], optional): List of existing profile contents that were mentioned/referenced
    """

    add: list[ProfileAddItem] | None = Field(
        default=None,
        description="List of new profiles to be added with their content, time to live, and optional metadata",
    )
    delete: list[str] | None = Field(
        default=None,
        description="List of existing profile contents to be deleted (profiles that are contradicted by new interactions)",
    )
    mention: list[str] | None = Field(
        default=None,
        description="List of existing profile contents that were mentioned or referenced in the new interactions",
    )

    # OpenAI schema parsing requires explicitly forbidding additional properties
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class StructuredProfilesOutput(BaseModel):
    """
    Output schema for extraction-only profile extraction.
    Only extracts profiles — no delete/mention operations.

    Attributes:
        profiles (list[ProfileAddItem], optional): List of extracted profiles with content, time_to_live, and optional metadata
    """

    profiles: list[ProfileAddItem] | None = Field(
        default=None,
        description="List of extracted profiles with content, time_to_live, and optional metadata",
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


def calculate_expiration_timestamp(
    last_modified_timestamp: int, profile_time_to_live: ProfileTimeToLive
) -> int:
    """
    Calculate the expiration timestamp for a profile based on the last modified timestamp and the profile time to live.
    Args:
        last_modified_timestamp (int): The timestamp of the last modification of the profile.
        profile_time_to_live (ProfileTimeToLive): The time to live of the profile.
    Returns:
        The expiration timestamp for the profile.
    """
    expiration_timestamp = datetime.max
    last_modified_datetime = datetime.fromtimestamp(last_modified_timestamp)

    if profile_time_to_live == ProfileTimeToLive.ONE_DAY:
        expiration_timestamp = last_modified_datetime + timedelta(days=1)
    elif profile_time_to_live == ProfileTimeToLive.ONE_WEEK:
        expiration_timestamp = last_modified_datetime + timedelta(days=7)
    elif profile_time_to_live == ProfileTimeToLive.ONE_MONTH:
        expiration_timestamp = last_modified_datetime + timedelta(days=30)
    elif profile_time_to_live == ProfileTimeToLive.ONE_QUARTER:
        expiration_timestamp = last_modified_datetime + timedelta(days=90)
    elif profile_time_to_live == ProfileTimeToLive.ONE_YEAR:
        expiration_timestamp = last_modified_datetime + timedelta(days=365)
    elif profile_time_to_live == ProfileTimeToLive.INFINITY:
        expiration_timestamp = datetime.max
    else:
        raise ValueError(f"Invalid profile time to live: {profile_time_to_live}")
    try:
        return int(expiration_timestamp.timestamp())
    except (OverflowError, OSError, ValueError):
        import sys

        return sys.maxsize


def check_string_token_overlap(str1: str, str2: str, threshold: float = 0.7) -> bool:
    """
    Check if two strings have significant token overlap, indicating they might be referring to the same thing.
    This is useful for fuzzy matching when exact string matching is too strict.

    Args:
        str1 (str): First string to compare
        str2 (str): Second string to compare
        threshold (float): Minimum overlap ratio required to consider strings as matching (0.0 to 1.0)

    Returns:
        bool: True if strings have significant overlap, False otherwise
    """
    # Normalize strings: lowercase and split into tokens
    tokens1 = set(str1.lower().split())
    tokens2 = set(str2.lower().split())

    # Calculate overlap
    if not tokens1 or not tokens2:
        return False

    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))

    # Calculate overlap ratio
    overlap_ratio = max(intersection / len(tokens1), intersection / len(tokens2))

    return overlap_ratio >= threshold


def construct_profile_extraction_messages_from_sessions(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    existing_profiles: list[UserProfile],
    agent_context_prompt: str,
    context_prompt: str,
    extraction_definition_prompt: str,
    metadata_definition_prompt: str | None = None,
) -> list[dict]:
    """
    Construct LLM messages for profile extraction from sessions.

    This function uses the shared message construction interface to build messages
    with a system prompt and a final user prompt specific to profile extraction.
    Interactions are formatted grouped by session.

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_interaction_data_models: List of request interaction groups to extract profiles from
        existing_profiles: List of existing user profiles for context
        agent_context_prompt: Context about the agent for system message
        context_prompt: Additional context for system message
        extraction_definition_prompt: Definition of what profiles should contain
        metadata_definition_prompt: Optional definition for profile metadata

    Returns:
        list[dict]: List of messages ready for profile extraction
    """
    # Format existing profiles for the final prompt
    formatted_existing_profiles = ", ".join(
        [profile.content for profile in existing_profiles]
    )

    # Configure system message (before interactions)
    # Stable content (instructions, examples, definitions) goes in system message for token caching
    system_config = PromptConfig(
        prompt_id=ProfileGenerationServiceConstants.PROFILE_UPDATE_INSTRUCTION_START_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "context_prompt": context_prompt,
            "extraction_definition_prompt": extraction_definition_prompt,
            "metadata_definition_prompt": metadata_definition_prompt,
        },
    )

    # Configure final user message (after interactions)
    # Only dynamic per-call data goes in user message
    user_config = PromptConfig(
        prompt_id=ProfileGenerationServiceConstants.PROFILE_UPDATE_MAIN_PROMPT_ID,
        variables={
            "existing_profiles": formatted_existing_profiles,
            "interactions": format_sessions_to_history_string(
                request_interaction_data_models
            ),
        },
    )

    # Extract flat interactions for message construction (needed for image handling)
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


def construct_incremental_profile_extraction_messages(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    existing_profiles: list[UserProfile],
    agent_context_prompt: str,
    context_prompt: str,
    extraction_definition_prompt: str,
    previously_extracted: list[list[UserProfile]],
    metadata_definition_prompt: str | None = None,
) -> list[dict]:
    """
    Construct LLM messages for incremental profile extraction.

    Uses incremental prompts that show what previous extractors already found,
    so this extractor focuses on finding additional information not already covered.

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_interaction_data_models: List of request interaction groups to extract profiles from
        existing_profiles: List of existing user profiles for context (refreshed from storage)
        agent_context_prompt: Context about the agent for system message
        context_prompt: Additional context for system message
        extraction_definition_prompt: Definition of what profiles should contain
        previously_extracted: List of profile lists from all previous extractors
        metadata_definition_prompt: Optional definition for profile metadata

    Returns:
        list[dict]: List of messages ready for incremental profile extraction
    """
    # Format existing profiles
    formatted_existing_profiles = ", ".join(
        [profile.content for profile in existing_profiles]
    )

    # Format previously extracted profiles
    previously_added = []
    for profile_list in previously_extracted:
        previously_added.extend(profile.content for profile in profile_list)

    formatted_previously_added = (
        "\n".join(f"- {content}" for content in previously_added)
        if previously_added
        else "(None)"
    )

    # Configure system message with incremental prompt
    system_config = PromptConfig(
        prompt_id=ProfileGenerationServiceConstants.PROFILE_UPDATE_INSTRUCTION_INCREMENTAL_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "context_prompt": context_prompt,
            "extraction_definition_prompt": extraction_definition_prompt,
            "metadata_definition_prompt": metadata_definition_prompt,
        },
    )

    # Configure final user message with incremental prompt
    user_config = PromptConfig(
        prompt_id=ProfileGenerationServiceConstants.PROFILE_UPDATE_MAIN_INCREMENTAL_PROMPT_ID,
        variables={
            "existing_profiles": formatted_existing_profiles,
            "previously_added_profiles": formatted_previously_added,
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
