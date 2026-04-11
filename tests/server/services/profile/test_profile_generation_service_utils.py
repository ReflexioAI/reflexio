"""Tests for profile generation service utility functions."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserProfile,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.profile.profile_generation_service_utils import (
    construct_profile_extraction_messages_from_sessions,
)


def test_construct_profile_extraction_messages_with_sessions():
    """Test that construct_profile_extraction_messages_from_sessions formats interactions correctly in the rendered prompt."""
    # Create test interactions with both content and actions
    timestamp = int(datetime.now(UTC).timestamp())
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="user_123",
            request_id="req_1",
            content="I love Italian food",
            role="user",
            created_at=timestamp,
            user_action="none",
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="user_123",
            request_id="req_1",
            content="I also enjoy sushi",
            role="user",
            created_at=timestamp,
            user_action="click",
            user_action_description="restaurant menu",
        ),
    ]

    # Create request interaction group
    request = Request(
        request_id="req_1",
        user_id="user_123",
        created_at=timestamp,
    )
    sessions = [
        RequestInteractionDataModel(
            session_id="session_1",
            request=request,
            interactions=interactions,
        )
    ]

    # Create existing profiles
    existing_profiles = [
        UserProfile(
            profile_id="profile_1",
            user_id="user_123",
            content="likes Mexican food",
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_0",
        )
    ]

    # Create prompt manager
    prompt_manager = PromptManager()

    # Call the function
    messages = construct_profile_extraction_messages_from_sessions(
        prompt_manager=prompt_manager,
        request_interaction_data_models=sessions,
        existing_profiles=existing_profiles,
        agent_context_prompt="Test agent context",
        context_prompt="Test context",
        extraction_definition_prompt="food preferences",
        metadata_definition_prompt="cuisine type",
    )

    # Validate that messages were created
    assert len(messages) > 0, "No messages were created"

    # Find the user message that contains the interactions
    found_interactions = False
    for message in messages:
        # Messages are dicts with 'role' and 'content' keys
        if isinstance(message, dict) and "content" in message:
            # Content can be a string or a list of content blocks
            content = message.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                extracted_text = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        extracted_text += item.get("text", "")
                content = extracted_text
            else:
                content = str(content)

            # Check if this message contains the interaction section
            if (
                "[Interaction start]" in content
                or "User and agent interactions:" in content
                or "=== Session:" in content
                or "user: ```I love Italian food```"
                in content  # Check directly for content
            ):
                # Validate the interactions are formatted correctly in the rendered prompt
                assert "user: ```I love Italian food```" in content, (
                    "Expected 'user: ```I love Italian food```' in prompt"
                )
                assert "user: ```I also enjoy sushi```" in content, (
                    "Expected 'user: ```I also enjoy sushi```' in prompt"
                )
                assert "user: ```click restaurant menu```" in content, (
                    "Expected 'user: ```click restaurant menu```' in prompt"
                )

                # Also verify existing profiles are in the prompt
                assert "likes Mexican food" in content, (
                    "Expected existing profile in prompt"
                )

                found_interactions = True
                break

    assert found_interactions, "Did not find interactions in the rendered prompt"


def test_construct_profile_extraction_messages_with_empty_sessions():
    """Test that construct_profile_extraction_messages_from_sessions handles empty sessions."""
    # Empty sessions list
    sessions = []

    # Create prompt manager
    prompt_manager = PromptManager()

    # Call the function
    messages = construct_profile_extraction_messages_from_sessions(
        prompt_manager=prompt_manager,
        request_interaction_data_models=sessions,
        existing_profiles=[],
        agent_context_prompt="Test agent context",
        context_prompt="Test context",
        extraction_definition_prompt="food preferences",
        metadata_definition_prompt="cuisine type",
    )

    # Should still create messages (system message + user message with prompt)
    assert len(messages) > 0, "No messages were created for empty sessions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
