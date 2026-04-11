"""Tests for agent success evaluation utility functions."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    construct_agent_success_evaluation_messages_from_sessions,
)


def test_construct_agent_success_evaluation_messages_with_sessions():
    """Test that construct_agent_success_evaluation_messages_from_sessions formats interactions correctly in the rendered prompt."""
    # Create test interactions with both content and actions
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="user_123",
            request_id="req_1",
            content="The agent helped me complete my task successfully",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="none",
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="user_123",
            request_id="req_1",
            content="I used the search tool",
            role="assistant",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="none",
            user_action_description="",
        ),
        Interaction(
            interaction_id=3,
            user_id="user_123",
            request_id="req_1",
            content="Great!",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="click",
            user_action_description="search button",
        ),
    ]

    # Create test request
    test_request = Request(
        request_id="req_1",
        user_id="user_123",
        source="test",
        agent_version="v1.0",
        session_id="test_group",
        created_at=int(datetime.now(UTC).timestamp()),
    )

    # Create RequestInteractionDataModel
    request_interaction_data_models = [
        RequestInteractionDataModel(
            request=test_request,
            interactions=interactions,
            session_id="test_group",
        )
    ]

    # Create prompt manager
    prompt_manager = PromptManager()

    # Call the function
    messages = construct_agent_success_evaluation_messages_from_sessions(
        prompt_manager=prompt_manager,
        request_interaction_data_models=request_interaction_data_models,
        agent_context_prompt="Test agent context",
        success_definition_prompt="Evaluate if the agent successfully completed the task",
        tool_can_use="search, calculator",
        metadata_definition_prompt="Include tool usage statistics",
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
                "[Interactions]" in content
                or "User and agent interactions:" in content
                or "user: ```The agent helped me complete my task successfully```"
                in content  # Check directly
            ):
                # Validate the interactions are formatted correctly in the rendered prompt
                assert (
                    "user: ```The agent helped me complete my task successfully```"
                    in content
                ), (
                    "Expected 'user: ```The agent helped me complete my task successfully```' in prompt"
                )
                assert "assistant: ```I used the search tool```" in content, (
                    "Expected 'assistant: ```I used the search tool```' in prompt"
                )
                assert "user: ```Great!```" in content, (
                    "Expected 'user: ```Great!```' in prompt"
                )
                assert "user: ```click search button```" in content, (
                    "Expected 'user: ```click search button```' in prompt"
                )

                # Also verify success definition and tools are in the content
                assert (
                    "Evaluate if the agent successfully completed the task" in content
                ), "Expected success definition in prompt"
                assert "search, calculator" in content, "Expected tools in prompt"

                found_interactions = True
                break

    assert found_interactions, "Did not find interactions in the rendered prompt"


def test_construct_agent_success_evaluation_messages_with_empty_sessions():
    """Test that construct_agent_success_evaluation_messages_from_sessions handles empty sessions."""
    # Empty sessions list
    request_interaction_data_models = []

    # Create prompt manager
    prompt_manager = PromptManager()

    # Call the function
    messages = construct_agent_success_evaluation_messages_from_sessions(
        prompt_manager=prompt_manager,
        request_interaction_data_models=request_interaction_data_models,
        agent_context_prompt="Test agent context",
        success_definition_prompt="Evaluate if the agent successfully completed the task",
        tool_can_use="search, calculator",
        metadata_definition_prompt="Include tool usage statistics",
    )

    # Should still create messages (user message with prompt)
    assert len(messages) > 0, "No messages were created for empty sessions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
