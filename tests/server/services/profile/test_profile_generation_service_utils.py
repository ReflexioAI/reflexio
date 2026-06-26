"""Tests for profile generation service utility functions."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.common import NEVER_EXPIRES_TIMESTAMP
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    ProfileTimeToLive,
    Request,
    UserActionType,
    UserProfile,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.profile.profile_generation_service_utils import (
    calculate_expiration_timestamp,
    construct_profile_extraction_messages_from_sessions,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from tests import test_data
from tests.server.test_utils import encode_image_to_base64


def _mime_type_from_image_fixture(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise AssertionError(f"Unsupported test fixture image type: {image_path}")


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
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="user_123",
            request_id="req_1",
            content="I also enjoy sushi",
            role="user",
            created_at=timestamp,
            user_action=UserActionType.CLICK,
            user_action_description="restaurant menu",
        ),
    ]

    # Create request interaction group
    request = Request(
        request_id="req_1",
        user_id="user_123",
        session_id="test_session",
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


def test_profile_extraction_prompt_keeps_image_encoding_after_storage_round_trip():
    timestamp = int(datetime.now(UTC).timestamp())
    user_id = "visual-user"
    request = Request(
        request_id="visual-request",
        user_id=user_id,
        created_at=timestamp,
        source="api",
        agent_version="v1",
        session_id="visual-session",
    )
    image_path = Path(test_data.__file__).parent / "sushi.png"
    expected_mime_type = _mime_type_from_image_fixture(image_path)
    image_encoding = encode_image_to_base64(str(image_path))
    interaction = Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id=request.request_id,
        content="what I had for lunch today",
        role="user",
        created_at=timestamp,
        image_encoding=image_encoding,
    )

    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        storage = SQLiteStorage(org_id="visual-test", db_path=f"{temp_dir}/test.db")
        storage.add_request(request)
        storage.add_user_interaction(user_id, interaction)

        sessions, flat = storage.get_last_k_interactions_grouped(user_id, 10)
        assert flat[0].image_encoding == image_encoding

        messages = construct_profile_extraction_messages_from_sessions(
            prompt_manager=PromptManager(),
            request_interaction_data_models=sessions,
            existing_profiles=[],
            agent_context_prompt="Test agent context",
            context_prompt="Test context",
            extraction_definition_prompt="food preferences",
        )

    user_message = messages[-1]
    assert isinstance(user_message["content"], list)
    image_blocks = [
        block
        for block in user_message["content"]
        if isinstance(block, dict) and block.get("type") == "image_url"
    ]
    assert len(image_blocks) == 1
    assert (
        image_blocks[0]["image_url"]["url"]
        == f"data:{expected_mime_type};base64,{image_encoding}"
    )


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
    )

    # Should still create messages (system message + user message with prompt)
    assert len(messages) > 0, "No messages were created for empty sessions"


def test_calculate_expiration_timestamp_infinity_returns_sentinel():
    """Infinity TTL must return the NEVER_EXPIRES_TIMESTAMP sentinel (Jan 1 2100),
    not a `datetime.max`-derived year-9999 integer that would render as
    'Jan 1, 10000' after timezone conversion on the frontend.
    """
    now = int(datetime.now(UTC).timestamp())
    assert (
        calculate_expiration_timestamp(now, ProfileTimeToLive.INFINITY)
        == NEVER_EXPIRES_TIMESTAMP
    )


@pytest.mark.parametrize(
    "ttl, expected_delta_seconds",
    [
        (ProfileTimeToLive.ONE_DAY, 1 * 24 * 3600),
        (ProfileTimeToLive.ONE_WEEK, 7 * 24 * 3600),
        (ProfileTimeToLive.ONE_MONTH, 30 * 24 * 3600),
        (ProfileTimeToLive.ONE_QUARTER, 90 * 24 * 3600),
        (ProfileTimeToLive.ONE_YEAR, 365 * 24 * 3600),
    ],
)
def test_calculate_expiration_timestamp_finite_ttls(ttl, expected_delta_seconds):
    """Finite TTLs must shift last_modified forward by their documented delta."""
    now = int(datetime.now(UTC).timestamp())
    expiration = calculate_expiration_timestamp(now, ttl)
    assert expiration == now + expected_delta_seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
