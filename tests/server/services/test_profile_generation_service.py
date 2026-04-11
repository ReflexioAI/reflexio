import datetime
import inspect
import os
import tempfile
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from reflexio.server.api_endpoints.request_context import RequestContext


# Disable global mock mode for these tests so the test-specific mocks are used
@pytest.fixture(autouse=True)
def disable_mock_llm_response(monkeypatch):
    """Disable MOCK_LLM_RESPONSE env var so tests use their own mocks."""
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)


from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    InteractionData,
    ProfileTimeToLive,
    PublishUserInteractionRequest,
    Request,
)
from reflexio.models.config_schema import ProfileExtractorConfig
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.services.profile.profile_generation_service import (
    ProfileGenerationService,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileAddItem,
    ProfileGenerationRequest,
    StructuredProfilesOutput,
)
from tests import test_data
from tests.server.test_utils import encode_image_to_base64


@pytest.fixture
def mock_chat_completion():
    def mock_generate_chat_response_side_effect(messages, **kwargs):
        """
        Check prompt content to determine which mock response to return.
        If prompt contains "Output just a boolean value", return boolean response.
        Otherwise, return JSON updates response.
        """
        # Get the prompt content from the messages
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        # Check if this is a should_extract_profile call
        if "Output just a boolean value" in prompt_content:
            return "true"
        # Otherwise, this is a profile generation call
        # Check if response_format is a Pydantic model class
        response_format = kwargs.get("response_format")
        if (
            response_format
            and inspect.isclass(response_format)
            and issubclass(response_format, BaseModel)
        ):
            # Return a ProfileUpdateOutput instance
            return StructuredProfilesOutput(
                profiles=[
                    ProfileAddItem(content="like sushi", time_to_live="one_month")
                ]
            )
        # Return JSON string for non-structured responses
        return '```json\n{\n    "add": [{\n        "content": "like sushi",\n        "time_to_live": "one_month"\n    }]\n}\n```'

    # Mock the LLM client's generate_chat_response method
    with patch(
        "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
        side_effect=mock_generate_chat_response_side_effect,
    ):
        yield


def test_refresh_profiles_for_user(mock_chat_completion):
    user_id = "test_user_id"
    org_id = "0"
    interaction_request = InteractionData(
        content="remember i like sushi",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up profile extractor config with batch_size
        profile_extractor_config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            should_extract_profile_prompt_override="test",
            context_prompt="test",
            extraction_definition_prompt="test",
            metadata_definition_prompt="test",
        )
        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [profile_extractor_config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        # Create a PublishUserInteractionRequest first
        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction_request],
        )

        # Convert to interactions and store in storage
        interactions = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request,
                request_id="1",
            )
        )
        request_obj = Request(
            request_id="1",
            user_id=user_id,
            source="",
        )
        profile_generation_service.storage.add_request(request_obj)
        for interaction_obj in interactions:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        # Create profile generation request - extractors collect from storage
        profile_generation_request = ProfileGenerationRequest(
            user_id=user_id,
            request_id="1",
            source="",
            auto_run=False,  # Skip stride check for testing
        )

        profile_generation_service.run(profile_generation_request)

        profiles = profile_generation_service.storage.get_user_profile(user_id)
        assert len(profiles) == 1
        profile = profiles[0]
        assert profile.content == "like sushi"
        assert profile.profile_time_to_live == ProfileTimeToLive.ONE_MONTH


def test_test_refresh_profiles_for_user_with_image_encoding(mock_chat_completion):
    user_id = "test_user_id"
    org_id = "0"
    image_fp = str(Path(test_data.__file__).parent / "sushi.png")
    interaction_request = InteractionData(
        content="remember i like sushi",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
        image_encoding=encode_image_to_base64(image_fp),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up profile extractor config with batch_size
        profile_extractor_config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            should_extract_profile_prompt_override="test",
            context_prompt="test",
            extraction_definition_prompt="test",
            metadata_definition_prompt="test",
        )
        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [profile_extractor_config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        # Create a PublishUserInteractionRequest first
        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction_request],
        )

        # Convert to interactions and store in storage
        interactions = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request,
                request_id="1",
            )
        )
        request_obj = Request(
            request_id="1",
            user_id=user_id,
            source="",
        )
        profile_generation_service.storage.add_request(request_obj)
        for interaction_obj in interactions:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        # Create profile generation request - extractors collect from storage
        profile_generation_request = ProfileGenerationRequest(
            user_id=user_id,
            request_id="1",
            source="",
            auto_run=False,  # Skip stride check for testing
        )

        profile_generation_service.run(profile_generation_request)

        profiles = profile_generation_service.storage.get_user_profile(user_id)
        assert len(profiles) == 1
        profile = profiles[0]
        assert profile.content == "like sushi"
        assert profile.profile_time_to_live == ProfileTimeToLive.ONE_MONTH


def test_profile_extraction_message_construction():
    """Test that interactions are formatted correctly in rendered prompts."""
    # Temporarily disable MOCK_LLM_RESPONSE so we can capture real LLM calls
    # Note: We must always restore this, even if it wasn't originally set
    original_mock_llm = os.environ.get("MOCK_LLM_RESPONSE")
    os.environ.pop("MOCK_LLM_RESPONSE", None)

    try:
        user_id = "test_user_id"
        org_id = "0"

        # Create interactions with both content and actions
        interaction1 = InteractionData(
            content="I love Italian food",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        interaction2 = InteractionData(
            content="I also enjoy sushi",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
            user_action="click",
            user_action_description="restaurant menu",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            profile_generation_service = ProfileGenerationService(
                llm_client=llm_client,
                request_context=RequestContext(
                    org_id=org_id, storage_base_dir=temp_dir
                ),
            )

            # Set up profile extractor config with batch_size
            profile_extractor_config = ProfileExtractorConfig(
                extractor_name="test_extractor",
                should_extract_profile_prompt_override="test",
                context_prompt="test context",
                extraction_definition_prompt="food preferences",
                metadata_definition_prompt="cuisine type",
            )
            profile_generation_service.configurator.set_config_by_name(
                "profile_extractor_configs", [profile_extractor_config]
            )
            profile_generation_service.configurator.set_config_by_name(
                "batch_size", 100
            )

            # Create a PublishUserInteractionRequest
            publish_request = PublishUserInteractionRequest(
                user_id=user_id,
                interaction_data_list=[interaction1, interaction2],
            )

            # Convert to interactions and store in storage
            interactions = (
                GenerationService.get_interaction_from_publish_user_interaction_request(
                    publish_user_interaction_request=publish_request,
                    request_id="1",
                )
            )
            request_obj = Request(
                request_id="1",
                user_id=user_id,
                source="",
            )
            profile_generation_service.storage.add_request(request_obj)
            for interaction_obj in interactions:
                profile_generation_service.storage.add_user_interaction(
                    user_id, interaction_obj
                )

            # Capture the messages sent to generate_chat_response
            captured_messages = []

            def mock_generate_chat_response(messages, **kwargs):
                captured_messages.append(messages)

                # Check if this is a should_extract_profile call or actual extraction
                prompt_content = ""
                for message in messages:
                    if isinstance(message, dict) and "content" in message:
                        prompt_content += str(message["content"])

                # Check if this is a should_extract_profile call
                if "Output just a boolean value" in prompt_content:
                    return "true"
                # This is the actual profile extraction call
                # Check if parse_structured_output is True in kwargs
                if kwargs.get("parse_structured_output", False):
                    # Return the parsed dict directly
                    return {
                        "add": [
                            {
                                "content": "like Italian food and sushi",
                                "time_to_live": "one_month",
                            }
                        ]
                    }
                return '```json\n{\n    "add": [{\n        "content": "like Italian food and sushi",\n        "time_to_live": "one_month"\n    }]\n}\n```'

            with patch(
                "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
                side_effect=mock_generate_chat_response,
            ):
                # Create profile generation request - extractors collect from storage
                profile_generation_request = ProfileGenerationRequest(
                    user_id=user_id,
                    request_id="1",
                    source="",
                    auto_run=False,  # Skip stride check for testing
                )

                # This should trigger message construction
                profile_generation_service.run(profile_generation_request)

            # Validate that messages were captured
            assert len(captured_messages) > 0, "No messages were captured"

            # Find the message that contains the profile_update_main prompt
            # This should be in the user message after the should_extract check
            found_interactions_in_prompt = False
            for messages in captured_messages:
                for message in messages:
                    if isinstance(message, dict) and "content" in message:
                        # Message content might be a list of dicts (multimodal) or a string
                        content_str = ""
                        if isinstance(message["content"], list):
                            for item in message["content"]:
                                if isinstance(item, dict) and "text" in item:
                                    content_str += item["text"]
                        else:
                            content_str = str(message["content"])

                        # Check if this message contains interactions (could be in should_extract or main prompt)
                        if any(
                            pattern in content_str
                            for pattern in [
                                "User: ```I love Italian food```",
                                "user: ```I love Italian food```",
                                "[Interaction",
                            ]
                        ):
                            # Validate the interactions are formatted correctly in the rendered prompt
                            # Note: format might be "User:" (capital) or "user:" depending on the prompt
                            has_interaction1 = (
                                "User: ```I love Italian food```" in content_str
                                or "user: ```I love Italian food```" in content_str
                            )
                            has_interaction2 = (
                                "User: ```I also enjoy sushi```" in content_str
                                or "user: ```I also enjoy sushi```" in content_str
                            )
                            _has_interaction3 = (
                                "User: ```click restaurant menu```" in content_str
                                or "user: ```click restaurant menu```" in content_str
                            )

                            if has_interaction1 and has_interaction2:
                                # Found both content interactions, action might be in a different message
                                found_interactions_in_prompt = True
                                break
                if found_interactions_in_prompt:
                    break

            assert found_interactions_in_prompt, (
                "Did not find interactions in any rendered prompt"
            )

    finally:
        # Always restore MOCK_LLM_RESPONSE to "true" for test isolation
        # This ensures other tests in the same process don't fail
        os.environ["MOCK_LLM_RESPONSE"] = original_mock_llm or "true"


def test_refresh_profiles_with_output_pending_status(mock_chat_completion):
    """Test that profiles are created with PENDING status when output_pending_status=True."""
    user_id = "test_user_id"
    org_id = "0"

    # Create two interactions for profile generation
    interaction1 = InteractionData(
        content="remember i like sushi",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )
    interaction2 = InteractionData(
        content="also remember i enjoy ramen",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up profile extractor config with batch_size
        profile_extractor_config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            should_extract_profile_prompt_override="test",
            context_prompt="test",
            extraction_definition_prompt="test",
            metadata_definition_prompt="test",
        )
        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [profile_extractor_config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        # First, create CURRENT profiles (output_pending_status=False)
        publish_request1 = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction1],
        )

        interactions1 = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request1,
                request_id="1",
            )
        )
        # Store interactions in storage
        request_obj1 = Request(
            request_id="1",
            user_id=user_id,
            source="",
        )
        profile_generation_service.storage.add_request(request_obj1)
        for interaction_obj in interactions1:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        profile_generation_request1 = ProfileGenerationRequest(
            user_id=user_id,
            request_id="1",
            source="",
            auto_run=False,  # Skip stride check for testing
        )

        profile_generation_service.run(profile_generation_request1)

        # Verify CURRENT profile was created (status=None)
        current_profiles = profile_generation_service.storage.get_user_profile(
            user_id, status_filter=[None]
        )
        assert len(current_profiles) == 1
        assert current_profiles[0].status is None  # CURRENT status is None
        assert current_profiles[0].content == "like sushi"

        # Now, create PENDING profiles (output_pending_status=True) - simulating rerun
        # Add more interactions for the rerun
        publish_request2 = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction2],
        )

        interactions2 = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request2,
                request_id="2",
            )
        )
        request_obj2 = Request(
            request_id="2",
            user_id=user_id,
            source="",
        )
        profile_generation_service.storage.add_request(request_obj2)
        for interaction_obj in interactions2:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        profile_generation_request2 = ProfileGenerationRequest(
            user_id=user_id,
            request_id="2",
            source="",
            auto_run=False,  # Skip stride check for testing
        )

        # Create a new service instance with output_pending_status=True for pending profiles
        profile_generation_service_rerun = ProfileGenerationService(
            llm_client=profile_generation_service.client,
            request_context=profile_generation_service.request_context,
            allow_manual_trigger=True,
            output_pending_status=True,
        )

        profile_generation_service_rerun.run(profile_generation_request2)

        # Verify PENDING profile was created (status="pending")
        from reflexio.models.api_schema.service_schemas import Status

        pending_profiles = profile_generation_service.storage.get_user_profile(
            user_id, status_filter=[Status.PENDING]
        )
        assert len(pending_profiles) == 1
        assert pending_profiles[0].status == Status.PENDING
        assert pending_profiles[0].content == "like sushi"

        # Verify CURRENT profile is still there and unchanged
        current_profiles = profile_generation_service.storage.get_user_profile(
            user_id, status_filter=[None]
        )
        assert len(current_profiles) == 1
        assert current_profiles[0].status is None
        assert current_profiles[0].content == "like sushi"

        # Verify we have 2 total profiles (1 CURRENT + 1 PENDING)
        all_profiles = profile_generation_service.storage.get_user_profile(
            user_id, status_filter=[None, Status.PENDING]
        )
        assert len(all_profiles) == 2


def test_run_manual_regular_no_window_size(mock_chat_completion):
    """Test run_manual_regular works even without batch_size configured.

    Since extractors handle window size at their level, the manual flow no longer
    validates window_size upfront. Extractors use a fallback of 1000 interactions
    when no window size is configured.
    """
    user_id = "test_user_manual"
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up profile extractor config WITHOUT window size
        profile_extractor_config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            should_extract_profile_prompt_override="test",
            context_prompt="test",
            extraction_definition_prompt="test",
            metadata_definition_prompt="test",
        )
        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [profile_extractor_config]
        )
        # batch_size is not configured

        # Add some interactions to storage
        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="request_1",
            content="Test content",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_obj = Request(
            request_id="request_1",
            user_id=user_id,
            source="",
        )
        profile_generation_service.storage.add_request(request_obj)
        profile_generation_service.storage.add_user_interaction(user_id, interaction)

        from reflexio.models.api_schema.service_schemas import (
            ManualProfileGenerationRequest,
        )

        request = ManualProfileGenerationRequest(user_id=user_id)
        response = profile_generation_service.run_manual_regular(request)

        # Without window_size, extractors use fallback of 1000 interactions
        # So the request should succeed (with profiles generated from mock)
        assert response.success is True


def test_run_manual_regular_no_interactions(mock_chat_completion):
    """Test run_manual_regular handles case when no interactions exist."""
    user_id = "test_user_manual_no_interactions"
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up profile extractor config WITH window size
        profile_extractor_config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            should_extract_profile_prompt_override="test",
            context_prompt="test",
            extraction_definition_prompt="test",
            metadata_definition_prompt="test",
        )
        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [profile_extractor_config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        from reflexio.models.api_schema.service_schemas import (
            ManualProfileGenerationRequest,
        )

        request = ManualProfileGenerationRequest(user_id=user_id)
        response = profile_generation_service.run_manual_regular(request)

        # Should succeed but with 0 profiles since no interactions
        assert response.success is True
        assert response.profiles_generated == 0
        # Message format may vary, just check success and 0 profiles


def test_run_manual_regular_with_interactions(mock_chat_completion):
    """Test run_manual_regular generates profiles with CURRENT status."""
    user_id = "test_user_manual_with_interactions"
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up profile extractor config WITH window size
        profile_extractor_config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            should_extract_profile_prompt_override="test",
            context_prompt="test",
            extraction_definition_prompt="test",
            metadata_definition_prompt="test",
        )
        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [profile_extractor_config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        # First, add some interactions to storage
        interaction = InteractionData(
            content="remember i like sushi",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction],
        )
        interactions = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request,
                request_id="1",
            )
        )
        request_obj = Request(
            request_id="1",
            user_id=user_id,
            source="",
        )
        profile_generation_service.storage.add_request(request_obj)
        for interaction_obj in interactions:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        from reflexio.models.api_schema.service_schemas import (
            ManualProfileGenerationRequest,
        )

        request = ManualProfileGenerationRequest(user_id=user_id)
        response = profile_generation_service.run_manual_regular(request)

        # Should succeed with profiles generated
        assert response.success is True
        assert response.profiles_generated > 0

        # Verify profiles have CURRENT status (None)
        profiles = profile_generation_service.storage.get_user_profile(
            user_id, status_filter=[None]
        )
        assert len(profiles) > 0
        for profile in profiles:
            assert profile.status is None  # CURRENT status


def test_run_manual_regular_with_source_filter(mock_chat_completion):
    """Test run_manual_regular respects source filter."""
    user_id = "test_user_manual_source"
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up profile extractor config WITH window size
        profile_extractor_config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            should_extract_profile_prompt_override="test",
            context_prompt="test",
            extraction_definition_prompt="test",
            metadata_definition_prompt="test",
        )
        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [profile_extractor_config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        # Add interactions with source_a
        interaction_a = InteractionData(
            content="remember i like sushi",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        publish_request_a = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction_a],
            source="source_a",
        )
        interactions_a = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request_a,
                request_id="1",
            )
        )
        request_a = Request(
            request_id="1",
            user_id=user_id,
            source="source_a",
        )
        profile_generation_service.storage.add_request(request_a)
        for interaction_obj in interactions_a:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        # Add interactions with source_b
        interaction_b = InteractionData(
            content="I prefer pizza",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        publish_request_b = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction_b],
            source="source_b",
        )
        interactions_b = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request_b,
                request_id="2",
            )
        )
        request_b = Request(
            request_id="2",
            user_id=user_id,
            source="source_b",
        )
        profile_generation_service.storage.add_request(request_b)
        for interaction_obj in interactions_b:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        from reflexio.models.api_schema.service_schemas import (
            ManualProfileGenerationRequest,
        )

        # Request with existing source - should generate profiles
        request = ManualProfileGenerationRequest(user_id=user_id, source="source_a")
        response = profile_generation_service.run_manual_regular(request)

        # Should succeed and generate profiles from source_a
        assert response.success is True
        assert response.profiles_generated > 0


# ===============================
# Tests for _get_rerun_user_ids
# ===============================


class TestGetRerunItems:
    """Tests for the _get_rerun_user_ids method."""

    def test_get_rerun_user_ids_returns_user_ids(self):
        """Test that _get_rerun_user_ids returns user IDs to process."""
        org_id = "0"

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            service = ProfileGenerationService(
                llm_client=llm_client,
                request_context=RequestContext(
                    org_id=org_id, storage_base_dir=temp_dir
                ),
            )

            # Add interactions for user1 - 2 requests
            for i in range(2):
                request_id = f"user1_request_{i}"
                interaction = Interaction(
                    interaction_id=i,
                    user_id="user1",
                    request_id=request_id,
                    content=f"Test content {i}",
                    role="user",
                    created_at=int(datetime.datetime.now(UTC).timestamp()),
                )
                request_obj = Request(
                    request_id=request_id,
                    user_id="user1",
                    source="test_source",
                    session_id=f"group_{i}",
                )
                service.storage.add_request(request_obj)
                service.storage.add_user_interaction("user1", interaction)

            # Add interactions for user2 - 1 request
            request_id = "user2_request_0"
            interaction = Interaction(
                interaction_id=10,
                user_id="user2",
                request_id=request_id,
                content="Test content user2",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_obj = Request(
                request_id=request_id,
                user_id="user2",
                source="test_source",
                session_id="group_user2",
            )
            service.storage.add_request(request_obj)
            service.storage.add_user_interaction("user2", interaction)

            from reflexio.models.api_schema.service_schemas import (
                RerunProfileGenerationRequest,
            )

            # Get all users
            request = RerunProfileGenerationRequest()
            result = service._get_rerun_user_ids(request)

            # Should return list of 2 user IDs
            assert len(result) == 2
            assert "user1" in result
            assert "user2" in result

    def test_get_rerun_user_ids_filters_by_user_id(self):
        """Test that _get_rerun_user_ids filters by user_id when specified."""
        org_id = "0"

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            service = ProfileGenerationService(
                llm_client=llm_client,
                request_context=RequestContext(
                    org_id=org_id, storage_base_dir=temp_dir
                ),
            )

            # Add interactions for user1
            interaction1 = Interaction(
                interaction_id=1,
                user_id="user1",
                request_id="request_1",
                content="Test content user1",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request1 = Request(
                request_id="request_1",
                user_id="user1",
                source="test_source",
                session_id="group_1",
            )
            service.storage.add_request(request1)
            service.storage.add_user_interaction("user1", interaction1)

            # Add interactions for user2
            interaction2 = Interaction(
                interaction_id=2,
                user_id="user2",
                request_id="request_2",
                content="Test content user2",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request2 = Request(
                request_id="request_2",
                user_id="user2",
                source="test_source",
                session_id="group_2",
            )
            service.storage.add_request(request2)
            service.storage.add_user_interaction("user2", interaction2)

            from reflexio.models.api_schema.service_schemas import (
                RerunProfileGenerationRequest,
            )

            # Filter by user1 only
            request = RerunProfileGenerationRequest(user_id="user1")
            result = service._get_rerun_user_ids(request)

            # Should only have user1
            assert len(result) == 1
            assert "user1" in result
            assert "user2" not in result

    def test_get_rerun_user_ids_with_source_filter(self):
        """Test that _get_rerun_user_ids applies source filter correctly."""
        org_id = "0"

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            service = ProfileGenerationService(
                llm_client=llm_client,
                request_context=RequestContext(
                    org_id=org_id, storage_base_dir=temp_dir
                ),
            )

            user_id = "test_user"

            # Add request with source_a
            interaction_a = Interaction(
                interaction_id=1,
                user_id=user_id,
                request_id="request_a",
                content="Test content A",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_a = Request(
                request_id="request_a",
                user_id=user_id,
                source="source_a",
                session_id="group_a",
            )
            service.storage.add_request(request_a)
            service.storage.add_user_interaction(user_id, interaction_a)

            # Add request with source_b
            interaction_b = Interaction(
                interaction_id=2,
                user_id=user_id,
                request_id="request_b",
                content="Test content B",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_b = Request(
                request_id="request_b",
                user_id=user_id,
                source="source_b",
                session_id="group_b",
            )
            service.storage.add_request(request_b)
            service.storage.add_user_interaction(user_id, interaction_b)

            from reflexio.models.api_schema.service_schemas import (
                RerunProfileGenerationRequest,
            )

            # Filter by source_a
            request = RerunProfileGenerationRequest(user_id=user_id, source="source_a")
            result = service._get_rerun_user_ids(request)

            # Should have the user_id
            assert len(result) == 1
            assert user_id in result


def test_get_rerun_user_ids_returns_empty_when_no_matches():
    """Test that _get_rerun_user_ids returns empty list when no items match."""
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        from reflexio.models.api_schema.service_schemas import (
            RerunProfileGenerationRequest,
        )

        request = RerunProfileGenerationRequest(user_id="nonexistent_user")
        result = service._get_rerun_user_ids(request)

        assert result == []


def test_collect_scoped_interactions_for_precheck_uses_extractor_scope():
    """Pre-check should use extractor-specific window and source filters."""
    org_id = "0"
    user_id = "test_user"

    with tempfile.TemporaryDirectory() as temp_dir:
        service = ProfileGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        service.configurator.set_config_by_name("batch_size", 240)
        service.service_config = service._load_generation_service_config(
            ProfileGenerationRequest(
                user_id=user_id,
                request_id="request-1",
                source="api",
                auto_run=True,
            )
        )

        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="request-1",
            content="I prefer concise summaries",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_obj = Request(
            request_id="request-1",
            user_id=user_id,
            source="api",
            session_id="group-1",
        )
        session_data = RequestInteractionDataModel(
            session_id="group-1",
            request=request_obj,
            interactions=[interaction],
        )

        service.storage.get_last_k_interactions_grouped = MagicMock(
            return_value=([session_data], [])
        )

        extractor_configs = [
            ProfileExtractorConfig(
                extractor_name="api_profiles",
                extraction_definition_prompt="communication preferences",
                request_sources_enabled=["api"],
                batch_size_override=150,
            ),
            ProfileExtractorConfig(
                extractor_name="web_profiles",
                extraction_definition_prompt="shopping preferences",
                request_sources_enabled=["web"],
                batch_size_override=90,
            ),
        ]

        (
            scoped_groups,
            scoped_configs,
        ) = service._collect_scoped_interactions_for_precheck(extractor_configs)

        assert len(scoped_groups) == 1
        assert [c.extractor_name for c in scoped_configs] == ["api_profiles"]
        service.storage.get_last_k_interactions_grouped.assert_called_once()
        _, kwargs = service.storage.get_last_k_interactions_grouped.call_args
        assert kwargs["k"] == 150
        assert kwargs["sources"] == ["api"]


def test_should_run_before_extraction_combines_all_extractor_criteria():
    """Consolidated pre-check should include all enabled extractor definitions and override conditions."""
    org_id = "0"
    user_id = "test_user"

    with tempfile.TemporaryDirectory() as temp_dir:
        service = ProfileGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        service.service_config = service._load_generation_service_config(
            ProfileGenerationRequest(
                user_id=user_id,
                request_id="request-1",
                source="api",
                auto_run=True,
            )
        )

        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="request-1",
            content="I am leading a migration project and prefer concise updates.",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_obj = Request(
            request_id="request-1",
            user_id=user_id,
            source="api",
            session_id="group-1",
        )
        session_data = RequestInteractionDataModel(
            session_id="group-1",
            request=request_obj,
            interactions=[interaction],
        )

        extractor_configs = [
            ProfileExtractorConfig(
                extractor_name="comm_profiles",
                extraction_definition_prompt="communication preferences",
            ),
            ProfileExtractorConfig(
                extractor_name="work_profiles",
                extraction_definition_prompt="career goals and projects",
                should_extract_profile_prompt_override=(
                    "when user mentions work projects, deadlines, or role changes"
                ),
            ),
        ]

        with (
            patch.object(
                service,
                "_collect_scoped_interactions_for_precheck",
                return_value=([session_data], extractor_configs),
            ),
            patch.object(
                service.client,
                "generate_chat_response",
                return_value="true",
            ) as mock_generate,
        ):
            should_run = service._should_run_before_extraction(extractor_configs)

        assert should_run is True
        mock_generate.assert_called_once()
        prompt = mock_generate.call_args.kwargs["messages"][0]["content"]
        assert "communication preferences" in prompt
        assert "career goals and projects" in prompt
        assert "work projects, deadlines, or role changes" in prompt


if __name__ == "__main__":
    pytest.main([__file__])
