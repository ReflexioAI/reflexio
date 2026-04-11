"""Test request source filtering for profile extractors"""

import datetime
import tempfile
from datetime import UTC
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
    Request,
)
from reflexio.models.config_schema import ProfileExtractorConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.services.profile.profile_generation_service import (
    ProfileGenerationService,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
)


@pytest.fixture
def mock_chat_completion():
    def mock_generate_chat_response_side_effect(messages, **kwargs):
        """Mock LLM responses"""
        # Get the prompt content from the messages
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        # Check if this is a should_extract_profile call
        if "Output just a boolean value" in prompt_content:
            return "true"
        # Return parsed dict for structured output
        if kwargs.get("parse_structured_output", False):
            return {"add": [{"content": "test profile", "time_to_live": "one_month"}]}
        return '{"add": [{"content": "test profile", "time_to_live": "one_month"}]}'

    with patch(
        "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
        side_effect=mock_generate_chat_response_side_effect,
    ):
        yield


def test_profile_extractor_filters_by_source_api(mock_chat_completion):
    """Test that profile extractor only runs when source matches request_sources_enabled"""
    user_id = "test_user_id"
    org_id = "0"
    interaction_request = InteractionData(
        content="test content",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up two profile extractor configs with batch_size
        # Config 1: only enabled for "api" source
        config1 = ProfileExtractorConfig(
            extractor_name="api_extractor",
            extraction_definition_prompt="API config",
            request_sources_enabled=["api"],
        )

        # Config 2: only enabled for "webhook" source
        config2 = ProfileExtractorConfig(
            extractor_name="webhook_extractor",
            extraction_definition_prompt="Webhook config",
            request_sources_enabled=["webhook"],
        )

        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [config1, config2]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        # Create a PublishUserInteractionRequest
        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction_request],
        )

        # Convert to interactions and store in storage with source="api"
        interactions = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request,
                request_id="1",
            )
        )
        request_obj = Request(
            request_id="1",
            user_id=user_id,
            source="api",
        )
        profile_generation_service.storage.add_request(request_obj)
        for interaction_obj in interactions:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        profile_generation_request = ProfileGenerationRequest(
            user_id=user_id,
            request_id="1",
            source="api",
            auto_run=False,  # Skip stride check for testing
        )

        profile_generation_service.run(profile_generation_request)

        # Profile should be created (config1 should run)
        profiles = profile_generation_service.storage.get_user_profile(user_id)
        assert len(profiles) == 1


def test_profile_extractor_filters_by_source_webhook(mock_chat_completion):
    """Test that profile extractor only runs when source matches"""
    user_id = "test_user_id"
    org_id = "0"
    interaction_request = InteractionData(
        content="test content",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up config only enabled for "api" source with batch_size
        config = ProfileExtractorConfig(
            extractor_name="api_only_extractor",
            extraction_definition_prompt="API only config",
            request_sources_enabled=["api"],
        )

        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction_request],
        )

        # Convert to interactions and store in storage with source="webhook"
        interactions = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request,
                request_id="1",
            )
        )
        request_obj = Request(
            request_id="1",
            user_id=user_id,
            source="webhook",
        )
        profile_generation_service.storage.add_request(request_obj)
        for interaction_obj in interactions:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        profile_generation_request = ProfileGenerationRequest(
            user_id=user_id,
            request_id="1",
            source="webhook",
            auto_run=False,  # Skip stride check for testing
        )

        profile_generation_service.run(profile_generation_request)

        # No profile should be created (config should be filtered out)
        profiles = profile_generation_service.storage.get_user_profile(user_id)
        assert len(profiles) == 0


def test_profile_extractor_none_enables_all_sources(mock_chat_completion):
    """Test that request_sources_enabled=None enables all sources"""
    user_id = "test_user_id"
    org_id = "0"
    interaction_request = InteractionData(
        content="test content",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Config with request_sources_enabled=None (default) and batch_size
        config = ProfileExtractorConfig(
            extractor_name="all_sources_extractor",
            extraction_definition_prompt="All sources config",
            request_sources_enabled=None,
        )

        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction_request],
        )

        # Test with any source - store in storage
        interactions = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request,
                request_id="1",
            )
        )
        request_obj = Request(
            request_id="1",
            user_id=user_id,
            source="random_source",
        )
        profile_generation_service.storage.add_request(request_obj)
        for interaction_obj in interactions:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        profile_generation_request = ProfileGenerationRequest(
            user_id=user_id,
            request_id="1",
            source="random_source",
            auto_run=False,  # Skip stride check for testing
        )

        profile_generation_service.run(profile_generation_request)

        # Profile should be created (None means all sources enabled)
        profiles = profile_generation_service.storage.get_user_profile(user_id)
        assert len(profiles) == 1


def test_profile_extractor_empty_list_enables_all_sources(mock_chat_completion):
    """Test that request_sources_enabled=[] enables all sources"""
    user_id = "test_user_id"
    org_id = "0"
    interaction_request = InteractionData(
        content="test content",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        profile_generation_service = ProfileGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Config with request_sources_enabled=[] and batch_size
        config = ProfileExtractorConfig(
            extractor_name="empty_sources_extractor",
            extraction_definition_prompt="All sources config",
            request_sources_enabled=[],
        )

        profile_generation_service.configurator.set_config_by_name(
            "profile_extractor_configs", [config]
        )
        profile_generation_service.configurator.set_config_by_name("batch_size", 100)

        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction_request],
        )

        # Test with any source - store in storage
        interactions = (
            GenerationService.get_interaction_from_publish_user_interaction_request(
                publish_user_interaction_request=publish_request,
                request_id="1",
            )
        )
        request_obj = Request(
            request_id="1",
            user_id=user_id,
            source="another_random_source",
        )
        profile_generation_service.storage.add_request(request_obj)
        for interaction_obj in interactions:
            profile_generation_service.storage.add_user_interaction(
                user_id, interaction_obj
            )

        profile_generation_request = ProfileGenerationRequest(
            user_id=user_id,
            request_id="1",
            source="another_random_source",
            auto_run=False,  # Skip stride check for testing
        )

        profile_generation_service.run(profile_generation_request)

        # Profile should be created (empty list means all sources enabled)
        profiles = profile_generation_service.storage.get_user_profile(user_id)
        assert len(profiles) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
