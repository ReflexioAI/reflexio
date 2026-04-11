import datetime
import tempfile
from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)
from reflexio.models.config_schema import (
    PlaybookAggregatorConfig,
    PlaybookConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.playbook.playbook_generation_service import (
    PlaybookGenerationService,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
)


def create_request_interaction_data_model(
    user_id: str, request_id: str, interactions: list[Interaction]
) -> RequestInteractionDataModel:
    """Helper function to create a RequestInteractionDataModel for testing."""
    request = Request(
        request_id=request_id,
        user_id=user_id,
        source="test",
        agent_version="1.0",
        session_id="session_1",
    )
    return RequestInteractionDataModel(
        session_id="session_1",
        request=request,
        interactions=interactions,
    )


@pytest.fixture
def mock_chat_completion():
    # Mock response for should_generate_playbook call
    mock_should_generate_response = "true"

    # Mock response for extract_playbook call
    mock_extract_response = '```json\n{\n    "playbook": "The agent was helpful and provided accurate information",\n    "type": "positive"\n}\n```'

    def mock_generate_chat_response_side_effect(messages, **kwargs):
        """
        Check prompt content to determine which mock response to return.
        If prompt contains "Output just a boolean value", return boolean response.
        Otherwise, return JSON playbook response.
        """
        # Get the prompt content from the messages
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        # Check if this is a should_generate_playbook call
        if "Output just a boolean value" in prompt_content:
            return mock_should_generate_response
        # Otherwise, this is a playbook extraction call
        return mock_extract_response

    # Mock the LiteLLM client's generate_chat_response method
    with patch(
        "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
        side_effect=mock_generate_chat_response_side_effect,
    ):
        yield


def test_generate_playbook(mock_chat_completion):
    user_id = "test_user_id"
    org_id = "0"
    interaction = Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id="test_request_id",
        content="The agent was very helpful in explaining the process",
        role="user",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up playbook config with batch_size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_configs", [playbook_config]
        )
        playbook_generation_service.configurator.set_config_by_name("batch_size", 100)

        # Store interactions in storage first
        request_obj = Request(
            request_id="test_request_id",
            user_id=user_id,
            source="test",
            agent_version="1.0",
            session_id="session_1",
        )
        playbook_generation_service.storage.add_request(request_obj)
        playbook_generation_service.storage.add_user_interaction(user_id, interaction)

        # Create playbook generation request - extractors collect from storage
        playbook_request = PlaybookGenerationRequest(
            request_id="test_request_id",
            agent_version="1.0",
            user_id=user_id,
            auto_run=False,  # Skip stride check for testing
        )

        playbook_generation_service.run(playbook_request)

        # Verify playbook was saved
        user_playbooks = playbook_generation_service.storage.get_user_playbooks()
        assert len(user_playbooks) > 0
        playbook = user_playbooks[0]
        assert playbook.request_id == "test_request_id"


def test_empty_interactions(mock_chat_completion):
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up playbook config
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_configs", [playbook_config]
        )

        # Create playbook generation request with empty request interaction data models
        playbook_request = PlaybookGenerationRequest(
            request_id="test_request_id",
            agent_version="1.0",
            request_interaction_data_models=[],
        )

        playbook_generation_service.run(playbook_request)

        # Verify no playbook was generated
        user_playbooks = playbook_generation_service.storage.get_user_playbooks()
        assert len(user_playbooks) == 0


def test_missing_configs(mock_chat_completion):
    user_id = "test_user_id"
    org_id = "0"
    interaction = Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id="test_request_id",
        content="The agent was very helpful in explaining the process",
        role="user",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Create request interaction data model
        request_interaction_data_model = create_request_interaction_data_model(
            user_id=user_id,
            request_id="test_request_id",
            interactions=[interaction],
        )

        # Create playbook generation request without setting up configs
        playbook_request = PlaybookGenerationRequest(
            request_id="test_request_id",
            agent_version="1.0",
            request_interaction_data_models=[request_interaction_data_model],
        )

        playbook_generation_service.run(playbook_request)

        # Verify no playbook was generated
        user_playbooks = playbook_generation_service.storage.get_user_playbooks()
        assert len(user_playbooks) == 0


def test_error_handling(mock_chat_completion):
    user_id = "test_user_id"
    org_id = "0"
    interaction = Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id="test_request_id",
        content="The agent was very helpful in explaining the process",
        role="user",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up playbook config
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_configs", [playbook_config]
        )

        # Create request interaction data model
        request_interaction_data_model = create_request_interaction_data_model(
            user_id=user_id,
            request_id="test_request_id",
            interactions=[interaction],
        )

        # Create playbook generation request
        playbook_request = PlaybookGenerationRequest(
            request_id="test_request_id",
            agent_version="1.0",
            request_interaction_data_models=[request_interaction_data_model],
        )

        # Mock storage.save_user_playbooks to raise an exception
        with patch.object(
            playbook_generation_service.storage,
            "save_user_playbooks",
            side_effect=Exception("Storage error"),
        ):
            # The service should handle the error gracefully
            playbook_generation_service.run(playbook_request)

            # Verify no playbook was saved
            user_playbooks = playbook_generation_service.storage.get_user_playbooks()
            assert len(user_playbooks) == 0


def test_run_manual_regular_no_window_size(mock_chat_completion):
    """Test run_manual_regular works even without batch_size configured.

    Since extractors handle window size at their level, the manual flow no longer
    validates window_size upfront. Extractors use a fallback of 1000 interactions
    when no window size is configured.
    """
    org_id = "0"
    user_id = "test_user"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITHOUT window size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_configs", [playbook_config]
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
        playbook_generation_service.storage.add_request(request_obj)
        playbook_generation_service.storage.add_user_interaction(user_id, interaction)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
        )

        request = ManualPlaybookGenerationRequest(agent_version=agent_version)
        response = playbook_generation_service.run_manual_regular(request)

        # Without window_size, extractors use fallback of 1000 interactions
        # So the request should succeed
        assert response.success is True


def test_run_manual_regular_no_interactions(mock_chat_completion):
    """Test run_manual_regular handles case when no interactions exist."""
    org_id = "0"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITH window size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_configs", [playbook_config]
        )
        playbook_generation_service.configurator.set_config_by_name("batch_size", 100)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
        )

        request = ManualPlaybookGenerationRequest(agent_version=agent_version)
        response = playbook_generation_service.run_manual_regular(request)

        # Should succeed but with 0 playbooks since no interactions
        assert response.success is True
        assert response.playbooks_generated == 0
        assert "No interactions found" in response.msg


def test_run_manual_regular_with_interactions(mock_chat_completion):
    """Test run_manual_regular generates playbooks with CURRENT status."""
    user_id = "test_user_id"
    org_id = "0"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITH window size and allow_manual_trigger
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
            allow_manual_trigger=True,  # Required for manual generation
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_configs", [playbook_config]
        )
        playbook_generation_service.configurator.set_config_by_name("batch_size", 100)

        # First, add some interactions to storage
        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="test_request_id",
            content="The agent was very helpful",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_obj = Request(
            request_id="test_request_id",
            user_id=user_id,
            source="",
            agent_version=agent_version,
        )
        playbook_generation_service.storage.add_request(request_obj)
        playbook_generation_service.storage.add_user_interaction(user_id, interaction)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
        )

        request = ManualPlaybookGenerationRequest(agent_version=agent_version)
        response = playbook_generation_service.run_manual_regular(request)

        # Should succeed (playbooks generated depends on mock)
        assert response.success is True
        # Note: playbooks_generated may be 0 if mock returns no playbook
        # The key is that the method runs without error


def test_run_manual_regular_with_source_filter(mock_chat_completion):
    """Test run_manual_regular respects source filter."""
    user_id = "test_user_id"
    org_id = "0"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITH window size and allow_manual_trigger
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
            allow_manual_trigger=True,  # Required for manual generation
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_configs", [playbook_config]
        )
        playbook_generation_service.configurator.set_config_by_name("batch_size", 100)

        # Add interactions with source_a
        interaction_a = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="request_a",
            content="The agent was helpful",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_a = Request(
            request_id="request_a",
            user_id=user_id,
            source="source_a",
            agent_version=agent_version,
        )
        playbook_generation_service.storage.add_request(request_a)
        playbook_generation_service.storage.add_user_interaction(user_id, interaction_a)

        # Add interactions with source_b
        interaction_b = Interaction(
            interaction_id=2,
            user_id=user_id,
            request_id="request_b",
            content="The agent was not helpful",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_b = Request(
            request_id="request_b",
            user_id=user_id,
            source="source_b",
            agent_version=agent_version,
        )
        playbook_generation_service.storage.add_request(request_b)
        playbook_generation_service.storage.add_user_interaction(user_id, interaction_b)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
        )

        # Request with non-existent source
        request = ManualPlaybookGenerationRequest(
            agent_version=agent_version, source="non_existent_source"
        )
        response = playbook_generation_service.run_manual_regular(request)

        # Should succeed but with 0 playbooks since no matching source
        assert response.success is True
        assert response.playbooks_generated == 0


def test_run_manual_regular_output_pending_status_false(mock_chat_completion):
    """Test that run_manual_regular outputs CURRENT status when output_pending_status=False."""
    user_id = "test_user_id"
    org_id = "0"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)

        # Create service with output_pending_status=False (default for manual regular)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITH window size and allow_manual_trigger
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
            allow_manual_trigger=True,  # Required for manual generation
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_configs", [playbook_config]
        )
        playbook_generation_service.configurator.set_config_by_name("batch_size", 100)

        # Add interaction
        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="test_request_id",
            content="The agent was very helpful",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_obj = Request(
            request_id="test_request_id",
            user_id=user_id,
            source="",
            agent_version=agent_version,
        )
        playbook_generation_service.storage.add_request(request_obj)
        playbook_generation_service.storage.add_user_interaction(user_id, interaction)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
            Status,
        )

        request = ManualPlaybookGenerationRequest(agent_version=agent_version)
        response = playbook_generation_service.run_manual_regular(request)

        # Should succeed (playbooks generated depends on mock)
        assert response.success is True
        # Note: playbooks_generated may be 0 if mock returns no playbook

        # Verify no PENDING playbooks (output_pending_status=False)
        pending_playbooks = playbook_generation_service.storage.get_user_playbooks(
            status_filter=[Status.PENDING]
        )

        assert len(pending_playbooks) == 0, (
            "Manual generation should not create PENDING playbooks"
        )


# ===============================
# Tests for _get_rerun_user_ids
# ===============================


class TestGetRerunItems:
    """Tests for the _get_rerun_user_ids method."""

    def test_get_rerun_user_ids_returns_user_ids(self):
        """Test that _get_rerun_user_ids returns user IDs."""
        org_id = "0"

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            service = PlaybookGenerationService(
                llm_client=llm_client,
                request_context=RequestContext(
                    org_id=org_id, storage_base_dir=temp_dir
                ),
            )

            # Add interactions with different users
            agent_version = "1.0"

            # User 1 - 2 requests
            user_id_1 = "test_user_1"
            for i in range(2):
                request_id = f"request_user1_{i}"
                interaction = Interaction(
                    interaction_id=i,
                    user_id=user_id_1,
                    request_id=request_id,
                    content=f"Test content {i}",
                    role="user",
                    created_at=int(datetime.datetime.now(UTC).timestamp()),
                )
                request_obj = Request(
                    request_id=request_id,
                    user_id=user_id_1,
                    source="test_source",
                    agent_version=agent_version,
                    session_id="group_1",
                )
                service.storage.add_request(request_obj)
                service.storage.add_user_interaction(user_id_1, interaction)

            # User 2 - 1 request
            user_id_2 = "test_user_2"
            request_id = "request_user2_0"
            interaction = Interaction(
                interaction_id=10,
                user_id=user_id_2,
                request_id=request_id,
                content="Test content user 2",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_obj = Request(
                request_id=request_id,
                user_id=user_id_2,
                source="test_source",
                agent_version=agent_version,
                session_id="group_2",
            )
            service.storage.add_request(request_obj)
            service.storage.add_user_interaction(user_id_2, interaction)

            from reflexio.models.api_schema.service_schemas import (
                RerunPlaybookGenerationRequest,
            )

            request = RerunPlaybookGenerationRequest(agent_version=agent_version)
            result = service._get_rerun_user_ids(request)

            # Should return list of 2 user IDs
            assert len(result) == 2
            assert user_id_1 in result
            assert user_id_2 in result

    def test_get_rerun_user_ids_with_source_filter(self):
        """Test that _get_rerun_user_ids applies source filter correctly."""
        org_id = "0"

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            service = PlaybookGenerationService(
                llm_client=llm_client,
                request_context=RequestContext(
                    org_id=org_id, storage_base_dir=temp_dir
                ),
            )

            agent_version = "1.0"

            # Add request with source_a for user_a
            user_id_a = "test_user_a"
            interaction_a = Interaction(
                interaction_id=1,
                user_id=user_id_a,
                request_id="request_a",
                content="Test content A",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_a = Request(
                request_id="request_a",
                user_id=user_id_a,
                source="source_a",
                agent_version=agent_version,
                session_id="group_a",
            )
            service.storage.add_request(request_a)
            service.storage.add_user_interaction(user_id_a, interaction_a)

            # Add request with source_b for user_b
            user_id_b = "test_user_b"
            interaction_b = Interaction(
                interaction_id=2,
                user_id=user_id_b,
                request_id="request_b",
                content="Test content B",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_b = Request(
                request_id="request_b",
                user_id=user_id_b,
                source="source_b",
                agent_version=agent_version,
                session_id="group_b",
            )
            service.storage.add_request(request_b)
            service.storage.add_user_interaction(user_id_b, interaction_b)

            from reflexio.models.api_schema.service_schemas import (
                RerunPlaybookGenerationRequest,
            )

            # Filter by source_a - should only include user_a
            request = RerunPlaybookGenerationRequest(
                agent_version=agent_version, source="source_a"
            )
            result = service._get_rerun_user_ids(request)

            # Should only have user_a (who has source_a requests)
            assert len(result) == 1
            assert user_id_a in result
            assert user_id_b not in result


def test_get_rerun_user_ids_returns_empty_when_no_matches():
    """Test that _get_rerun_user_ids returns empty list when no items match."""
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        from reflexio.models.api_schema.service_schemas import (
            RerunPlaybookGenerationRequest,
        )

        request = RerunPlaybookGenerationRequest(agent_version="1.0")
        result = service._get_rerun_user_ids(request)

        assert result == []


def test_collect_scoped_interactions_for_precheck_uses_extractor_scope():
    """Pre-check should use extractor-specific window and source filters."""
    org_id = "0"
    user_id = "test_user"

    with tempfile.TemporaryDirectory() as temp_dir:
        service = PlaybookGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        service.configurator.set_config_by_name("batch_size", 200)
        service.service_config = service._load_generation_service_config(
            PlaybookGenerationRequest(
                request_id="request-1",
                agent_version="1.0",
                user_id=user_id,
                source="api",
                auto_run=True,
            )
        )

        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="request-1",
            content="user corrected the agent behavior",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        session_id = create_request_interaction_data_model(
            user_id=user_id,
            request_id="request-1",
            interactions=[interaction],
        )

        service.storage.get_last_k_interactions_grouped = MagicMock(
            return_value=([session_id], [])
        )

        extractor_configs = [
            PlaybookConfig(
                extractor_name="api_playbook",
                extraction_definition_prompt="extract api-related playbook",
                request_sources_enabled=["api"],
                batch_size_override=120,
                aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
            ),
            PlaybookConfig(
                extractor_name="web_playbook",
                extraction_definition_prompt="extract web-related playbook",
                request_sources_enabled=["web"],
                batch_size_override=80,
                aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
            ),
        ]

        (
            scoped_groups,
            scoped_configs,
        ) = service._collect_scoped_interactions_for_precheck(extractor_configs)

        assert len(scoped_groups) == 1
        assert [c.extractor_name for c in scoped_configs] == ["api_playbook"]
        service.storage.get_last_k_interactions_grouped.assert_called_once()
        _, kwargs = service.storage.get_last_k_interactions_grouped.call_args
        assert kwargs["k"] == 120
        assert kwargs["sources"] == ["api"]


if __name__ == "__main__":
    pytest.main([__file__])
