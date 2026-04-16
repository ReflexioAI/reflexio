"""Integration tests for PlaybookGenerationService."""

import contextlib
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.integration


# Disable global mock mode for the message construction test so LLM client mock is used
@pytest.fixture
def disable_mock_llm_response(monkeypatch):
    """Disable MOCK_LLM_RESPONSE env var so tests use their own mocks."""
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)


from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    PlaybookAggregatorConfig,
    PlaybookConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.playbook.playbook_generation_service import (
    PlaybookGenerationService,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
    StructuredPlaybookContent,
    StructuredPlaybookList,
)
from tests.server.test_utils import skip_in_precommit, skip_low_priority


def create_request_interaction_data_model(
    user_id: str,
    request_id: str,
    interactions: list[Interaction],
    agent_version: str = "test_agent_1",
) -> RequestInteractionDataModel:
    """Helper function to create a RequestInteractionDataModel for testing."""
    request = Request(
        request_id=request_id,
        user_id=user_id,
        source="test",
        agent_version=agent_version,
        session_id="session_1",
    )
    return RequestInteractionDataModel(
        session_id="session_1",
        request=request,
        interactions=interactions,
    )


@pytest.fixture
def mock_request_context():
    """Create a mock request context with necessary components."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    context = MagicMock(spec=RequestContext)
    context.org_id = "test_org_123"
    context.storage = MagicMock()
    # Mock get_operation_state to return None by default (no in-progress state)
    context.storage.get_operation_state.return_value = None
    # Mock try_acquire_in_progress_lock to return success
    context.storage.try_acquire_in_progress_lock.return_value = {"acquired": True}
    # Mock get_user_playbooks to return empty list (for existing playbooks check)
    context.storage.get_user_playbooks.return_value = []
    # Mock get_last_k_interactions_grouped to return empty by default
    context.storage.get_last_k_interactions_grouped.return_value = ([], [])
    context.configurator = MagicMock()
    context.configurator.get_config.return_value.user_playbook_extractor_configs = [
        PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="Test playbook definition",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
    ]
    # Mock batch_size for extractor
    context.configurator.get_config.return_value.batch_size = 100
    context.prompt_manager = PromptManager()
    return context


@pytest.fixture
def playbook_generation_service(mock_request_context):
    """Create a PlaybookGenerationService instance with mocked dependencies."""
    mock_client = MagicMock()
    service = PlaybookGenerationService(
        llm_client=mock_client, request_context=mock_request_context
    )
    return service  # noqa: RET504


@pytest.fixture
def test_interactions():
    """Create test interactions for playbook generation."""
    return [
        Interaction(
            interaction_id=1,
            user_id="test_user_123",
            request_id="test_request_1",
            content="I need help with my account",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="click",
            user_action_description="Clicked help button",
            interacted_image_url="https://example.com/help",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user_123",
            request_id="test_request_1",
            content="Thank you for your help!",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="click",
            user_action_description="Clicked thank you button",
            interacted_image_url="https://example.com/thank-you",
        ),
    ]


def _setup_mock_chat_completion(
    service,
    should_generate=True,
    content="The agent was helpful and provided accurate information",
):
    """Helper function to set up mock chat completion responses."""

    def mock_generate_chat_response(messages, **kwargs):
        """
        Check prompt content to determine which mock response to return.
        If prompt contains boolean output instruction, return boolean response.
        Otherwise, return structured JSON playbook response.
        """
        # Get the prompt content from the messages
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        # Check if this is a should_generate_playbook call
        # Support both old and new prompt formats
        if (
            "Output just a boolean value" in prompt_content
            or "Return only true or false" in prompt_content
        ):
            return "true" if should_generate else "false"
        # Otherwise, this is a playbook extraction call - return list-shaped output
        return StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    trigger="interacting with users",
                    content=content,
                )
            ]
        )

    service.client.generate_chat_response = MagicMock(
        side_effect=mock_generate_chat_response
    )


@skip_in_precommit
def test_playbook_generation_with_storage(
    playbook_generation_service,
    mock_request_context,
    test_interactions,
    disable_mock_llm_response,
):
    """Test playbook generation with mocked storage."""
    _setup_mock_chat_completion(playbook_generation_service)

    # Create request interaction data model
    request_interaction_data_model = create_request_interaction_data_model(
        user_id="test_user_123",
        request_id="test_request_1",
        interactions=test_interactions,
    )

    # Mock storage to return interactions when extractor calls get_last_k_interactions_grouped
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = (
        [request_interaction_data_model],
        test_interactions,
    )

    # Create playbook generation request with new API
    request = PlaybookGenerationRequest(
        request_id="test_request_1",
        agent_version="test_agent_1",
        user_id="test_user_123",
        auto_run=False,  # Skip stride check for testing
    )

    # Run playbook generation
    playbook_generation_service.run(request)

    # Verify storage was called with correct playbook
    mock_request_context.storage.save_user_playbooks.assert_called_once()
    saved_playbooks = mock_request_context.storage.save_user_playbooks.call_args[0][0]

    assert len(saved_playbooks) == 1
    playbook = saved_playbooks[0]
    assert isinstance(playbook, UserPlaybook)
    assert playbook.agent_version == "test_agent_1"
    assert playbook.request_id == "test_request_1"
    # Verify top-level fields are populated
    assert playbook.trigger == "interacting with users"
    assert playbook.content == "The agent was helpful and provided accurate information"


@skip_in_precommit
@skip_low_priority
def test_playbook_generation_with_empty_interactions(
    playbook_generation_service, mock_request_context
):
    """Test playbook generation with empty interactions."""
    # No need to set up mock since service should return early with empty interactions

    # Storage returns empty interactions
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = ([], [])

    # Create playbook generation request with new API
    request = PlaybookGenerationRequest(
        request_id="test_request_1",
        agent_version="test_agent_1",
        user_id="test_user_123",
        auto_run=False,  # Skip stride check for testing
    )

    # Run playbook generation
    playbook_generation_service.run(request)

    # Verify storage was not called
    mock_request_context.storage.save_user_playbooks.assert_not_called()


@skip_in_precommit
@skip_low_priority
def test_playbook_generation_with_no_playbook_config(
    playbook_generation_service, mock_request_context, test_interactions
):
    """Test playbook generation with no playbook config."""
    _setup_mock_chat_completion(playbook_generation_service)

    # Set empty playbook config
    mock_request_context.configurator.get_config.return_value.user_playbook_extractor_configs = []

    # Create request interaction data model
    request_interaction_data_model = create_request_interaction_data_model(
        user_id="test_user_123",
        request_id="test_request_1",
        interactions=test_interactions,
    )

    # Mock storage to return interactions
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = (
        [request_interaction_data_model],
        test_interactions,
    )

    # Create playbook generation request with new API
    request = PlaybookGenerationRequest(
        request_id="test_request_1",
        agent_version="test_agent_1",
        user_id="test_user_123",
        auto_run=False,  # Skip stride check for testing
    )

    # Run playbook generation
    playbook_generation_service.run(request)

    # Verify storage was not called
    mock_request_context.storage.save_user_playbooks.assert_not_called()


@skip_in_precommit
@skip_low_priority
def test_playbook_generation_with_should_not_generate(
    playbook_generation_service,
    mock_request_context,
    test_interactions,
    disable_mock_llm_response,
):
    """Test playbook generation when should_generate_playbook returns false."""
    _setup_mock_chat_completion(playbook_generation_service, should_generate=False)

    # Create request interaction data model
    request_interaction_data_model = create_request_interaction_data_model(
        user_id="test_user_123",
        request_id="test_request_1",
        interactions=test_interactions,
    )

    # Mock storage to return interactions
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = (
        [request_interaction_data_model],
        test_interactions,
    )

    # Create playbook generation request with new API
    request = PlaybookGenerationRequest(
        request_id="test_request_1",
        agent_version="test_agent_1",
        user_id="test_user_123",
        auto_run=False,  # Skip stride check for testing
    )

    # Run playbook generation
    playbook_generation_service.run(request)

    # Verify storage was not called
    mock_request_context.storage.save_user_playbooks.assert_not_called()


def test_playbook_message_construction_with_interactions(
    mock_request_context,
    disable_mock_llm_response,
):
    """Test that interactions are formatted correctly in rendered playbook prompts."""
    from unittest.mock import patch

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager

    # Create test interactions
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user_123",
            request_id="test_request_1",
            content="I need help with my account",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="click",
            user_action_description="help button",
            interacted_image_url="https://example.com/help",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user_123",
            request_id="test_request_1",
            content="Thank you for your help!",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="none",
            user_action_description="",
        ),
    ]

    # Create request interaction data model
    request_interaction_data_model = create_request_interaction_data_model(
        user_id="test_user_123",
        request_id="test_request_1",
        interactions=interactions,
    )

    # Add mock prompt_manager to request context
    mock_request_context.prompt_manager = PromptManager()

    # Mock storage to return interactions when extractor calls get_last_k_interactions_grouped
    mock_request_context.storage.get_last_k_interactions_grouped.return_value = (
        [request_interaction_data_model],
        interactions,
    )

    # Use a real LiteLLMClient
    llm_config = LiteLLMConfig(model="gpt-4o-mini")
    llm_client = LiteLLMClient(llm_config)
    service = PlaybookGenerationService(
        llm_client=llm_client, request_context=mock_request_context
    )

    # Capture the messages sent to generate_chat_response
    captured_messages = []

    def mock_generate_chat_response(messages, **kwargs):
        captured_messages.append(messages)
        # Return appropriate responses based on content
        # Check if this is should_generate or playbook extraction
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        if "Output just a boolean value" in prompt_content:
            return "true"
        # Return list-shaped output expected by the new extractor schema
        return StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    trigger="assisting users",
                    content="The agent was helpful",
                )
            ]
        )

    with patch(
        "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
        side_effect=mock_generate_chat_response,
    ):
        # Create playbook generation request with new API
        request = PlaybookGenerationRequest(
            request_id="test_request_1",
            agent_version="test_agent_1",
            user_id="test_user_123",
            auto_run=False,  # Skip stride check for testing
        )

        # Run playbook generation
        with contextlib.suppress(Exception):
            # We're just validating message construction, errors are ok
            service.run(request)

    # Validate that messages were captured
    assert len(captured_messages) > 0, "No messages were captured"

    # Find the message that contains the playbook_extraction_main prompt
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

                # Check if this message contains interactions
                if any(
                    pattern in content_str
                    for pattern in [
                        "User: ```I need help",
                        "user: ```I need help",
                        "[Interaction",
                        "Session:",
                    ]
                ):
                    # Validate the interactions are formatted correctly in the rendered prompt
                    # Note: format might be "User:" (capital) or "user:" depending on the prompt
                    # Content is wrapped in backticks in the prompt template
                    has_interaction1 = (
                        "User: ```I need help with my account```" in content_str
                        or "user: ```I need help with my account```" in content_str
                    )
                    has_interaction2 = (
                        "User: ```Thank you for your help!```" in content_str
                        or "user: ```Thank you for your help!```" in content_str
                    )

                    if has_interaction1 and has_interaction2:
                        # Found both content interactions
                        found_interactions_in_prompt = True
                        break
        if found_interactions_in_prompt:
            break

    assert found_interactions_in_prompt, (
        "Did not find interactions in any rendered prompt"
    )
