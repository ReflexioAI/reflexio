"""
Unit tests for PlaybookExtractor.

Tests the extractor's new responsibilities for:
- Operation state key generation (not user-scoped)
- Interaction collection with window/stride across all users
- Source filtering
- Operation state updates
- Integration of run() method
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    BlockingIssue,
    BlockingIssueKind,
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.config_schema import PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.playbook.playbook_extractor import PlaybookExtractor
from reflexio.server.services.playbook.playbook_generation_service import (
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
    StructuredPlaybookList,
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LiteLLMClient)
    client.generate_chat_response.return_value = "true"
    return client


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def request_context(temp_storage_dir):
    """Create a request context with mock storage."""
    context = RequestContext(org_id="test_org", storage_base_dir=temp_storage_dir)
    context.storage = MagicMock()
    return context


@pytest.fixture
def extractor_config():
    """Create a playbook extractor config."""
    return PlaybookConfig(
        extractor_name="quality_playbook",
        extraction_definition_prompt="Evaluate agent quality",
    )


@pytest.fixture
def service_config():
    """Create a service config."""
    return PlaybookGenerationServiceConfig(
        agent_version="1.0.0",
        request_id="test_request",
        source="api",
    )


@pytest.fixture
def sample_interactions():
    """Create sample interactions from multiple users for testing."""
    return [
        Interaction(
            interaction_id=1,
            user_id="user1",
            content="The agent helped me well",
            request_id="req1",
            created_at=1000,
            role="user",
        ),
        Interaction(
            interaction_id=2,
            user_id="user1",
            content="Glad I could help!",
            request_id="req1",
            created_at=1001,
            role="assistant",
        ),
        Interaction(
            interaction_id=3,
            user_id="user2",
            content="Could be faster",
            request_id="req2",
            created_at=1002,
            role="user",
        ),
    ]


@pytest.fixture
def sample_request_interaction_models(sample_interactions):
    """Create sample RequestInteractionDataModel objects."""
    request1 = Request(
        request_id="req1",
        user_id="user1",
        created_at=1000,
        source="api",
    )
    request2 = Request(
        request_id="req2",
        user_id="user2",
        created_at=1002,
        source="api",
    )
    return [
        RequestInteractionDataModel(
            session_id="req1",
            request=request1,
            interactions=sample_interactions[:2],
        ),
        RequestInteractionDataModel(
            session_id="req2",
            request=request2,
            interactions=[sample_interactions[2]],
        ),
    ]


# ===============================
# Test: Operation State Key
# ===============================


class TestOperationStateKey:
    """Tests for operation state key generation."""

    def test_state_manager_key_does_not_include_user_id(
        self, request_context, mock_llm_client, extractor_config, service_config
    ):
        """Test that playbook extractor state manager builds keys without user_id (not user-scoped)."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        mgr = extractor._create_state_manager()

        assert mgr.service_name == "playbook_extractor"
        assert mgr.org_id == "test_org"
        # Verify the bookmark key format does NOT include user_id
        key = mgr._bookmark_key(name="quality_playbook")
        assert "playbook_extractor" in key
        assert "test_org" in key
        assert "quality_playbook" in key
        assert key == "playbook_extractor::test_org::quality_playbook"

    def test_different_playbook_names_have_different_keys(
        self, request_context, mock_llm_client, service_config
    ):
        """Test that different playbook names get different operation state keys."""
        config1 = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Quality prompt",
        )
        config2 = PlaybookConfig(
            extractor_name="speed_playbook",
            extraction_definition_prompt="Speed prompt",
        )

        extractor1 = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config1,
            service_config=service_config,
            agent_context="Test agent",
        )
        extractor2 = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config2,
            service_config=service_config,
            agent_context="Test agent",
        )

        mgr1 = extractor1._create_state_manager()
        mgr2 = extractor2._create_state_manager()
        key1 = mgr1._bookmark_key(name="quality_playbook")
        key2 = mgr2._bookmark_key(name="speed_playbook")
        assert key1 != key2


# ===============================
# Test: Get Interactions (Not User-Scoped)
# ===============================


class TestGetInteractions:
    """Tests for interaction collection logic (not user-scoped).

    Note: Batch interval checking is handled upstream by BaseGenerationService._filter_configs_by_batch_interval()
    before the extractor is created, so batch_interval tests are at the service level.
    """

    def test_passes_none_user_id_to_storage(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that user_id from service_config is passed to get_last_k_interactions_grouped."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate agent quality",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        extractor._get_interactions()

        # Verify user_id from service_config was passed to storage
        call_kwargs = request_context.storage.get_last_k_interactions_grouped.call_args[
            1
        ]
        assert call_kwargs["user_id"] is None  # service_config.user_id is None

    def test_returns_interactions(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that interactions are returned from storage."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate agent quality",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        result = extractor._get_interactions()

        assert result is not None
        assert len(result) == 2  # Two sessions

    def test_uses_window_size_with_none_user_id(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that window size is used with user_id=None for all users."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate agent quality",
            batch_size_override=50,
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        extractor._get_interactions()

        # Verify get_last_k_interactions_grouped was called with user_id=None
        request_context.storage.get_last_k_interactions_grouped.assert_called_once()
        call_kwargs = request_context.storage.get_last_k_interactions_grouped.call_args[
            1
        ]
        assert call_kwargs["user_id"] is None
        assert call_kwargs["k"] == 50

    def test_none_sources_enabled_gets_all_sources(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that request_sources_enabled=None gets interactions from all sources."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate quality",
            request_sources_enabled=None,  # Get all sources
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        extractor._get_interactions()

        # Verify sources filter is None (get all sources) in get_last_k_interactions_grouped
        call_kwargs = request_context.storage.get_last_k_interactions_grouped.call_args[
            1
        ]
        assert call_kwargs["sources"] is None


# ===============================
# Test: Update Operation State
# ===============================


class TestUpdateOperationState:
    """Tests for operation state update logic."""

    def test_updates_state_with_all_users_interactions(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that operation state is updated with interactions from all users."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        extractor._update_operation_state(sample_request_interaction_models)

        # Verify upsert was called
        request_context.storage.upsert_operation_state.assert_called_once()

        # Verify state contains all interaction IDs (from both users)
        call_args = request_context.storage.upsert_operation_state.call_args
        state = call_args[0][1]

        assert 1 in state["last_processed_interaction_ids"]
        assert 2 in state["last_processed_interaction_ids"]
        assert 3 in state["last_processed_interaction_ids"]


# ===============================
# Test: Run Integration
# ===============================


class TestRun:
    """Integration tests for the run() method."""

    def test_run_collects_interactions_from_all_users(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that run() collects interactions from all users."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate agent quality",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            extractor.run()

        # Verify storage was queried with user_id=None
        call_kwargs = request_context.storage.get_last_k_interactions_grouped.call_args[
            1
        ]
        assert call_kwargs["user_id"] is None

    def test_run_returns_user_playbook(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that run() returns UserPlaybook objects."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate agent quality",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            result = extractor.run()

        assert result is not None
        assert len(result) > 0
        assert all(isinstance(f, UserPlaybook) for f in result)

    def test_mock_mode_includes_source_interaction_ids(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that mock mode populates source_interaction_ids from input interactions."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate agent quality",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            result = extractor.run()

        assert len(result) == 1
        assert result[0].source_interaction_ids == [1, 2, 3]

    def test_run_returns_empty_when_no_interactions(
        self,
        request_context,
        mock_llm_client,
        service_config,
    ):
        """Test that run() returns empty list when no interactions available."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate agent quality",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            [],
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        result = extractor.run()

        assert result == []

    def test_run_updates_operation_state_on_success(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that operation state is updated after successful extraction."""
        config = PlaybookConfig(
            extractor_name="quality_playbook",
            extraction_definition_prompt="Evaluate agent quality",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            result = extractor.run()

        # Verify operation state was updated
        if result:
            request_context.storage.upsert_operation_state.assert_called()


# ===============================
# Test: Structured AgentPlaybook Extraction
# ===============================


class TestStructuredPlaybookExtraction:
    """Tests for structured playbook extraction with JSON output."""

    def test_extracts_structured_playbook_with_all_fields(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that structured playbook with all fields is correctly extracted."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        # Mock LLM response with flat fields (new schema)
        mock_llm_client.generate_chat_response.return_value = StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    instruction="ask for CLI preference",
                    pitfall="assume GUI workflows by default",
                    trigger="assisting technical users",
                )
            ]
        )

        # Mock prompt manager
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.return_value = "mock prompt"
        request_context.prompt_manager.get_active_version.return_value = "1.2.0"

        # Disable mock mode to use the mocked LLM response
        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        assert result[0].structured_data.instruction == "ask for CLI preference"
        assert result[0].structured_data.pitfall == "assume GUI workflows by default"
        assert result[0].structured_data.trigger == "assisting technical users"
        assert result[0].structured_data.embedding_text == "assisting technical users"
        assert 'Trigger: "assisting technical users"' in result[0].content
        assert 'Instruction: "ask for CLI preference"' in result[0].content
        assert 'Pitfall: "assume GUI workflows by default"' in result[0].content
        assert result[0].source_interaction_ids == [1, 2, 3]

    def test_extracts_structured_playbook_with_only_do_action(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that structured playbook with only do_action is correctly extracted."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        # Mock LLM response with only instruction (flat fields)
        mock_llm_client.generate_chat_response.return_value = StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    instruction="provide step-by-step instructions",
                    pitfall=None,
                    trigger="user asks for help",
                )
            ]
        )

        # Mock prompt manager
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.return_value = "mock prompt"
        request_context.prompt_manager.get_active_version.return_value = "1.2.0"

        # Disable mock mode to use the mocked LLM response
        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        assert (
            result[0].structured_data.instruction == "provide step-by-step instructions"
        )
        assert result[0].structured_data.pitfall is None
        assert result[0].structured_data.trigger == "user asks for help"

    def test_returns_empty_when_playbook_is_null(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that empty list is returned when playbook is null."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        # Mock LLM response with no playbook (empty list)
        mock_llm_client.generate_chat_response.return_value = StructuredPlaybookList(
            playbooks=[]
        )

        # Mock prompt manager
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.return_value = "mock prompt"
        request_context.prompt_manager.get_active_version.return_value = "1.2.0"

        # Disable mock mode to use the mocked LLM response
        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert result == []

    def test_returns_empty_on_invalid_response_format(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that empty list is returned when response format is invalid."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        # Mock LLM response with invalid format (string instead of dict)
        mock_llm_client.generate_chat_response.return_value = "invalid response"

        # Mock prompt manager
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.return_value = "mock prompt"
        request_context.prompt_manager.get_active_version.return_value = "1.2.0"

        # Disable mock mode to use the mocked LLM response
        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert result == []


# ===============================
# Test: _build_user_playbook + _process_structured_response_list Unit Tests
# ===============================


class TestBuildUserPlaybook:
    """
    Direct unit tests for the per-entry _build_user_playbook helper and the
    list-processing _process_structured_response_list method.
    """

    def test_builds_user_playbook_from_single_entry(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that _build_user_playbook correctly handles a single StructuredPlaybookContent entry."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        entry = StructuredPlaybookContent(
            instruction="validate inputs",
            pitfall="trust user data blindly",
            trigger="processing external data",
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is not None
        assert result.structured_data.instruction == "validate inputs"
        assert result.structured_data.pitfall == "trust user data blindly"
        assert result.structured_data.trigger == "processing external data"
        assert result.playbook_name == extractor_config.extractor_name

    def test_returns_none_for_entry_without_content(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that _build_user_playbook returns None when entry has no usable content."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        # No playbook: trigger and content both None
        entry = StructuredPlaybookContent()

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is None

    def test_passes_source_interaction_ids(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that _build_user_playbook attaches the supplied source_interaction_ids."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        entry = StructuredPlaybookContent(
            instruction="validate inputs",
            trigger="processing external data",
        )

        result = extractor._build_user_playbook(
            entry, source_interaction_ids=[10, 20, 30]
        )

        assert result is not None
        assert result.source_interaction_ids == [10, 20, 30]

    def test_process_structured_response_list_returns_empty_for_empty_list(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """An empty StructuredPlaybookList yields no UserPlaybook entries."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        response = StructuredPlaybookList(playbooks=[])

        result = extractor._process_structured_response_list(
            response, source_interaction_ids=[]
        )

        assert result == []

    def test_process_structured_response_list_filters_invalid_entries(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Entries without usable content are dropped while valid ones are kept."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        response = StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    instruction="validate inputs",
                    trigger="processing external data",
                ),
                # No content + no trigger → has_content == False, must be filtered out
                StructuredPlaybookContent(),
            ]
        )

        result = extractor._process_structured_response_list(
            response, source_interaction_ids=[7, 8]
        )

        assert len(result) == 1
        assert result[0].structured_data.trigger == "processing external data"
        assert result[0].source_interaction_ids == [7, 8]

    def test_process_structured_response_list_emits_multiple_user_playbooks(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Multiple valid entries become multiple UserPlaybook objects sharing source IDs."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        response = StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    instruction="explain root cause first",
                    trigger="user asks for help debugging an error",
                    content="When users ask for debugging help, explain the root cause before proposing fixes.",
                ),
                StructuredPlaybookContent(
                    instruction="deliver corrections without unnecessary apologies",
                    trigger="agent provides a factual correction during debugging",
                    content="Reserve apologies for genuine mistakes, not routine corrections.",
                ),
            ]
        )

        result = extractor._process_structured_response_list(
            response, source_interaction_ids=[1, 2, 3]
        )

        assert len(result) == 2
        triggers = {p.structured_data.trigger for p in result}
        assert triggers == {
            "user asks for help debugging an error",
            "agent provides a factual correction during debugging",
        }
        assert all(p.source_interaction_ids == [1, 2, 3] for p in result)
        assert all(p.playbook_name == extractor_config.playbook_name for p in result)

    def test_mock_mode_routes_through_process_structured_response_list(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """The MOCK_LLM_RESPONSE branch must build a StructuredPlaybookList
        and feed it through _process_structured_response_list — pinning the
        contract that mock-mode and real-mode share the same UserPlaybook
        construction path.
        """
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with (
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}),
            patch.object(
                extractor,
                "_process_structured_response_list",
                wraps=extractor._process_structured_response_list,
            ) as spy_process,
        ):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert spy_process.call_count == 1
        ((response_arg,), kwargs) = spy_process.call_args
        assert isinstance(response_arg, StructuredPlaybookList)
        assert len(response_arg.playbooks) == 1
        # source_interaction_ids must come from the input interactions, not
        # be re-derived inside _process_structured_response_list
        assert kwargs["source_interaction_ids"] == [1, 2, 3]
        assert len(result) == 1
        assert result[0].source_interaction_ids == [1, 2, 3]


# ===============================
# Test: Blocking Issue Round-Trip
# ===============================


class TestBlockingIssueRoundTrip:
    """Tests for blocking_issue field in structured playbook extraction."""

    def test_build_user_playbook_with_blocking_issue(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that _build_user_playbook correctly populates blocking_issue on UserPlaybook."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        entry = StructuredPlaybookContent(
            instruction="inform user that file deletion requires admin approval",
            pitfall="attempt to delete files without permission",
            trigger="user asks to delete shared files",
            blocking_issue=BlockingIssue(
                kind=BlockingIssueKind.PERMISSION_DENIED,
                details="Agent lacks admin-level file deletion permissions on shared drives",
            ),
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is not None
        assert result.structured_data.blocking_issue is not None
        assert (
            result.structured_data.blocking_issue.kind
            == BlockingIssueKind.PERMISSION_DENIED
        )
        assert (
            "admin-level file deletion" in result.structured_data.blocking_issue.details
        )
        assert "Blocked by:" in result.content
        assert "[permission_denied]" in result.content

    def test_build_user_playbook_without_blocking_issue(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that _build_user_playbook works correctly when blocking_issue is None."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        entry = StructuredPlaybookContent(
            instruction="validate inputs",
            trigger="processing external data",
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is not None
        assert result.structured_data.blocking_issue is None
        assert "Blocked by:" not in result.content

    def test_extracts_playbook_with_blocking_issue_end_to_end(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test end-to-end extraction with blocking_issue included in LLM response."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        mock_llm_client.generate_chat_response.return_value = StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    instruction="suggest using the API endpoint instead",
                    pitfall="attempt to access the database directly",
                    trigger="user requests direct database access",
                    blocking_issue=BlockingIssue(
                        kind=BlockingIssueKind.MISSING_TOOL,
                        details="No direct database query tool available",
                    ),
                )
            ]
        )

        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.return_value = "mock prompt"
        request_context.prompt_manager.get_active_version.return_value = "2.0.0"

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        assert result[0].structured_data.blocking_issue is not None
        assert (
            result[0].structured_data.blocking_issue.kind
            == BlockingIssueKind.MISSING_TOOL
        )
        assert (
            result[0].structured_data.blocking_issue.details
            == "No direct database query tool available"
        )
        assert "Blocked by: [missing_tool]" in result[0].content


# ===============================
# Test: Rationale Field Round-Trip
# ===============================


class TestRationaleRoundTrip:
    """Tests for rationale field flowing through the playbook extraction pipeline."""

    def test_extraction_preserves_rationale(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that rationale flows from LLM response through to UserPlaybook structured_data."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        mock_llm_client.generate_chat_response.return_value = StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    rationale="Users need to understand the approach before seeing code",
                    trigger="User asks for debugging help",
                    instruction="Outline strategy before writing code",
                    pitfall="Jumping straight to code fixes",
                )
            ]
        )

        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.return_value = "mock prompt"
        request_context.prompt_manager.get_active_version.return_value = "1.2.0"

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        playbook = result[0]

        # Verify rationale is preserved in structured_data
        assert (
            playbook.structured_data.rationale
            == "Users need to understand the approach before seeing code"
        )

        # Verify the other structured_data fields are populated correctly
        assert playbook.structured_data.trigger == "User asks for debugging help"
        assert (
            playbook.structured_data.instruction
            == "Outline strategy before writing code"
        )
        assert playbook.structured_data.pitfall == "Jumping straight to code fixes"
        assert playbook.structured_data.embedding_text == "User asks for debugging help"

        # Verify rationale appears in the formatted playbook content string
        assert (
            'Rationale: "Users need to understand the approach before seeing code"'
            in playbook.content
        )
        assert 'Trigger: "User asks for debugging help"' in playbook.content
        assert 'Instruction: "Outline strategy before writing code"' in playbook.content
        assert 'Pitfall: "Jumping straight to code fixes"' in playbook.content


# ===============================
# Test: Freeform AgentPlaybook Extraction
# ===============================


class TestPlaybookContentExtraction:
    """Tests for playbook content (freeform summary) handling in _build_user_playbook."""

    def test_playbook_content_used_as_primary_content(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that LLM-provided playbook content is used directly (not derived from structured fields)."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        entry = StructuredPlaybookContent(
            content="Agent should check accounts directly when users report persistent login issues after prior attempts.",
            trigger="User reports a login issue after already trying password reset",
            instruction="Check account directly instead of suggesting generic steps",
            pitfall="Suggest password reset when user already tried it",
            rationale="The agent ignored the user's prior attempt, causing frustration.",
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is not None
        # playbook content is the LLM's freeform summary, NOT formatted structured fields
        assert (
            result.content
            == "Agent should check accounts directly when users report persistent login issues after prior attempts."
        )
        assert "Trigger:" not in result.content
        # structured_data fields are all populated
        assert (
            result.structured_data.trigger
            == "User reports a login issue after already trying password reset"
        )
        assert (
            result.structured_data.instruction
            == "Check account directly instead of suggesting generic steps"
        )
        assert (
            result.structured_data.pitfall
            == "Suggest password reset when user already tried it"
        )
        assert (
            result.structured_data.rationale
            == "The agent ignored the user's prior attempt, causing frustration."
        )
        assert result.structured_data.embedding_text == result.structured_data.trigger

    def test_fallback_to_formatted_structured_when_no_playbook_content(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test backward compatibility: if LLM omits playbook content, fall back to formatted structured fields."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        entry = StructuredPlaybookContent(
            trigger="User asks for help debugging",
            instruction="Outline strategy before writing code",
            pitfall="Jump straight to code fixes",
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is not None
        # Falls back to formatted structured fields
        assert "Trigger:" in result.content
        assert "Instruction:" in result.content

    def test_playbook_content_only_still_works(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that playbook content alone (no structured fields) still produces a valid UserPlaybook."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        entry = StructuredPlaybookContent(
            content="Agent over-apologizes when delivering factual corrections",
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is not None
        assert (
            result.content
            == "Agent over-apologizes when delivering factual corrections"
        )
        assert result.structured_data.embedding_text == result.content

    def test_end_to_end_with_playbook_content(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test end-to-end extraction where LLM returns playbook content + structured fields."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        mock_llm_client.generate_chat_response.return_value = StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    content="Agent should limit apologies and focus on clear, concise responses during billing inquiries.",
                    trigger="User reports a billing concern",
                    instruction="Limit apologies and provide concise responses",
                    pitfall="Over-apologizing in routine support interactions",
                )
            ]
        )

        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.return_value = "mock prompt"
        request_context.prompt_manager.get_active_version.return_value = "3.0.0"

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        assert (
            result[0].content
            == "Agent should limit apologies and focus on clear, concise responses during billing inquiries."
        )
        assert result[0].structured_data.trigger == "User reports a billing concern"
