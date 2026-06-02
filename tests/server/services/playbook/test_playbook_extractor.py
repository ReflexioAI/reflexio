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
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    PlaybookConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.resumable_agent import (
    FINISH_EXTRACTION_TOOL_NAME,
)
from reflexio.server.services.playbook.playbook_extractor import PlaybookExtractor
from reflexio.server.services.playbook.playbook_generation_service import (
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
    StructuredPlaybookList,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import AgentRunStatus

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
def sqlite_storage(temp_storage_dir):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(
            org_id="test_org", db_path=f"{temp_storage_dir}/reflexio.db"
        )


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

    Note: Stride checking is handled upstream by BaseGenerationService._filter_configs_by_stride()
    before the extractor is created, so stride_size tests are at the service level.
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
            window_size_override=50,
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

        assert isinstance(result, list)
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

        assert isinstance(result, list)
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


class TestResumableAgentPath:
    """Tests for the config-gated resumable playbook extraction path."""

    def test_generates_playbook_and_finalizes_agent_run(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        request_context.storage = sqlite_storage
        request_context.configurator.get_config = MagicMock(
            return_value=Config(
                storage_config=StorageConfigSQLite(),
                pending_tool_call_config=PendingToolCallConfig(enabled=True),
            )
        )
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.side_effect = (
            lambda prompt_id, variables: f"{prompt_id}: {variables}"
        )

        make_tc, _make_stop = tool_call_completion
        response = make_tc(
            FINISH_EXTRACTION_TOOL_NAME,
            {
                "playbooks": [
                    {
                        "trigger": "User asks about deployments",
                        "content": "Prefer ECS deployment guidance.",
                        "rationale": "The workspace uses AWS ECS.",
                    }
                ]
            },
        )
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
                return_value=True,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
            playbooks = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(playbooks) == 1
        assert playbooks[0].content == "Prefer ECS deployment guidance."
        assert playbooks[0].source_interaction_ids == [1, 2, 3]
        row = sqlite_storage.conn.execute("SELECT id FROM _agent_runs").fetchone()
        assert row is not None
        run = sqlite_storage.get_agent_run(row["id"])
        assert run is not None
        assert run.status == AgentRunStatus.AGENT_COMPLETED
        assert run.binding.org_id == "test_org"
        assert run.binding.user_id is None
        assert run.binding.extractor_kind == "playbook"
        assert run.binding.source_interaction_ids == [1, 2, 3]


# ===============================
# Test: Structured AgentPlaybook Extraction
# ===============================


class TestStructuredPlaybookExtraction:
    """Structured playbook extraction now routes through the always-on
    ``finish_extraction`` tool loop. The model emits its playbooks in the
    ``finish_extraction`` tool-call payload; the loop reads ``resp.tool_calls``
    and the extractor materialises ``result.output`` into ``UserPlaybook``
    entries. A degenerate loop that produces no usable finish_extraction output
    leaves ``result.output is None`` and the extractor returns ``[]``.
    """

    def _make_extractor(
        self, request_context, sqlite_storage, extractor_config, service_config
    ):
        request_context.storage = sqlite_storage
        request_context.configurator.get_config = MagicMock(
            return_value=Config(
                storage_config=StorageConfigSQLite(),
                pending_tool_call_config=PendingToolCallConfig(enabled=True),
            )
        )
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.side_effect = (
            lambda prompt_id, variables: f"{prompt_id}: {variables}"
        )
        request_context.prompt_manager.get_active_version.return_value = "1.2.0"
        return PlaybookExtractor(
            request_context=request_context,
            llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

    def test_extracts_structured_playbook_with_all_fields(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """A finish_extraction payload with a fully-specified playbook is
        materialised into a UserPlaybook entry."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        extractor = self._make_extractor(
            request_context, sqlite_storage, extractor_config, service_config
        )

        make_tc, _make_stop = tool_call_completion
        response = make_tc(
            FINISH_EXTRACTION_TOOL_NAME,
            {
                "playbooks": [
                    {
                        "trigger": "assisting technical users",
                        "content": "ask for CLI preference before proceeding",
                    }
                ]
            },
        )

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
                return_value=True,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        assert result[0].trigger == "assisting technical users"
        assert result[0].content == "ask for CLI preference before proceeding"
        assert result[0].source_interaction_ids == [1, 2, 3]

    def test_extracts_structured_playbook_with_only_do_action(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """A trigger+content playbook is materialised correctly."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        extractor = self._make_extractor(
            request_context, sqlite_storage, extractor_config, service_config
        )

        make_tc, _make_stop = tool_call_completion
        response = make_tc(
            FINISH_EXTRACTION_TOOL_NAME,
            {
                "playbooks": [
                    {
                        "trigger": "user asks for help",
                        "content": "provide step-by-step instructions",
                    }
                ]
            },
        )

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
                return_value=True,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        assert result[0].content == "provide step-by-step instructions"
        assert result[0].trigger == "user asks for help"

    def test_returns_empty_when_no_playbooks_emitted(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """A finish_extraction payload with an empty playbook list yields []."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        extractor = self._make_extractor(
            request_context, sqlite_storage, extractor_config, service_config
        )

        make_tc, _make_stop = tool_call_completion
        response = make_tc(FINISH_EXTRACTION_TOOL_NAME, {"playbooks": []})

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
                return_value=True,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert result == []

    def test_returns_empty_on_degenerate_loop(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """A degenerate loop where the model never emits usable finish_extraction
        output leaves ``result.output is None`` and the extractor returns []."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        extractor = self._make_extractor(
            request_context, sqlite_storage, extractor_config, service_config
        )

        # Model stops with plain text and never calls finish_extraction, so the
        # loop terminates with no structured output.
        _make_tc, make_stop = tool_call_completion
        response = make_stop()

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
                return_value=True,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
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
            trigger="processing external data",
            content="validate inputs before processing",
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is not None
        assert result.trigger == "processing external data"
        assert result.content == "validate inputs before processing"
        # Singleton extraction: playbook_name is always the singleton constant.
        assert result.playbook_name == "playbook"

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
            trigger="processing external data",
            content="validate inputs",
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
                    trigger="processing external data",
                    content="validate inputs",
                ),
                # No content + no trigger → has_content == False, must be filtered out
                StructuredPlaybookContent(),
            ]
        )

        result = extractor._process_structured_response_list(
            response, source_interaction_ids=[7, 8]
        )

        assert len(result) == 1
        assert result[0].trigger == "processing external data"
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
                    trigger="user asks for help debugging an error",
                    content="When users ask for debugging help, explain the root cause before proposing fixes.",
                ),
                StructuredPlaybookContent(
                    trigger="agent provides a factual correction during debugging",
                    content="Reserve apologies for genuine mistakes, not routine corrections.",
                ),
            ]
        )

        result = extractor._process_structured_response_list(
            response, source_interaction_ids=[1, 2, 3]
        )

        assert len(result) == 2
        triggers = {p.trigger for p in result}
        assert triggers == {
            "user asks for help debugging an error",
            "agent provides a factual correction during debugging",
        }
        assert all(p.source_interaction_ids == [1, 2, 3] for p in result)
        # Singleton extraction: playbook_name is always the singleton constant.
        assert all(p.playbook_name == "playbook" for p in result)

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
# Test: Rationale Field Round-Trip
# ===============================


class TestRationaleRoundTrip:
    """Tests for rationale field flowing through the playbook extraction pipeline."""

    def test_extraction_preserves_rationale(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """Rationale flows from the finish_extraction payload through to the
        UserPlaybook top-level fields."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        request_context.storage = sqlite_storage
        request_context.configurator.get_config = MagicMock(
            return_value=Config(
                storage_config=StorageConfigSQLite(),
                pending_tool_call_config=PendingToolCallConfig(enabled=True),
            )
        )
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.side_effect = (
            lambda prompt_id, variables: f"{prompt_id}: {variables}"
        )
        request_context.prompt_manager.get_active_version.return_value = "1.2.0"
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        make_tc, _make_stop = tool_call_completion
        response = make_tc(
            FINISH_EXTRACTION_TOOL_NAME,
            {
                "playbooks": [
                    {
                        "rationale": "Users need to understand the approach before seeing code",
                        "trigger": "User asks for debugging help",
                        "content": "Outline strategy before writing code",
                    }
                ]
            },
        )

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
                return_value=True,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        playbook = result[0]

        # Verify rationale is preserved as top-level field
        assert (
            playbook.rationale
            == "Users need to understand the approach before seeing code"
        )

        # Verify the other top-level fields are populated correctly
        assert playbook.trigger == "User asks for debugging help"
        assert playbook.content == "Outline strategy before writing code"


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
            rationale="The agent ignored the user's prior attempt, causing frustration.",
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        assert result is not None
        # playbook content is the LLM's freeform summary
        assert (
            result.content
            == "Agent should check accounts directly when users report persistent login issues after prior attempts."
        )
        # top-level fields are populated
        assert (
            result.trigger
            == "User reports a login issue after already trying password reset"
        )
        assert (
            result.rationale
            == "The agent ignored the user's prior attempt, causing frustration."
        )

    def test_fallback_to_formatted_structured_when_no_playbook_content(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Entry with trigger but no content should be rejected (content is required)."""
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        entry = StructuredPlaybookContent(
            trigger="User asks for help debugging",
        )

        result = extractor._build_user_playbook(entry, source_interaction_ids=[])

        # Without content, the entry has no actionable content and is rejected
        assert result is None

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

    def test_end_to_end_with_playbook_content(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """End-to-end extraction (through the finish_extraction loop) where the
        finish payload carries playbook content + structured fields."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        request_context.storage = sqlite_storage
        request_context.configurator.get_config = MagicMock(
            return_value=Config(
                storage_config=StorageConfigSQLite(),
                pending_tool_call_config=PendingToolCallConfig(enabled=True),
            )
        )
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.side_effect = (
            lambda prompt_id, variables: f"{prompt_id}: {variables}"
        )
        request_context.prompt_manager.get_active_version.return_value = "3.0.0"
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        make_tc, _make_stop = tool_call_completion
        response = make_tc(
            FINISH_EXTRACTION_TOOL_NAME,
            {
                "playbooks": [
                    {
                        "content": "Agent should limit apologies and focus on clear, concise responses during billing inquiries.",
                        "trigger": "User reports a billing concern",
                    }
                ]
            },
        )

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
                return_value=True,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
            result = extractor.extract_playbook_entries(
                sample_request_interaction_models
            )

        assert len(result) == 1
        assert (
            result[0].content
            == "Agent should limit apologies and focus on clear, concise responses during billing inquiries."
        )
        assert result[0].trigger == "User reports a billing concern"
