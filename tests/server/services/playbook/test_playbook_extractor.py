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
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.playbook.components.extractor import (
    PlaybookExtractor,
)
from reflexio.server.services.playbook.playbook_evidence import (
    candidate_rejection_reason,
    resolve_verbatim_source_span,
    strict_output_validation_errors,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookEvidenceSource,
    PlaybookEvidenceUnit,
    StructuredPlaybookContent,
    StructuredPlaybookEvidence,
    StructuredPlaybookList,
    StructuredReferencedExtractedPlaybookContent,
    StructuredReferencedExtractedPlaybookList,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import AgentRunStatus
from reflexio.test_support.llm_mock import make_structured_finish


def _evidence_unit(
    evidence_ref: str,
    source_span: str,
    *,
    interaction_id: int = 1,
    role: str = "user",
) -> PlaybookEvidenceUnit:
    turn_ref = evidence_ref.split(":", maxsplit=1)[0]
    return PlaybookEvidenceUnit(
        evidence_ref=evidence_ref,
        turn_ref=turn_ref,
        source_span=source_span,
        interaction_id=interaction_id,
        request_id=f"request-{interaction_id}",
        role=role,
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
        session_id="test_session",
        created_at=1000,
        source="api",
    )
    request2 = Request(
        request_id="req2",
        user_id="user2",
        session_id="test_session",
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

    def test_run_bookmark_advance_carries_all_users_interactions(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """run()'s outcome.bookmark_advance carries interactions from all users.

        The extractor no longer self-advances the bookmark (F1); it defers the
        advance onto the ExtractionOutcome for persist to apply atomically with
        the playbook row writes.
        """
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

        # Bookmark is NOT self-advanced anymore — no upsert during run().
        request_context.storage.upsert_operation_state.assert_not_called()

        assert isinstance(result, ExtractionOutcome)
        advance = result.bookmark_advance
        assert advance is not None
        processed_ids = [
            interaction.interaction_id for interaction in advance.processed_interactions
        ]
        assert 1 in processed_ids
        assert 2 in processed_ids
        assert 3 in processed_ids


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

        assert isinstance(result, ExtractionOutcome)
        assert len(result.items) > 0
        assert all(isinstance(f, UserPlaybook) for f in result.items)

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

        assert isinstance(result, ExtractionOutcome)
        assert len(result.items) == 1
        assert result.items[0].source_interaction_ids == [3]

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

    def test_run_carries_bookmark_advance_on_success(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """After successful extraction the outcome carries a bookmark advance.

        The extractor no longer writes the bookmark itself (F1) — it defers the
        advance onto the ExtractionOutcome for persist to apply.
        """
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

        assert isinstance(result, ExtractionOutcome)
        assert result.bookmark_advance is not None
        # The advance is deferred, not applied inside run().
        request_context.storage.upsert_operation_state.assert_not_called()


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
        request_context.prompt_manager.get_active_version.side_effect = (
            lambda prompt_id: (
                "4.6.0" if prompt_id == "playbook_extraction_context" else "1.2.0"
            )
        )
        sample_request_interaction_models[0].interactions[
            0
        ].content = "I prefer ECS deployment guidance"

        make_tc, _make_stop = tool_call_completion
        response = make_structured_finish(
            {
                "playbooks": [
                    {
                        "trigger": "User asks about deployments",
                        "content": "Prefer ECS deployment guidance.",
                        "rationale": "The user's request provides a concrete deployment signal that applies to future deployment tasks.",
                        "evidence_kind": "preference",
                        "evidence_refs": ["T1"],
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
        # The model's wording stays: it does not reach past the cited span, so
        # there is nothing to narrow.
        assert playbooks[0].content == "Prefer ECS deployment guidance."
        assert playbooks[0].source_interaction_ids == [1]
        assert playbooks[0].source_span == "I prefer ECS deployment guidance"
        row = sqlite_storage.conn.execute("SELECT id FROM _agent_runs").fetchone()
        assert row is not None
        run = sqlite_storage.get_agent_run(row["id"])
        assert run is not None
        assert run.status == AgentRunStatus.AGENT_COMPLETED
        assert run.binding.org_id == "test_org"
        assert run.binding.user_id is None
        assert run.binding.extractor_kind == "playbook"
        assert run.binding.source_interaction_ids == [1, 2, 3]
        assert run.generation_request_snapshot["output_schema_name"] == (
            "StructuredReferencedExtractedPlaybookList"
        )


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
        request_context.prompt_manager.get_active_version.side_effect = (
            lambda prompt_id: (
                "4.6.0" if prompt_id == "playbook_extraction_context" else "1.2.0"
            )
        )
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
        response = make_structured_finish(
            {
                "playbooks": [
                    {
                        "trigger": "assisting technical users",
                        "content": "ask for CLI preference before proceeding",
                        "rationale": "The user correction shows that this preference changes the successful path for similar tasks.",
                        "evidence_kind": "correction",
                        "evidence_refs": ["T3"],
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
        assert result[0].source_interaction_ids == [3]

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
        response = make_structured_finish(
            {
                "playbooks": [
                    {
                        "trigger": "user asks for help",
                        "content": "provide step-by-step instructions",
                        "rationale": "The explicit correction shows a reusable need for ordered instructions on similar tasks.",
                        "evidence_kind": "correction",
                        "evidence_refs": ["T3"],
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
        response = make_structured_finish({"playbooks": []})

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

    def test_degenerate_loop_is_terminal_extraction_failure(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """Missing structured output must not masquerade as a valid empty result."""
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
            pytest.raises(
                RuntimeError,
                match="Playbook extraction failed without structured output",
            ),
        ):
            extractor.extract_playbook_entries(sample_request_interaction_models)


# ===============================
# Test: _build_user_playbook + _process_structured_response_list Unit Tests
# ===============================


class TestBuildUserPlaybook:
    """
    Direct unit tests for the per-entry _build_user_playbook helper and the
    list-processing _process_structured_response_list method.
    """

    def test_resolves_validated_local_evidence_to_persisted_provenance(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )
        entry = StructuredPlaybookContent(
            rationale="The explicit correction applies to future deployment reviews and prevents skipping the required verification.",
            evidence_kind="correction",
            future_task_class="deployment reviews",
            improvement_mechanism="prevents skipping the required verification",
            trigger="when reviewing a deployment change",
            content="Run the regional deployment checklist before approval.",
            reader_angle="correction",
            notes="Applies only to regional deployments.",
            evidence=[
                StructuredPlaybookEvidence(
                    turn_ref="T7", source_span="Use the regional checklist"
                )
            ],
        )
        sources = {
            "T7": PlaybookEvidenceSource(
                interaction_id=712,
                request_id="request-44",
                evidence_texts=("Please: Use the regional checklist before approval.",),
            )
        }

        result = extractor._build_user_playbook(
            entry, source_interaction_ids=[], evidence_sources=sources
        )

        assert result is not None
        assert result.source_interaction_ids == [712]
        assert result.source_span == "Use the regional checklist"
        assert result.reader_angle == "correction"
        assert result.notes == "Applies only to regional deployments."

    def test_resolves_evidence_unit_refs_without_model_authored_quotes(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )
        entry = StructuredReferencedExtractedPlaybookContent(
            rationale="The correction supports a reusable deployment check.",
            evidence_kind="correction",
            trigger="When reviewing a regional deployment",
            content="Run the regional checklist before approval.",
            evidence_refs=["T7"],
        )
        sources = {
            "T7": PlaybookEvidenceSource(
                interaction_id=712,
                request_id="request-44",
                evidence_texts=("Use the regional checklist",),
                role="user",
            )
        }
        units = {
            "T7": _evidence_unit("T7", "Use the regional checklist", interaction_id=712)
        }

        result = extractor._build_user_playbook(
            StructuredPlaybookContent.model_validate(entry.model_dump()),
            source_interaction_ids=[],
            evidence_sources=sources,
            evidence_units=units,
        )

        assert result is not None
        assert result.source_interaction_ids == [712]
        assert result.source_span == "Use the regional checklist"

    def test_rejects_unknown_evidence_unit_ref(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )
        entry = StructuredPlaybookContent(
            rationale="The correction supports a reusable deployment check.",
            evidence_kind="correction",
            trigger="When reviewing a regional deployment",
            content="Run the regional checklist before approval.",
            evidence_refs=["T99"],
        )

        result = extractor._build_user_playbook(
            entry,
            source_interaction_ids=[],
            evidence_sources={},
            evidence_units={},
        )

        assert result is None

    @pytest.mark.parametrize(
        ("entry_changes", "source_changes"),
        [
            ({"evidence": []}, {}),
            ({"evidence": [{"turn_ref": "T99", "source_span": "corrected"}]}, {}),
            ({"evidence": [{"turn_ref": "T1", "source_span": "paraphrased"}]}, {}),
            ({"rationale": None}, {}),
            ({"evidence_kind": None}, {}),
            ({"trigger": None}, {}),
            ({"content": None}, {}),
            ({}, {"interaction_id": 0}),
        ],
    )
    def test_rejects_malformed_or_unsupported_evidence(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        entry_changes,
        source_changes,
    ):
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )
        entry_data = {
            "rationale": "The correction applies to future support tasks and prevents repeating the omission.",
            "evidence_kind": "correction",
            "future_task_class": "support tasks",
            "improvement_mechanism": "prevents repeating the omission",
            "trigger": "when handling this support issue",
            "content": "Include the corrected field.",
            "reader_angle": "correction",
            "evidence": [{"turn_ref": "T1", "source_span": "corrected"}],
        }
        entry_data.update(entry_changes)
        source_data = {
            "interaction_id": 11,
            "request_id": "request-11",
            "evidence_texts": ("The user corrected this field.",),
        }
        source_data.update(source_changes)

        result = extractor._build_user_playbook(
            StructuredPlaybookContent.model_validate(entry_data),
            source_interaction_ids=[],
            evidence_sources={"T1": PlaybookEvidenceSource(**source_data)},
        )

        assert result is None

    def test_derives_reader_angle_from_evidence_kind(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )
        entry = StructuredPlaybookContent(
            rationale="The user's durable preference changes future output choices.",
            evidence_kind="preference",
            trigger="when selecting visuals for this user",
            content="Use low-motion visuals.",
            evidence=[
                StructuredPlaybookEvidence(
                    turn_ref="T1", source_span="I prefer low-motion visuals"
                )
            ],
        )

        result = extractor._build_user_playbook(
            entry,
            source_interaction_ids=[],
            evidence_sources={
                "T1": PlaybookEvidenceSource(
                    interaction_id=12,
                    request_id="request-12",
                    evidence_texts=("I prefer low-motion visuals",),
                    role="user",
                    request_source="create_agent",
                )
            },
        )

        assert result is not None
        assert result.reader_angle == "preference"

    def test_preference_citing_agent_turn_is_rejected_not_repaired(self):
        """A preference must quote the user, not the agent restating the user.

        The whole batch is invalid here, so the validator DOES ask for one
        corrective turn — but the reason code is what drops the row.
        """
        output = StructuredReferencedExtractedPlaybookList(
            playbooks=[
                StructuredReferencedExtractedPlaybookContent(
                    rationale="A direct preference changes future story generation.",
                    evidence_kind="preference",
                    trigger="When generating fantasy stories for this user",
                    content="Use quest-driven fantasy settings.",
                    evidence_refs=["T2"],
                )
            ]
        )
        sources = {
            "T2": PlaybookEvidenceSource(
                interaction_id=22,
                request_id="request-22",
                evidence_texts=("An epic quest-driven fantasy brief",),
                role="agent",
            )
        }
        units = {
            "T2": _evidence_unit(
                "T2",
                "An epic quest-driven fantasy brief",
                interaction_id=22,
                role="agent",
            )
        }

        assert strict_output_validation_errors(output, sources, units) == (
            "playbooks[0]: preference_without_direct_user_evidence",
        )
        assert (
            candidate_rejection_reason(output.playbooks[0], sources, units)
            == "preference_without_direct_user_evidence"
        )

    def test_candidate_rejects_turn_reference_in_persisted_rationale(self):
        entry = StructuredReferencedExtractedPlaybookContent(
            rationale="T1 contains the request, but T2 proves the repeated failure.",
            evidence_kind="observed-failure",
            trigger="When the user requests a generated artifact",
            content="Deliver the requested artifact visibly.",
            evidence_refs=["T1"],
        )
        sources = {
            "T1": PlaybookEvidenceSource(
                interaction_id=31,
                request_id="request-31",
                evidence_texts=("Create it",),
                role="user",
            )
        }

        assert (
            candidate_rejection_reason(
                entry,
                sources,
                {"T1": _evidence_unit("T1", "Create it", interaction_id=31)},
            )
            == "turn_reference_in_persisted_prose"
        )

    def test_candidate_rejects_cited_turn_reference_in_persisted_rationale(self):
        evidence_sources = {
            "T1": PlaybookEvidenceSource(
                interaction_id=11,
                request_id="request-1",
                evidence_texts=("The user supplied the required value.",),
                role="user",
                request_source="test",
            )
        }
        candidate = StructuredReferencedExtractedPlaybookContent(
            evidence_kind="correction",
            rationale="T1 shows that the value was supplied.",
            trigger="When the required value is supplied",
            content="Use the supplied value without asking again.",
            evidence_refs=["T1"],
        )

        assert (
            candidate_rejection_reason(
                candidate,
                evidence_sources,
                {
                    "T1": _evidence_unit(
                        "T1",
                        "The user supplied the required value.",
                        interaction_id=11,
                    )
                },
            )
            == "turn_reference_in_persisted_prose"
        )

    def test_candidate_leaves_repetition_semantics_to_reviewer(self):
        entry = StructuredPlaybookContent(
            rationale=(
                "The status-only response forced the user to keep re-issuing the "
                "request."
            ),
            evidence_kind="correction",
            trigger="When the user requests a generated artifact",
            content="Deliver the artifact so the user does not have to repeat it.",
            evidence=[
                StructuredPlaybookEvidence(turn_ref="T1", source_span="Create it"),
                StructuredPlaybookEvidence(turn_ref="T2", source_span="status=success"),
            ],
        )
        sources = {
            "T1": PlaybookEvidenceSource(
                interaction_id=32,
                request_id="request-32",
                evidence_texts=("Create it",),
                role="user",
            ),
            "T2": PlaybookEvidenceSource(
                interaction_id=33,
                request_id="request-33",
                evidence_texts=("status=success",),
                role="agent",
            ),
        }

        # The evidence references and spans are structurally valid. Whether the
        # prose overclaims repetition is a semantic judgment for the reviewer,
        # not a phrase-list heuristic in provenance validation.
        assert candidate_rejection_reason(entry, sources) is None

    def test_candidate_accepts_user_repetition_claim_with_two_user_turns(self):
        entry = StructuredPlaybookContent(
            rationale="The same request recurred after only a status was visible.",
            evidence_kind="observed-failure",
            trigger="When the user requests a generated artifact",
            content="Deliver the requested artifact visibly.",
            evidence=[
                StructuredPlaybookEvidence(turn_ref="T1", source_span="Create it"),
                StructuredPlaybookEvidence(turn_ref="T2", source_span="Create it"),
            ],
        )
        sources = {
            "T1": PlaybookEvidenceSource(
                interaction_id=34,
                request_id="request-34",
                evidence_texts=("Create it",),
                role="user",
            ),
            "T2": PlaybookEvidenceSource(
                interaction_id=35,
                request_id="request-35",
                evidence_texts=("Create it",),
                role="user",
            ),
        }

        assert candidate_rejection_reason(entry, sources) is None

    def test_invalid_preference_does_not_erase_valid_sibling(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )
        correction = "Please execute the requested task now."
        synthesized = "An epic quest-driven fantasy brief"
        output = StructuredReferencedExtractedPlaybookList(
            playbooks=[
                StructuredReferencedExtractedPlaybookContent(
                    rationale="The direct correction prevents repeated explanation.",
                    evidence_kind="correction",
                    trigger="When the user requests this task",
                    content="Execute the requested task.",
                    evidence_refs=["T1"],
                ),
                StructuredReferencedExtractedPlaybookContent(
                    rationale="The task brief states a reusable preference.",
                    evidence_kind="preference",
                    trigger="When writing fantasy content",
                    content="Use epic quest-driven fantasy.",
                    evidence_refs=["T2"],
                ),
            ]
        )
        sources = {
            "T1": PlaybookEvidenceSource(
                interaction_id=31,
                request_id="request-31",
                evidence_texts=(correction,),
                role="user",
                request_source="create_agent",
            ),
            "T2": PlaybookEvidenceSource(
                interaction_id=32,
                request_id="request-32",
                evidence_texts=(synthesized,),
                role="agent",
            ),
        }
        units = {
            "T1": _evidence_unit("T1", correction, interaction_id=31, role="user"),
            "T2": _evidence_unit("T2", synthesized, interaction_id=32, role="agent"),
        }

        # One bad sibling must not cost the batch a repair turn: the valid
        # candidate is still usable, so there is nothing for repair to fix.
        assert not strict_output_validation_errors(output, sources, units)
        result = extractor._process_structured_response_list(
            output,
            evidence_sources=sources,
            evidence_units=units,
        )

        assert len(result) == 1
        assert result[0].content == "Execute the requested task."
        assert result[0].source_interaction_ids == [31]

    def test_preference_validator_accepts_direct_user_evidence(self):
        output = StructuredReferencedExtractedPlaybookList(
            playbooks=[
                StructuredReferencedExtractedPlaybookContent(
                    rationale="A direct preference changes future analysis output.",
                    evidence_kind="preference",
                    trigger="When preparing future analysis for this user",
                    content="Lead with a compact table.",
                    evidence_refs=["T1"],
                )
            ]
        )
        sources = {
            "T1": PlaybookEvidenceSource(
                interaction_id=21,
                request_id="request-21",
                evidence_texts=("For future analyses, lead with a compact table",),
                role="user",
                request_source="create_agent",
            )
        }

        assert not strict_output_validation_errors(
            output,
            sources,
            {
                "T1": _evidence_unit(
                    "T1",
                    "For future analyses, lead with a compact table",
                    interaction_id=21,
                )
            },
        )

    def test_build_playbook_does_not_semantically_rewrite_model_content(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )
        entry = StructuredPlaybookContent(
            rationale="The direct preference changes future analysis output.",
            evidence_kind="preference",
            trigger="When preparing future analysis for this user",
            content="Lead with a compact table and include citations.",
            evidence=[
                StructuredPlaybookEvidence(
                    turn_ref="T1",
                    source_span="For future analyses, lead with a compact table",
                )
            ],
        )

        result = extractor._build_user_playbook(
            entry,
            source_interaction_ids=[],
            evidence_sources={
                "T1": PlaybookEvidenceSource(
                    interaction_id=21,
                    request_id="request-21",
                    evidence_texts=("For future analyses, lead with a compact table",),
                    role="user",
                    request_source="create_agent",
                )
            },
        )

        assert result is not None
        # Provenance validation does not guess which words are semantically
        # unsupported. The reviewer receives this candidate and must narrow or
        # reject it based on the complete chronology.
        assert result.content == "Lead with a compact table and include citations."
        assert result.trigger == "When preparing future analysis for this user"

    def test_preference_validator_ignores_synthesized_memory_signal(self):
        output = StructuredReferencedExtractedPlaybookList(playbooks=[])
        sources = {
            "T4": PlaybookEvidenceSource(
                interaction_id=24,
                request_id="request-24",
                evidence_texts=("I prefer concise status updates",),
                role="user",
                request_source="store_memory_tool",
            )
        }

        assert not strict_output_validation_errors(output, sources)

    def test_resolves_typographic_quotes_to_exact_persisted_source(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )
        source = PlaybookEvidenceSource(
            interaction_id=91,
            request_id="request-91",
            evidence_texts=("Deliver the “finished artifact” now.",),
        )
        entry = StructuredPlaybookContent(
            rationale="The user explicitly required visible delivery.",
            evidence_kind="correction",
            trigger="when an artifact is requested",
            content="Deliver the artifact itself.",
            evidence=[
                StructuredPlaybookEvidence(
                    turn_ref="T1",
                    source_span='Deliver the "finished artifact" now.',
                )
            ],
        )

        result = extractor._build_user_playbook(
            entry,
            source_interaction_ids=[],
            evidence_sources={"T1": source},
        )

        assert result is not None
        assert result.source_span == "Deliver the “finished artifact” now."

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
        # The LLM sees only local turn labels; production code resolves them
        # through the retained evidence map after generation.
        assert set(kwargs["evidence_sources"]) == {"T1", "T2", "T3"}
        assert len(result) == 1
        assert result[0].source_interaction_ids == [3]


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
        response = make_structured_finish(
            {
                "playbooks": [
                    {
                        "rationale": "Users need to understand the approach before seeing code",
                        "evidence_kind": "correction",
                        "future_task_class": "debugging tasks",
                        "improvement_mechanism": "clarifies the approach before implementation",
                        "trigger": "User asks for debugging help",
                        "content": "Outline strategy before writing code",
                        "reader_angle": "correction",
                        "evidence_refs": ["T3"],
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
        response = make_structured_finish(
            {
                "playbooks": [
                    {
                        "content": "Agent should limit apologies and focus on clear, concise responses during billing inquiries.",
                        "trigger": "User reports a billing concern",
                        "rationale": "The user's correction demonstrates a response-shape problem that can recur in similar support tasks.",
                        "evidence_kind": "correction",
                        "future_task_class": "billing support tasks",
                        "improvement_mechanism": "keeps the response focused on resolution",
                        "reader_angle": "correction",
                        "evidence_refs": ["T3"],
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


# ===============================
# Evidence-grounding primitives
# ===============================


class TestVerbatimSpanResolution:
    """Spans must resolve to exactly one real substring, never fuzzily."""

    def test_exact_substring_resolves(self):
        assert (
            resolve_verbatim_source_span("dark mode", ("I prefer dark mode.",))
            == "dark mode"
        )

    def test_typographic_quotes_resolve_to_the_original_text(self):
        # The model re-typed a curly apostrophe as a straight one.
        resolved = resolve_verbatim_source_span(
            "don't ship it",
            ("Please don’t ship it today.",),  # noqa: RUF001
        )
        assert resolved == "don’t ship it"  # noqa: RUF001

    def test_ambiguous_normalized_match_is_rejected(self):
        """Two normalized-only positions: we cannot say which was quoted."""
        resolved = resolve_verbatim_source_span(
            "don't",
            ("don’t ship it and don’t merge it",),  # noqa: RUF001
        )
        assert resolved is None

    def test_exact_match_wins_over_normalized_ambiguity(self):
        """A literal occurrence is unambiguous even if variants also appear."""
        resolved = resolve_verbatim_source_span(
            "don't",
            ("don’t ship it and don't merge it",),  # noqa: RUF001
        )
        assert resolved == "don't"

    def test_absent_span_is_rejected(self):
        assert resolve_verbatim_source_span("never said", ("something else",)) is None


class TestStrictBatchValidation:
    """The validator asks for repair only when the whole batch is unusable."""

    @staticmethod
    def _sources(role: str = "user") -> dict[str, PlaybookEvidenceSource]:
        return {
            "T1": PlaybookEvidenceSource(
                interaction_id=1,
                request_id="r1",
                evidence_texts=("Please execute the task now.",),
                role=role,
            )
        }

    @staticmethod
    def _units(role: str = "user") -> dict[str, PlaybookEvidenceUnit]:
        return {"T1": _evidence_unit("T1", "Please execute the task now.", role=role)}

    @staticmethod
    def _candidate(**overrides) -> StructuredReferencedExtractedPlaybookContent:
        fields = {
            "rationale": "The correction applies to future runs of this task.",
            "evidence_kind": "correction",
            "trigger": "When the user requests this task",
            "content": "Execute the requested task.",
            "evidence_refs": ["T1"],
        }
        fields.update(overrides)
        return StructuredReferencedExtractedPlaybookContent(**fields)

    def test_empty_batch_is_a_valid_answer(self):
        output = StructuredReferencedExtractedPlaybookList(playbooks=[])
        assert (
            strict_output_validation_errors(output, self._sources(), self._units())
            == ()
        )

    def test_fully_valid_batch_passes(self):
        output = StructuredReferencedExtractedPlaybookList(
            playbooks=[self._candidate()]
        )
        assert (
            strict_output_validation_errors(output, self._sources(), self._units())
            == ()
        )

    def test_wholly_ungrounded_batch_reports_every_candidate(self):
        bad = self._candidate(evidence_refs=["T9"])
        output = StructuredReferencedExtractedPlaybookList(
            playbooks=[bad, bad.model_copy()]
        )

        errors = strict_output_validation_errors(output, self._sources(), self._units())

        assert errors == (
            "playbooks[0]: unknown_evidence_ref",
            "playbooks[1]: unknown_evidence_ref",
        )

    def test_one_bad_sibling_does_not_trigger_repair(self):
        bad = self._candidate(evidence_refs=["T9"])
        output = StructuredReferencedExtractedPlaybookList(
            playbooks=[self._candidate(), bad]
        )

        # A usable candidate survives, so repair has nothing to fix.
        assert (
            strict_output_validation_errors(output, self._sources(), self._units())
            == ()
        )

    def test_validator_does_not_mutate_the_output(self):
        output = StructuredReferencedExtractedPlaybookList(
            playbooks=[self._candidate()]
        )
        before = output.model_dump_json()

        strict_output_validation_errors(output, self._sources(), self._units())

        assert output.model_dump_json() == before
