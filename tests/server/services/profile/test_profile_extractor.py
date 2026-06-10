"""
Unit tests for ProfileExtractor.

Tests the extractor's new responsibilities for:
- Operation state key generation
- Interaction collection with window/stride
- Source filtering
- Operation state updates
- Integration of run() method
"""

import os
import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.extraction.resumable_agent import (
    FINISH_EXTRACTION_TOOL_NAME,
)
from reflexio.server.services.profile.profile_extractor import ProfileExtractor
from reflexio.server.services.profile.profile_generation_service import (
    ProfileGenerationServiceConfig,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LiteLLMClient)
    # Return an empty StructuredProfilesOutput for profile extraction
    client.generate_chat_response.return_value = StructuredProfilesOutput()
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
    # Mock the storage
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
    """Create a profile extractor config."""
    return ProfileExtractorConfig(
        extractor_name="test_extractor",
        extraction_definition_prompt="Extract user preferences",
    )


@pytest.fixture
def service_config():
    """Create a service config."""
    return ProfileGenerationServiceConfig(
        user_id="test_user",
        request_id="test_request",
        source="api",
    )


@pytest.fixture
def sample_interactions():
    """Create sample interactions for testing."""
    return [
        Interaction(
            interaction_id=1,
            user_id="test_user",
            content="I prefer dark mode",
            request_id="req1",
            created_at=1000,
            role="user",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user",
            content="Got it, I'll remember that preference",
            request_id="req1",
            created_at=1001,
            role="assistant",
        ),
    ]


@pytest.fixture
def sample_request_interaction_models(sample_interactions):
    """Create sample RequestInteractionDataModel objects."""
    request = Request(
        request_id="req1",
        user_id="test_user",
        session_id="test_session",
        created_at=1000,
        source="api",
    )
    return [
        RequestInteractionDataModel(
            session_id="req1",
            request=request,
            interactions=sample_interactions,
        )
    ]


# ===============================
# Test: Operation State Key
# ===============================


class TestOperationStateKey:
    """Tests for operation state key generation."""

    def test_state_manager_includes_user_id_in_bookmark_key(
        self, request_context, mock_llm_client, extractor_config, service_config
    ):
        """Test that profile extractor state manager builds keys with user_id (user-scoped)."""
        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        mgr = extractor._create_state_manager()

        assert mgr.service_name == "profile_extractor"
        assert mgr.org_id == "test_org"
        # Verify the bookmark key format includes user_id
        key = mgr._bookmark_key(name="test_extractor", scope_id=service_config.user_id)
        assert "profile_extractor" in key
        assert "test_org" in key
        assert "test_user" in key
        assert "test_extractor" in key
        assert key == "profile_extractor::test_org::test_user::test_extractor"

    def test_different_users_have_different_keys(
        self, request_context, mock_llm_client, extractor_config
    ):
        """Test that different users get different operation state keys."""
        config1 = ProfileGenerationServiceConfig(
            user_id="user1", request_id="req1", source="api"
        )
        config2 = ProfileGenerationServiceConfig(
            user_id="user2", request_id="req2", source="api"
        )

        extractor1 = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=config1,
            agent_context="Test agent",
        )
        extractor2 = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=config2,
            agent_context="Test agent",
        )

        mgr1 = extractor1._create_state_manager()
        mgr2 = extractor2._create_state_manager()
        key1 = mgr1._bookmark_key(name="test_extractor", scope_id=config1.user_id)
        key2 = mgr2._bookmark_key(name="test_extractor", scope_id=config2.user_id)
        assert key1 != key2


# ===============================
# Test: Get Interactions
# ===============================


class TestGetInteractions:
    """Tests for interaction collection logic.

    Note: Stride checking is handled upstream by BaseGenerationService._filter_configs_by_stride()
    before the extractor is created, so stride_size tests are at the service level.
    """

    def test_returns_interactions(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that interactions are returned from storage."""
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        result = extractor._get_interactions()

        assert result is not None
        assert len(result) == 1  # One session

    def test_uses_window_size_when_configured(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that window size is used to fetch interactions."""
        # Configure extractor with window size
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
            window_size_override=50,
        )

        # Mock storage
        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        extractor._get_interactions()

        # Verify get_last_k_interactions_grouped was called with correct window size
        request_context.storage.get_last_k_interactions_grouped.assert_called_once()
        call_kwargs = request_context.storage.get_last_k_interactions_grouped.call_args
        assert call_kwargs[1]["k"] == 50

    def test_returns_none_when_source_filter_skips(
        self,
        request_context,
        mock_llm_client,
        sample_request_interaction_models,
    ):
        """Test that None is returned when source filter causes skip."""
        # Configure extractor with specific sources
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
            request_sources_enabled=["mobile", "desktop"],
        )

        # Service config has source="api" which is not in enabled list
        service_config = ProfileGenerationServiceConfig(
            user_id="test_user",
            request_id="test_request",
            source="api",  # Not in enabled list
        )

        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        result = extractor._get_interactions()

        assert result is None

    def test_passes_correct_user_id_to_storage(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that user_id is passed to storage methods (user-scoped)."""
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        extractor._get_interactions()

        # Verify user_id was passed to get_last_k_interactions_grouped
        call_kwargs = request_context.storage.get_last_k_interactions_grouped.call_args[
            1
        ]
        assert call_kwargs["user_id"] == "test_user"


# ===============================
# Test: Update Operation State
# ===============================


class TestUpdateOperationState:
    """Tests for operation state update logic."""

    def test_updates_state_after_processing(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that operation state is updated with processed interactions."""
        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        extractor._update_operation_state(sample_request_interaction_models)

        # Verify upsert was called
        request_context.storage.upsert_operation_state.assert_called_once()

        # Verify state contains interaction IDs
        call_args = request_context.storage.upsert_operation_state.call_args
        state_key = call_args[0][0]
        state = call_args[0][1]

        assert "profile_extractor" in state_key
        assert "last_processed_interaction_ids" in state
        assert 1 in state["last_processed_interaction_ids"]
        assert 2 in state["last_processed_interaction_ids"]


# ===============================
# Test: Run Integration
# ===============================


class TestRun:
    """Integration tests for the run() method."""

    def test_run_collects_own_interactions_when_not_provided(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that run() collects interactions when not provided in service config."""
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )

        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        # Enable mock mode for LLM responses
        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            extractor.run()

        # Verify storage was queried for interactions
        request_context.storage.get_last_k_interactions_grouped.assert_called()

    def test_run_returns_empty_when_no_interactions(
        self,
        request_context,
        mock_llm_client,
        service_config,
    ):
        """Test that run() returns None when no interactions available."""
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
        )

        # Return empty interactions
        request_context.storage.get_last_k_interactions_grouped.return_value = (
            [],
            [],
        )

        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        result = extractor.run()

        assert result is None

    def test_run_does_not_update_bookmark_when_extraction_fails(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Run should raise and leave bookmark unchanged when extraction fails."""
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
        )
        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )
        request_context.storage.get_user_profile.return_value = []

        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )
        extractor._generate_raw_updates_from_sessions = MagicMock(
            side_effect=RuntimeError("llm timeout")
        )

        with pytest.raises(RuntimeError):
            extractor.run()

        request_context.storage.upsert_operation_state.assert_not_called()

    def test_run_updates_operation_state_on_success(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """Test that operation state is updated after successful extraction."""
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
        )

        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )
        request_context.storage.get_user_profile.return_value = []

        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            result = extractor.run()

        # Verify operation state was updated
        if result is not None:
            request_context.storage.upsert_operation_state.assert_called()


class TestResumableAgentPath:
    """Tests for the config-gated resumable profile extraction path."""

    def test_generates_profiles_and_finalizes_agent_run(
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
                "profiles": [
                    {
                        "content": "User prefers dark mode.",
                        "time_to_live": "infinity",
                    }
                ]
            },
        )
        extractor = ProfileExtractor(
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
            raw_profiles = extractor._generate_raw_updates_from_sessions(
                request_interaction_data_models=sample_request_interaction_models,
                existing_profiles=[],
            )

        assert raw_profiles[0]["content"] == "User prefers dark mode."
        row = sqlite_storage.conn.execute("SELECT id FROM _agent_runs").fetchone()
        assert row is not None
        run = sqlite_storage.get_agent_run(row["id"])
        assert run is not None
        assert run.status == AgentRunStatus.AGENT_COMPLETED
        assert run.binding.org_id == "test_org"
        assert run.binding.user_id == "test_user"
        assert run.binding.extractor_kind == "profile"
        assert run.binding.source_interaction_ids == [1, 2]

    def test_loop_still_runs_when_pending_tools_disabled(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """The extraction loop is always-on. When the pending-tool feature flag
        is disabled, the loop still runs (only ask_human/attach_pending_info
        tools are withheld) and produces profiles via finish_extraction.
        """
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
                "profiles": [
                    {
                        "content": "Loop extraction path.",
                        "time_to_live": "infinity",
                    }
                ]
            },
        )
        extractor = ProfileExtractor(
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
                return_value=False,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
            raw_profiles = extractor._generate_raw_updates_from_sessions(
                request_interaction_data_models=sample_request_interaction_models,
                existing_profiles=[],
            )

        assert raw_profiles[0]["content"] == "Loop extraction path."

    def test_attach_pending_info_request_is_org_scoped_and_run_still_finalizes(
        self,
        monkeypatch,
        request_context,
        sqlite_storage,
        extractor_config,
        service_config,
        sample_request_interaction_models,
        tool_call_completion,
    ):
        """Post-#107 a profile extractor exposes attach_pending_info_request, not
        ask_human. The attach tool does not CREATE a pending tool call; it attaches
        the current run to an existing org-scoped Prior Knowledge (ask_human)
        request. We seed that org-scoped pending call, have the agent attach to it,
        and assert the org-scoping holds and the run still finalizes wired to it.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        request_context.storage = sqlite_storage
        request_context.configurator.get_config = MagicMock(
            return_value=Config(
                storage_config=StorageConfigSQLite(),
                pending_tool_call_config=PendingToolCallConfig(
                    enabled=True,
                ),
            )
        )
        request_context.prompt_manager = MagicMock()
        request_context.prompt_manager.render_prompt.side_effect = (
            lambda prompt_id, variables: f"{prompt_id}: {variables}"
        )

        # Seed an existing org-scoped ask_human pending tool call. The attach tool
        # only succeeds against a record whose org/scope/tool_name match
        # human_feedback_scope(org_id) + tool_name="ask_human".
        org_scope = human_feedback_scope("test_org")
        seeded_question = "Which deployment standard should be treated as canonical?"
        seeded_call = PendingToolCallRecord(
            id="ptc_seeded_prior_knowledge",
            org_id="test_org",
            user_id="test_user",
            scope=org_scope,
            scope_hash=build_scope_hash(org_scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text=seeded_question,
                answer_format="short text",
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=seeded_question,
            answer_format="short text",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            cache_until=datetime.now(UTC) + timedelta(hours=1),
            valid_until=datetime.now(UTC) + timedelta(hours=1),
        )
        sqlite_storage.create_pending_tool_call(seeded_call)

        make_tc, _ = tool_call_completion
        attach_response = make_tc(
            "attach_pending_info_request",
            {
                "pending_tool_call_id": seeded_call.id,
                "why_relevant": "Canonical deployment standard affects this profile.",
            },
        )
        finish_response = make_tc(
            FINISH_EXTRACTION_TOOL_NAME,
            {
                "profiles": [
                    {
                        "content": "User prefers dark mode.",
                        "time_to_live": "infinity",
                    }
                ]
            },
        )
        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with (
            patch("litellm.completion", side_effect=[attach_response, finish_response]),
            patch(
                "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
                return_value=True,
            ),
            patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}),
        ):
            raw_profiles = extractor._generate_raw_updates_from_sessions(
                request_interaction_data_models=sample_request_interaction_models,
                existing_profiles=[],
            )

        assert raw_profiles[0]["content"] == "User prefers dark mode."

        # The attach tool does not create new pending calls; the only one is the
        # seeded org-scoped Prior Knowledge request.
        pending_calls = sqlite_storage.list_pending_tool_calls()
        assert len(pending_calls) == 1
        pending_call = pending_calls[0]
        assert pending_call.scope == {"org_id": "test_org", "scope_kind": "org"}
        assert pending_call.user_id == "test_user"

        row = sqlite_storage.conn.execute("SELECT id FROM _agent_runs").fetchone()
        assert row is not None
        run = sqlite_storage.get_agent_run(row["id"])
        assert run is not None
        assert run.status == AgentRunStatus.AGENT_COMPLETED

        # attach_pending_info_request wires the run to the existing pending call via
        # a run-tool dependency (it returns a synchronous Completed result, so it is
        # NOT recorded in run.pending_tool_call_ids the way ask_human's AsyncAccepted
        # would be). Assert the durable dependency edge instead.
        assert run.pending_tool_call_ids == []
        dependencies = sqlite_storage.list_run_tool_dependencies(run.id)
        assert [dep.pending_tool_call_id for dep in dependencies] == [pending_call.id]

    def test_run_resumable_empty_output_still_surfaces_run_id(
        self,
        request_context,
        mock_llm_client,
        service_config,
        sample_request_interaction_models,
    ):
        """A resumable run that finishes with EMPTY output (agent asked a human
        and produced no profiles yet) must still surface its run_id so the
        generation service can finalize the run to FINALIZED_PENDING_TOOL.

        Regression: previously run() returned None for empty output, dropping
        the run_id and orphaning the run in AGENT_COMPLETED so the resolve ->
        resume chain could never fire.
        """
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
        )
        request_context.storage.get_last_k_interactions_grouped.return_value = (
            sample_request_interaction_models,
            [],
        )
        request_context.storage.get_user_profile.return_value = []
        extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )
        # Simulate the resumable agent finishing with empty output while a
        # durable run row was created (and a follow-up ask persisted).
        extractor._generate_raw_updates_from_sessions = MagicMock(return_value=[])
        extractor._last_resumable_run_id = "run_empty_followup"

        result = extractor.run()

        assert isinstance(result, ExtractionOutcome)
        assert result.run_id == "run_empty_followup"
        assert result.items == []


# ===============================
# Test: Convert Raw to User Profiles
# ===============================


class TestConvertRawToUserProfiles:
    """Tests for converting raw profile dicts to UserProfile objects."""

    def _make_extractor(self, request_context, mock_llm_client, service_config):
        config = ProfileExtractorConfig(
            extractor_name="test_extractor",
            extraction_definition_prompt="Extract user preferences",
        )
        return ProfileExtractor(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

    def test_converts_valid_profiles(
        self, request_context, mock_llm_client, service_config
    ):
        """Test converting valid raw profile dicts."""
        extractor = self._make_extractor(
            request_context, mock_llm_client, service_config
        )

        raw_profiles = [
            {"content": "User prefers dark mode", "time_to_live": "one_month"},
            {"content": "User's name is John", "time_to_live": "infinity"},
        ]

        result = extractor._convert_raw_to_user_profiles(
            raw_profiles=raw_profiles,
            user_id="test_user",
            request_id="test_request",
        )

        assert len(result) == 2
        assert result[0].content == "User prefers dark mode"
        assert result[0].user_id == "test_user"
        # Singleton extraction no longer records per-extractor provenance.
        assert result[0].extractor_names is None
        assert result[1].content == "User's name is John"

    def test_skips_invalid_profiles(
        self, request_context, mock_llm_client, service_config
    ):
        """Test that invalid profile dicts are skipped."""
        extractor = self._make_extractor(
            request_context, mock_llm_client, service_config
        )

        raw_profiles = [
            {"content": "Valid profile", "time_to_live": "one_month"},
            {"no_content_key": "Invalid"},
            "not_a_dict",
        ]

        result = extractor._convert_raw_to_user_profiles(
            raw_profiles=raw_profiles,
            user_id="test_user",
            request_id="test_request",
        )

        assert len(result) == 1
        assert result[0].content == "Valid profile"

    def test_custom_features_extracted(
        self, request_context, mock_llm_client, service_config
    ):
        """Test that extra fields become custom_features."""
        extractor = self._make_extractor(
            request_context, mock_llm_client, service_config
        )

        raw_profiles = [
            {
                "content": "Likes pizza",
                "time_to_live": "one_month",
                "metadata": "pizza",
            },
        ]

        result = extractor._convert_raw_to_user_profiles(
            raw_profiles=raw_profiles,
            user_id="test_user",
            request_id="test_request",
        )

        assert len(result) == 1
        assert result[0].custom_features == {"metadata": "pizza"}
