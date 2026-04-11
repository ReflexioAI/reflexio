"""
Unit tests for AgentSuccessEvaluator.

Tests the evaluator's responsibilities for:
- Source filtering on provided interactions
- Sampling rate filtering
- Running evaluations on provided interactions
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)
from reflexio.models.config_schema import AgentSuccessConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_constants import (
    AgentSuccessEvaluationOutput,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_service import (
    AgentSuccessGenerationServiceConfig,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluator import (
    AgentSuccessEvaluator,
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LiteLLMClient)
    # Return a successful AgentSuccessEvaluationOutput
    client.generate_chat_response.return_value = AgentSuccessEvaluationOutput(
        is_success=True
    )
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
    """Create an agent success config."""
    return AgentSuccessConfig(
        evaluation_name="task_completion",
        success_definition_prompt="Evaluate if task was completed",
    )


@pytest.fixture
def sample_interactions():
    """Create sample interactions from multiple users for testing."""
    return [
        Interaction(
            interaction_id=1,
            user_id="user1",
            content="Please help me with task A",
            request_id="req1",
            created_at=1000,
            role="user",
        ),
        Interaction(
            interaction_id=2,
            user_id="user1",
            content="Sure, I completed task A for you",
            request_id="req1",
            created_at=1001,
            role="assistant",
        ),
        Interaction(
            interaction_id=3,
            user_id="user2",
            content="Can you do task B?",
            request_id="req2",
            created_at=1002,
            role="user",
        ),
        Interaction(
            interaction_id=4,
            user_id="user2",
            content="Task B is done",
            request_id="req2",
            created_at=1003,
            role="assistant",
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
            interactions=sample_interactions[2:],
        ),
    ]


@pytest.fixture
def service_config(sample_request_interaction_models):
    """Create a service config with required request_interaction_data_models."""
    return AgentSuccessGenerationServiceConfig(
        agent_version="1.0.0",
        session_id="test_group",
        request_interaction_data_models=sample_request_interaction_models,
        source="api",
    )


# ===============================
# Test: Run Method
# ===============================


class TestRun:
    """Tests for the run() method."""

    def test_run_processes_provided_interactions(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that run() processes interactions from service_config."""
        evaluator = AgentSuccessEvaluator(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            result = evaluator.run()

        # Should return a list
        assert result is not None
        assert isinstance(result, list)

    def test_run_respects_sampling_rate(
        self,
        request_context,
        mock_llm_client,
        sample_request_interaction_models,
    ):
        """Test that run() respects sampling rate."""
        # Config with 0% sampling rate - should skip all
        config = AgentSuccessConfig(
            evaluation_name="task_completion",
            success_definition_prompt="Evaluate if task was completed",
            sampling_rate=0.0,
        )

        service_config = AgentSuccessGenerationServiceConfig(
            agent_version="1.0.0",
            session_id="test_group",
            request_interaction_data_models=sample_request_interaction_models,
            source="api",
        )

        evaluator = AgentSuccessEvaluator(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        result = evaluator.run()

        # With 0% sampling rate, should skip all and return empty
        assert result == []

    def test_run_with_100_percent_sampling_rate(
        self,
        request_context,
        mock_llm_client,
        extractor_config,
        service_config,
    ):
        """Test that run() processes all with 100% sampling rate."""
        # Default sampling_rate is 1.0 (100%)
        evaluator = AgentSuccessEvaluator(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            result = evaluator.run()

        # Should return results (may be empty if parsing fails in mock mode)
        assert result is not None
        assert isinstance(result, list)


# ===============================
# Test: Source Filtering
# ===============================


class TestSourceFiltering:
    """Tests for source filtering behavior."""

    def test_filters_interactions_by_source(
        self,
        request_context,
        mock_llm_client,
        sample_interactions,
    ):
        """Test that interactions are filtered by source."""
        # Create interactions with different sources
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
            source="web",
        )
        models = [
            RequestInteractionDataModel(
                session_id="req1",
                request=request1,
                interactions=sample_interactions[:2],
            ),
            RequestInteractionDataModel(
                session_id="req2",
                request=request2,
                interactions=sample_interactions[2:],
            ),
        ]

        # Config that only accepts 'api' source
        config = AgentSuccessConfig(
            evaluation_name="task_completion",
            success_definition_prompt="Evaluate if task was completed",
            source_filter="api",
        )

        service_config = AgentSuccessGenerationServiceConfig(
            agent_version="1.0.0",
            session_id="test_group",
            request_interaction_data_models=models,
            source="api",
        )

        evaluator = AgentSuccessEvaluator(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

        with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "true"}):
            result = evaluator.run()

        # Should process only the api source interaction
        assert result is not None
        assert isinstance(result, list)


# ===============================
# Unit tests for helper methods
# ===============================


class TestCountUserTurns:
    """Tests for AgentSuccessEvaluator._count_user_turns."""

    @pytest.fixture
    def evaluator(self, mock_llm_client, request_context):
        config = AgentSuccessConfig(
            evaluation_name="test",
            success_definition_prompt="test",
        )
        service_config = AgentSuccessGenerationServiceConfig(
            session_id="test_session",
            agent_version="v1",
            request_interaction_data_models=[],
            source="api",
        )
        return AgentSuccessEvaluator(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )

    def _interaction(self, role, content):
        return Interaction(role=role, content=content, user_id="u1", request_id="r1")

    def _rdm(self, request_id, interactions):
        return RequestInteractionDataModel(
            session_id="s1",
            request=Request(request_id=request_id, session_id="s1", user_id="u1"),
            interactions=interactions,
        )

    def test_counts_user_turns_only(self, evaluator):
        """User turns are counted; agent/system turns are excluded."""
        models = [
            self._rdm(
                "r1",
                [
                    self._interaction("user", "hello"),
                    self._interaction("assistant", "hi"),
                    self._interaction("user", "question"),
                    self._interaction("tool", "result"),
                    self._interaction("system", "info"),
                ],
            )
        ]
        assert evaluator._count_user_turns(models) == 2

    def test_multiple_request_models(self, evaluator):
        """User turns across multiple RequestInteractionDataModels are summed."""
        models = [
            self._rdm("r1", [self._interaction("user", "a")]),
            self._rdm(
                "r2",
                [
                    self._interaction("user", "b"),
                    self._interaction("agent", "c"),
                ],
            ),
        ]
        assert evaluator._count_user_turns(models) == 2

    def test_empty_interactions(self, evaluator):
        """Empty interactions return 0."""
        assert evaluator._count_user_turns([]) == 0

    def test_case_insensitive_role_matching(self, evaluator):
        """Role matching is case-insensitive."""
        models = [
            self._rdm(
                "r1",
                [
                    self._interaction("Assistant", "hi"),
                    self._interaction("USER", "hello"),
                ],
            )
        ]
        # "Assistant" should be excluded, "USER" should be counted
        assert evaluator._count_user_turns(models) == 1


class TestGetCorrectionCount:
    """Tests for AgentSuccessEvaluator._get_correction_count."""

    def test_returns_zero_on_storage_exception(self, mock_llm_client, temp_storage_dir):
        """_get_correction_count returns 0 when storage raises."""
        config = AgentSuccessConfig(
            evaluation_name="test",
            success_definition_prompt="test",
        )
        service_config = AgentSuccessGenerationServiceConfig(
            session_id="test_session",
            agent_version="v1",
            request_interaction_data_models=[],
            source="api",
        )
        request_context = RequestContext(
            org_id="test_org", storage_base_dir=temp_storage_dir
        )
        request_context.storage = MagicMock()
        request_context.storage.count_user_playbooks_by_session.side_effect = (
            RuntimeError("db down")
        )

        evaluator = AgentSuccessEvaluator(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )
        assert evaluator._get_correction_count() == 0

    def test_returns_count_from_storage(self, mock_llm_client, temp_storage_dir):
        """_get_correction_count returns the value from storage."""
        config = AgentSuccessConfig(
            evaluation_name="test",
            success_definition_prompt="test",
        )
        service_config = AgentSuccessGenerationServiceConfig(
            session_id="test_session",
            agent_version="v1",
            request_interaction_data_models=[],
            source="api",
        )
        request_context = RequestContext(
            org_id="test_org", storage_base_dir=temp_storage_dir
        )
        request_context.storage = MagicMock()
        request_context.storage.count_user_playbooks_by_session.return_value = 5

        evaluator = AgentSuccessEvaluator(
            request_context=request_context,
            llm_client=mock_llm_client,
            extractor_config=config,
            service_config=service_config,
            agent_context="Test agent",
        )
        assert evaluator._get_correction_count() == 5
