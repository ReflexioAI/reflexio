"""Integration tests for direct shadow-content grading.

Verifies the evaluator runs a second-pass LLM call when shadow_mode_enabled
is True and the session has shadow_content, and populates the new outcome
fields without disturbing the regular evaluation path.

Fixture adaptations vs the plan template:
- AgentSuccessEvaluator.__init__ requires positional args ``extractor_config``
  (AgentSuccessConfig) and ``service_config`` (AgentSuccessGenerationServiceConfig)
  plus ``agent_context: str``.  A real RequestContext is constructed with a
  temp dir and a mocked storage layer (same pattern as test_agent_success_evaluator.py).
- Config.shadow_mode_enabled is a plain bool field on Config — no setattr tricks needed.
- AgentSuccessEvaluator.run() is the public entrypoint; the shadow grade fires
  inside _evaluate_group which run() calls.
"""

import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import AgentSuccessConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_constants import (
    AgentSuccessEvaluationOutput,
    AgentSuccessEvaluationWithComparisonOutput,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_service import (
    AgentSuccessGenerationServiceConfig,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluator import (
    AgentSuccessEvaluator,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request_with_shadow(
    *, with_shadow: bool
) -> list[RequestInteractionDataModel]:
    """Build a single-session request list, optionally with shadow_content on the assistant turn."""
    return [
        RequestInteractionDataModel(
            session_id="s1",
            request=Request(request_id="r1", user_id="u1", created_at=1000),
            interactions=[
                Interaction(
                    interaction_id=1,
                    user_id="u1",
                    request_id="r1",
                    role="user",
                    content="hi",
                    created_at=1000,
                ),
                Interaction(
                    interaction_id=2,
                    user_id="u1",
                    request_id="r1",
                    role="assistant",
                    content="hello (regular)",
                    shadow_content="hello (shadow)" if with_shadow else "",
                    created_at=1001,
                ),
            ],
        )
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_evaluator(
    temp_dir: str,
    *,
    shadow_mode_enabled: bool = True,
    with_shadow: bool = True,
) -> AgentSuccessEvaluator:
    """
    Build a fully-wired AgentSuccessEvaluator.

    The RequestContext uses a real temp dir (so the configurator initialises)
    but its storage is replaced with a MagicMock so no DB I/O happens.
    The LLM client is mocked to return a successful evaluation for both the
    regular path and (when needed) the combined comparison path.
    """
    request_interaction_data_models = _make_request_with_shadow(with_shadow=with_shadow)

    context = RequestContext(org_id="test_org_shadow", storage_base_dir=temp_dir)
    context.storage = MagicMock()
    context.storage.count_user_playbooks_by_session.return_value = 0

    # Patch get_config so it returns a Config with shadow_mode_enabled as requested
    mock_config = MagicMock()
    mock_config.shadow_mode_enabled = shadow_mode_enabled
    mock_config.tool_can_use = None
    context.configurator = MagicMock()
    context.configurator.get_config.return_value = mock_config

    extractor_config = AgentSuccessConfig(
        evaluation_name="overall_success",
        success_definition_prompt="any non-empty answer is success",
    )

    service_config = AgentSuccessGenerationServiceConfig(
        agent_version="v1",
        session_id="s1",
        request_interaction_data_models=request_interaction_data_models,
        source="test",
    )

    # The mock LLM client returns valid outputs for all three call types
    mock_client = MagicMock(spec=LiteLLMClient)
    mock_client.generate_chat_response.side_effect = _llm_side_effect

    return AgentSuccessEvaluator(
        request_context=context,
        llm_client=mock_client,
        extractor_config=extractor_config,
        service_config=service_config,
        agent_context="test agent",
    )


def _llm_side_effect(*args, **kwargs):
    """Return the right mock output based on response_format."""
    response_format = kwargs.get("response_format")
    if response_format is AgentSuccessEvaluationWithComparisonOutput:
        return AgentSuccessEvaluationWithComparisonOutput(
            is_success=True,
            better_request="1",
            is_significantly_better=False,
        )
    # Default: AgentSuccessEvaluationOutput
    return AgentSuccessEvaluationOutput(is_success=True, is_escalated=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shadow_grade_runs_when_enabled_and_shadow_present(temp_dir):
    """run() triggers _run_direct_shadow_grade when flag on + shadow_content present."""
    evaluator = _make_evaluator(temp_dir, shadow_mode_enabled=True, with_shadow=True)

    with patch.object(
        evaluator,
        "_run_direct_shadow_grade",
        return_value=(False, True),
    ) as mock_shadow:
        results = evaluator.run()

    assert mock_shadow.called
    assert len(results) == 1
    result = results[0]
    assert result.shadow_is_success is False
    assert result.shadow_is_escalated is True


def test_shadow_grade_skipped_when_no_shadow_content(temp_dir):
    """run() does not call _run_direct_shadow_grade when no interaction has shadow_content."""
    evaluator = _make_evaluator(temp_dir, shadow_mode_enabled=True, with_shadow=False)

    with patch.object(evaluator, "_run_direct_shadow_grade") as mock_shadow:
        results = evaluator.run()

    assert not mock_shadow.called
    assert len(results) == 1
    result = results[0]
    assert result.shadow_is_success is None
    assert result.shadow_is_escalated is None


def test_shadow_grade_skipped_when_flag_off(temp_dir):
    """run() does not call _run_direct_shadow_grade when shadow_mode_enabled=False."""
    evaluator = _make_evaluator(temp_dir, shadow_mode_enabled=False, with_shadow=True)

    with patch.object(evaluator, "_run_direct_shadow_grade") as mock_shadow:
        results = evaluator.run()

    assert not mock_shadow.called
    assert len(results) == 1
    result = results[0]
    assert result.shadow_is_success is None
    assert result.shadow_is_escalated is None


def test_shadow_grade_calls_use_shadow_true_and_populates_result(temp_dir):
    """_run_direct_shadow_grade body executes and threads use_shadow=True through."""
    evaluator = _make_evaluator(temp_dir, shadow_mode_enabled=True, with_shadow=True)

    captured: dict[str, object] = {}

    real_construct = (
        "reflexio.server.services.agent_success_evaluation"
        ".agent_success_evaluator"
        ".construct_agent_success_evaluation_messages_from_sessions"
    )

    def capture_construct(*args: object, **kwargs: object) -> list[dict[str, str]]:
        captured["use_shadow"] = kwargs.get("use_shadow")
        return [{"role": "user", "content": "test"}]

    # call 1 — _evaluate_with_shadow_comparison (uses AgentSuccessEvaluationWithComparisonOutput)
    # call 2 — _run_direct_shadow_grade (uses AgentSuccessEvaluationOutput)
    valid_comparison = AgentSuccessEvaluationWithComparisonOutput(
        is_success=True,
        better_request="1",
        is_significantly_better=False,
    )
    valid_output = AgentSuccessEvaluationOutput(is_success=True, is_escalated=False)
    call_count: dict[str, int] = {"n": 0}

    def llm_side_effect(*args: object, **kwargs: object) -> object:
        call_count["n"] += 1
        response_format = kwargs.get("response_format")
        if response_format is AgentSuccessEvaluationWithComparisonOutput:
            return valid_comparison
        return valid_output

    evaluator.client.generate_chat_response = Mock(side_effect=llm_side_effect)

    with patch(real_construct, side_effect=capture_construct):
        results = evaluator.run()

    assert captured.get("use_shadow") is True
    assert len(results) == 1
    result = results[0]
    assert result.shadow_is_success is True
    assert result.shadow_is_escalated is False


def test_shadow_grade_returns_none_when_llm_raises(temp_dir):
    """_run_direct_shadow_grade returns (None, None) when the LLM call raises."""
    evaluator = _make_evaluator(temp_dir, shadow_mode_enabled=True, with_shadow=True)

    valid_comparison = AgentSuccessEvaluationWithComparisonOutput(
        is_success=True,
        better_request="1",
        is_significantly_better=False,
    )
    call_count: dict[str, int] = {"n": 0}

    def side_effect(*args: object, **kwargs: object) -> object:
        call_count["n"] += 1
        response_format = kwargs.get("response_format")
        if response_format is AgentSuccessEvaluationWithComparisonOutput:
            # First call — regular combined comparison — succeeds
            return valid_comparison
        # Second call — shadow grade — fails
        raise RuntimeError("simulated provider failure")

    evaluator.client.generate_chat_response = Mock(side_effect=side_effect)

    results = evaluator.run()

    assert results is not None
    assert len(results) == 1
    result = results[0]
    assert result.shadow_is_success is None
    assert result.shadow_is_escalated is None
