"""Integration tests for ShadowComparisonJudge.

The LLM client and prompt manager are mocked at the seam (MagicMock) — we
exercise the judge's orchestration logic, position recording, and failure
handling, but not the network. Real-LLM coverage of the prompt itself
lives in the mock-compliance suite under tests/prompt/.
"""

from __future__ import annotations

import random

import pytest

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.eval_overview_schema import ShadowComparisonOutput
from reflexio.server.services.shadow_comparison.judge import ShadowComparisonJudge

pytestmark = pytest.mark.integration


# --- helpers ------------------------------------------------------------


def _make_interaction(shadow_content: str = "SHADOW") -> Interaction:
    """Build an Interaction with the agent-side content set to 'REFLEX'.

    Args:
        shadow_content (str): Shadow-side content. Empty string is the
            "shadow missing" signal the judge filters on.

    Returns:
        Interaction: A populated agent interaction with `shadow_content` set.
    """
    return Interaction(
        interaction_id=42,
        user_id="u1",
        request_id="r1",
        role="agent",
        content="REFLEX",
        shadow_content=shadow_content,
    )


def _stub_llm_client(
    output: ShadowComparisonOutput | None = None,
    exc: Exception | None = None,
) -> object:
    """Build a stub LiteLLMClient whose generate_chat_response is controllable.

    Args:
        output (ShadowComparisonOutput | None): Value to return.
        exc (Exception | None): Exception to raise instead.

    Returns:
        object: A MagicMock-like stub with a `generate_chat_response` method.
    """
    from unittest.mock import MagicMock

    client = MagicMock()
    if exc is not None:
        client.generate_chat_response.side_effect = exc
    else:
        client.generate_chat_response.return_value = (
            output
            if output is not None
            else ShadowComparisonOutput(
                better_request="1",
                is_significantly_better=True,
                comparison_reason="Request 1 was direct.",
            )
        )
    return client


def _stub_prompt_manager() -> object:
    """Build a stub PromptManager that records its render_prompt calls."""
    from unittest.mock import MagicMock

    pm = MagicMock()
    pm.render_prompt.return_value = "rendered-prompt-body"
    return pm


# --- tests --------------------------------------------------------------


def test_judge_returns_verdict_for_interaction_with_shadow_content():
    llm_client = _stub_llm_client()
    prompt_manager = _stub_prompt_manager()
    judge = ShadowComparisonJudge(
        llm_client=llm_client,  # type: ignore[arg-type]
        prompt_manager=prompt_manager,  # type: ignore[arg-type]
        prompt_version="v1.0.0",
    )

    verdict = judge.judge_turn(
        interaction=_make_interaction(),
        session_id="s1",
        agent_version="v1",
        rng=random.Random(0),  # noqa: S311 — test seed, not crypto
        user_message="hello",
    )

    assert verdict is not None
    assert verdict.interaction_id == "42"
    assert verdict.session_id == "s1"
    assert verdict.agent_version == "v1"
    assert verdict.judge_prompt_version == "v1.0.0"
    assert verdict.output.better_request == "1"
    # verdict_id is 0 (storage assigns); storage layer is tested separately.
    assert verdict.verdict_id == 0


def test_judge_returns_none_when_shadow_content_missing():
    llm_client = _stub_llm_client()
    prompt_manager = _stub_prompt_manager()
    judge = ShadowComparisonJudge(
        llm_client=llm_client,  # type: ignore[arg-type]
        prompt_manager=prompt_manager,  # type: ignore[arg-type]
        prompt_version="v1.0.0",
    )

    verdict = judge.judge_turn(
        interaction=_make_interaction(shadow_content=""),
        session_id="s1",
        agent_version="v1",
        rng=random.Random(0),  # noqa: S311 — test seed, not crypto
    )

    assert verdict is None
    # The LLM was never called: empty shadow short-circuits before the call.
    assert not llm_client.generate_chat_response.called  # type: ignore[attr-defined]


def test_judge_returns_none_on_llm_failure():
    """LLM exception is swallowed so the regen worker can continue other turns."""
    llm_client = _stub_llm_client(exc=RuntimeError("rate limit"))
    prompt_manager = _stub_prompt_manager()
    judge = ShadowComparisonJudge(
        llm_client=llm_client,  # type: ignore[arg-type]
        prompt_manager=prompt_manager,  # type: ignore[arg-type]
        prompt_version="v1.0.0",
    )

    verdict = judge.judge_turn(
        interaction=_make_interaction(),
        session_id="s1",
        agent_version="v1",
        rng=random.Random(0),  # noqa: S311 — test seed, not crypto
    )

    assert verdict is None


def test_judge_records_position_assignment_correctly():
    """reflexio_is_request_1 on the verdict must match the rendered prompt."""
    llm_client = _stub_llm_client()
    prompt_manager = _stub_prompt_manager()
    judge = ShadowComparisonJudge(
        llm_client=llm_client,  # type: ignore[arg-type]
        prompt_manager=prompt_manager,  # type: ignore[arg-type]
        prompt_version="v1.0.0",
    )

    verdict = judge.judge_turn(
        interaction=_make_interaction(),
        session_id="s1",
        agent_version="v1",
        rng=random.Random(0),  # noqa: S311 — test seed, not crypto
    )

    assert verdict is not None
    # PromptManager.render_prompt(prompt_id, variables) — variables is the
    # second positional argument (no keyword) or the `variables=` kwarg.
    call = prompt_manager.render_prompt.call_args  # type: ignore[attr-defined]
    variables = (
        call.kwargs.get("variables")
        if call.kwargs.get("variables") is not None
        else call.args[1]
    )
    if verdict.reflexio_is_request_1:
        assert variables["request_1_response"] == "REFLEX"
        assert variables["request_2_response"] == "SHADOW"
    else:
        assert variables["request_1_response"] == "SHADOW"
        assert variables["request_2_response"] == "REFLEX"


def test_judge_called_with_correct_prompt_id():
    """The judge must use prompt_id='shadow_comparison'."""
    llm_client = _stub_llm_client()
    prompt_manager = _stub_prompt_manager()
    judge = ShadowComparisonJudge(
        llm_client=llm_client,  # type: ignore[arg-type]
        prompt_manager=prompt_manager,  # type: ignore[arg-type]
        prompt_version="v1.0.0",
    )

    judge.judge_turn(
        interaction=_make_interaction(),
        session_id="s1",
        agent_version="v1",
        rng=random.Random(0),  # noqa: S311 — test seed, not crypto
    )

    call = prompt_manager.render_prompt.call_args  # type: ignore[attr-defined]
    prompt_id = call.kwargs.get("prompt_id") or (call.args[0] if call.args else None)
    assert prompt_id == "shadow_comparison"


def test_judge_passes_response_format_to_llm():
    """LLM is called with response_format=ShadowComparisonOutput for structured output."""
    llm_client = _stub_llm_client()
    prompt_manager = _stub_prompt_manager()
    judge = ShadowComparisonJudge(
        llm_client=llm_client,  # type: ignore[arg-type]
        prompt_manager=prompt_manager,  # type: ignore[arg-type]
        prompt_version="v1.0.0",
    )

    judge.judge_turn(
        interaction=_make_interaction(),
        session_id="s1",
        agent_version="v1",
        rng=random.Random(0),  # noqa: S311 — test seed, not crypto
    )

    call = llm_client.generate_chat_response.call_args  # type: ignore[attr-defined]
    response_format = call.kwargs.get("response_format")
    assert response_format is ShadowComparisonOutput
