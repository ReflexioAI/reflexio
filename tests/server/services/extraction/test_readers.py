"""Unit tests for ProfileReader / PlaybookReader angle-specialist readers."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.readers import (
    PLAYBOOK_READER_TOOLS,
    PROFILE_READER_TOOLS,
    PlaybookReader,
    ProfileReader,
    ReaderCtx,
    ReaderInputs,
)


@pytest.fixture
def real_client(monkeypatch):
    """Real LiteLLMClient configured for anthropic — matches tool-loop test fixtures."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    config = LiteLLMConfig(model="claude-sonnet-4-6")
    return LiteLLMClient(config)


def _stub_pm(expected_key: str) -> MagicMock:
    pm = MagicMock()
    pm.render_prompt.return_value = f"stub prompt for {expected_key}"
    return pm


def test_profile_reader_collects_emits(real_client, tool_call_completion):
    """ProfileReader should collect emitted candidates and stop on finish."""
    make_tc, _ = tool_call_completion
    pm = _stub_pm("profile_reader_facts")
    reader = ProfileReader(angle="facts", client=real_client, prompt_manager=pm)
    responses = [
        make_tc(
            "emit_profile",
            {
                "content": "User uses polars.",
                "time_to_live": "infinity",
                "source_span": "I use polars not pandas",
                "notes": "confidence=0.95;tag=tool",
                "reader_angle": "facts",
            },
        ),
        make_tc("finish", {}),
    ]

    with patch("litellm.completion", side_effect=responses):
        candidates = reader.read(
            ReaderInputs(sessions="USER: I use polars not pandas.")
        )

    assert len(candidates) == 1
    assert candidates[0].content == "User uses polars."
    assert candidates[0].reader_angle == "facts"
    pm.render_prompt.assert_called_once_with(
        "profile_reader_facts",
        variables={"sessions": "USER: I use polars not pandas."},
    )


def test_playbook_reader_collects_emits(real_client, tool_call_completion):
    """PlaybookReader should collect emitted candidates and stop on finish."""
    make_tc, _ = tool_call_completion
    pm = _stub_pm("playbook_reader_behavior")
    reader = PlaybookReader(angle="behavior", client=real_client, prompt_manager=pm)
    responses = [
        make_tc(
            "emit_playbook",
            {
                "trigger": "user says 'ship'",
                "content": "skip tests",
                "rationale": "",
                "source_span": "When I say 'ship', skip tests",
                "notes": "confidence=0.7;strength=soft",
                "reader_angle": "behavior",
            },
        ),
        make_tc("finish", {}),
    ]

    with patch("litellm.completion", side_effect=responses):
        candidates = reader.read(
            ReaderInputs(sessions="USER: When I say 'ship', skip tests.")
        )

    assert len(candidates) == 1
    assert candidates[0].trigger == "user says 'ship'"
    assert candidates[0].content == "skip tests"
    assert candidates[0].reader_angle == "behavior"


def test_profile_reader_ctx_isolated_across_runs(real_client, tool_call_completion):
    """Each ProfileReader.read() call should start with a fresh ReaderCtx."""
    make_tc, _ = tool_call_completion
    pm = _stub_pm("profile_reader_context")
    reader = ProfileReader(angle="context", client=real_client, prompt_manager=pm)

    responses_run_1 = [
        make_tc(
            "emit_profile",
            {
                "content": "User is shipping on Friday.",
                "time_to_live": "one_week",
                "reader_angle": "context",
            },
        ),
        make_tc("finish", {}),
    ]
    responses_run_2 = [make_tc("finish", {})]

    with patch("litellm.completion", side_effect=responses_run_1):
        run_1 = reader.read(ReaderInputs(sessions="USER: Shipping Friday."))
    with patch("litellm.completion", side_effect=responses_run_2):
        run_2 = reader.read(ReaderInputs(sessions="USER: nothing."))

    assert len(run_1) == 1
    assert run_2 == []  # fresh ctx — no leakage from the first run


def test_profile_reader_tools_registry_advertises_both_tools():
    """PROFILE_READER_TOOLS should expose emit_profile and finish."""
    spec_names = {s["function"]["name"] for s in PROFILE_READER_TOOLS.openai_specs()}
    assert spec_names == {"emit_profile", "finish"}


def test_playbook_reader_tools_registry_advertises_both_tools():
    """PLAYBOOK_READER_TOOLS should expose emit_playbook and finish."""
    spec_names = {s["function"]["name"] for s in PLAYBOOK_READER_TOOLS.openai_specs()}
    assert spec_names == {"emit_playbook", "finish"}


def test_reader_ctx_defaults():
    """ReaderCtx should default to empty candidates and not-finished."""
    ctx = ReaderCtx()
    assert ctx.candidates == []
    assert ctx.finished is False
