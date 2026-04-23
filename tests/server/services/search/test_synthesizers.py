"""Unit tests for ProfileSynthesizer and PlaybookSynthesizer."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.critics import CrossEntityFlag
from reflexio.server.services.search.synthesizers import (
    PlaybookSynthesizer,
    ProfileSynthesizer,
    _candidates_to_block,
)


@pytest.fixture
def real_client(monkeypatch):
    """Real LiteLLMClient with anthropic creds — matches test_tools.py pattern."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    return LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))


def _pm(render_return: str = "synth prompt") -> MagicMock:
    pm = MagicMock()
    pm.render_prompt.return_value = render_return
    return pm


# ---------------- _candidates_to_block ---------------- #


def test_candidates_to_block_empty_returns_sentinel():
    assert _candidates_to_block([]) == "(no candidates)"


def test_candidates_to_block_renders_batches():
    block = _candidates_to_block(
        [
            {"ids": ["p1", "p2"], "why": "direct"},
            {"ids": ["p3"], "why": "context"},
        ]
    )
    assert "[direct] -> p1, p2" in block
    assert "[context] -> p3" in block


# ---------------- ProfileSynthesizer ---------------- #


def test_profile_synth_ranks(real_client, tool_call_completion):
    """Synthesizer emits a ranked ID list and finishes cleanly."""
    make_tc, _ = tool_call_completion
    candidates = [
        {"ids": ["p1", "p2"], "why": "direct"},
        {"ids": ["p3"], "why": "context"},
    ]
    responses = [
        make_tc("rank", {"ordered_ids": ["p2", "p3", "p1"]}),
        make_tc("finish", {}),
    ]
    synth = ProfileSynthesizer(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        ordered, flags = synth.rank(
            query="polars", candidates=candidates, other_lane_summary=""
        )
    assert ordered == ["p2", "p3", "p1"]
    assert flags == []


def test_profile_synth_drop_and_flag(real_client, tool_call_completion):
    """Drop excludes candidates; flag raises a CrossEntityFlag tagged 'profile'."""
    make_tc, _ = tool_call_completion
    candidates = [{"ids": ["p1", "p2"], "why": "direct"}]
    responses = [
        make_tc("drop", {"id": "p2", "reason": "stale"}),
        make_tc(
            "flag_cross_entity_conflict",
            {"id": "p1", "reason": "contradicts playbook"},
        ),
        make_tc("rank", {"ordered_ids": ["p1"]}),
        make_tc("finish", {}),
    ]
    synth = ProfileSynthesizer(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        ordered, flags = synth.rank(
            query="q", candidates=candidates, other_lane_summary="- b0"
        )
    assert ordered == ["p1"]
    assert len(flags) == 1
    assert isinstance(flags[0], CrossEntityFlag)
    assert flags[0].lane == "profile"
    assert "contradicts playbook" in flags[0].reason


# ---------------- PlaybookSynthesizer ---------------- #


def test_playbook_synth_ranks(real_client, tool_call_completion):
    """Playbook synthesizer produces a ranked list; flags default empty."""
    make_tc, _ = tool_call_completion
    candidates = [{"ids": ["b1", "b2"], "why": "direct"}]
    responses = [
        make_tc("rank", {"ordered_ids": ["b1", "b2"]}),
        make_tc("finish", {}),
    ]
    synth = PlaybookSynthesizer(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        ordered, flags = synth.rank(
            query="q", candidates=candidates, other_lane_summary=""
        )
    assert ordered == ["b1", "b2"]
    assert flags == []


def test_playbook_synth_flag_tagged_with_playbook_lane(
    real_client, tool_call_completion
):
    """Flags raised in playbook synth are tagged with lane='playbook'."""
    make_tc, _ = tool_call_completion
    responses = [
        make_tc(
            "flag_cross_entity_conflict",
            {"id": "b1", "reason": "contradicts profile"},
        ),
        make_tc("rank", {"ordered_ids": ["b1"]}),
        make_tc("finish", {}),
    ]
    synth = PlaybookSynthesizer(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        _, flags = synth.rank(
            query="q",
            candidates=[{"ids": ["b1"], "why": "direct"}],
            other_lane_summary="- p0",
        )
    assert len(flags) == 1
    assert flags[0].lane == "playbook"
