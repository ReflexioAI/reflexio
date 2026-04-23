"""Unit tests for ProfileSearchAgent and PlaybookSearchAgent."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.search.search_agents import (
    PlaybookSearchAgent,
    ProfileSearchAgent,
    SearchCtx,
)


@pytest.fixture
def real_client(monkeypatch):
    """Real LiteLLMClient with anthropic creds — matches test_tools.py pattern."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    return LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))


def _pm(render_return: str = "search prompt") -> MagicMock:
    pm = MagicMock()
    pm.render_prompt.return_value = render_return
    return pm


# ---------------- ProfileSearchAgent ---------------- #


def test_profile_search_agent_submits_candidates(real_client, tool_call_completion):
    """Direct intent: one search call then submit_candidates terminates the loop."""
    make_tc, _ = tool_call_completion
    storage = MagicMock()
    storage.search_user_profile.return_value = [
        MagicMock(profile_id="p1"),
        MagicMock(profile_id="p2"),
    ]
    req = MagicMock()
    req.user_id = "u1"
    agent = ProfileSearchAgent(
        "direct", client=real_client, prompt_manager=_pm(), storage=storage
    )
    responses = [
        make_tc(
            "search_profiles",
            {"query": "polars", "top_k": 10, "respect_ttl": True},
        ),
        make_tc("submit_candidates", {"ids": ["p1", "p2"], "why": "direct match"}),
    ]
    with patch("litellm.completion", side_effect=responses):
        ctx = agent.run(query="polars", req=req)

    assert isinstance(ctx, SearchCtx)
    assert ctx.ids == ["p1", "p2"]
    assert ctx.why == "direct match"
    assert ctx.finished is True
    storage.search_user_profile.assert_called_once()
    call_args = storage.search_user_profile.call_args
    assert call_args.args[0].user_id == "u1"
    assert call_args.args[0].query == "polars"
    assert call_args.kwargs["status_filter"] == [None]


def test_profile_search_agent_reformulate_then_submit(
    real_client, tool_call_completion
):
    """Reformulate mutates ctx.query; next search sees the new query."""
    make_tc, _ = tool_call_completion
    storage = MagicMock()
    storage.search_user_profile.return_value = [MagicMock(profile_id="p1")]
    req = MagicMock()
    req.user_id = "u1"
    agent = ProfileSearchAgent(
        "context", client=real_client, prompt_manager=_pm(), storage=storage
    )
    responses = [
        make_tc("reformulate", {"new_query": "data frame library"}),
        make_tc(
            "search_profiles",
            {"query": "data frame library", "top_k": 15, "respect_ttl": True},
        ),
        make_tc("submit_candidates", {"ids": ["p1"], "why": "broadened"}),
    ]
    with patch("litellm.completion", side_effect=responses):
        ctx = agent.run(query="polars", req=req)

    assert ctx.ids == ["p1"]
    assert ctx.query == "data frame library"


def test_profile_search_agent_temporal_disables_ttl(real_client, tool_call_completion):
    """Temporal intent should be free to pass respect_ttl=False."""
    make_tc, _ = tool_call_completion
    storage = MagicMock()
    storage.search_user_profile.return_value = []
    req = MagicMock()
    req.user_id = "u1"
    agent = ProfileSearchAgent(
        "temporal", client=real_client, prompt_manager=_pm(), storage=storage
    )
    responses = [
        make_tc(
            "search_profiles",
            {"query": "prev db", "top_k": 10, "respect_ttl": False},
        ),
        make_tc("submit_candidates", {"ids": [], "why": "nothing relevant"}),
    ]
    with patch("litellm.completion", side_effect=responses):
        agent.run(query="prev db", req=req)

    assert storage.search_user_profile.call_args.kwargs["status_filter"] is None


def test_profile_search_agent_missing_user_id_short_circuits(
    real_client, tool_call_completion
):
    """When req.user_id is falsy, search returns 0 hits without hitting storage."""
    make_tc, _ = tool_call_completion
    storage = MagicMock()
    req = MagicMock()
    req.user_id = None
    agent = ProfileSearchAgent(
        "direct", client=real_client, prompt_manager=_pm(), storage=storage
    )
    responses = [
        make_tc("search_profiles", {"query": "x"}),
        make_tc("submit_candidates", {"ids": [], "why": "no user"}),
    ]
    with patch("litellm.completion", side_effect=responses):
        agent.run(query="x", req=req)

    storage.search_user_profile.assert_not_called()


# ---------------- PlaybookSearchAgent ---------------- #


def test_playbook_search_agent_submits_candidates(real_client, tool_call_completion):
    """Playbook direct intent: one search, then submit."""
    make_tc, _ = tool_call_completion
    storage = MagicMock()
    storage.search_user_playbooks.return_value = [
        MagicMock(user_playbook_id="b1"),
        MagicMock(user_playbook_id="b2"),
    ]
    req = MagicMock()
    req.user_id = "u1"
    agent = PlaybookSearchAgent(
        "direct", client=real_client, prompt_manager=_pm(), storage=storage
    )
    responses = [
        make_tc(
            "search_playbooks",
            {"query": "run tests", "top_k": 10, "respect_ttl": True},
        ),
        make_tc("submit_candidates", {"ids": ["b1", "b2"], "why": "literal"}),
    ]
    with patch("litellm.completion", side_effect=responses):
        ctx = agent.run(query="run tests", req=req)

    assert ctx.ids == ["b1", "b2"]
    assert ctx.why == "literal"
    storage.search_user_playbooks.assert_called_once()
    sent = storage.search_user_playbooks.call_args.args[0]
    assert sent.user_id == "u1"
    assert sent.query == "run tests"
    assert sent.status_filter == [None]


def test_playbook_search_agent_missing_user_id_short_circuits(
    real_client, tool_call_completion
):
    """When req.user_id is falsy, playbook search returns 0 hits without hitting storage."""
    make_tc, _ = tool_call_completion
    storage = MagicMock()
    req = MagicMock()
    req.user_id = None
    agent = PlaybookSearchAgent(
        "direct", client=real_client, prompt_manager=_pm(), storage=storage
    )
    responses = [
        make_tc("search_playbooks", {"query": "x"}),
        make_tc("submit_candidates", {"ids": [], "why": "no user"}),
    ]
    with patch("litellm.completion", side_effect=responses):
        agent.run(query="x", req=req)

    storage.search_user_playbooks.assert_not_called()


def test_playbook_search_agent_temporal_includes_archived(
    real_client, tool_call_completion
):
    """Temporal intent: status_filter is None so archived items are in scope."""
    make_tc, _ = tool_call_completion
    storage = MagicMock()
    storage.search_user_playbooks.return_value = []
    req = MagicMock()
    req.user_id = "u1"
    agent = PlaybookSearchAgent(
        "temporal", client=real_client, prompt_manager=_pm(), storage=storage
    )
    responses = [
        make_tc(
            "search_playbooks",
            {"query": "x", "top_k": 10, "respect_ttl": False},
        ),
        make_tc("submit_candidates", {"ids": [], "why": "none"}),
    ]
    with patch("litellm.completion", side_effect=responses):
        agent.run(query="x", req=req)

    sent = storage.search_user_playbooks.call_args.args[0]
    assert sent.status_filter is None
