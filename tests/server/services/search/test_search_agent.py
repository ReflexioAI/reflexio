"""Integration tests for SearchAgent (read-only single loop)."""

import json
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.search.search_agent import SearchAgent


@pytest.fixture
def temp_storage(tmp_path):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    # NOTE: SQLiteStorage requires org_id + db_path kwargs (not a single positional).
    return SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "srch.db"))


@pytest.fixture
def prompt_manager():
    from reflexio.server.prompt.prompt_manager import PromptManager

    return PromptManager()


@pytest.fixture
def llm_client():
    c = MagicMock()
    c.config = MagicMock()
    c.config.api_key_config = None
    return c


def _mk_tc(id_, name, args):
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _mk_resp(tool_calls, content=None):
    r = MagicMock()
    r.tool_calls = tool_calls
    r.content = content
    return r


def test_search_agent_returns_answer_from_finish(
    temp_storage, prompt_manager, llm_client
):
    llm_client.generate_chat_response.side_effect = [
        _mk_resp(
            [_mk_tc("c1", "search_user_profiles", {"query": "food", "top_k": 10})]
        ),
        _mk_resp([_mk_tc("c2", "finish", {"answer": "no evidence in memory"})]),
    ]

    agent = SearchAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        enable_agent_answer=True,
    )
    result = agent.run(
        user_id="u_1", agent_version="v1", query="what do I like to eat?"
    )
    assert result.answer == "no evidence in memory"


def test_search_agent_reads_agent_playbooks(temp_storage, prompt_manager, llm_client):
    """Search agent can fall through to AgentPlaybooks."""
    llm_client.generate_chat_response.side_effect = [
        _mk_resp([_mk_tc("c1", "search_user_playbooks", {"query": "x", "top_k": 10})]),
        _mk_resp([_mk_tc("c2", "search_agent_playbooks", {"query": "x", "top_k": 10})]),
        _mk_resp([_mk_tc("c3", "finish", {"answer": "fallback answer"})]),
    ]
    agent = SearchAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        enable_agent_answer=True,
    )
    r = agent.run(user_id="u_1", agent_version="v1", query="x")
    assert r.answer == "fallback answer"


def test_search_agent_reports_budget_exceeded_on_max_steps(
    temp_storage, prompt_manager, llm_client
):
    """Loop hits max_steps without ever calling finish — budget_exceeded is True."""
    llm_client.generate_chat_response.side_effect = [
        _mk_resp([_mk_tc(f"c{i}", "search_user_profiles", {"query": "x", "top_k": 10})])
        for i in range(5)
    ]
    agent = SearchAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        max_steps=2,
        enable_agent_answer=True,
    )
    r = agent.run(user_id="u_1", agent_version="v1", query="x")
    assert r.outcome == "max_steps"
    assert r.budget_exceeded is True
    assert r.answer == "no answer"


def test_search_agent_search_only_mode_returns_none_answer(
    temp_storage, prompt_manager, llm_client
):
    """When ``enable_agent_answer=False`` (default), the agent's answer is
    forced to None even if the LLM produced one. Callers (the host) synthesize
    the final response from the entities harvested by the search agent.
    """
    llm_client.generate_chat_response.side_effect = [
        _mk_resp([_mk_tc("c1", "search_user_profiles", {"query": "x", "top_k": 10})]),
        # LLM still emits an answer in the mock; the agent must drop it.
        _mk_resp([_mk_tc("c2", "finish", {"answer": "ignored"})]),
    ]
    agent = SearchAgent(
        client=llm_client, storage=temp_storage, prompt_manager=prompt_manager
    )
    r = agent.run(user_id="u_so", agent_version="v1", query="anything?")
    assert r.answer is None
    # Search-only mode must still let the agent finish cleanly.
    assert r.outcome == "finish_tool"


def test_search_agent_prompt_includes_search_only_block_when_disabled(prompt_manager):
    """Rendered prompt carries the search-only mode flag verbatim so the LLM
    can branch its finish() call accordingly.
    """
    rendered = prompt_manager.render_prompt(
        "search_agent",
        variables={
            "query": "x",
            "max_steps": "3",
            "enable_agent_answer": "false",
        },
    )
    assert "enable_agent_answer = false" in rendered
    assert "Search-only output rule" in rendered


def test_search_agent_prompt_includes_answer_block_when_enabled(prompt_manager):
    """Rendered prompt carries the synthesis flag when the host opts in."""
    rendered = prompt_manager.render_prompt(
        "search_agent",
        variables={
            "query": "x",
            "max_steps": "3",
            "enable_agent_answer": "true",
        },
    )
    assert "enable_agent_answer = true" in rendered
    assert "Expected answer format" in rendered


def test_search_agent_trace_captures_harvested_ids(
    temp_storage, prompt_manager, llm_client
):
    """Trace contains search turn results — used by AgenticSearchService for entity harvesting."""
    from reflexio.models.api_schema.domain.entities import (
        NEVER_EXPIRES_TIMESTAMP,
        UserProfile,
    )
    from reflexio.models.api_schema.domain.enums import ProfileTimeToLive

    temp_storage.add_user_profile(
        "u_1",
        [
            UserProfile(
                profile_id="p_seed_1",
                user_id="u_1",
                content="user likes sushi",
                last_modified_timestamp=0,
                generated_from_request_id="r_1",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                extractor_names=["test"],
            ),
        ],
    )

    llm_client.generate_chat_response.side_effect = [
        _mk_resp(
            [_mk_tc("c1", "search_user_profiles", {"query": "food", "top_k": 10})]
        ),
        _mk_resp([_mk_tc("c2", "finish", {"answer": "user likes sushi"})]),
    ]

    agent = SearchAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        enable_agent_answer=True,
    )
    result = agent.run(user_id="u_1", agent_version="v1", query="what does user like?")

    # trace.turns should contain at least the search turn
    assert len(result.trace.turns) >= 1
    search_turns = [
        t for t in result.trace.turns if t.tool_name == "search_user_profiles"
    ]
    assert search_turns


def test_search_agent_prompt_frames_agent_improvement(prompt_manager):
    """Sanity: search prompt opening must frame retrieval around informing
    the agent's next action, not 'memory query'."""
    out = prompt_manager.render_prompt(
        "search_agent",
        variables={
            "query": "what does user like?",
            "max_steps": "3",
            "enable_agent_answer": "false",
        },
    )
    assert "helping an AI agent" in out or "inform" in out
    assert "memory query agent" not in out.lower()


def test_search_agent_emits_summary_info_line(
    caplog, temp_storage, prompt_manager, llm_client
):
    """Each run emits ONE INFO line starting with 'search_agent ' that
    contains elapsed_ms, turns, outcome, answer_len, and usage."""
    import logging

    llm_client.generate_chat_response.side_effect = [
        _mk_resp(
            [_mk_tc("c1", "search_user_profiles", {"query": "food", "top_k": 10})]
        ),
        _mk_resp([_mk_tc("c2", "finish", {"answer": "user likes sushi"})]),
    ]

    agent = SearchAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        enable_agent_answer=True,
    )

    with caplog.at_level(
        logging.INFO, logger="reflexio.server.services.search.search_agent"
    ):
        agent.run(user_id="u_summary", agent_version="v1", query="what do I like?")

    summary = [r for r in caplog.records if r.getMessage().startswith("search_agent ")]
    assert len(summary) == 1, (
        f"Expected 1 summary line, got: {[r.getMessage() for r in summary]}"
    )
    msg = summary[0].getMessage()
    assert "elapsed_ms=" in msg
    assert "turns=" in msg
    assert "outcome=" in msg
    assert "answer_len=" in msg
    assert "usage={" in msg
