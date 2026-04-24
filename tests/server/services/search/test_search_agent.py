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
        client=llm_client, storage=temp_storage, prompt_manager=prompt_manager
    )
    result = agent.run(
        user_id="u_1", agent_version="v1", query="what do I like to eat?"
    )
    assert result["answer"] == "no evidence in memory"


def test_search_agent_reads_agent_playbooks(temp_storage, prompt_manager, llm_client):
    """Search agent can fall through to AgentPlaybooks."""
    llm_client.generate_chat_response.side_effect = [
        _mk_resp([_mk_tc("c1", "search_user_playbooks", {"query": "x", "top_k": 10})]),
        _mk_resp([_mk_tc("c2", "search_agent_playbooks", {"query": "x", "top_k": 10})]),
        _mk_resp([_mk_tc("c3", "finish", {"answer": "fallback answer"})]),
    ]
    agent = SearchAgent(
        client=llm_client, storage=temp_storage, prompt_manager=prompt_manager
    )
    r = agent.run(user_id="u_1", agent_version="v1", query="x")
    assert r["answer"] == "fallback answer"


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
    )
    r = agent.run(user_id="u_1", agent_version="v1", query="x")
    assert r["outcome"] == "max_steps"
    assert r["budget_exceeded"] is True
    assert r["answer"] == "no answer"
