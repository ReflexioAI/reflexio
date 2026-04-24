"""Integration tests for ExtractionAgent. Uses mocked LLM + real SQLite storage."""

import json
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.extraction.extraction_agent import ExtractionAgent


@pytest.fixture
def temp_storage(tmp_path):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    return SQLiteStorage(org_id="test-org", db_path=str(tmp_path / "ext.db"))


@pytest.fixture
def prompt_manager():
    from reflexio.server.prompt.prompt_manager import PromptManager

    return PromptManager()


@pytest.fixture
def llm_client():
    """Mocked LLM client that returns scripted tool calls."""
    client = MagicMock()
    client.config = MagicMock()
    client.config.api_key_config = None
    return client


def _mk_tool_response(tool_calls, content=None):
    """Construct a fake LLM response shape matching run_tool_loop expectations."""
    resp = MagicMock()
    resp.tool_calls = tool_calls
    resp.content = content
    return resp


def _mk_tool_call(id_, name, args_dict):
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args_dict)
    return tc


def test_extraction_agent_happy_path_new_profile(
    temp_storage, prompt_manager, llm_client
):
    """Session: user states a new fact. Agent searches (empty), creates, finishes."""
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1",
                    "search_user_profiles",
                    {"query": "food preferences", "top_k": 10},
                )
            ]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_profile",
                    {
                        "content": "user likes sushi",
                        "ttl": "infinity",
                        "source_span": "I love sushi",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
    ]

    agent = ExtractionAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        max_steps=12,
    )
    result = agent.run(
        user_id="u_1",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="Extract food preferences.",
        sessions_text="User: I love sushi",
    )

    assert result.outcome == "finish_tool"
    assert len(result.applied) == 1
    # Profile landed in storage
    assert len(temp_storage.get_user_profile("u_1")) == 1


def test_extraction_agent_invariant_blocks_ungrounded_create(
    temp_storage, prompt_manager, llm_client
):
    """Agent skips search, tries to create — invariant A drops it."""
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1",
                    "create_user_profile",
                    {
                        "content": "x",
                        "ttl": "infinity",
                        "source_span": "y",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c2", "finish", {})]),
    ]

    agent = ExtractionAgent(
        client=llm_client, storage=temp_storage, prompt_manager=prompt_manager
    )
    result = agent.run(
        user_id="u_1",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="x",
        sessions_text="User: whatever",
    )
    assert result.outcome == "finish_tool"
    assert len(result.applied) == 0
    assert any(v.code == "A" for v in result.violations)


def test_extraction_agent_max_steps_still_commits_valid_ops(
    temp_storage, prompt_manager, llm_client
):
    """Loop hits max_steps with partially valid plan — plan commits per spec §7."""

    # Script 3 turns that each do search + create, never call finish
    def _turn_script(query):
        return _mk_tool_response(
            [
                _mk_tool_call(
                    "c", "search_user_profiles", {"query": query, "top_k": 10}
                ),
                _mk_tool_call(
                    "c2",
                    "create_user_profile",
                    {
                        "content": f"fact about {query}",
                        "ttl": "infinity",
                        "source_span": query,
                    },
                ),
            ]
        )

    llm_client.generate_chat_response.side_effect = [
        _turn_script(f"q_{i}") for i in range(5)
    ]

    agent = ExtractionAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        max_steps=3,  # force max_steps before finish
    )
    result = agent.run(
        user_id="u_1",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="x",
        sessions_text="User: test",
    )
    assert result.outcome == "max_steps"
    assert len(result.applied) >= 1


def test_extraction_agent_prompt_frames_self_improvement(prompt_manager):
    """Sanity: extraction prompt opening must frame extraction around agent
    self-improvement, not 'memory storage'."""
    out = prompt_manager.render_prompt(
        "extraction_agent",
        variables={
            "sessions": "User: hi",
            "extraction_criteria": "extract facts",
            "extraction_kind": "UserProfile",
        },
    )
    assert "improve over time" in out or "self-improv" in out
    assert "memory extractor" not in out.lower()


def test_extraction_agent_prompt_forbids_profile_rule_overlap(prompt_manager):
    """Sanity (v1.3.0): prompt must carry the anti-pattern examples for
    rule-shaped profile content and the 'no overlap' rule. Guards against
    regression to the earlier bundled-fact / rule-in-profile behaviour."""
    out = prompt_manager.render_prompt(
        "extraction_agent",
        variables={
            "sessions": "User: hi",
            "extraction_criteria": "extract facts",
            "extraction_kind": "UserProfile",
        },
    )
    # One-fact-per-profile rule must be present.
    assert "One fact per profile" in out
    # No-overlap rule between profile and playbook.
    assert "No overlap between profile and playbook" in out
    # Concrete anti-pattern example showing rule leaking into profile.
    assert "prefers no code review scheduling before 10am" in out


def test_extraction_agent_emits_summary_info_line(
    caplog, temp_storage, prompt_manager, llm_client
):
    """Each run emits ONE INFO line starting with 'extraction_agent[' that
    contains elapsed_ms, turns, tools, outcome, applied, violations, usage."""
    import logging

    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1",
                    "search_user_profiles",
                    {"query": "food preferences", "top_k": 10},
                )
            ]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_profile",
                    {
                        "content": "user likes sushi",
                        "ttl": "infinity",
                        "source_span": "I love sushi",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
    ]

    agent = ExtractionAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        max_steps=12,
    )

    with caplog.at_level(
        logging.INFO, logger="reflexio.server.services.extraction.extraction_agent"
    ):
        agent.run(
            user_id="u_summary",
            agent_version="v1",
            extractor_name="food",
            extraction_criteria="Extract food preferences.",
            sessions_text="User: I love sushi",
        )

    summary = [
        r for r in caplog.records if r.getMessage().startswith("extraction_agent[")
    ]
    assert len(summary) == 1, (
        f"Expected 1 summary line, got: {[r.getMessage() for r in summary]}"
    )
    msg = summary[0].getMessage()
    assert "elapsed_ms=" in msg
    assert "turns=" in msg
    assert "tools={" in msg
    assert "outcome=" in msg
    assert "applied=" in msg
    assert "violations=" in msg
    assert "usage={" in msg
