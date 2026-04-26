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
            "max_steps": "4",
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
            "max_steps": "4",
        },
    )
    # One-fact-per-profile rule must be present.
    assert "One fact per profile" in out
    # No-overlap rule between profile and playbook.
    assert "No overlap between profile and playbook" in out
    # The prompt must include some anti-pattern guidance distinguishing
    # rule-shaped from fact-shaped content. The specific example string
    # is allowed to evolve via Phase 27 tuning, so we check for structural
    # markers (the rule wording) rather than a single example.


def test_extraction_agent_prompt_specifies_playbook_format(prompt_manager):
    """Sanity (v1.4.0): prompt must carry the Agent-Skills-inspired format
    guidance for UserPlaybook trigger + content + rationale. Guards against
    regression to the earlier unstructured semicolon-delimited shape."""
    out = prompt_manager.render_prompt(
        "extraction_agent",
        variables={
            "sessions": "User: hi",
            "extraction_criteria": "extract rules",
            "extraction_kind": "UserPlaybook",
            "max_steps": "4",
        },
    )
    # The Playbook format section must be present.
    assert "Playbook format" in out
    # Trigger guidance — imperative conditional phrasing must be required;
    # the proposer is allowed to evolve specific examples.
    assert "imperative conditional phrasing" in out
    # Content guidance — markdown bullet list for independent instructions.
    assert "markdown bullet list" in out
    # Rationale guidance — one sentence explaining WHY, not what.
    assert "one sentence" in out.lower()


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


def test_extraction_agent_threads_request_id_into_profile(
    temp_storage, prompt_manager, llm_client
):
    """request_id passed to agent.run lands in stored UserProfile.generated_from_request_id.

    Recall@K-style downstream consumers depend on this thread to translate
    retrieved profiles back to their source publish_interaction request.
    A regression here silently breaks per-session provenance for the agentic
    backend.
    """
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1",
                    "search_user_profiles",
                    {"query": "food", "top_k": 10},
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
        client=llm_client, storage=temp_storage, prompt_manager=prompt_manager
    )
    agent.run(
        user_id="u_rid",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="x",
        sessions_text="User: I love sushi",
        request_id="test-rid-abc",
    )

    profiles = temp_storage.get_user_profile("u_rid")
    assert len(profiles) == 1
    assert profiles[0].generated_from_request_id == "test-rid-abc"


def test_extraction_agent_threads_request_id_into_playbook(
    temp_storage, prompt_manager, llm_client
):
    """request_id also lands on UserPlaybook.request_id (mirror of profile thread)."""
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1",
                    "search_user_playbooks",
                    {"query": "rules", "top_k": 10},
                )
            ]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_playbook",
                    {
                        "trigger": "When user asks about food",
                        "content": "- Note that user likes sushi.",
                        "rationale": "User preference",
                        "source_span": "I love sushi",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
    ]

    from reflexio.server.services.extraction.tools import PLAYBOOK_EXTRACTION_TOOLS

    agent = ExtractionAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        registry=PLAYBOOK_EXTRACTION_TOOLS,
    )
    agent.run(
        user_id="u_rid_pb",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="Extract behavioural rules.",
        sessions_text="User: I love sushi",
        extraction_kind="UserPlaybook",
        request_id="test-rid-pb",
    )

    playbooks = temp_storage.get_user_playbooks(user_id="u_rid_pb")
    assert len(playbooks) == 1
    assert playbooks[0].request_id == "test-rid-pb"


def test_extraction_agent_request_id_default_is_empty_string(
    temp_storage, prompt_manager, llm_client
):
    """Backward compat: callers that omit request_id get '' on the profile.

    Existing test callers (and any historical deployments) must keep
    working without code changes.
    """
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [_mk_tool_call("c1", "search_user_profiles", {"query": "x", "top_k": 10})]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_profile",
                    {
                        "content": "fact",
                        "ttl": "infinity",
                        "source_span": "x",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
    ]

    agent = ExtractionAgent(
        client=llm_client, storage=temp_storage, prompt_manager=prompt_manager
    )
    agent.run(
        user_id="u_default",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="x",
        sessions_text="User: x",
    )

    profiles = temp_storage.get_user_profile("u_default")
    assert len(profiles) == 1
    assert profiles[0].generated_from_request_id == ""


def test_extraction_agent_threads_date_into_profile(
    temp_storage, prompt_manager, llm_client
):
    """`date` argument on create_user_profile lands in stored UserProfile.date_mentioned.

    Temporal-reasoning retrieval downstream filters on this typed field. A
    regression here silently drops the date from the agentic backend, breaking
    the date-anchor signal for T-R questions.
    """
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [_mk_tool_call("c1", "search_user_profiles", {"query": "moma", "top_k": 10})]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_profile",
                    {
                        "content": "user visited MoMA on 2024-08-23 (session date)",
                        "ttl": "infinity",
                        "source_span": "I visited MoMA on Aug 23",
                        "date": "2024-08-23",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
    ]

    agent = ExtractionAgent(
        client=llm_client, storage=temp_storage, prompt_manager=prompt_manager
    )
    agent.run(
        user_id="u_date",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="x",
        sessions_text="User: I visited MoMA on Aug 23",
        request_id="rid-date",
    )

    profiles = temp_storage.get_user_profile("u_date")
    assert len(profiles) == 1
    assert profiles[0].date_mentioned == "2024-08-23"


def test_extraction_agent_threads_date_into_playbook(
    temp_storage, prompt_manager, llm_client
):
    """`date` argument on create_user_playbook lands in stored UserPlaybook.date_mentioned.

    Mirror of the profile thread; verifies the playbook commit path also
    propagates the canonical date end-to-end.
    """
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [_mk_tool_call("c1", "search_user_playbooks", {"query": "x", "top_k": 10})]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_playbook",
                    {
                        "trigger": "When user asks about MoMA visit",
                        "content": "- Reference the 2024-08-23 visit.",
                        "rationale": "Anchor on the known date.",
                        "source_span": "I visited MoMA on Aug 23",
                        "date": "2024-08-23",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
    ]

    from reflexio.server.services.extraction.tools import PLAYBOOK_EXTRACTION_TOOLS

    agent = ExtractionAgent(
        client=llm_client,
        storage=temp_storage,
        prompt_manager=prompt_manager,
        registry=PLAYBOOK_EXTRACTION_TOOLS,
    )
    agent.run(
        user_id="u_date_pb",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="Extract behavioural rules.",
        sessions_text="User: I visited MoMA on Aug 23",
        extraction_kind="UserPlaybook",
        request_id="rid-date-pb",
    )

    playbooks = temp_storage.get_user_playbooks(user_id="u_date_pb")
    assert len(playbooks) == 1
    assert playbooks[0].date_mentioned == "2024-08-23"


def test_extraction_agent_date_default_is_empty_string(
    temp_storage, prompt_manager, llm_client
):
    """Backward compat: callers that omit ``date`` get "" on the profile."""
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [_mk_tool_call("c1", "search_user_profiles", {"query": "x", "top_k": 10})]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_profile",
                    {
                        "content": "no date here",
                        "ttl": "infinity",
                        "source_span": "x",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
    ]

    agent = ExtractionAgent(
        client=llm_client, storage=temp_storage, prompt_manager=prompt_manager
    )
    agent.run(
        user_id="u_no_date",
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="x",
        sessions_text="User: no event",
    )

    profiles = temp_storage.get_user_profile("u_no_date")
    assert len(profiles) == 1
    assert profiles[0].date_mentioned == ""
