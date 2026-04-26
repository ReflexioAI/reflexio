"""Tests for the agentic-v2 AgenticExtractionRunner adapter.

Three required tests (per Task 12 spec):
1. test_agentic_adapter_end_to_end_creates_profile  — scripted LLM, real SQLite
2. test_agentic_adapter_triggers_playbook_aggregator — mocked aggregator
3. test_agentic_adapter_pre_filter_rejects_short_session — pre-flight gate

Additional unit tests cover:
- force_extraction bypasses pre-filter
- multiple extractor configs each invoke ExtractionAgent
- skip_aggregation short-circuits aggregator
- agent failure degrades to warning (not exception)
- hard violations surface as warnings
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.service_schemas import (
    PublishUserInteractionRequest,
    Request,
)
from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    UserPlaybookExtractorConfig,
)
from reflexio.server.services.extraction.agentic_adapter import AgenticExtractionRunner
from reflexio.server.services.extraction.plan import CommitResult, Violation

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _make_interaction(role: str, content: str, user_id: str = "u_test") -> Interaction:
    return Interaction(
        interaction_id=0,
        user_id=user_id,
        request_id="req_abc",
        role=role,
        content=content,
    )


def _make_request(session_id: str = "s1") -> Request:
    return Request(
        request_id="req_abc",
        user_id="u_test",
        source="cli",
        agent_version="v1",
        session_id=session_id,
    )


def _make_publish_request(
    *,
    force_extraction: bool = False,
    skip_aggregation: bool = False,
    user_id: str = "u_test",
) -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        user_id=user_id,
        interaction_data_list=[{"role": "User", "content": "hi"}],  # type: ignore[list-item]
        source="cli",
        agent_version="v1",
        force_extraction=force_extraction,
        skip_aggregation=skip_aggregation,
    )


def _make_runner(
    storage: object = None,
) -> AgenticExtractionRunner:
    """Build a runner with a mocked request_context."""
    rc = MagicMock()
    rc.storage = storage if storage is not None else MagicMock()
    rc.prompt_manager = MagicMock()
    rc.prompt_manager.render_prompt.return_value = "stub prompt"
    rc.configurator = MagicMock()
    rc.org_id = "test-org"

    return AgenticExtractionRunner(
        llm_client=MagicMock(),
        request_context=rc,
    )


def _mk_tool_call(id_: str, name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _mk_tool_response(tool_calls: list, content: str | None = None) -> MagicMock:
    resp = MagicMock()
    resp.tool_calls = tool_calls
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Test 1: end-to-end creates profile (real SQLite, scripted LLM)
# ---------------------------------------------------------------------------


def test_agentic_adapter_end_to_end_creates_profile(tmp_path):
    """Scripted 3-turn LLM: search → create → finish.

    Invokes the runner with real SQLite storage; asserts the profile lands in
    storage after the run completes.
    """
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    user_id = "u_adapter_e2e"
    store = SQLiteStorage(
        org_id="test-org-e2e", db_path=str(tmp_path / "adapter_e2e.db")
    )

    # Real client (key doesn't matter — LLM is mocked via generate_chat_response)
    import os

    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    pm = PromptManager()

    rc = MagicMock()
    rc.storage = store
    rc.prompt_manager = pm
    rc.configurator = MagicMock()
    rc.org_id = "test-org-e2e"

    runner = AgenticExtractionRunner(
        llm_client=client,
        request_context=rc,
    )

    # Script: search (empty result) → create profile → finish
    scripted = [
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1", "search_user_profiles", {"query": "food", "top_k": 10}
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

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="test_profile_extractor",
                extraction_definition_prompt="Extract food preferences.",
            )
        ],
        user_playbook_extractor_configs=[],
    )

    with patch.object(client, "generate_chat_response", side_effect=scripted):
        warnings = runner.run(
            publish_request=_make_publish_request(
                force_extraction=True, user_id=user_id
            ),
            request_id="req_e2e",
            new_interactions=[_make_interaction("User", "I love sushi", user_id)],
            new_request=Request(
                request_id="req_e2e",
                user_id=user_id,
                source="cli",
                agent_version="v1",
                session_id="s_e2e",
            ),
            config=cfg,
        )

    assert isinstance(warnings, list)
    profiles = store.get_user_profile(user_id)
    assert len(profiles) == 1, f"Expected 1 profile, got {len(profiles)}: {profiles}"
    assert profiles[0].content == "user likes sushi"


# ---------------------------------------------------------------------------
# Test 2: aggregation triggered for configs with aggregation_config
# ---------------------------------------------------------------------------


def test_agentic_adapter_triggers_playbook_aggregator():
    """Runner triggers PlaybookAggregator.run once per config that has aggregation_config."""
    runner = _make_runner()

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[],
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="with_agg",
                extraction_definition_prompt="Extract playbook rules.",
                aggregation_config=PlaybookAggregatorConfig(),
            ),
            UserPlaybookExtractorConfig(
                extractor_name="without_agg",
                extraction_definition_prompt="Extract playbook rules.",
            ),
        ],
    )

    # Stub ExtractionAgent.run to return empty CommitResult (no LLM calls needed)
    empty_result = CommitResult(applied=[], violations=[], outcome="finish_tool")
    fake_agg_cls = MagicMock()
    fake_agg_cls.return_value.run.return_value = {}

    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent.run",
            return_value=empty_result,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookAggregator",
            fake_agg_cls,
        ),
    ):
        runner.run(
            publish_request=_make_publish_request(force_extraction=True),
            request_id="req_agg",
            new_interactions=[
                _make_interaction("User", "Trigger aggregation test"),
            ],
            new_request=_make_request(),
            config=cfg,
        )

    # Aggregator constructed + run called exactly once (only "with_agg" has aggregation_config)
    assert fake_agg_cls.return_value.run.call_count == 1
    call_arg = fake_agg_cls.return_value.run.call_args.args[0]
    assert call_arg.playbook_name == "with_agg"


# ---------------------------------------------------------------------------
# Test 3: pre-filter rejects short session
# ---------------------------------------------------------------------------


def test_agentic_adapter_pre_filter_rejects_short_session():
    """When _cheap_should_run_reject returns a reason, runner exits early.

    ExtractionAgent must not be invoked.
    """
    runner = _make_runner()

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="default",
                extraction_definition_prompt="Extract facts.",
            )
        ],
        user_playbook_extractor_configs=[],
    )

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent.run"
    ) as mock_agent_run:
        warnings = runner.run(
            publish_request=_make_publish_request(
                force_extraction=False
            ),  # pre-filter active
            request_id="req_prefilter",
            new_interactions=[
                _make_interaction("Agent", "only agent turn, no user turn")
            ],
            new_request=_make_request(),
            config=cfg,
        )

    assert warnings == []
    mock_agent_run.assert_not_called()


# ---------------------------------------------------------------------------
# Additional unit tests
# ---------------------------------------------------------------------------


def test_runner_force_extraction_bypasses_pre_filter():
    """force_extraction=True calls ExtractionAgent even with no User turns."""
    runner = _make_runner()

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="only_profile",
                extraction_definition_prompt="Extract facts.",
            )
        ],
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="only_playbook",
                extraction_definition_prompt="Extract rules.",
            )
        ],
    )

    empty_result = CommitResult(applied=[], violations=[], outcome="finish_tool")

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent.run",
        return_value=empty_result,
    ) as mock_agent_run:
        runner.run(
            publish_request=_make_publish_request(force_extraction=True),
            request_id="req_force",
            new_interactions=[_make_interaction("Agent", "no user turn")],
            new_request=_make_request(),
            config=cfg,
        )

    # 1 profile + 1 playbook config = 2 total agent calls; pre-filter was bypassed
    assert mock_agent_run.call_count == 2


def test_runner_iterates_all_extractor_configs():
    """Runner calls ExtractionAgent once per config across both profile + playbook lists."""
    runner = _make_runner()

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="profile_one",
                extraction_definition_prompt="profile prompt",
            ),
            ProfileExtractorConfig(
                extractor_name="profile_two",
                extraction_definition_prompt="profile prompt 2",
            ),
        ],
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="playbook_one",
                extraction_definition_prompt="playbook prompt",
            ),
        ],
    )

    empty_result = CommitResult(applied=[], violations=[], outcome="finish_tool")

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent.run",
        return_value=empty_result,
    ) as mock_agent_run:
        runner.run(
            publish_request=_make_publish_request(force_extraction=True),
            request_id="req_multi",
            new_interactions=[_make_interaction("User", "test content")],
            new_request=_make_request(),
            config=cfg,
        )

    # 2 profile configs + 1 playbook config = 3 total agent calls
    assert mock_agent_run.call_count == 3
    called_names = {c.kwargs["extractor_name"] for c in mock_agent_run.call_args_list}
    assert called_names == {"profile_one", "profile_two", "playbook_one"}


def test_runner_skip_aggregation_short_circuits():
    """skip_aggregation=True → PlaybookAggregator never constructed."""
    runner = _make_runner()

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[],
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="with_agg",
                extraction_definition_prompt="p",
                aggregation_config=PlaybookAggregatorConfig(),
            ),
        ],
    )

    empty_result = CommitResult(applied=[], violations=[], outcome="finish_tool")
    fake_agg_cls = MagicMock()

    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent.run",
            return_value=empty_result,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookAggregator",
            fake_agg_cls,
        ),
    ):
        runner.run(
            publish_request=_make_publish_request(
                force_extraction=True, skip_aggregation=True
            ),
            request_id="req_skip_agg",
            new_interactions=[_make_interaction("User", "hi")],
            new_request=_make_request(),
            config=cfg,
        )

    fake_agg_cls.assert_not_called()


def test_runner_agent_failure_becomes_warning():
    """Exception from ExtractionAgent.run is caught and surfaced as a warning."""
    runner = _make_runner()

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="failing_extractor",
                extraction_definition_prompt="Extract facts.",
            )
        ],
        user_playbook_extractor_configs=[],
    )

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent.run",
        side_effect=RuntimeError("LLM timeout"),
    ):
        warnings = runner.run(
            publish_request=_make_publish_request(force_extraction=True),
            request_id="req_fail",
            new_interactions=[_make_interaction("User", "test")],
            new_request=_make_request(),
            config=cfg,
        )

    assert any("failing_extractor" in w and "LLM timeout" in w for w in warnings)


def test_runner_hard_violation_surfaces_as_warning():
    """Hard invariant violations in CommitResult are appended to warnings."""
    runner = _make_runner()

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="default",
                extraction_definition_prompt="Extract facts.",
            )
        ],
        user_playbook_extractor_configs=[],
    )

    violation = Violation(
        code="A",
        severity="hard",
        affected_op_indices=[0],
        msg="create without prior search",
    )
    result_with_violation = CommitResult(
        applied=[], violations=[violation], outcome="finish_tool"
    )

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent.run",
        return_value=result_with_violation,
    ):
        warnings = runner.run(
            publish_request=_make_publish_request(force_extraction=True),
            request_id="req_violation",
            new_interactions=[_make_interaction("User", "test")],
            new_request=_make_request(),
            config=cfg,
        )

    assert any("violation A" in w for w in warnings)


def test_runner_soft_violation_does_not_surface_as_warning():
    """Soft invariant violations are logged but not added to warnings."""
    runner = _make_runner()

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="default",
                extraction_definition_prompt="Extract facts.",
            )
        ],
        user_playbook_extractor_configs=[],
    )

    soft_violation = Violation(
        # E (`inv_E_no_duplicate_creates`) is genuinely a soft invariant per
        # invariants.py — using "B" here mismatched its real severity ("hard")
        # and would have hidden a regression where soft violations were
        # mistakenly upgraded to hard.
        code="E",
        severity="soft",
        affected_op_indices=[0],
        msg="soft warning",
    )
    result_with_soft = CommitResult(
        applied=[], violations=[soft_violation], outcome="finish_tool"
    )

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent.run",
        return_value=result_with_soft,
    ):
        warnings = runner.run(
            publish_request=_make_publish_request(force_extraction=True),
            request_id="req_soft",
            new_interactions=[_make_interaction("User", "test")],
            new_request=_make_request(),
            config=cfg,
        )

    # Soft violations must NOT appear in warnings
    assert not any("violation" in w for w in warnings)


# ---------------------------------------------------------------------------
# Regression tests: per-kind tool constraint
# ---------------------------------------------------------------------------


def test_runner_profile_extractor_cannot_emit_playbook_ops(tmp_path):
    """Profile extractor runs with PROFILE_EXTRACTION_TOOLS.

    A scripted create_user_playbook call from the LLM (in the profile extractor
    turn) is rejected with 'unknown tool' by the registry; no playbook lands in
    storage.

    Note: Config with ``user_playbook_extractor_configs=[]`` triggers the
    schema validator which injects a default playbook extractor.  We account
    for that by scripting a second set of 2 turns (search → finish) for the
    default playbook extractor so the scripted list is not exhausted early.
    """
    import os

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    user_id = "u_profile_constraint"
    store = SQLiteStorage(
        org_id="test-org-pc", db_path=str(tmp_path / "profile_constraint.db")
    )

    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    pm = PromptManager()

    rc = MagicMock()
    rc.storage = store
    rc.prompt_manager = pm
    rc.configurator = MagicMock()
    rc.org_id = "test-org-pc"

    runner = AgenticExtractionRunner(llm_client=client, request_context=rc)

    # Turn order (2 extractors run in sequence — profile first, playbook second):
    # Profile extractor turns (PROFILE_EXTRACTION_TOOLS):
    #   1. search_user_profiles
    #   2. create_user_playbook ← forbidden, returns {"error": "unknown tool: ..."}
    #   3. finish
    # Default playbook extractor turns (PLAYBOOK_EXTRACTION_TOOLS):
    #   4. search_user_playbooks
    #   5. finish
    scripted = [
        # --- profile extractor ---
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1", "search_user_profiles", {"query": "food", "top_k": 10}
                )
            ]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_playbook",  # forbidden in PROFILE_EXTRACTION_TOOLS
                    {
                        "trigger": "ask about food",
                        "content": "suggest sushi",
                        "source_span": "I love sushi",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
        # --- default playbook extractor (no ops) ---
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c4", "search_user_playbooks", {"query": "food", "top_k": 10}
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c5", "finish", {})]),
    ]

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="profile_only",
                extraction_definition_prompt="Extract food preferences.",
            )
        ],
        # Empty list triggers default playbook extractor injection via schema validator.
        # This is expected behaviour; we script for it explicitly above.
        user_playbook_extractor_configs=[],
    )

    with patch.object(client, "generate_chat_response", side_effect=scripted):
        runner.run(
            publish_request=_make_publish_request(
                force_extraction=True, user_id=user_id
            ),
            request_id="req_pc",
            new_interactions=[_make_interaction("User", "I love sushi", user_id)],
            new_request=Request(
                request_id="req_pc",
                user_id=user_id,
                source="cli",
                agent_version="v1",
                session_id="s_pc",
            ),
            config=cfg,
        )

    # The forbidden create_user_playbook was rejected — zero playbooks in storage.
    playbooks = store.get_user_playbooks(user_id=user_id)
    assert playbooks == [], (
        f"Profile extractor must not emit playbooks; got: {playbooks}"
    )


def test_runner_playbook_extractor_cannot_emit_profile_ops(tmp_path):
    """Playbook extractor runs with PLAYBOOK_EXTRACTION_TOOLS.

    A scripted create_user_profile call from the LLM (in the playbook extractor
    turn) is rejected with 'unknown tool' by the registry; no profile lands in
    storage.

    Note: Config with ``profile_extractor_configs=[]`` triggers the schema
    validator which injects a default profile extractor.  We account for that
    by scripting a first set of 2 turns (search → finish) for the default
    profile extractor, then 3 turns for the explicit playbook extractor.
    """
    import os

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    user_id = "u_playbook_constraint"
    store = SQLiteStorage(
        org_id="test-org-plc", db_path=str(tmp_path / "playbook_constraint.db")
    )

    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    pm = PromptManager()

    rc = MagicMock()
    rc.storage = store
    rc.prompt_manager = pm
    rc.configurator = MagicMock()
    rc.org_id = "test-org-plc"

    runner = AgenticExtractionRunner(llm_client=client, request_context=rc)

    # Turn order (2 extractors run in sequence — profile first, playbook second):
    # Default profile extractor turns (PROFILE_EXTRACTION_TOOLS, no ops):
    #   1. search_user_profiles
    #   2. finish
    # Playbook extractor turns (PLAYBOOK_EXTRACTION_TOOLS):
    #   3. search_user_playbooks
    #   4. create_user_profile ← forbidden, returns {"error": "unknown tool: ..."}
    #   5. finish
    scripted = [
        # --- default profile extractor (no ops) ---
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1", "search_user_profiles", {"query": "food", "top_k": 10}
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c2", "finish", {})]),
        # --- playbook extractor ---
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c3", "search_user_playbooks", {"query": "food", "top_k": 10}
                )
            ]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c4",
                    "create_user_profile",  # forbidden in PLAYBOOK_EXTRACTION_TOOLS
                    {
                        "content": "user likes sushi",
                        "ttl": "infinity",
                        "source_span": "I love sushi",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c5", "finish", {})]),
    ]

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        # Empty list triggers default profile extractor injection via schema validator.
        # This is expected behaviour; we script for it explicitly above.
        profile_extractor_configs=[],
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="playbook_only",
                extraction_definition_prompt="Extract behavioral rules.",
            )
        ],
    )

    with patch.object(client, "generate_chat_response", side_effect=scripted):
        runner.run(
            publish_request=_make_publish_request(
                force_extraction=True, user_id=user_id
            ),
            request_id="req_plc",
            new_interactions=[_make_interaction("User", "I love sushi", user_id)],
            new_request=Request(
                request_id="req_plc",
                user_id=user_id,
                source="cli",
                agent_version="v1",
                session_id="s_plc",
            ),
            config=cfg,
        )

    # The forbidden create_user_profile was rejected — zero profiles in storage.
    profiles = store.get_user_profile(user_id)
    assert profiles == [], f"Playbook extractor must not emit profiles; got: {profiles}"
