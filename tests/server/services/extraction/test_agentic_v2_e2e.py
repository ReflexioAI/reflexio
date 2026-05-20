"""End-to-end test for agentic-v2 via GenerationService.run.

Exercises the full publish flow (gate -> config iteration -> windowing
-> ExtractionAgent -> commit -> aggregator trigger) with a mocked LLM.
Verifies storage state + aggregator invocation.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    UserPlaybookExtractorConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mk_tool_call(id_: str, name: str, args: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _mk_resp(tool_calls: list, content: str | None = None) -> MagicMock:
    r = MagicMock()
    r.tool_calls = tool_calls
    r.content = content
    return r


def _make_agentic_config() -> Config:
    return Config(
        extraction_backend="agentic",
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="e2e_profile",
                extraction_definition_prompt="Extract user facts from the session.",
            ),
        ],
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="e2e_playbook",
                extraction_definition_prompt="Extract behavioral preferences.",
                aggregation_config=PlaybookAggregatorConfig(),
            ),
        ],
    )


def _make_scripted_client(responses: list) -> LiteLLMClient:
    """Build a real LiteLLMClient whose generate_chat_response is scripted.

    Scopes ``OPENAI_API_KEY`` to client construction via ``patch.dict`` so
    the env mutation does not leak into other tests in the same process
    (which would make test ordering matter).
    """
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
        client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    client.generate_chat_response = MagicMock(side_effect=responses)  # type: ignore[method-assign]
    return client


# ---------------------------------------------------------------------------
# Test 1: full flow — profile + playbook created, aggregator triggered
# ---------------------------------------------------------------------------


def test_e2e_agentic_v2_full_flow(tmp_path):
    """Publish a session with extraction_backend='agentic'; verify storage + aggregator.

    Scripts 6 LLM turns (3 per extractor: search -> create -> finish) and
    asserts that:
      - A profile with the expected content is written to storage.
      - A user playbook with the expected content is written to storage.
      - PlaybookAggregator.run is invoked at least once.
      - No unexpected warnings are returned.
    """
    user_id = "e2e_user"
    org_id = "e2e_org"

    # 6 scripted turns: 3 for profile extractor, 3 for playbook extractor.
    scripted = [
        # --- profile extractor ---
        _mk_resp(
            [
                _mk_tool_call(
                    "c1",
                    "search_user_profiles",
                    {"query": "food preferences", "top_k": 10},
                )
            ]
        ),
        _mk_resp(
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
        _mk_resp([_mk_tool_call("c3", "finish", {})]),
        # --- playbook extractor ---
        _mk_resp(
            [
                _mk_tool_call(
                    "c4",
                    "search_user_playbooks",
                    {"query": "food preferences", "top_k": 10},
                )
            ]
        ),
        _mk_resp(
            [
                _mk_tool_call(
                    "c5",
                    "create_user_playbook",
                    {
                        "trigger": "user asks about food",
                        "content": "suggest sushi-related options",
                        "source_span": "I love sushi",
                    },
                )
            ]
        ),
        _mk_resp([_mk_tool_call("c6", "finish", {})]),
    ]

    client = _make_scripted_client(scripted)

    with tempfile.TemporaryDirectory() as temp_dir:
        request_context = RequestContext(org_id=org_id, storage_base_dir=temp_dir)
        gs = GenerationService(llm_client=client, request_context=request_context)
        # Inject agentic Config; bypass disk-based configurator.
        gs.configurator.get_config = MagicMock(return_value=_make_agentic_config())  # type: ignore[method-assign]

        with patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookAggregator"
        ) as mock_agg_cls:
            mock_agg = MagicMock()
            mock_agg_cls.return_value = mock_agg

            request = PublishUserInteractionRequest(
                user_id=user_id,
                interaction_data_list=[
                    InteractionData(
                        role="User",
                        content="I love sushi — please always recommend it when I ask about food.",
                    ),
                    InteractionData(
                        role="Assistant",
                        content="Noted! I'll keep your sushi preference in mind.",
                    ),
                ],
                session_id="e2e_sid",
                force_extraction=True,
            )
            result = gs.run(request)

        # --- profile assertion ---
        assert request_context.storage is not None
        profiles = request_context.storage.get_user_profile(user_id)
        assert any("sushi" in (p.content or "").lower() for p in profiles), (
            f"expected a sushi profile; got: {[p.content for p in profiles]}"
        )

        # Provenance: agentic-extracted profiles must carry the publish
        # request_id so retrieval can trace back to the source publish (this
        # is what LongMemEval-style recall@K depends on).
        for p in profiles:
            assert p.generated_from_request_id == result.request_id, (
                f"profile {p.profile_id} has stale generated_from_request_id "
                f"{p.generated_from_request_id!r}, expected {result.request_id!r}"
            )

        # --- playbook assertion ---
        playbooks = request_context.storage.get_user_playbooks(user_id=user_id)
        assert any("sushi" in (pb.content or "").lower() for pb in playbooks), (
            f"expected a sushi playbook; got: {[pb.content for pb in playbooks]}"
        )

        # Mirror provenance assertion for playbooks.
        for pb in playbooks:
            assert pb.request_id == result.request_id, (
                f"playbook {pb.user_playbook_id} has stale request_id "
                f"{pb.request_id!r}, expected {result.request_id!r}"
            )

        # --- aggregator triggered ---
        assert mock_agg.run.call_count >= 1, (
            "PlaybookAggregator.run should have been called at least once"
        )

        # --- no unexpected warnings ---
        assert not result.warnings, f"unexpected warnings: {result.warnings}"


# ---------------------------------------------------------------------------
# Test 2: extraction skipped when pre-filter rejects short session
# ---------------------------------------------------------------------------


def test_e2e_agentic_v2_extraction_agent_not_invoked_for_trivial_session(tmp_path):
    """Pre-filter rejects short-content session; ExtractionAgent is never called.

    Uses force_extraction=False with very short user content (< 30 chars) to
    trigger the 'all_user_turns_too_short' pre-filter path inside
    AgenticExtractionRunner.  ExtractionAgent must not be constructed or called.

    Choice: we exercise the real _cheap_should_run_reject path (not empty
    interaction_data_list, which would be rejected by Pydantic min_length=1).
    """
    user_id = "e2e_user2"
    org_id = "e2e_org2"

    # No LLM turns should be consumed.
    client = _make_scripted_client([])

    with tempfile.TemporaryDirectory() as temp_dir:
        request_context = RequestContext(org_id=org_id, storage_base_dir=temp_dir)
        gs = GenerationService(llm_client=client, request_context=request_context)
        gs.configurator.get_config = MagicMock(return_value=_make_agentic_config())  # type: ignore[method-assign]

        with patch(
            "reflexio.server.services.extraction.agentic_adapter.ExtractionAgent"
        ) as mock_agent_cls:
            request = PublishUserInteractionRequest(
                user_id=user_id,
                interaction_data_list=[
                    # Short user content (< 30 chars) → pre-filter rejects.
                    InteractionData(role="User", content="hi"),
                ],
                session_id="e2e_sid2",
                force_extraction=False,  # pre-filter active
            )
            result = gs.run(request)

        # ExtractionAgent was never instantiated.
        mock_agent_cls.assert_not_called()

    # No profiles persisted.
    assert request_context.storage is not None
    profiles = request_context.storage.get_user_profile(user_id)
    assert profiles == [], f"expected no profiles; got {profiles}"

    # Result must not have raised (warnings may be empty or trivial).
    assert result.request_id is not None


def test_e2e_publish_stores_interactions_but_skips_extraction_when_stalled(tmp_path):
    """An auth/billing stall pauses automatic learning retries, not raw storage."""
    user_id = "stalled_user"
    org_id = "stalled_org"

    with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False
    ):
        request_context = RequestContext(org_id=org_id, storage_base_dir=temp_dir)
        assert request_context.storage is not None
        request_context.storage.llm_client.get_embeddings = MagicMock(  # type: ignore[method-assign]
            return_value=[[], []]
        )
        request_context.storage.upsert_stall_state(
            reason="auth_error",
            stalled_at=datetime.now(UTC),
            reset_estimate=None,
            error_message="401 Invalid authentication credentials",
        )

        gs = GenerationService(
            llm_client=_make_scripted_client([]),
            request_context=request_context,
        )
        gs.configurator.get_config = MagicMock(return_value=_make_agentic_config())  # type: ignore[method-assign]

        request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[
                InteractionData(
                    role="User",
                    content=(
                        "I prefer implementation work to pause automatic learning "
                        "when provider authentication is broken."
                    ),
                ),
                InteractionData(
                    role="Assistant",
                    content="I will surface the authentication issue instead.",
                ),
            ],
            session_id="stalled_sid",
            force_extraction=True,
        )

        with (
            patch(
                "reflexio.server.services.generation_service.ReflectionService.run"
            ) as mock_reflection_run,
            patch(
                "reflexio.server.services.extraction.agentic_adapter."
                "AgenticExtractionRunner.run"
            ) as mock_agentic_run,
        ):
            result = gs.run(request)

        assert result.request_id is not None
        assert result.warnings
        assert "auth_error" in result.warnings[0]
        assert "paused" in result.warnings[0]
        assert mock_reflection_run.call_count == 0
        assert mock_agentic_run.call_count == 0

        stored = request_context.storage.get_user_interaction(user_id)
        assert len(stored) == 2
        assert {item.role for item in stored} == {"User", "Assistant"}


def test_e2e_stalled_forced_publish_runs_with_explicit_override(tmp_path):
    """A forced publish can retry a stall only with the explicit override flag."""
    user_id = "stalled_override_user"
    org_id = "stalled_override_org"

    with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
        os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False
    ):
        request_context = RequestContext(org_id=org_id, storage_base_dir=temp_dir)
        assert request_context.storage is not None
        request_context.storage.llm_client.get_embeddings = MagicMock(  # type: ignore[method-assign]
            return_value=[[], []]
        )
        request_context.storage.upsert_stall_state(
            reason="auth_error",
            stalled_at=datetime.now(UTC),
            reset_estimate=None,
            error_message="401 Invalid authentication credentials",
        )

        gs = GenerationService(
            llm_client=_make_scripted_client([]),
            request_context=request_context,
        )
        gs.configurator.get_config = MagicMock(return_value=_make_agentic_config())  # type: ignore[method-assign]

        request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[
                InteractionData(
                    role="User",
                    content="Retry extraction explicitly after fixing provider authentication.",
                ),
                InteractionData(role="Assistant", content="Retrying extraction now."),
            ],
            session_id="stalled_override_sid",
            force_extraction=True,
            override_learning_stall=True,
        )

        with (
            patch(
                "reflexio.server.services.generation_service.ReflectionService.run"
            ) as mock_reflection_run,
            patch(
                "reflexio.server.services.extraction.agentic_adapter."
                "AgenticExtractionRunner.run",
                return_value=[],
            ) as mock_agentic_run,
        ):
            result = gs.run(request)

        assert result.request_id is not None
        assert result.warnings == []
        assert mock_reflection_run.call_count == 1
        assert mock_agentic_run.call_count == 1


# ---------------------------------------------------------------------------
# Test 3: one rule → exactly one playbook (tool constraint regression)
# ---------------------------------------------------------------------------


def test_e2e_one_rule_produces_exactly_one_playbook(tmp_path):
    """Single publish, single behavioural rule, two extractor configs enabled.

    Profile extractor: search_user_profiles → create_user_profile → finish.
    Playbook extractor: search_user_playbooks → create_user_playbook → finish.

    Because PROFILE_EXTRACTION_TOOLS forbids create_user_playbook, the profile
    extractor cannot accidentally emit a second playbook even if the scripted LLM
    tried to.  Only the playbook extractor's create_user_playbook call succeeds,
    so exactly one UserPlaybook lands in storage.
    """
    user_id = "e2e_user3"
    org_id = "e2e_org3"

    # 6 scripted turns:
    # profile extractor (3): search_user_profiles → create_profile → finish
    # playbook extractor (3): search_playbooks → create_playbook → finish
    scripted = [
        # --- profile extractor: only emits a profile ---
        _mk_resp(
            [
                _mk_tool_call(
                    "c1",
                    "search_user_profiles",
                    {"query": "on-call schedule", "top_k": 10},
                )
            ]
        ),
        _mk_resp(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_profile",
                    {
                        "content": "user is on-call this week",
                        "ttl": "one_week",
                        "source_span": "on-call this week",
                    },
                )
            ]
        ),
        _mk_resp([_mk_tool_call("c3", "finish", {})]),
        # --- playbook extractor: emits one playbook ---
        _mk_resp(
            [
                _mk_tool_call(
                    "c4",
                    "search_user_playbooks",
                    {"query": "code review scheduling", "top_k": 10},
                )
            ]
        ),
        _mk_resp(
            [
                _mk_tool_call(
                    "c5",
                    "create_user_playbook",
                    {
                        "trigger": "code review scheduling",
                        "content": "avoid scheduling code reviews before 10am",
                        "source_span": "no code review before 10am",
                    },
                )
            ]
        ),
        _mk_resp([_mk_tool_call("c6", "finish", {})]),
    ]

    client = _make_scripted_client(scripted)

    config = Config(
        extraction_backend="agentic",
        storage_config=StorageConfigSQLite(),
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="oncall_profile",
                extraction_definition_prompt="Extract on-call and schedule facts.",
            ),
        ],
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="scheduling_rules",
                extraction_definition_prompt="Extract scheduling behavioural rules.",
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        request_context = RequestContext(org_id=org_id, storage_base_dir=temp_dir)
        gs = GenerationService(llm_client=client, request_context=request_context)
        gs.configurator.get_config = MagicMock(return_value=config)  # type: ignore[method-assign]

        request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[
                InteractionData(
                    role="User",
                    content=(
                        "I'm on-call this week. "
                        "Please avoid scheduling code reviews before 10am for me."
                    ),
                ),
                InteractionData(
                    role="Assistant",
                    content="Noted — I'll avoid scheduling code reviews before 10am.",
                ),
            ],
            session_id="e2e_sid3",
            force_extraction=True,
        )
        result = gs.run(request)

    # Exactly one playbook — the profile extractor's PROFILE_EXTRACTION_TOOLS
    # forbids create_user_playbook so only the playbook extractor's call lands.
    assert request_context.storage is not None
    playbooks = request_context.storage.get_user_playbooks(user_id=user_id)
    assert len(playbooks) == 1, (
        f"Expected exactly 1 playbook; got {len(playbooks)}: {[pb.content for pb in playbooks]}"
    )

    # Profile content must not contain behavioural guidance markers.
    profiles = request_context.storage.get_user_profile(user_id)
    assert len(profiles) == 1, (
        f"Expected exactly 1 profile; got {len(profiles)}: {[p.content for p in profiles]}"
    )

    # No unexpected warnings.
    assert not result.warnings, f"unexpected warnings: {result.warnings}"
