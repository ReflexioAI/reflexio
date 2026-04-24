"""End-to-end test for agentic-v2 via GenerationService.run.

Exercises the full publish flow (gate -> config iteration -> windowing
-> ExtractionAgent -> commit -> aggregator trigger) with a mocked LLM.
Verifies storage state + aggregator invocation.
"""

from __future__ import annotations

import json
import os
import tempfile
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
    """Build a real LiteLLMClient whose generate_chat_response is scripted."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
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

        # --- playbook assertion ---
        playbooks = request_context.storage.get_user_playbooks(user_id=user_id)
        assert any("sushi" in (pb.content or "").lower() for pb in playbooks), (
            f"expected a sushi playbook; got: {[pb.content for pb in playbooks]}"
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
