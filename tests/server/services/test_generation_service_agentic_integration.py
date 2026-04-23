"""Integration test: GenerationService.run routes through the agentic adapter.

The orchestrator's 6-reader / 2-critic / reconciler cascade is covered by
``test_agentic_backend_pipeline_integration.py``. This test focuses on the
dispatcher glue — config flag set to ``"agentic"`` → publish → persisted
profiles / playbooks carry ``reader_angle`` / ``source_span``; classic config
still runs the classic pipeline.

LLM calls within ``AgenticExtractionService`` are stubbed at the service
boundary so the test doesn't need to thread through the tool-call sequencing
of 6+2+reconciler; that's a concern of the dedicated orchestrator test.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import SearchUserProfileRequest
from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.extraction.agentic_extraction_service import (
    ExtractionResult,
)
from reflexio.server.services.extraction.critics import VettedPlaybook, VettedProfile

pytestmark = pytest.mark.integration


def _make_publish_request() -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        user_id="u_test",
        interaction_data_list=[
            InteractionData(
                role="User",
                content=(
                    "I'm a senior Go engineer. This week I'm on-call, "
                    "avoid scheduling reviews before 10am."
                ),
            ),
            InteractionData(
                role="Agent",
                content="Got it — routing review requests after 10am while you're on-call.",
            ),
        ],
        source="cli",
        agent_version="v1",
    )


def _fake_extraction_result() -> ExtractionResult:
    """Two vetted items that exercise both lanes + both new agentic fields."""
    return ExtractionResult(
        profiles=[
            VettedProfile(
                content="User is a senior Go engineer.",
                time_to_live="infinity",
                source_span="senior Go engineer",
                reader_angle="facts",
            ),
            VettedProfile(
                content="User is on-call this week.",
                time_to_live="one_week",
                source_span="This week I'm on-call",
                reader_angle="context",
            ),
        ],
        playbooks=[
            VettedPlaybook(
                trigger="scheduling a review during user's on-call week",
                content="avoid times before 10am",
                rationale="user is on-call this week",
                reader_angle="behavior",
            ),
        ],
    )


def _install_agentic_config(reflexio: Reflexio) -> None:
    """Overwrite the configurator's in-memory config with agentic backends on."""
    cfg = Config(
        storage_config=StorageConfigSQLite(),
        extraction_backend="agentic",
        search_backend="agentic",
    )
    reflexio.request_context.configurator.config = cfg


def test_generation_service_run_agentic_path_persists_with_agentic_fields(
    tmp_path, monkeypatch
):
    """End-to-end: config.extraction_backend=agentic → profiles persisted with reader_angle."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    monkeypatch.setenv("REFLEXIO_STORAGE", "sqlite")

    reflexio = Reflexio(
        org_id="test-agentic-dispatch",
        storage_base_dir=str(tmp_path),
    )
    _install_agentic_config(reflexio)

    # Stub the agentic orchestrator's LLM-driven run() so the test doesn't
    # depend on exact tool-call sequencing. The orchestrator itself has its
    # own integration test.
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.AgenticExtractionService"
        ) as mock_service_cls,
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=False,
        ),
    ):
        mock_service_cls.return_value.run.return_value = _fake_extraction_result()
        reflexio.publish_interaction(_make_publish_request())

    # Verify profiles persisted with the agentic fields set
    storage = reflexio.request_context.storage
    assert storage is not None
    results = storage.search_user_profile(
        SearchUserProfileRequest(user_id="u_test", top_k=10)
    )
    assert len(results) == 2, f"expected 2 profiles, got {len(results)}"

    angles = {p.reader_angle for p in results}
    assert angles == {"facts", "context"}, angles
    assert all(p.source_span for p in results), "source_span populated on every profile"
    assert all(p.extractor_names == ["agentic"] for p in results)

    # Verify playbook persisted with reader_angle
    playbooks = storage.get_user_playbooks(user_id="u_test", limit=10)
    assert len(playbooks) == 1
    assert playbooks[0].reader_angle == "behavior"
    assert playbooks[0].trigger == "scheduling a review during user's on-call week"


def test_generation_service_run_classic_path_does_not_call_agentic_runner(
    tmp_path, monkeypatch
):
    """Regression guard: classic config must not invoke the agentic adapter."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    monkeypatch.setenv("REFLEXIO_STORAGE", "sqlite")

    reflexio = Reflexio(
        org_id="test-classic-dispatch",
        storage_base_dir=str(tmp_path),
    )
    # Default config → extraction_backend="classic".
    assert reflexio.request_context.configurator.config.extraction_backend == "classic"

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.AgenticExtractionService"
    ) as mock_service_cls:
        mock_service_cls.return_value.run.return_value = _fake_extraction_result()
        # Force extraction to bypass the classic cheap pre-filter for this test
        # (we don't care about the classic LLM call succeeding — we only care
        # that the agentic adapter was NOT invoked).
        req = _make_publish_request()
        req.force_extraction = True
        # Classic extractors may fail without real LLM keys — that's fine,
        # we're only asserting the agentic adapter wasn't touched.
        with contextlib.suppress(Exception):
            reflexio.publish_interaction(req)

    mock_service_cls.assert_not_called()


def test_runner_returns_warnings_from_aggregator_failure(tmp_path, monkeypatch):
    """If the PlaybookAggregator raises, the publish still succeeds with a warning."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    monkeypatch.setenv("REFLEXIO_STORAGE", "sqlite")

    reflexio = Reflexio(
        org_id="test-aggregator-fail",
        storage_base_dir=str(tmp_path),
    )

    from reflexio.models.config_schema import (
        PlaybookAggregatorConfig,
        UserPlaybookExtractorConfig,
    )

    reflexio.request_context.configurator.config = Config(
        storage_config=StorageConfigSQLite(),
        extraction_backend="agentic",
        search_backend="agentic",
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="agg_playbook",
                extraction_definition_prompt="x",
                aggregation_config=PlaybookAggregatorConfig(),
            ),
        ],
    )

    failing_aggregator = MagicMock()
    failing_aggregator.return_value.run.side_effect = RuntimeError("aggregator down")
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.AgenticExtractionService"
        ) as mock_service_cls,
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=False,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookAggregator",
            failing_aggregator,
        ),
    ):
        mock_service_cls.return_value.run.return_value = _fake_extraction_result()
        # publish_interaction returns the GenerationServiceResult — check warnings.
        response = reflexio.publish_interaction(_make_publish_request())

    # Playbook was still saved despite the aggregator blowing up.
    storage = reflexio.request_context.storage
    assert storage is not None
    playbooks = storage.get_user_playbooks(user_id="u_test", limit=10)
    assert len(playbooks) == 1
    # And the failure surfaced as a warning (non-fatal).
    warnings_list = getattr(response, "warnings", None) or []
    assert any("aggregation failed for agg_playbook" in w for w in warnings_list)
