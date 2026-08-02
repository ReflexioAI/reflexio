import datetime
import tempfile
from datetime import UTC
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    ManualPlaybookGenerationRequest,
    Request,
    RerunPlaybookGenerationRequest,
)
from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    PlaybookConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
    PlaybookGenerationServiceConfig,
)


def create_request_interaction_data_model(
    user_id: str, request_id: str, interactions: list[Interaction]
) -> RequestInteractionDataModel:
    """Helper function to create a RequestInteractionDataModel for testing."""
    request = Request(
        request_id=request_id,
        user_id=user_id,
        source="test",
        agent_version="1.0",
        session_id="session_1",
    )
    return RequestInteractionDataModel(
        session_id="session_1",
        request=request,
        interactions=interactions,
    )


def _storage(service: PlaybookGenerationService) -> Any:
    assert service.storage is not None
    return cast(Any, service.storage)


def _aggregation_enabled_config() -> Config:
    return Config(
        storage_config=StorageConfigSQLite(),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        ),
    )


def _service_for_inline_aggregation(configurator: Any) -> PlaybookGenerationService:
    ctx = MagicMock()
    ctx.configurator = configurator
    ctx.org_id = "test-org"
    service = PlaybookGenerationService(llm_client=MagicMock(), request_context=ctx)
    service.service_config = PlaybookGenerationServiceConfig(
        request_id="req-1",
        agent_version="v1",
        user_id="user-1",
    )
    return service


def test_inline_aggregation_default_path_schedules_durably():
    configurator = MagicMock()
    configurator.get_config.return_value = _aggregation_enabled_config()
    service = _service_for_inline_aggregation(configurator)
    with patch(
        "reflexio.server.services.playbook.aggregation_trigger."
        "ensure_local_playbook_aggregation_scheduler"
    ):
        service._trigger_playbook_aggregation()

    _storage(service).schedule_playbook_aggregation.assert_called_once_with("v1")


def test_maybe_trigger_user_playbook_aggregation_durably_schedules():
    from reflexio.server.services.playbook.aggregation_trigger import (
        maybe_trigger_user_playbook_aggregation,
    )

    configurator = MagicMock()
    configurator.get_config.return_value = _aggregation_enabled_config()
    ctx = MagicMock()
    ctx.configurator = configurator
    ctx.org_id = "test-org"
    ctx.storage = MagicMock()
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.playbook.aggregation_trigger."
        "ensure_local_playbook_aggregation_scheduler"
    ) as ensure_scheduler:
        result = maybe_trigger_user_playbook_aggregation(
            request_context=ctx,
            llm_client=llm_client,
            agent_version="v1",
            reason="unit_test",
        )

    assert result.status == "scheduled"
    assert result.reason == "unit_test"
    assert result.agent_version == "v1"
    ctx.storage.schedule_playbook_aggregation.assert_called_once_with("v1")
    ensure_scheduler.assert_called_once_with(ctx)


def test_maybe_trigger_user_playbook_aggregation_reports_no_config():
    from reflexio.server.services.playbook.aggregation_trigger import (
        maybe_trigger_user_playbook_aggregation,
    )

    configurator = MagicMock()
    configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        user_playbook_extractor_config=None,
    )
    ctx = MagicMock(configurator=configurator, org_id="test-org")

    result = maybe_trigger_user_playbook_aggregation(
        request_context=ctx,
        llm_client=MagicMock(),
        agent_version="v1",
        reason="missing_config",
    )

    assert result.status == "skipped_no_config"
    assert result.reason == "missing_config"
    assert result.agent_version == "v1"


def test_maybe_trigger_user_playbook_aggregation_reports_schedule_failure():
    from reflexio.server.services.playbook.aggregation_trigger import (
        maybe_trigger_user_playbook_aggregation,
    )

    configurator = MagicMock()
    configurator.get_config.return_value = _aggregation_enabled_config()
    ctx = MagicMock(configurator=configurator, org_id="test-org")
    ctx.storage.schedule_playbook_aggregation.side_effect = RuntimeError("boom")

    result = maybe_trigger_user_playbook_aggregation(
        request_context=ctx,
        llm_client=MagicMock(),
        agent_version="v1",
        reason="post_commit",
    )

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"


@pytest.fixture
def mock_chat_completion():
    # Mock response for should_generate_playbook call
    mock_should_generate_response = "true"

    # Mock response for extract_playbook call
    mock_extract_response = '```json\n{\n    "playbook": "The agent was helpful and provided accurate information",\n    "type": "positive"\n}\n```'

    def mock_generate_chat_response_side_effect(messages, **kwargs):
        """
        Check prompt content to determine which mock response to return.
        If prompt contains "Output just a boolean value", return boolean response.
        Otherwise, return JSON playbook response.
        """
        # Get the prompt content from the messages
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        # Check if this is a should_generate_playbook call
        if "Output just a boolean value" in prompt_content:
            return mock_should_generate_response
        # Otherwise, this is a playbook extraction call
        return mock_extract_response

    # Mock the LiteLLM client's generate_chat_response method
    with patch(
        "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
        side_effect=mock_generate_chat_response_side_effect,
    ):
        yield


def test_generate_playbook(mock_chat_completion):
    user_id = "test_user_id"
    org_id = "0"
    interaction = Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id="test_request_id",
        content="The agent was very helpful in explaining the process",
        role="user",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up playbook config with window_size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )
        playbook_generation_service.configurator.set_config_by_name("window_size", 100)

        # Store interactions in storage first
        request_obj = Request(
            request_id="test_request_id",
            user_id=user_id,
            source="test",
            agent_version="1.0",
            session_id="session_1",
        )
        _storage(playbook_generation_service).add_request(request_obj)
        _storage(playbook_generation_service).add_user_interaction(user_id, interaction)

        # Create playbook generation request - extractors collect from storage
        playbook_request = PlaybookGenerationRequest(
            request_id="test_request_id",
            agent_version="1.0",
            user_id=user_id,
            auto_run=False,  # Skip stride check for testing
        )

        playbook_generation_service.run(playbook_request)

        # Verify playbook was saved
        user_playbooks = _storage(playbook_generation_service).get_user_playbooks()
        assert len(user_playbooks) > 0
        playbook = user_playbooks[0]
        assert playbook.request_id == "test_request_id"


def test_empty_interactions(mock_chat_completion):
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up playbook config
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )

        playbook_request = PlaybookGenerationRequest(
            request_id="test_request_id",
            agent_version="1.0",
            user_id="test_user_id",
            auto_run=False,
        )

        playbook_generation_service.run(playbook_request)

        # Verify no playbook was generated
        user_playbooks = _storage(playbook_generation_service).get_user_playbooks()
        assert len(user_playbooks) == 0


def test_missing_configs(mock_chat_completion):
    user_id = "test_user_id"
    org_id = "0"
    interaction = Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id="test_request_id",
        content="The agent was very helpful in explaining the process",
        role="user",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        request_obj = Request(
            request_id="test_request_id",
            user_id=user_id,
            source="test",
            agent_version="1.0",
            session_id="session_1",
        )
        _storage(playbook_generation_service).add_request(request_obj)
        _storage(playbook_generation_service).add_user_interaction(user_id, interaction)

        # Create playbook generation request without setting up configs
        playbook_request = PlaybookGenerationRequest(
            request_id="test_request_id",
            agent_version="1.0",
            user_id=user_id,
            auto_run=False,
        )

        with patch.object(
            playbook_generation_service, "_load_extractor_config", return_value=None
        ):
            playbook_generation_service.run(playbook_request)

        # Verify no playbook was generated
        user_playbooks = _storage(playbook_generation_service).get_user_playbooks()
        assert len(user_playbooks) == 0


def test_error_handling(mock_chat_completion):
    user_id = "test_user_id"
    org_id = "0"
    interaction = Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id="test_request_id",
        content="The agent was very helpful in explaining the process",
        role="user",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        # Set up playbook config
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )

        request_obj = Request(
            request_id="test_request_id",
            user_id=user_id,
            source="test",
            agent_version="1.0",
            session_id="session_1",
        )
        _storage(playbook_generation_service).add_request(request_obj)
        _storage(playbook_generation_service).add_user_interaction(user_id, interaction)

        # Create playbook generation request
        playbook_request = PlaybookGenerationRequest(
            request_id="test_request_id",
            agent_version="1.0",
            user_id=user_id,
            auto_run=False,
        )

        # Mock the lineage-aware save to raise an exception
        with patch.object(
            _storage(playbook_generation_service),
            "save_user_playbooks",
            side_effect=Exception("Storage error"),
        ):
            # The service should handle the error gracefully
            playbook_generation_service.run(playbook_request)

            # Verify no playbook was saved
            user_playbooks = _storage(playbook_generation_service).get_user_playbooks()
            assert len(user_playbooks) == 0


def test_finalize_drops_empty_and_same_batch_duplicates_with_dedup_flag_off():
    """The write path should be idempotent even without LLM deduplication."""
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )
        playbook_generation_service.service_config = PlaybookGenerationServiceConfig(
            request_id="test_request_id",
            agent_version="1.0",
            user_id="test_user",
            source="test_source",
        )

        first = UserPlaybook(
            agent_version="1.0",
            request_id="test_request_id",
            content="Run the narrow verification first.",
            trigger="When debugging",
        )
        duplicate = UserPlaybook(
            agent_version="1.0",
            request_id="test_request_id",
            content="  run the narrow verification first.  ",
            trigger=" when debugging ",
        )
        blank = UserPlaybook(
            agent_version="1.0",
            request_id="test_request_id",
            content="   ",
            trigger="When debugging",
        )

        with (
            patch(
                "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator",
            ) as mock_dedup_cls,
            patch.object(
                _storage(playbook_generation_service),
                "save_user_playbooks",
            ) as save_user_playbooks,
            patch.object(
                playbook_generation_service, "_enqueue_user_playbook_optimization"
            ),
        ):
            mock_dedup_cls.return_value.deduplicate.side_effect = (
                lambda results, *_args, **_kwargs: (
                    [p for r in results for p in r],
                    [],
                    [],
                )
            )
            playbook_generation_service._finalize_extracted_items(
                [first, duplicate, blank],
                model_provenance=ModelProvenance(
                    model_name="served-model",
                    provider="provider",
                ),
            )

        save_user_playbooks.assert_called_once()
        saved_playbooks = save_user_playbooks.call_args.args[0]
        assert saved_playbooks == [first]
        assert first.status is None
        assert first.source == "test_source"
        context = save_user_playbooks.call_args.kwargs["lineage_contexts"][0]
        assert context.op_kind == "create"
        assert context.model_name == "served-model"


def test_finalize_without_provenance_emits_create_with_null_model_fields():
    """Opaque routes still write create lineage; model fields stay null."""
    from reflexio.models.api_schema.domain.entities import LineageContext

    with tempfile.TemporaryDirectory() as temp_dir:
        service = PlaybookGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id="0", storage_base_dir=temp_dir),
        )
        service.service_config = PlaybookGenerationServiceConfig(
            request_id="legacy-request",
            agent_version="1.0",
            user_id="test-user",
            source="test",
        )
        playbook = UserPlaybook(
            agent_version="1.0",
            request_id="legacy-request",
            content="Preserve the old output.",
            trigger="When resuming a legacy run",
        )

        with (
            patch(
                "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator",
            ) as mock_dedup_cls,
            patch.object(_storage(service), "save_user_playbooks") as save,
            patch.object(service, "_enqueue_user_playbook_optimization"),
        ):
            mock_dedup_cls.return_value.deduplicate.return_value = (
                [playbook],
                [],
                [],
            )
            mock_dedup_cls.return_value.model_provenance = None
            mock_dedup_cls.return_value.consolidated_output_indices = set()
            service._finalize_extracted_items([playbook], model_provenance=None)

        contexts = save.call_args.kwargs["lineage_contexts"]
        assert len(contexts) == 1
        assert contexts[0] == LineageContext(
            op_kind="create",
            actor="extractor",
            request_id="legacy-request",
            model_name=None,
            provider=None,
        )


def test_resolve_write_plan_honors_reject_all_consolidation():
    """An intentionally empty consolidation result must not restore candidates."""
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PlaybookGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id="0", storage_base_dir=temp_dir),
        )
        service.service_config = PlaybookGenerationServiceConfig(
            request_id="test_request_id",
            agent_version="1.0",
            user_id="test_user",
            source="test_source",
        )
        candidate = UserPlaybook(
            agent_version="1.0",
            request_id="test_request_id",
            content="A candidate already covered by an existing playbook.",
            trigger="When handling the same task",
        )

        with (
            patch(
                "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator",
            ) as consolidator_class,
            patch.object(
                _storage(service), "precompute_user_playbook_embeddings"
            ) as precompute,
        ):
            consolidator_class.return_value.deduplicate.return_value = ([], [], [])
            plan = service._resolve_write_plan([[candidate]])

        assert plan is None
        precompute.assert_not_called()


def test_resolve_write_plan_reviews_grounded_normal_candidates_before_consolidation():
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PlaybookGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id="0", storage_base_dir=temp_dir),
        )
        service.service_config = PlaybookGenerationServiceConfig(
            request_id="test_request_id",
            agent_version="1.0",
            user_id="test_user",
            source="test_source",
            auto_run=False,
        )
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="Review grounded lessons",
        )
        service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )
        request = Request(
            request_id="source_request",
            user_id="test_user",
            source="test_source",
            agent_version="1.0",
            session_id="session_1",
        )
        interaction = Interaction(
            interaction_id=44,
            user_id="test_user",
            request_id="source_request",
            content="Use the answer I already supplied.",
            role="user",
            created_at=44,
        )
        _storage(service).add_request(request)
        _storage(service).add_user_interaction("test_user", interaction)
        candidate = UserPlaybook(
            agent_version="1.0",
            request_id="test_request_id",
            user_id="test_user",
            content="Use the supplied answer.",
            trigger="When the user supplies the requested answer",
            rationale="The user explicitly corrected the repeated question.",
            source_interaction_ids=[44],
            source_span="Use the answer I already supplied.",
            reader_angle="correction",
        )
        reviewed = candidate.model_copy(
            update={"content": "Treat the supplied answer as binding."}
        )

        with (
            patch(
                "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator",
            ) as consolidator_class,
            patch(
                "reflexio.server.services.playbook.components.reviewer.PlaybookCandidateReviewer",
            ) as reviewer_class,
            patch.object(_storage(service), "precompute_user_playbook_embeddings"),
            patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}),
        ):
            consolidator = consolidator_class.return_value
            consolidator.retrieve_existing_playbooks.return_value = []
            consolidator.deduplicate.side_effect = lambda results, *_args, **_kwargs: (
                [item for result in results for item in result],
                [],
                [],
            )
            reviewer = reviewer_class.return_value
            reviewer.is_enabled.return_value = True
            reviewer.review.return_value = [reviewed]

            plan = service._resolve_write_plan([[candidate]])

        assert plan is not None
        assert plan.new_playbooks == [reviewed]
        reviewer.review.assert_called_once()
        consolidator.deduplicate.assert_called_once()
        assert consolidator.deduplicate.call_args.args[0] == [[reviewed]]


def test_review_interaction_window_raises_when_source_filter_excludes_window():
    service = PlaybookGenerationService(
        llm_client=MagicMock(), request_context=MagicMock()
    )
    service.service_config = PlaybookGenerationServiceConfig(
        request_id="request",
        agent_version="v1",
    )
    playbook_config = PlaybookConfig(
        extractor_name="test_playbook",
        extraction_definition_prompt="Review grounded lessons",
    )
    extractor = MagicMock()
    extractor._get_interactions.return_value = None

    with (
        patch.object(service, "_create_extractor", return_value=extractor),
        pytest.raises(RuntimeError, match="source filter excluded"),
    ):
        service._review_interaction_window(playbook_config)


def test_review_interaction_window_returns_honest_empty_window():
    service = PlaybookGenerationService(
        llm_client=MagicMock(), request_context=MagicMock()
    )
    service.service_config = PlaybookGenerationServiceConfig(
        request_id="request",
        agent_version="v1",
    )
    playbook_config = PlaybookConfig(
        extractor_name="test_playbook",
        extraction_definition_prompt="Review grounded lessons",
    )
    extractor = MagicMock()
    extractor._get_interactions.return_value = []

    with patch.object(service, "_create_extractor", return_value=extractor):
        assert service._review_interaction_window(playbook_config) == []


def test_automatic_review_reloads_the_configured_extraction_window():
    request_context = MagicMock()
    service = PlaybookGenerationService(
        llm_client=MagicMock(), request_context=request_context
    )
    service.service_config = PlaybookGenerationServiceConfig(
        request_id="request",
        agent_version="v1",
        user_id="user-1",
        source="chat",
    )
    playbook_config = PlaybookConfig(
        extractor_name="test_playbook",
        extraction_definition_prompt="Review grounded lessons",
        request_sources_enabled=["chat"],
        window_size_override=3,
    )
    expected_window = [MagicMock(spec=RequestInteractionDataModel)]
    extractor = MagicMock()
    extractor._get_interactions.return_value = expected_window

    with patch.object(
        service, "_create_extractor", return_value=extractor
    ) as create_extractor:
        assert service._review_interaction_window(playbook_config) == expected_window

    create_extractor.assert_called_once_with(playbook_config, service.service_config)
    extractor._get_interactions.assert_called_once_with()
    request_context.storage.get_interactions_by_ids.assert_not_called()


def test_loading_a_new_generation_request_clears_the_review_window_cache():
    service = PlaybookGenerationService(
        llm_client=MagicMock(), request_context=MagicMock()
    )
    service._review_window_cache = [MagicMock(spec=RequestInteractionDataModel)]

    config = service._load_generation_service_config(
        PlaybookGenerationRequest(
            request_id="request-2",
            agent_version="v1",
            user_id="user-2",
        )
    )

    assert config.request_id == "request-2"
    assert service._review_window_cache is None


def test_resolve_write_plan_fails_closed_for_strict_candidate_without_evidence():
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PlaybookGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id="0", storage_base_dir=temp_dir),
        )
        service.service_config = PlaybookGenerationServiceConfig(
            request_id="test_request_id",
            agent_version="1.0",
            user_id="test_user",
            source="test_source",
            auto_run=False,
        )
        service.configurator.set_config_by_name(
            "user_playbook_extractor_config",
            PlaybookConfig(
                extractor_name="test_playbook",
                extraction_definition_prompt="Review grounded lessons",
            ),
        )
        request = Request(
            request_id="source_request",
            user_id="test_user",
            source="test_source",
            agent_version="1.0",
            session_id="session_1",
        )
        interaction = Interaction(
            interaction_id=45,
            user_id="test_user",
            request_id="source_request",
            content="Use the answer I already supplied.",
            role="user",
            created_at=45,
        )
        _storage(service).add_request(request)
        _storage(service).add_user_interaction("test_user", interaction)
        candidate = UserPlaybook(
            agent_version="1.0",
            request_id="test_request_id",
            user_id="test_user",
            content="Use the supplied answer.",
            trigger="When the user supplies the requested answer",
            rationale="The user explicitly corrected the repeated question.",
            source_interaction_ids=[45],
            source_span=None,
            reader_angle="correction",
        )

        with (
            patch(
                "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator",
            ),
            patch(
                "reflexio.server.services.playbook.components.reviewer.PlaybookCandidateReviewer",
            ) as reviewer_class,
            patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}),
        ):
            reviewer_class.return_value.is_enabled.return_value = True
            with pytest.raises(
                RuntimeError,
                match="missing validated evidence metadata: C1",
            ):
                service._resolve_write_plan([[candidate]])


def test_run_manual_regular_no_window_size(mock_chat_completion):
    """Test run_manual_regular works even without window_size configured.

    Since extractors handle window size at their level, the manual flow no longer
    validates window_size upfront. Extractors use a fallback of 1000 interactions
    when no window size is configured.
    """
    org_id = "0"
    user_id = "test_user"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITHOUT window size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )
        # window_size is not configured

        # Add some interactions to storage
        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="request_1",
            content="Test content",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_obj = Request(
            request_id="request_1",
            user_id=user_id,
            session_id="test_session",
            source="",
        )
        _storage(playbook_generation_service).add_request(request_obj)
        _storage(playbook_generation_service).add_user_interaction(user_id, interaction)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
        )

        request = ManualPlaybookGenerationRequest(agent_version=agent_version)
        response = playbook_generation_service.run_manual_regular(request)

        # Without window_size, extractors use fallback of 1000 interactions
        # So the request should succeed
        assert response.success is True


def test_run_manual_regular_no_interactions(mock_chat_completion):
    """Test run_manual_regular handles case when no interactions exist."""
    org_id = "0"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITH window size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )
        playbook_generation_service.configurator.set_config_by_name("window_size", 100)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
        )

        request = ManualPlaybookGenerationRequest(agent_version=agent_version)
        response = playbook_generation_service.run_manual_regular(request)

        # Should succeed but with 0 playbooks since no interactions
        assert response.success is True
        assert response.playbooks_generated == 0
        assert response.msg is not None
        assert "No interactions found" in response.msg


def test_run_manual_regular_with_interactions(mock_chat_completion):
    """Test run_manual_regular generates playbooks with CURRENT status."""
    user_id = "test_user_id"
    org_id = "0"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITH window size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )
        playbook_generation_service.configurator.set_config_by_name("window_size", 100)

        # First, add some interactions to storage
        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="test_request_id",
            content="The agent was very helpful",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_obj = Request(
            request_id="test_request_id",
            user_id=user_id,
            session_id="test_session",
            source="",
            agent_version=agent_version,
        )
        _storage(playbook_generation_service).add_request(request_obj)
        _storage(playbook_generation_service).add_user_interaction(user_id, interaction)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
        )

        request = ManualPlaybookGenerationRequest(agent_version=agent_version)
        response = playbook_generation_service.run_manual_regular(request)

        # Should succeed (playbooks generated depends on mock)
        assert response.success is True
        # Note: playbooks_generated may be 0 if mock returns no playbook
        # The key is that the method runs without error


def test_run_manual_regular_with_source_filter(mock_chat_completion):
    """Test run_manual_regular respects source filter."""
    user_id = "test_user_id"
    org_id = "0"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITH window size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )
        playbook_generation_service.configurator.set_config_by_name("window_size", 100)

        # Add interactions with source_a
        interaction_a = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="request_a",
            content="The agent was helpful",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_a = Request(
            request_id="request_a",
            user_id=user_id,
            session_id="test_session",
            source="source_a",
            agent_version=agent_version,
        )
        _storage(playbook_generation_service).add_request(request_a)
        _storage(playbook_generation_service).add_user_interaction(
            user_id, interaction_a
        )

        # Add interactions with source_b
        interaction_b = Interaction(
            interaction_id=2,
            user_id=user_id,
            request_id="request_b",
            content="The agent was not helpful",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_b = Request(
            request_id="request_b",
            user_id=user_id,
            session_id="test_session",
            source="source_b",
            agent_version=agent_version,
        )
        _storage(playbook_generation_service).add_request(request_b)
        _storage(playbook_generation_service).add_user_interaction(
            user_id, interaction_b
        )

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
        )

        # Request with non-existent source
        request = ManualPlaybookGenerationRequest(
            agent_version=agent_version, source="non_existent_source"
        )
        response = playbook_generation_service.run_manual_regular(request)

        # Should succeed but with 0 playbooks since no matching source
        assert response.success is True
        assert response.playbooks_generated == 0


def test_run_manual_regular_output_pending_status_false(mock_chat_completion):
    """Test that run_manual_regular outputs CURRENT status when output_pending_status=False."""
    user_id = "test_user_id"
    org_id = "0"
    agent_version = "1.0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)

        # Create service with output_pending_status=False (default for manual regular)
        playbook_generation_service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
            allow_manual_trigger=True,
            output_pending_status=False,
        )

        # Set up playbook config WITH window size
        playbook_config = PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="test",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2,
            ),
        )
        playbook_generation_service.configurator.set_config_by_name(
            "user_playbook_extractor_config", playbook_config
        )
        playbook_generation_service.configurator.set_config_by_name("window_size", 100)

        # Add interaction
        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="test_request_id",
            content="The agent was very helpful",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        request_obj = Request(
            request_id="test_request_id",
            user_id=user_id,
            session_id="test_session",
            source="",
            agent_version=agent_version,
        )
        _storage(playbook_generation_service).add_request(request_obj)
        _storage(playbook_generation_service).add_user_interaction(user_id, interaction)

        from reflexio.models.api_schema.service_schemas import (
            ManualPlaybookGenerationRequest,
            Status,
        )

        request = ManualPlaybookGenerationRequest(agent_version=agent_version)
        response = playbook_generation_service.run_manual_regular(request)

        # Should succeed (playbooks generated depends on mock)
        assert response.success is True
        # Note: playbooks_generated may be 0 if mock returns no playbook

        # Verify no PENDING playbooks (output_pending_status=False)
        pending_playbooks = _storage(playbook_generation_service).get_user_playbooks(
            status_filter=[Status.PENDING]
        )

        assert len(pending_playbooks) == 0, (
            "Manual generation should not create PENDING playbooks"
        )


# ===============================
# Tests for _get_rerun_user_ids
# ===============================


class TestGetRerunItems:
    """Tests for the _get_rerun_user_ids method."""

    def test_get_rerun_user_ids_returns_user_ids(self):
        """Test that _get_rerun_user_ids returns user IDs."""
        org_id = "0"

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            service = PlaybookGenerationService(
                llm_client=llm_client,
                request_context=RequestContext(
                    org_id=org_id, storage_base_dir=temp_dir
                ),
            )

            # Add interactions with different users
            agent_version = "1.0"

            # User 1 - 2 requests
            user_id_1 = "test_user_1"
            for i in range(2):
                request_id = f"request_user1_{i}"
                interaction = Interaction(
                    interaction_id=i,
                    user_id=user_id_1,
                    request_id=request_id,
                    content=f"Test content {i}",
                    role="user",
                    created_at=int(datetime.datetime.now(UTC).timestamp()),
                )
                request_obj = Request(
                    request_id=request_id,
                    user_id=user_id_1,
                    source="test_source",
                    agent_version=agent_version,
                    session_id="group_1",
                )
                _storage(service).add_request(request_obj)
                _storage(service).add_user_interaction(user_id_1, interaction)

            # User 2 - 1 request
            user_id_2 = "test_user_2"
            request_id = "request_user2_0"
            interaction = Interaction(
                interaction_id=10,
                user_id=user_id_2,
                request_id=request_id,
                content="Test content user 2",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_obj = Request(
                request_id=request_id,
                user_id=user_id_2,
                source="test_source",
                agent_version=agent_version,
                session_id="group_2",
            )
            _storage(service).add_request(request_obj)
            _storage(service).add_user_interaction(user_id_2, interaction)

            from reflexio.models.api_schema.service_schemas import (
                RerunPlaybookGenerationRequest,
            )

            request = RerunPlaybookGenerationRequest(agent_version=agent_version)
            result = service._get_rerun_user_ids(request)

            # Should return list of 2 user IDs
            assert len(result) == 2
            assert user_id_1 in result
            assert user_id_2 in result

    def test_get_rerun_user_ids_with_source_filter(self):
        """Test that _get_rerun_user_ids applies source filter correctly."""
        org_id = "0"

        with tempfile.TemporaryDirectory() as temp_dir:
            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            service = PlaybookGenerationService(
                llm_client=llm_client,
                request_context=RequestContext(
                    org_id=org_id, storage_base_dir=temp_dir
                ),
            )

            agent_version = "1.0"

            # Add request with source_a for user_a
            user_id_a = "test_user_a"
            interaction_a = Interaction(
                interaction_id=1,
                user_id=user_id_a,
                request_id="request_a",
                content="Test content A",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_a = Request(
                request_id="request_a",
                user_id=user_id_a,
                source="source_a",
                agent_version=agent_version,
                session_id="group_a",
            )
            _storage(service).add_request(request_a)
            _storage(service).add_user_interaction(user_id_a, interaction_a)

            # Add request with source_b for user_b
            user_id_b = "test_user_b"
            interaction_b = Interaction(
                interaction_id=2,
                user_id=user_id_b,
                request_id="request_b",
                content="Test content B",
                role="user",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
            request_b = Request(
                request_id="request_b",
                user_id=user_id_b,
                source="source_b",
                agent_version=agent_version,
                session_id="group_b",
            )
            _storage(service).add_request(request_b)
            _storage(service).add_user_interaction(user_id_b, interaction_b)

            from reflexio.models.api_schema.service_schemas import (
                RerunPlaybookGenerationRequest,
            )

            # Filter by source_a - should only include user_a
            request = RerunPlaybookGenerationRequest(
                agent_version=agent_version, source="source_a"
            )
            result = service._get_rerun_user_ids(request)

            # Should only have user_a (who has source_a requests)
            assert len(result) == 1
            assert user_id_a in result
            assert user_id_b not in result


def test_get_rerun_user_ids_returns_empty_when_no_matches():
    """Test that _get_rerun_user_ids returns empty list when no items match."""
    org_id = "0"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        service = PlaybookGenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        from reflexio.models.api_schema.service_schemas import (
            RerunPlaybookGenerationRequest,
        )

        request = RerunPlaybookGenerationRequest(agent_version="1.0")
        result = service._get_rerun_user_ids(request)

        assert result == []


def test_create_run_request_for_item_uses_per_user_operation_request_ids():
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PlaybookGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id="0", storage_base_dir=temp_dir),
        )
        request = ManualPlaybookGenerationRequest(agent_version="v1", source="api")

        first = service._create_run_request_for_item("user_1", request)
        second = service._create_run_request_for_item("user_2", request)

    assert first.request_id.startswith("manual_")
    assert second.request_id.startswith("manual_")
    assert first.request_id != second.request_id


def test_create_run_request_for_item_uses_distinct_rerun_operation_request_ids():
    with tempfile.TemporaryDirectory() as temp_dir:
        service = PlaybookGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id="0", storage_base_dir=temp_dir),
        )
        request = RerunPlaybookGenerationRequest(agent_version="v1", source="api")

        first = service._create_run_request_for_item("user_1", request)
        second = service._create_run_request_for_item("user_2", request)

    assert first.request_id.startswith("rerun_playbook_")
    assert second.request_id.startswith("rerun_playbook_")
    assert first.request_id != second.request_id
    assert first.auto_run is False
    assert second.auto_run is False


def test_collect_scoped_interactions_for_precheck_uses_extractor_scope():
    """Pre-check should use extractor-specific window and source filters."""
    org_id = "0"
    user_id = "test_user"

    with tempfile.TemporaryDirectory() as temp_dir:
        service = PlaybookGenerationService(
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        service.configurator.set_config_by_name("window_size", 200)
        service.service_config = service._load_generation_service_config(
            PlaybookGenerationRequest(
                request_id="request-1",
                agent_version="1.0",
                user_id=user_id,
                source="api",
                auto_run=True,
            )
        )

        interaction = Interaction(
            interaction_id=1,
            user_id=user_id,
            request_id="request-1",
            content="user corrected the agent behavior",
            role="user",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        session_id = create_request_interaction_data_model(
            user_id=user_id,
            request_id="request-1",
            interactions=[interaction],
        )

        _storage(service).get_last_k_interactions_grouped = MagicMock(
            return_value=([session_id], [])
        )

        extractor_config = PlaybookConfig(
            extractor_name="api_playbook",
            extraction_definition_prompt="extract api-related playbook",
            request_sources_enabled=["api"],
            window_size_override=120,
            aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
        )

        (
            scoped_groups,
            scoped_config,
        ) = service._collect_scoped_interactions_for_precheck(extractor_config)

        assert len(scoped_groups) == 1
        assert scoped_config.extractor_name == "api_playbook"
        _storage(service).get_last_k_interactions_grouped.assert_called_once()
        _, kwargs = _storage(service).get_last_k_interactions_grouped.call_args
        assert kwargs["k"] == 120
        assert kwargs["sources"] == ["api"]


if __name__ == "__main__":
    pytest.main([__file__])
