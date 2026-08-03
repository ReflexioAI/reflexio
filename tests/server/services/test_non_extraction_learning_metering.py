from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserPlaybook
from reflexio.models.config_schema import PlaybookAggregatorConfig, PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.deferred_learning_plan import FinalizationResult
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
)
from reflexio.server.usage_metrics import UsageEvent, configure_usage_event_recorder


@pytest.fixture(autouse=True)
def _reset_usage_recorder():
    configure_usage_event_recorder(None)
    yield
    configure_usage_event_recorder(None)


def _request_context() -> RequestContext:
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = "org-1"
    ctx.storage = MagicMock()
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = None
    return ctx


def _agent_run(*, extractor_kind: str) -> AgentRunRecord:
    return AgentRunRecord(
        id="run-1",
        binding=AgentBinding(
            org_id="org-1",
            extractor_kind=extractor_kind,
            user_id="user-1",
            request_id="req-1",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZING,
        generation_request_snapshot={},
    )


def test_resumable_profile_bills_only_ids_returned_by_finalization() -> None:
    """A preassigned candidate ID is not billed when finalization drops it."""
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="profile")

    dropped_candidate = MagicMock(profile_id="dropped-before-persist")
    with patch(
        "reflexio.server.services.extraction.resume_worker.ProfileGenerationService"
    ) as service_class:
        finalizer = service_class.return_value._finalize_extracted_items_with_outcome
        finalizer.return_value = FinalizationResult([], won_receipt=True)
        worker._finalize_items(run, [dropped_candidate])

    finalizer.assert_called_once_with(
        [dropped_candidate], model_provenance=None, finalization_run_id=run.id
    )
    assert events == []


def test_resumable_fallback_reuses_its_event_key_on_finalization_retry() -> None:
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="profile")

    worker._record_finalized_learnings(run, [object()], entity_type="profile")
    worker._record_finalized_learnings(run, [object()], entity_type="profile")

    assert [event.event_key for event in events] == [
        "learn-batch:resumable:run-1:profile",
        "learn-batch:resumable:run-1:profile",
    ]


def test_resumable_finalization_emits_one_event_per_profile_id() -> None:
    """Finalization survivor IDs emit one entity-backed event per profile."""
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="profile")

    worker._record_finalized_learnings(
        run,
        ["prof-1", "prof-2"],
        entity_type="profile",
    )

    assert len(events) == 2
    assert sum(e.count_value for e in events) == 2  # total unchanged vs old count=2
    assert {e.event_key for e in events} == {
        "learn:profile:prof-1",
        "learn:profile:prof-2",
    }
    assert {e.entity_id for e in events} == {"prof-1", "prof-2"}
    for event in events:
        assert event.count_value == 1
        assert event.entity_type == "profile"
        assert event.source == "resumable_extraction"
        # tie key<->entity together so a swapped association can't pass on sets alone
        assert event.event_key == f"learn:{event.entity_type}:{event.entity_id}"


def test_resumable_finalization_emits_one_event_per_playbook_id() -> None:
    """Finalization survivor IDs emit one entity-backed event per playbook."""
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="playbook")

    worker._record_finalized_learnings(
        run,
        ["11", "12", "13"],
        entity_type="user_playbook",
    )

    assert len(events) == 3
    assert sum(e.count_value for e in events) == 3  # total unchanged vs old count=3
    assert {e.event_key for e in events} == {
        "learn:user_playbook:11",
        "learn:user_playbook:12",
        "learn:user_playbook:13",
    }
    assert {e.entity_id for e in events} == {"11", "12", "13"}
    for event in events:
        # tie key<->entity together so a swapped association can't pass on sets alone
        assert event.event_key == f"learn:{event.entity_type}:{event.entity_id}"


def test_resumable_playbook_bills_consolidation_replacement_id() -> None:
    """Billing follows the persisted replacement, not its input candidate."""
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="playbook")

    original_candidate = MagicMock(user_playbook_id=21)
    with patch(
        "reflexio.server.services.extraction.resume_worker.PlaybookGenerationService"
    ) as service_class:
        finalizer = service_class.return_value._finalize_extracted_items_with_outcome
        finalizer.return_value = FinalizationResult(["88"], won_receipt=True)
        worker._finalize_items(run, [original_candidate])

    finalizer.assert_called_once_with(
        [original_candidate], model_provenance=None, finalization_run_id=run.id
    )
    assert len(events) == 1
    assert events[0].count_value == 1
    assert events[0].event_key == "learn:user_playbook:88"
    assert events[0].entity_id == "88"


def test_aggregation_emits_no_learnings_generated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed aggregation remains observable but adds no billable learning."""
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    request_context = _request_context()
    storage = MagicMock()
    configurator = MagicMock()
    request_context.storage = storage
    request_context.configurator = configurator
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=request_context,
        agent_version="v1",
    )
    config = MagicMock()
    configurator.get_config.return_value = config
    config.user_playbook_extractor_config = PlaybookConfig(
        extractor_name="billing-boundary",
        extraction_definition_prompt="Extract user playbooks.",
        aggregation_config=PlaybookAggregatorConfig(
            min_cluster_size=2,
            reaggregation_trigger_count=2,
        ),
    )
    user_playbooks = [
        UserPlaybook(
            user_playbook_id=1,
            agent_version="v1",
            request_id="request-1",
            playbook_name="user_playbook",
            content="Document deployment decisions.",
        ),
        UserPlaybook(
            user_playbook_id=2,
            agent_version="v1",
            request_id="request-2",
            playbook_name="user_playbook",
            content="Verify deployment outcomes.",
        ),
    ]
    generated = AgentPlaybook(
        agent_playbook_id=101,
        playbook_name="user_playbook",
        agent_version="v1",
        content="Deploy changes and verify results.",
    )
    storage.count_user_playbooks.return_value = len(user_playbooks)
    storage.get_agent_playbooks.return_value = []
    storage.get_user_playbooks.return_value = user_playbooks
    storage.save_agent_playbooks.return_value = [generated]
    monkeypatch.setattr(
        aggregator,
        "get_clusters",
        lambda *_args: {0: user_playbooks},
    )
    monkeypatch.setattr(
        aggregator,
        "_generate_playbooks_with_source_clusters",
        lambda *_args, **_kwargs: [(generated, user_playbooks, None)],
    )
    monkeypatch.setattr(
        aggregator, "_enqueue_playbook_optimization", lambda _items: None
    )

    stats = aggregator.run(PlaybookAggregatorRequest(agent_version="v1", rerun=True))

    assert stats["playbooks_generated"] == 1
    assert any(event.event_name == "aggregation_succeeded" for event in events)
    assert not any(event.event_name == "learnings_generated" for event in events)
