from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionResult,
    ReflectionServiceRequest,
)
from reflexio.server.services.reflection.service import ReflectionService
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


def test_reflection_revision_records_learnings_generated() -> None:
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    service = ReflectionService(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    result = ReflectionResult(
        revised_count=2,
        cited_count=3,
        considered_count=2,
        trigger_revised_count=1,
        content_revised_count=2,
        ttl_changed_count=1,
    )

    service._record_learnings_generated(
        ReflectionServiceRequest(user_id="user-1", request_id="req-1"),
        result,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_name == "learnings_generated"
    assert event.event_category == "learning"
    assert event.count_value == 2
    assert event.pipeline == "reflection"
    assert event.source == "reflection"
    assert event.user_id == "user-1"
    assert event.request_id == "req-1"
    assert event.metadata["content_revised_count"] == 2


def test_reflection_zero_revisions_records_nothing() -> None:
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    service = ReflectionService(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )

    service._record_learnings_generated(
        ReflectionServiceRequest(user_id="user-1", request_id="req-1"),
        ReflectionResult(revised_count=0),
    )

    assert events == []


def test_resumable_finalization_records_learnings_generated() -> None:
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = AgentRunRecord(
        id="run-1",
        binding=AgentBinding(
            org_id="org-1",
            extractor_kind="profile",
            user_id="user-1",
            request_id="req-1",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZING,
        generation_request_snapshot={},
    )

    worker._record_finalized_learnings(
        run,
        [object(), object()],
        entity_type="profile",
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_name == "learnings_generated"
    assert event.count_value == 2
    assert event.pipeline == "profile"
    assert event.source == "resumable_extraction"
    assert event.entity_type == "profile"
    assert event.metadata == {"run_id": "run-1", "extractor_kind": "profile"}


def test_aggregation_records_attributed_learnings_generated() -> None:
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=_request_context(),
        agent_version="v1",
    )

    aggregator._record_learnings_generated(
        count=3,
        playbook_name="agent_rules",
        request_id="agg-run-1",
        metadata={"playbooks_generated": 3},
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_name == "learnings_generated"
    assert event.count_value == 3
    assert event.pipeline == "playbook"
    assert event.source == "aggregation"
    assert event.entity_type == "agent_playbook"
    assert event.agent_version == "v1"
    assert event.playbook_name == "agent_rules"


def test_metering_failure_is_isolated_from_the_product_path() -> None:
    """A metering-side exception must never propagate into the caller.

    All four non-extraction paths route through the shared, guarded
    ``emit_learnings_generated`` helper. Simulate config resolution blowing up
    (the one bit of pre-emit work that can raise) and assert the caller neither
    raises nor emits a phantom event.
    """
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    ctx = _request_context()
    configurator = MagicMock()
    configurator.get_config.side_effect = RuntimeError("config boom")
    ctx.configurator = configurator
    service = ReflectionService(request_context=ctx, llm_client=MagicMock())

    # Must not raise despite get_config() blowing up mid-emit.
    service._record_learnings_generated(
        ReflectionServiceRequest(user_id="user-1", request_id="req-1"),
        ReflectionResult(revised_count=2),
    )

    assert events == []
