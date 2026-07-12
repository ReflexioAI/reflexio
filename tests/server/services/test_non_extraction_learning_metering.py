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
        ReflectionServiceRequest(
            user_id="user-1", request_id="req-1", agent_version="agent-v9"
        ),
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
    assert event.agent_version == "agent-v9"
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


def test_resumable_finalization_falls_back_when_items_lack_ids() -> None:
    """Items with no durable id (e.g. plain objects) fall back to the
    count-based aggregate event -- Task A3's documented fallback, since there
    is no safe per-record id to key a dedup event on.
    """
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="profile")

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
    assert event.event_key is not None and event.event_key.startswith("learn-batch:")


def test_resumable_finalization_emits_one_event_per_profile_id() -> None:
    """When every item carries a durable ``profile_id`` (the common case --
    profile ids are assigned by the extractor before finalize runs), emit one
    entity-backed event per profile instead of the count-only fallback.
    """
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="profile")

    class _FakeProfile:
        def __init__(self, profile_id: str) -> None:
            self.profile_id = profile_id

    worker._record_finalized_learnings(
        run,
        [_FakeProfile("prof-1"), _FakeProfile("prof-2")],
        entity_type="profile",
    )

    assert len(events) == 2
    assert sum(e.count_value for e in events) == 2  # total unchanged vs old count=2
    assert {e.event_key for e in events} == {"learn:prof-1", "learn:prof-2"}
    assert {e.entity_id for e in events} == {"prof-1", "prof-2"}
    for event in events:
        assert event.count_value == 1
        assert event.entity_type == "profile"
        assert event.source == "resumable_extraction"


def test_resumable_finalization_emits_one_event_per_playbook_id() -> None:
    """Same as above for the playbook (``user_playbook_id``) kind."""
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="playbook")

    class _FakePlaybook:
        def __init__(self, user_playbook_id: int) -> None:
            self.user_playbook_id = user_playbook_id

    worker._record_finalized_learnings(
        run,
        [_FakePlaybook(11), _FakePlaybook(12), _FakePlaybook(13)],
        entity_type="user_playbook",
    )

    assert len(events) == 3
    assert sum(e.count_value for e in events) == 3  # total unchanged vs old count=3
    assert {e.event_key for e in events} == {"learn:11", "learn:12", "learn:13"}
    assert {e.entity_id for e in events} == {"11", "12", "13"}


def test_resumable_finalization_falls_back_when_a_playbook_id_is_unset() -> None:
    """A ``user_playbook_id=0`` (default, unset) mixed in with real ids means
    dedup dropped that item before persist -- fall back to the count-based
    aggregate rather than emit a colliding ``learn:0`` key or fabricate an id.
    """
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    worker = ExtractionResumeWorker(
        request_context=_request_context(),
        llm_client=MagicMock(),
    )
    run = _agent_run(extractor_kind="playbook")

    class _FakePlaybook:
        def __init__(self, user_playbook_id: int) -> None:
            self.user_playbook_id = user_playbook_id

    worker._record_finalized_learnings(
        run,
        [_FakePlaybook(21), _FakePlaybook(0)],
        entity_type="user_playbook",
    )

    assert len(events) == 1
    assert events[0].count_value == 2  # total unchanged vs old count=2
    assert events[0].event_key is not None and events[0].event_key.startswith(
        "learn-batch:"
    )


def test_aggregation_records_attributed_learnings_generated() -> None:
    """Aggregation emits one entity-backed event per generated playbook.

    ``saved_playbook_list`` entries always carry a real ``agent_playbook_id``
    (``save_agent_playbook_with_aggregate_event`` raises rather than
    returning a partial row) -- aggregator.py is the one caller with a clean,
    always-populated per-record id list, so it uses the entity-backed path
    (Task A3) rather than the count-only fallback.
    """
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=_request_context(),
        agent_version="v1",
    )

    aggregator._record_learnings_generated(
        learning_ids=["101", "102", "103"],
        playbook_name="agent_rules",
        request_id="agg-run-1",
        metadata={"playbooks_generated": 3},
    )

    assert len(events) == 3
    assert sum(e.count_value for e in events) == 3  # total unchanged vs old count=3
    assert {e.event_key for e in events} == {"learn:101", "learn:102", "learn:103"}
    assert {e.entity_id for e in events} == {"101", "102", "103"}
    for event in events:
        assert event.event_name == "learnings_generated"
        assert event.count_value == 1
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
