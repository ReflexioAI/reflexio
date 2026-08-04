from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
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
    assert {e.event_key for e in events} == {
        "learn:user_playbook:11",
        "learn:user_playbook:12",
        "learn:user_playbook:13",
    }
    assert {e.entity_id for e in events} == {"11", "12", "13"}
    for event in events:
        # tie key<->entity together so a swapped association can't pass on sets alone
        assert event.event_key == f"learn:{event.entity_type}:{event.entity_id}"


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
    (``save_agent_playbooks`` raises rather than
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
    assert {e.event_key for e in events} == {
        "learn:agent_playbook:101",
        "learn:agent_playbook:102",
        "learn:agent_playbook:103",
    }
    assert {e.entity_id for e in events} == {"101", "102", "103"}
    for event in events:
        assert event.event_name == "learnings_generated"
        assert event.count_value == 1
        assert event.pipeline == "playbook"
        assert event.source == "aggregation"
        assert event.entity_type == "agent_playbook"
        assert event.agent_version == "v1"
        assert event.playbook_name == "agent_rules"
        # tie key<->entity together so a swapped association can't pass on sets alone
        assert event.event_key == f"learn:{event.entity_type}:{event.entity_id}"


def test_aggregation_falls_back_when_an_agent_playbook_id_is_falsy() -> None:
    """Whole-branch-review finding (2): a falsy/0 ``agent_playbook_id`` mixed
    into the run must not mint a colliding ``learn:agent_playbook:0`` key --
    fall back to the count-based aggregate event instead, matching
    ``ExtractionResumeWorker``'s guard for the same failure mode.
    """
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    aggregator = PlaybookAggregator(
        llm_client=MagicMock(),
        request_context=_request_context(),
        agent_version="v1",
    )

    aggregator._record_learnings_generated(
        learning_ids=["201"],  # one id missing relative to total_count=2
        playbook_name="agent_rules",
        request_id="agg-run-2",
        metadata={"playbooks_generated": 2},
        total_count=2,
    )

    assert len(events) == 1
    assert events[0].count_value == 2  # total unchanged vs old count=2
    assert events[0].event_key is not None and events[0].event_key.startswith(
        "learn-batch:"
    )
    assert events[0].entity_type == "agent_playbook"
    assert events[0].source == "aggregation"
