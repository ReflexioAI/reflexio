from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserPlaybook
from reflexio.models.config_schema import PlaybookAggregatorConfig, PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
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
