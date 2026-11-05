from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.compute_admission import reserve_compute_attempt
from reflexio.server.services.extraction.history_inputs import HistoricalInputs
from reflexio.server.services.extraction.publication import _deliver
from reflexio.server.services.extraction.resume_worker import (
    ExtractionResumeWorker,
    _run_playbook_contract_selection,
    _run_uses_strict_playbook_evidence,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
)
from reflexio.server.services.profile.service import (
    ProfileGenerationService,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)
from reflexio.server.usage_metrics import (
    UsageEvent,
    UsageEventDeliveryStatus,
    configure_usage_event_recorder,
    exempt_usage_event_recorder,
)
from tests.server.services.extraction._publication_fixtures import (
    observed_financial_receipt,
)


@pytest.fixture(autouse=True)
def no_background_tagging(monkeypatch):
    # Tag scheduling has its own tests. Do not let delayed callbacks outlive
    # the isolated resume fixtures or write to a default local database.
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resume_worker.schedule_tagging",
        lambda **_kwargs: None,
    )


@pytest.fixture
def storage():
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id="org_1", db_path=f"{temp_dir}/reflexio.db")


@pytest.fixture(autouse=True)
def explicit_test_billing_exemption():
    configure_usage_event_recorder(exempt_usage_event_recorder)
    try:
        yield
    finally:
        configure_usage_event_recorder(None)


@pytest.fixture
def request_context(storage):
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = "org_1"
    ctx.storage = storage
    ctx.storage_base_dir = None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="Extract durable user deployment facts.",
        ),
        pending_tool_call_config=PendingToolCallConfig(
            pending_ttl_seconds=86_400,
            max_finalization_attempts=3,
        ),
    )
    ctx.configurator.get_agent_context.return_value = "Test agent context"
    ctx.prompt_manager = MagicMock()
    ctx.prompt_manager.render_prompt.side_effect = lambda prompt_id, variables: (
        f"{prompt_id}: {variables}"
    )
    return ctx


def _finalization_context(storage: SQLiteStorage) -> RequestContext:
    context = RequestContext.__new__(RequestContext)
    context.org_id = "org_1"
    context.storage = storage
    context.storage_base_dir = None
    context.configurator = MagicMock()
    context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="Extract durable user facts.",
        ),
        user_playbook_extractor_config=PlaybookConfig(
            extraction_definition_prompt="Extract durable operating rules.",
        ),
        pending_tool_call_config=PendingToolCallConfig(pending_ttl_seconds=86_400),
    )
    context.prompt_manager = MagicMock()
    context.prompt_manager.get_active_version.return_value = None
    return context


def _finalizing_run(
    *, run_id: str, extractor_kind: str, request_id: str
) -> AgentRunRecord:
    return AgentRunRecord(
        id=run_id,
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind=extractor_kind,
            user_id="user_1",
            request_id=request_id,
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZING,
        generation_request_snapshot={"request_id": request_id},
    )


def _persist_and_load_run(
    storage: SQLiteStorage, run: AgentRunRecord
) -> AgentRunRecord:
    storage.create_agent_run(run)
    persisted = storage.get_agent_run(run.id)
    assert persisted is not None
    assert persisted.created_at is not None
    return persisted


@pytest.mark.parametrize(
    "extractor_kind,entity_type",
    [("profile", "profile"), ("playbook", "user_playbook")],
)
@pytest.mark.parametrize(
    "status",
    [
        UsageEventDeliveryStatus.APPENDED,
        UsageEventDeliveryStatus.DUPLICATE,
        UsageEventDeliveryStatus.EXEMPT,
    ],
)
def test_receipt_learning_delivery_does_not_reread_model_configuration(
    storage, request_context, extractor_kind, entity_type, status
):
    run = _persist_and_load_run(
        storage,
        _finalizing_run(
            run_id="receipt-facts",
            extractor_kind=extractor_kind,
            request_id="request_1",
        ),
    )
    storage._ack_publication_receipt = MagicMock(return_value=True)
    request_context.configurator.get_config.side_effect = AssertionError(
        "Mutable model configuration is not a learning receipt input"
    )
    events = []

    def accept(event):
        events.append(event)
        return status

    configure_usage_event_recorder(accept)
    _deliver(
        storage,
        observed_financial_receipt(run, ["durable-learning"], entity_type=entity_type),
    )

    assert len(events) == 1
    assert events[0].event_key == f"learn:{entity_type}:durable-learning"
    assert run.created_at is not None
    assert events[0].created_at == run.created_at.timestamp()
    assert events[0].pipeline == extractor_kind
    assert events[0].platform_llm is None
    assert events[0].platform_storage is None


def _playbook_candidates(*, request_id: str, prefix: str) -> list[UserPlaybook]:
    return [
        UserPlaybook(
            user_id="user_1",
            agent_version="v1",
            request_id=request_id,
            content=f"Use deployment procedure {prefix}-{index}.",
            trigger=f"when deployment condition {prefix}-{index} occurs",
            rationale=f"Procedure {prefix}-{index} is required.",
            source="api",
            source_interaction_ids=[1, 2],
        )
        for index in range(2)
    ]


class _DeduplicatingUsageRecorder:
    def __init__(self) -> None:
        self.attempts: list[UsageEvent] = []
        self.events: list[UsageEvent] = []
        self._accepted_keys: set[str] = set()
        self._lock = threading.Lock()

    def __call__(self, event: UsageEvent) -> UsageEventDeliveryStatus:
        with self._lock:
            self.attempts.append(event)
            assert event.event_key is not None
            if event.event_key in self._accepted_keys:
                return UsageEventDeliveryStatus.DUPLICATE
            self._accepted_keys.add(event.event_key)
            self.events.append(event)
            return UsageEventDeliveryStatus.APPENDED


@pytest.mark.parametrize(
    ("schema_name", "expected"),
    [
        ("StructuredReferencedExtractedPlaybookList", True),
        ("StructuredExtractedPlaybookList", True),
        ("StructuredPlaybookList", False),
    ],
)
def test_playbook_resume_uses_schema_recorded_at_run_creation(
    request_context, schema_name, expected
):
    run = AgentRunRecord(
        id="playbook_run",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="playbook",
            user_id=None,
            request_id="request_1",
            agent_version="v1",
            source="api",
            source_interaction_ids=[1],
        ),
        status=AgentRunStatus.RUNNING,
        generation_request_snapshot={"output_schema_name": schema_name},
    )

    assert (
        _run_uses_strict_playbook_evidence(
            request_context,
            run,
            expert=False,
        )
        is expected
    )
    request_context.prompt_manager.get_active_version.assert_not_called()


@pytest.mark.parametrize(
    ("schema_name", "expected"),
    [
        ("StructuredReferencedExtractedPlaybookList", (True, True)),
        ("StructuredExtractedPlaybookList", (True, False)),
        ("StructuredPlaybookList", (False, False)),
    ],
)
def test_playbook_resume_pins_evidence_reference_mode(
    request_context, schema_name, expected
):
    run = AgentRunRecord(
        id="playbook_run",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="playbook",
            user_id=None,
            request_id="request_1",
            agent_version="v1",
            source="api",
            source_interaction_ids=[1],
        ),
        status=AgentRunStatus.RUNNING,
        generation_request_snapshot={"output_schema_name": schema_name},
    )

    assert (
        _run_playbook_contract_selection(request_context, run, expert=False) == expected
    )
    request_context.prompt_manager.get_active_version.assert_not_called()


def _seed_interactions(storage: SQLiteStorage) -> None:
    storage.add_request(
        Request(
            request_id="request_1",
            user_id="user_1",
            created_at=1_000,
            source="api",
            agent_version="v1",
            session_id="request_1",
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=1,
            user_id="user_1",
            request_id="request_1",
            created_at=1_000,
            role="user",
            content="Which deployment target should we use?",
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=2,
            user_id="user_1",
            request_id="request_1",
            created_at=1_001,
            role="assistant",
            content="I need the deployment standard.",
        )
    )


def _legacy_ownerless_playbook_run(source_interaction_ids: list[int]) -> AgentRunRecord:
    return AgentRunRecord(
        id="legacy_playbook_run",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="playbook",
            user_id=None,
            request_id="request_1",
            agent_version="v1",
            source="api",
            source_interaction_ids=source_interaction_ids,
        ),
        status=AgentRunStatus.FINALIZATION_FAILED,
        generation_request_snapshot={},
    )


def test_legacy_playbook_run_infers_owner_from_complete_source_evidence(
    request_context, storage
):
    _seed_interactions(storage)
    worker = ExtractionResumeWorker(request_context=request_context)

    resolved = worker._with_resolved_playbook_user_id(
        _legacy_ownerless_playbook_run([1, 2])
    )

    assert resolved.binding.user_id == "user_1"


def test_legacy_playbook_run_fails_closed_when_source_evidence_is_missing(
    request_context, storage
):
    _seed_interactions(storage)
    worker = ExtractionResumeWorker(request_context=request_context)

    with pytest.raises(ValueError, match=r"missing persisted.*interactions"):
        worker._with_resolved_playbook_user_id(_legacy_ownerless_playbook_run([1, 999]))


def test_legacy_playbook_run_fails_closed_for_multiple_source_owners(
    request_context, storage
):
    _seed_interactions(storage)
    storage.add_request(
        Request(
            request_id="request_2",
            user_id="user_2",
            created_at=1_002,
            source="api",
            agent_version="v1",
            session_id="request_2",
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=3,
            user_id="user_2",
            request_id="request_2",
            created_at=1_002,
            role="user",
            content="Use a different tenant's evidence.",
        )
    )
    worker = ExtractionResumeWorker(request_context=request_context)

    with pytest.raises(ValueError, match="unambiguous interaction owner"):
        worker._with_resolved_playbook_user_id(_legacy_ownerless_playbook_run([1, 3]))


def _seed_ready_run(storage: SQLiteStorage) -> None:
    storage.create_agent_run(
        AgentRunRecord(
            id="run_1",
            binding=AgentBinding(
                org_id="org_1",
                extractor_kind="profile",
                user_id="user_1",
                request_id="request_1",
                agent_version="v1",
                source="api",
                source_interaction_ids=[1, 2],
                window_start_interaction_id=1,
                window_end_interaction_id=2,
                extractor_config_hash="old_hash",
            ),
            status=AgentRunStatus.FINALIZED_PENDING_TOOL,
            generation_request_snapshot={"request_id": "request_1"},
        )
    )
    now = datetime.now(UTC)
    question = "What is the deployment target?"
    scope = human_feedback_scope("org_1")
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_1",
            org_id="org_1",
            user_id="user_1",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text=question,
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=question,
            args={"question": question},
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(minutes=5),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )
    storage.resolve_pending_tool_call(
        "ptc_1",
        result={"answer": "Use AWS ECS."},
        resolved_at=now,
        valid_for_seconds=3600,
    )


@pytest.mark.parametrize("pending_enabled", [False, True])
def test_resume_worker_resumes_profile_run_and_consumes_dependency(
    monkeypatch,
    request_context,
    storage,
    tool_call_completion,
    pending_enabled,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    request_context.configurator.get_config().pending_tool_call_config.pending_ttl_seconds = (
        86_400 if pending_enabled else 0
    )
    _seed_interactions(storage)
    _seed_ready_run(storage)
    _make_tc, make_stop = tool_call_completion
    response = make_stop(
        json.dumps(
            {
                "profiles": [
                    {
                        "content": "User deployment target is AWS ECS.",
                        "time_to_live": "infinity",
                    }
                ]
            }
        )
    )
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    worker = DurableLearningWorker(lambda _org: request_context)

    with (
        patch("litellm.completion", side_effect=[response]) as completion,
        patch(
            "reflexio.server.services.profile.components.consolidator.ProfileConsolidator",
        ) as mock_consolidator_cls,
    ):
        mock_consolidator_cls.return_value.deduplicate.side_effect = (
            lambda profiles, _user_id, _request_id: (profiles, [], [])
        )
        resumed = worker.drain_org("org_1", batch_size=1, lease_seconds=300)

    requested_tools = completion.call_args.kwargs.get("tools") or []
    names = {tool["function"]["name"] for tool in requested_tools}
    assert ("attach_pending_info_request" in names) is pending_enabled
    assert "ask_human" not in names
    assert resumed == 1
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZED
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is not None
    profiles = storage.get_user_profile("user_1")
    assert [profile.content for profile in profiles] == [
        "User deployment target is AWS ECS."
    ]


def test_resume_worker_retries_finalization_without_rerunning_agent(
    monkeypatch,
    request_context,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _seed_interactions(storage)
    _seed_ready_run(storage)
    _make_tc, make_stop = tool_call_completion
    response = make_stop(
        json.dumps(
            {
                "profiles": [
                    {
                        "content": "User deployment target is AWS ECS.",
                        "time_to_live": "infinity",
                    }
                ]
            }
        )
    )
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    worker = DurableLearningWorker(lambda _org: request_context)

    with (
        patch("litellm.completion", side_effect=[response]),
        patch(
            "reflexio.server.services.profile.service."
            "ProfileGenerationService._persist_write_plan",
            side_effect=RuntimeError("storage write failed"),
        ),
    ):
        resumed = worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)

    assert resumed == 1
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZATION_FAILED
    assert run.committed_output is not None
    assert run.finalization_attempts == 1
    assert run.next_resume_at is not None
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is None

    storage.update_agent_run_status(
        "run_1",
        AgentRunStatus.FINALIZATION_FAILED,
        next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
    )
    with (
        patch(
            "litellm.completion",
            side_effect=AssertionError("finalization retry must not call LLM"),
        ),
        patch(
            "reflexio.server.services.profile.service."
            "ProfileGenerationService._persist_write_plan",
            autospec=True,
            side_effect=ProfileGenerationService._persist_write_plan,
        ) as finalize,
    ):
        # The persisted class rotation may visit the empty learning class
        # before returning to finalization; one bounded rotation must finish.
        resumed = worker.drain_org(storage.org_id, batch_size=3, lease_seconds=300)

    assert resumed == 1
    finalize.assert_called_once()
    retried = storage.get_agent_run("run_1")
    assert retried is not None
    assert retried.status == AgentRunStatus.FINALIZED
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is not None


def test_resume_worker_tagging_schedule_failure_is_best_effort(
    request_context,
):
    run = AgentRunRecord(
        id="run_tagging",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="profile",
            user_id="user_1",
            request_id="request_1",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZED_PENDING_TOOL,
        generation_request_snapshot={"request_id": "request_1"},
    )
    worker = ExtractionResumeWorker(request_context=request_context)

    with patch(
        "reflexio.server.services.extraction.resume_worker.schedule_tagging",
        side_effect=RuntimeError("scheduler unavailable"),
    ):
        worker._schedule_finalized_tagging(run)


def test_resumable_finalization_bills_only_durable_ids_idempotently_on_retry(
    request_context,
):
    """A mixed batch charges its persisted profile once across finalization retries."""
    run_created_at = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    agent_completed_at = datetime(2026, 8, 31, 23, 59, tzinfo=UTC)
    run = AgentRunRecord(
        id="run_mixed_billing",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="profile",
            user_id="user_1",
            request_id="request_1",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZATION_FAILED,
        generation_request_snapshot={"request_id": "request_1"},
        agent_completed_at=agent_completed_at,
        created_at=run_created_at,
    )
    learning_ids = ["profile_1"]
    storage = request_context.storage
    storage._ack_publication_receipt = MagicMock(return_value=True)
    receipt = observed_financial_receipt(run, learning_ids, entity_type="profile")

    with (
        patch.object(
            storage,
            "get_agent_run",
            side_effect=AssertionError("billing must not re-read the run"),
        ),
        patch(
            "reflexio.server.metering_events.record_usage_event_strict",
            return_value=UsageEventDeliveryStatus.APPENDED,
        ) as record_event,
    ):
        _deliver(storage, receipt)
        _deliver(storage, receipt)

    assert [call.kwargs["event_key"] for call in record_event.call_args_list] == [
        "learn:profile:profile_1",
        "learn:profile:profile_1",
    ]
    assert [call.kwargs["count_value"] for call in record_event.call_args_list] == [
        1,
        1,
    ]
    assert [call.kwargs["entity_id"] for call in record_event.call_args_list] == [
        "profile_1",
        "profile_1",
    ]
    assert [call.kwargs["created_at"] for call in record_event.call_args_list] == [
        run_created_at.timestamp(),
        run_created_at.timestamp(),
    ]


@pytest.mark.parametrize(
    ("extractor_kind", "entity_type"),
    [("profile", "profile"), ("playbook", "user_playbook")],
)
def test_delivery_failure_after_receipt_commit_retries_billing_without_recompute(
    request_context,
    storage,
    extractor_kind,
    entity_type,
):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="Extract durable user facts.",
        ),
        user_playbook_extractor_config=PlaybookConfig(
            extraction_definition_prompt="Extract durable operating rules.",
        ),
        pending_tool_call_config=PendingToolCallConfig(pending_ttl_seconds=86_400),
    )
    request_id = f"request_{extractor_kind}_delivery_retry"
    run_id = f"run_{extractor_kind}_delivery_retry"
    committed_output = (
        {
            "profiles": [
                {
                    "content": "User deployment target is AWS ECS.",
                    "time_to_live": "infinity",
                }
            ]
        }
        if extractor_kind == "profile"
        else {
            "playbooks": [
                {
                    "content": "Prefer AWS ECS as the deployment target.",
                    "trigger": "when selecting a deployment target",
                    "rationale": "The team standardizes on AWS.",
                }
            ]
        }
    )
    run_created_at = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    _seed_completed_output(
        storage,
        AgentRunRecord(
            id=run_id,
            binding=AgentBinding(
                org_id="org_1",
                extractor_kind=extractor_kind,
                user_id="user_1",
                request_id=request_id,
                agent_version="v1",
                source="api",
                source_interaction_ids=[1, 2],
            ),
            status=AgentRunStatus.RUNNING,
            generation_request_snapshot={
                "request_id": request_id,
                "output_schema_name": "StructuredProfilesOutput"
                if extractor_kind == "profile"
                else "StructuredPlaybookList",
            },
            committed_output=None,
            next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            finalization_attempts=2,
            created_at=run_created_at,
        ),
        committed_output,
    )
    # SQLite creation owns its clock rather than accepting created_at from the
    # input dataclass. Establish this historical fixture explicitly before any
    # publication and assert the requested cycle actually persisted.
    storage.conn.execute(
        "UPDATE _agent_runs SET created_at=? WHERE id=?",
        (run_created_at.isoformat(), run_id),
    )
    storage.conn.commit()
    stored_run = storage.get_agent_run(run_id)
    assert stored_run is not None
    assert stored_run.created_at is not None
    durable_run_created_at = stored_run.created_at
    assert durable_run_created_at == run_created_at
    worker = DurableLearningWorker(lambda _org: request_context)
    attempts: list[UsageEvent] = []
    accepted: dict[str, UsageEvent] = {}
    fail_next = True

    def recorder(event: UsageEvent) -> UsageEventDeliveryStatus:
        nonlocal fail_next
        attempts.append(event)
        if fail_next:
            fail_next = False
            raise RuntimeError("billing sink unavailable")
        assert event.event_key is not None
        if event.event_key in accepted:
            return UsageEventDeliveryStatus.DUPLICATE
        accepted[event.event_key] = event
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(recorder)
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "litellm.completion",
                    side_effect=AssertionError("saved output cannot call model"),
                )
            )
            schedule_tagging = stack.enter_context(
                patch.object(ExtractionResumeWorker, "_schedule_finalized_tagging")
            )
            if extractor_kind == "profile":
                stack.enter_context(
                    patch(
                        "reflexio.server.services.profile.components.consolidator."
                        "ProfileConsolidator.deduplicate",
                        side_effect=lambda profiles, _user_id, _request_id: (
                            profiles,
                            [],
                            [],
                        ),
                    )
                )
                optimize = aggregate = None
            else:
                stack.enter_context(
                    patch.object(
                        PlaybookGenerationService,
                        "_configured_playbook_config",
                        return_value=None,
                    )
                )
                stack.enter_context(
                    patch(
                        "reflexio.server.services.playbook.components.consolidator."
                        "PlaybookConsolidator.deduplicate",
                        side_effect=lambda results, *_args, **_kwargs: (
                            [playbook for result in results for playbook in result],
                            [],
                            [],
                        ),
                    )
                )
                optimize = stack.enter_context(
                    patch.object(
                        PlaybookGenerationService,
                        "_enqueue_user_playbook_optimization",
                    )
                )
                aggregate = stack.enter_context(
                    patch.object(
                        PlaybookGenerationService,
                        "_trigger_playbook_aggregation",
                    )
                )

            assert (
                worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
            )
            failed = storage.get_agent_run(run_id)
            assert failed is not None
            assert failed.status == AgentRunStatus.FINALIZED
            assert failed.finalization_attempts == 2
            receipt_after_failure = storage.get_agent_run_finalization_receipt(
                run_id=run_id,
                entity_type=entity_type,
            )
            assert receipt_after_failure
            rows_after_failure = (
                [profile.profile_id for profile in storage.get_user_profile("user_1")]
                if extractor_kind == "profile"
                else [
                    str(row[0])
                    for row in storage.conn.execute(
                        "SELECT user_playbook_id FROM user_playbooks "
                        "WHERE request_id = ? ORDER BY user_playbook_id",
                        (request_id,),
                    ).fetchall()
                ]
            )

            assert failed.committed_output is not None
            assert failed.agent_completed_at is not None
            assert (
                failed.agent_completed_at.year,
                failed.agent_completed_at.month,
            ) != (durable_run_created_at.year, durable_run_created_at.month)
            with storage._stream_sql() as db:
                pending_receipt = storage._get_output_publication_receipt(
                    db, run_id, failed.committed_output["publication"]["id"]
                )
            assert (
                pending_receipt is not None
                and pending_receipt.delivery_state == "pending"
            )
            from reflexio.server.services.extraction.publication import _deliver

            with patch.object(
                ExtractionResumeWorker,
                "_items_from_committed_output",
                side_effect=AssertionError("delivery must not recompute"),
            ):
                _deliver(storage, pending_receipt)
            retry = storage.get_agent_run(run_id)
            assert retry == failed
            assert (
                worker.drain_org(storage.org_id, batch_size=3, lease_seconds=300) == 0
            )
            with storage._stream_sql() as db:
                delivered = storage._get_output_publication_receipt(
                    db, run_id, failed.committed_output["publication"]["id"]
                )
            assert (
                delivered is not None
                and delivered.delivery_state == "accepted"
                and delivered.financial_facts is None
            )

        assert retry is not None
        assert retry.status == AgentRunStatus.FINALIZED

        assert (
            storage.get_agent_run_finalization_receipt(
                run_id=run_id,
                entity_type=entity_type,
            )
            == receipt_after_failure
        )
        assert rows_after_failure == receipt_after_failure
        assert [event.event_key for event in attempts] == [
            f"learn:{entity_type}:{receipt_after_failure[0]}",
            f"learn:{entity_type}:{receipt_after_failure[0]}",
        ]
        assert list(accepted) == [f"learn:{entity_type}:{receipt_after_failure[0]}"]
        assert [event.created_at for event in attempts] == [
            durable_run_created_at.timestamp(),
            durable_run_created_at.timestamp(),
        ]
        schedule_tagging.assert_called_once()
        if optimize is not None and aggregate is not None:
            optimize.assert_called_once()
            aggregate.assert_called_once()
    finally:
        configure_usage_event_recorder(None)


def test_retry_after_billing_reuses_ids_without_replaying_playbook_schedulers(
    request_context,
    storage,
    monkeypatch,
):
    """A post-billing retry reuses durable IDs and skips derived schedulers."""
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="Extract durable user facts.",
        ),
        user_playbook_extractor_config=PlaybookConfig(
            extraction_definition_prompt="Extract durable operating rules.",
        ),
        pending_tool_call_config=PendingToolCallConfig(pending_ttl_seconds=86_400),
    )
    events: list[UsageEvent] = []
    delivery_attempts: list[UsageEvent] = []
    accepted_keys: set[str] = set()

    def deduplicating_recorder(event: UsageEvent) -> UsageEventDeliveryStatus:
        if event.event_name != "learnings_generated":
            return UsageEventDeliveryStatus.APPENDED
        delivery_attempts.append(event)
        assert event.event_key is not None
        if event.event_key in accepted_keys:
            return UsageEventDeliveryStatus.DUPLICATE
        accepted_keys.add(event.event_key)
        events.append(event)
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(deduplicating_recorder)

    def _run_failed_then_retried(
        run_id: str,
        *,
        assert_scheduler_calls: Callable[[], None] | None = None,
    ) -> None:
        worker = DurableLearningWorker(lambda _org: request_context)
        from reflexio.server.services.extraction.publication import _deliver

        with (
            patch.object(
                ExtractionResumeWorker, "_schedule_finalized_tagging"
            ) as tagging,
            patch(
                "litellm.completion",
                side_effect=AssertionError("saved output cannot call model"),
            ),
        ):
            with patch.object(
                storage, "_ack_publication_receipt", return_value=False
            ) as ack:
                # A prior output may have consumed finalization. Visit at
                # most one class rotation and stop at this output's commit,
                # before a subsequent turn can retry financial delivery.
                processed = 0
                first_attempt = None
                for _ in range(3):
                    processed += worker.drain_org(
                        storage.org_id, batch_size=1, lease_seconds=300
                    )
                    first_attempt = storage.get_agent_run(run_id)
                    assert first_attempt is not None
                    if first_attempt.status == AgentRunStatus.FINALIZED:
                        break
                assert processed == 1
            assert (
                first_attempt is not None
                and first_attempt.status == AgentRunStatus.FINALIZED
            )
            assert first_attempt.finalization_attempts == 0
            ack.assert_called_once()
            if assert_scheduler_calls is not None:
                assert_scheduler_calls()
            assert first_attempt.committed_output is not None
            with storage._stream_sql() as db:
                receipt = storage._get_output_publication_receipt(
                    db, run_id, first_attempt.committed_output["publication"]["id"]
                )
            assert receipt is not None and receipt.delivery_state == "pending"
            with patch.object(
                ExtractionResumeWorker,
                "_items_from_committed_output",
                side_effect=AssertionError("delivery must not recompute"),
            ):
                _deliver(storage, receipt)
            if assert_scheduler_calls is not None:
                assert_scheduler_calls()
            tagging.assert_called_once()
            assert storage.get_agent_run(run_id) == first_attempt
            assert (
                worker.drain_org(storage.org_id, batch_size=3, lease_seconds=300) == 0
            )
            with storage._stream_sql() as db:
                accepted = storage._get_output_publication_receipt(
                    db, run_id, first_attempt.committed_output["publication"]["id"]
                )
            assert accepted is not None and accepted.delivery_state == "accepted"

    try:
        profile_request_id = "request_profile_retry"
        _seed_completed_output(
            storage,
            AgentRunRecord(
                id="run_profile_retry",
                binding=AgentBinding(
                    org_id="org_1",
                    extractor_kind="profile",
                    user_id="user_1",
                    request_id=profile_request_id,
                    agent_version="v1",
                    source="api",
                    source_interaction_ids=[1, 2],
                ),
                status=AgentRunStatus.RUNNING,
                generation_request_snapshot={"request_id": profile_request_id},
                committed_output=None,
                next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            ),
            {
                "profiles": [
                    {
                        "content": "User deployment target is AWS ECS.",
                        "time_to_live": "infinity",
                    }
                ]
            },
        )
        with patch(
            "reflexio.server.services.profile.components.consolidator."
            "ProfileConsolidator.deduplicate",
            side_effect=lambda profiles, _user_id, _request_id: (
                profiles,
                [],
                [],
            ),
        ):
            _run_failed_then_retried("run_profile_retry")

        seed = UserPlaybook(
            user_id="user_1",
            agent_version="v1",
            request_id="seed_request",
            content="Prefer the current deployment default.",
            trigger="when selecting a deployment target",
            rationale="Existing operating rule.",
            source="api",
        )
        storage.save_user_playbooks([seed])
        playbook_request_id = "request_playbook_retry"
        _seed_completed_output(
            storage,
            AgentRunRecord(
                id="run_playbook_retry",
                binding=AgentBinding(
                    org_id="org_1",
                    extractor_kind="playbook",
                    user_id="user_1",
                    request_id=playbook_request_id,
                    agent_version="v1",
                    source="api",
                    source_interaction_ids=[1, 2],
                ),
                status=AgentRunStatus.RUNNING,
                generation_request_snapshot={
                    "request_id": playbook_request_id,
                    "output_schema_name": "StructuredPlaybookList",
                },
                committed_output=None,
                next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            ),
            {
                "playbooks": [
                    {
                        "content": "Prefer AWS ECS as the deployment target.",
                        "trigger": "when selecting a deployment target",
                        "rationale": "The team standardizes on AWS.",
                    }
                ]
            },
        )

        def consolidate_for_receipt_test(consolidator, results, *_args, **_kwargs):
            # This regression targets finalization receipt/scheduler replay,
            # independently of decision generation (as for the profile above).
            # Exercise the publisher with one replacement and real lineage.
            consolidator.consolidated_output_indices = {0}
            consolidator.lineage_sources_by_output_index = {
                0: [str(seed.user_playbook_id)]
            }
            return (
                [row for group in results for row in group],
                [seed.user_playbook_id],
                [(0, [seed.user_playbook_id])],
            )

        with (
            patch.object(
                PlaybookGenerationService,
                "_configured_playbook_config",
                return_value=None,
            ),
            patch(
                "reflexio.server.services.playbook.components.consolidator."
                "PlaybookConsolidator.deduplicate",
                autospec=True,
                side_effect=consolidate_for_receipt_test,
            ),
            patch.object(
                PlaybookGenerationService,
                "_enqueue_user_playbook_optimization",
            ) as enqueue_optimization,
            patch.object(
                PlaybookGenerationService,
                "_trigger_playbook_aggregation",
            ) as trigger_aggregation,
        ):

            def assert_playbook_scheduler_calls() -> None:
                enqueue_optimization.assert_called_once()
                trigger_aggregation.assert_called_once()

            _run_failed_then_retried(
                "run_playbook_retry",
                assert_scheduler_calls=assert_playbook_scheduler_calls,
            )
    finally:
        configure_usage_event_recorder(None)

    profile_keys = [
        event.event_key for event in events if event.entity_type == "profile"
    ]
    profile_event_ids = [
        event.entity_id for event in events if event.entity_type == "profile"
    ]
    playbook_keys = [
        event.event_key for event in events if event.entity_type == "user_playbook"
    ]
    playbook_event_ids = [
        event.entity_id for event in events if event.entity_type == "user_playbook"
    ]
    profile_survivor_ids = [
        row[0]
        for row in storage.conn.execute(
            "SELECT profile_id FROM profiles WHERE generated_from_request_id = ?",
            (profile_request_id,),
        ).fetchall()
    ]
    playbook_survivor_ids = [
        str(row[0])
        for row in storage.conn.execute(
            "SELECT user_playbook_id FROM user_playbooks WHERE request_id = ?",
            (playbook_request_id,),
        ).fetchall()
    ]
    profile_receipt_ids = storage.get_agent_run_finalization_receipt(
        run_id="run_profile_retry", entity_type="profile"
    )
    playbook_receipt_ids = storage.get_agent_run_finalization_receipt(
        run_id="run_playbook_retry", entity_type="user_playbook"
    )
    lineage_events = storage.get_lineage_events(request_id=playbook_request_id)
    assert sorted(event.op for event in lineage_events) == ["create", "merge"]
    assert {event.entity_id for event in lineage_events} == set(playbook_survivor_ids)
    playbook_lineage_ids = [event.event_id for event in lineage_events]
    assert len(set(playbook_lineage_ids)) == 2
    observed = {
        "profile_persisted": len(profile_survivor_ids),
        "profile_events": len(profile_keys),
        "profile_distinct_keys": len(set(profile_keys)),
        "playbook_persisted": len(playbook_survivor_ids),
        "playbook_lineage_events": len(playbook_lineage_ids),
        "playbook_events": len(playbook_keys),
        "playbook_distinct_keys": len(set(playbook_keys)),
    }
    assert observed == {
        "profile_persisted": 1,
        "profile_events": 1,
        "profile_distinct_keys": 1,
        "playbook_persisted": 1,
        "playbook_lineage_events": 2,
        "playbook_events": 1,
        "playbook_distinct_keys": 1,
    }
    assert profile_receipt_ids == profile_survivor_ids
    assert profile_event_ids == profile_survivor_ids
    assert profile_keys == [f"learn:profile:{profile_survivor_ids[0]}"]
    assert playbook_receipt_ids == playbook_survivor_ids
    assert playbook_event_ids == playbook_survivor_ids
    assert playbook_keys == [f"learn:user_playbook:{playbook_survivor_ids[0]}"]
    assert [event.event_key for event in delivery_attempts] == [
        profile_keys[0],
        profile_keys[0],
        playbook_keys[0],
        playbook_keys[0],
    ]


def _seed_completed_output(storage, run, payload):
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredPlaybookList,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )
    from tests.server.services.extraction._publication_fixtures import (
        write_completed_output,
    )

    shape = (
        StructuredProfilesOutput
        if run.binding.extractor_kind == "profile"
        else StructuredPlaybookList
    )
    written = write_completed_output(storage, run, shape.model_validate(payload))
    storage.conn.execute(
        "UPDATE _agent_runs SET updated_at=?,claimed_by=NULL,claimed_at=NULL WHERE id=?",
        ((datetime.now(UTC) - timedelta(days=1)).isoformat(), written.id),
    )
    storage.conn.commit()
    return storage.get_agent_run(written.id)


def _commit_resumed_output(
    worker,
    run,
    output,
    *,
    before_model: Callable[[AgentRunRecord], AgentRunRecord] | None = None,
):
    """Script output; retain the active caller's attempt and real output writer."""
    from dataclasses import asdict

    from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
    from reflexio.server.services.extraction.answer_inputs import answer_input_scope
    from reflexio.server.services.extraction.overflow_answers import (
        answer_witness,
        pin_extraction_answer,
    )
    from reflexio.server.services.extraction.resumable_agent import (
        ResumableExtractionAgent,
        decode_committed_output,
    )
    from reflexio.server.services.extraction.resume_worker import (
        _request_interaction_models_from_ids,
    )
    from reflexio.server.services.extraction.source_inputs import capture_source_inputs
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredPlaybookList,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    payload, provenance = decode_committed_output(output)
    shape = (
        StructuredProfilesOutput
        if run.binding.extractor_kind == "profile"
        else StructuredPlaybookList
    )
    sources = capture_source_inputs(
        worker.storage,
        run,
        _request_interaction_models_from_ids(
            worker.storage,
            run.binding.source_interaction_ids,
            fallback_source=run.binding.source,
            fallback_agent_version=run.binding.agent_version,
        ),
    )
    with answer_input_scope(worker.storage, run.id):
        answers = worker._load_resolved_tool_calls(run)
        for answer in answers:
            pin_extraction_answer(answer)
        dependencies = {
            answer.id: build_scope_hash(answer_witness(asdict(answer)))
            for answer in answers
        }

        def admitted_witness(_original):
            assert worker._active_claim is not None
            return worker._active_claim

        with patch(
            "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
            return_value=ToolLoopResult(
                ctx=None,
                trace=ToolLoopTrace(finished=True),
                finished_reason="structured_output",
                structured_output=shape.model_validate(payload),
                provenance=provenance,
                max_steps_remaining=8,
            ),
        ):
            result = ResumableExtractionAgent(
                client=worker.client, storage=worker.storage
            )._run(
                run=run,
                before_model=before_model or admitted_witness,
                publication_origin="resume",
                messages=[],
                history_inputs=HistoricalInputs(),
                output_schema=shape,
                source_inputs=sources,
                dependency_inputs=dependencies,
            )
    stored = result.execution_run
    assert stored is not None
    worker._active_claim = stored
    return stored


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_empty_output_receipt_retries_without_redispatch(storage, kind):
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        user_extraction_lease,
    )
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker
    from reflexio.server.services.extraction.publication import publish_resumed_output

    _seed_interactions(storage)
    entity_type = "profile" if kind == "profile" else "user_playbook"
    service_type = (
        ProfileGenerationService if kind == "profile" else PlaybookGenerationService
    )
    run = _seed_completed_output(
        storage,
        AgentRunRecord(
            id="run_empty_playbook",
            binding=AgentBinding(
                org_id="org_1",
                extractor_kind=kind,
                user_id="user_1",
                request_id="request_1",
                agent_version="v1",
                source="api",
                source_interaction_ids=[1, 2],
            ),
            status=AgentRunStatus.RUNNING,
            generation_request_snapshot={
                "output_schema_name": "StructuredProfilesOutput"
                if kind == "profile"
                else "StructuredPlaybookList"
            },
        ),
        {"profiles" if kind == "profile" else "playbooks": []},
    )
    assert run is not None
    context = _finalization_context(storage)
    worker = DurableLearningWorker(lambda _org: context)
    recorder = _DeduplicatingUsageRecorder()
    configure_usage_event_recorder(recorder)
    with (
        patch(
            "litellm.completion",
            side_effect=AssertionError("saved output cannot call model"),
        ),
        patch.object(
            service_type,
            "_resolve_write_plan",
            autospec=True,
            side_effect=service_type._resolve_write_plan,
        ) as resolve,
        patch.object(
            PlaybookGenerationService, "_enqueue_user_playbook_optimization"
        ) as optimize,
        patch.object(
            PlaybookGenerationService, "_trigger_playbook_aggregation"
        ) as aggregate,
        patch(
            "reflexio.server.services.extraction.resume_worker.schedule_tagging"
        ) as tagging,
    ):
        assert worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
        published = storage.get_agent_run(run.id)
        assert published is not None and published.status == AgentRunStatus.FINALIZED
        assert published.committed_output is not None
        assert worker.drain_org(storage.org_id, batch_size=3, lease_seconds=300) == 0
        tagging.assert_called_once()
        with storage._stream_sql() as db:
            receipt = storage._get_output_publication_receipt(
                db, published.id, published.committed_output["publication"]["id"]
            )
        assert receipt is not None and receipt.learning_ids == ()
        assert receipt.delivery_state == "accepted" and receipt.financial_facts is None
        no_prepare = MagicMock(
            side_effect=AssertionError("accepted output cannot recompute")
        )
        no_dispatch = MagicMock(
            side_effect=AssertionError("accepted output cannot redispatch")
        )
        with user_extraction_lease(storage, "user_1", max_wait_seconds=0):
            retained = adopt_agent_run_for_execution(storage, published)
            replay = publish_resumed_output(
                storage, retained, prepare=no_prepare, on_published=no_dispatch
            )
        assert replay.status == AgentRunStatus.FINALIZED
        no_prepare.assert_not_called()
        no_dispatch.assert_not_called()
        tagging.assert_called_once()
        resolve.assert_called_once()
        optimize.assert_not_called()
        aggregate.assert_not_called()
        assert (
            storage.get_agent_run_finalization_receipt(
                run_id=run.id, entity_type=entity_type
            )
            == []
        )
        with storage._stream_sql() as db:
            assert (
                storage._get_output_publication_receipt(
                    db, run.id, published.committed_output["publication"]["id"]
                )
                == receipt
            )
    # Historical empty evidence remains independently observable; it is not
    # an execution path and cannot stand in for a current output receipt.
    legacy = _persist_and_load_run(
        storage,
        _finalizing_run(
            run_id="legacy_empty",
            extractor_kind=kind,
            request_id="legacy_empty_request",
        ),
    )
    assert storage.save_agent_run_finalization_receipt(
        run_id=legacy.id, entity_type=entity_type, learning_ids=[]
    )
    historical_ids = storage.get_agent_run_finalization_receipt(
        run_id=legacy.id, entity_type=entity_type
    )
    assert type(historical_ids) is list and historical_ids == []
    assert not [
        event for event in recorder.events if event.event_name == "learnings_generated"
    ]


def test_resume_worker_fails_run_when_step_budget_exhausted(
    monkeypatch,
    request_context,
    storage,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _seed_interactions(storage)
    _seed_ready_run(storage)
    storage.update_agent_run_status(
        "run_1",
        AgentRunStatus.RESUME_READY,
        max_steps_remaining=0,
    )
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    worker = DurableLearningWorker(lambda _org: request_context)

    with patch(
        "litellm.completion",
        side_effect=AssertionError("exhausted budget must not call LLM"),
    ):
        resumed = worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)

    assert resumed == 1
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FAILED
    assert run.last_error == "Resumable extraction max-step budget exhausted"
    assert run.resume_attempts == 0
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is None


def test_resume_skips_owner_replaced_before_owned_selection(
    storage, request_context, monkeypatch
):
    from contextlib import contextmanager

    from reflexio.server.services.durable_learning import user_lease
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    _seed_ready_run(storage)
    before = storage.get_agent_run("run_1")
    worker = DurableLearningWorker(lambda _org: request_context)
    replacement = []
    owned = user_lease.owned_user_work_context

    @contextmanager
    def replace_owner(store, user, token, seconds):
        store.release_user_extraction(user, token)
        new_token = store.claim_user_extraction(user, "new-owner", seconds)
        assert new_token is not None and new_token != token
        replacement.append(new_token)
        with owned(store, user, token, seconds):
            yield

    monkeypatch.setattr(user_lease, "owned_user_work_context", replace_owner)
    try:
        with patch.object(
            ExtractionResumeWorker,
            "_process_claimed_run",
            side_effect=AssertionError("stale owner executed"),
        ) as process:
            assert (
                worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 0
            )
        process.assert_not_called()
        assert len(replacement) == 1
        assert storage.get_agent_run("run_1") == before
        assert storage.claim_user_extraction("user_1", "third-owner", 300) is None
    finally:
        for token in replacement:
            storage.release_user_extraction("user_1", token)


def test_lost_user_lease_cannot_rewind_finalized_resume(storage, request_context):
    from reflexio.server.services.durable_learning.user_lease import (
        user_extraction_lease,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
    )

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = ExtractionResumeWorker(request_context=request_context)
    with user_extraction_lease(storage, "user_1"):
        storage.update_agent_run_status("run_1", AgentRunStatus.FINALIZED)
        with storage._stream_sql() as db:
            db.query(
                "UPDATE learning_work SET lease_token='replacement' WHERE user_id='user_1'"
            )
        with pytest.raises(LeaseLostError):
            worker._update_claimed_status("run_1", AgentRunStatus.RESUME_READY)
    assert storage.get_agent_run("run_1").status == AgentRunStatus.FINALIZED


def test_bounded_user_lease_wait_does_not_claim_when_budget_busy(storage, monkeypatch):
    from reflexio.server.services.durable_learning import user_lease as module

    monkeypatch.setattr(module, "_reserve", lambda: False)
    with (
        pytest.raises(module.UserLeaseBusyError),
        module.user_extraction_lease(storage, "user_1", max_wait_seconds=0.01),
    ):
        pytest.fail("busy budget entered")
    assert storage.claim_extraction("automatic", 300) is None


@pytest.mark.parametrize("lease_seconds", [1, 90, 600])
def test_explicit_user_lease_uses_shared_duration_without_polling(
    storage, monkeypatch, lease_seconds
):
    from reflexio.server.services.durable_learning.user_lease import (
        user_extraction_lease,
    )

    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS", str(lease_seconds))
    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_POLL_SECONDS", "not-a-poll")
    with (
        user_extraction_lease(storage, "user_1"),
        storage._stream_sql(coordination=True) as db,
    ):
        row = db.query(
            "SELECT lease_until FROM learning_work WHERE org_id=? AND user_id=?",
            (storage.org_id, "user_1"),
        )[0]
        assert lease_seconds - 1 <= row["lease_until"] - db.now() <= lease_seconds


def test_explicit_user_lease_renews_configured_duration(storage, monkeypatch):
    from reflexio.server.services.durable_learning.user_lease import (
        user_extraction_lease,
    )

    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS", "1")
    renewed = threading.Event()
    durations = []
    original = storage.renew_extraction

    def renew(user_id, token, seconds):
        result = original(user_id, token, seconds)
        durations.append(seconds)
        renewed.set()
        return result

    monkeypatch.setattr(storage, "renew_extraction", renew)
    with user_extraction_lease(storage, "user_1"):
        assert renewed.wait(2), "configured one-second lease was not renewed"
        assert durations and set(durations) == {1}


def test_invalid_explicit_lease_fails_before_admission(storage, monkeypatch):
    from reflexio.server.services.durable_learning import user_lease as module

    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS", "0")
    reserve = MagicMock(side_effect=AssertionError("capacity was touched"))
    monkeypatch.setattr(module, "_reserve", reserve)
    with (
        pytest.raises(ValueError, match="REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS"),
        module.user_extraction_lease(storage, "user_1"),
    ):
        pytest.fail("invalid lease entered")
    reserve.assert_not_called()


def test_explicit_heartbeat_launch_failure_releases_lease_and_capacity(
    storage, monkeypatch
):
    from reflexio.server.services.durable_learning import user_lease as module
    from reflexio.server.services.durable_learning import worker

    monkeypatch.setattr(worker, "_budget", threading.BoundedSemaphore(1))
    original = threading.Thread.start

    def start(thread):
        if thread.name == "reflexio-extraction-heartbeat":
            raise RuntimeError("cannot start heartbeat")
        return original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    with (
        pytest.raises(RuntimeError, match="cannot start heartbeat"),
        module.user_extraction_lease(storage, "user_1"),
    ):
        pytest.fail("unrenewable lease entered")
    assert worker._reserve(), "failed launch leaked execution capacity"
    worker._release()
    token = storage.claim_user_extraction("user_1", "next-owner", 30)
    assert token is not None, "failed launch left the user leased"
    storage.release_user_extraction("user_1", token)


def test_busy_resume_claims_do_not_spend_execution_attempts(
    storage, request_context, monkeypatch
):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = DurableLearningWorker(lambda _org: request_context)
    before = storage.get_agent_run("run_1")
    parent = storage.claim_user_extraction("user_1", "busy-owner", 300)
    assert parent is not None
    executed = []

    def provider_failure(active_worker, run, _calls):
        executed.append(run.id)
        reserve_compute_attempt()
        raise RuntimeError("transient provider error")

    monkeypatch.setattr(ExtractionResumeWorker, "_resume_run", provider_failure)
    try:
        for _ in range(3):
            assert (
                worker.drain_org(storage.org_id, batch_size=3, lease_seconds=300) == 0
            )
            assert storage.get_agent_run("run_1") == before
        assert executed == []
    finally:
        storage.release_user_extraction("user_1", parent)
    assert worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None and result.resume_attempts == 1
    assert result.status == AgentRunStatus.RESUME_READY
    assert executed == ["run_1"]


def test_busy_user_does_not_end_resume_batch(storage, request_context, monkeypatch):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    _seed_ready_run(storage)
    first = storage.get_agent_run("run_1")
    assert first is not None
    storage.add_request(
        Request(
            request_id="request_2",
            user_id="user_2",
            created_at=1000,
            source="api",
            agent_version="v1",
            session_id="request_2",
        )
    )
    for identifier in (3, 4):
        storage._insert_interaction(
            Interaction(
                interaction_id=identifier,
                user_id="user_2",
                request_id="request_2",
                created_at=1000 + identifier,
                role="user",
                content="Independent synthetic deployment source",
            )
        )
    second = replace(
        first,
        id="run_2",
        binding=replace(
            first.binding,
            user_id="user_2",
            request_id="request_2",
            source_interaction_ids=[3, 4],
            window_start_interaction_id=3,
            window_end_interaction_id=4,
        ),
    )
    storage.create_agent_run(second)
    answer = storage.get_pending_tool_call("ptc_1")
    assert answer is not None
    second_answer = storage.create_pending_tool_call(
        replace(
            answer,
            id="ptc_2",
            user_id="user_2",
            dedup_key="second-user-deployment",
            question_text="Which deployment does user two need?",
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="run_2",
            pending_tool_call_id=second_answer.id,
            resolved_at=second_answer.resolved_at,
        )
    )
    worker = DurableLearningWorker(lambda _org: request_context)
    executed = []

    def execute(active_worker, run, _calls):
        executed.append(run.binding.user_id)
        return _commit_resumed_output(active_worker, run, {"profiles": []})

    parent = storage.claim_user_extraction("user_1", "busy-owner", 300)
    assert parent is not None
    monkeypatch.setattr(ExtractionResumeWorker, "_resume_run", execute)
    try:
        assert worker.drain_org(storage.org_id, batch_size=2, lease_seconds=300) == 1
        assert executed == ["user_2"]
        assert storage.get_agent_run("run_1") == first
        finished = storage.get_agent_run("run_2")
        assert finished is not None and finished.resume_attempts == 1
        assert finished.status == AgentRunStatus.FINALIZED
        assert storage.list_run_tool_dependencies("run_2")[0].consumed_at is not None
        assert storage.claim_user_extraction("user_1", "competitor", 300) is None
    finally:
        storage.release_user_extraction("user_1", parent)


def test_erased_source_cancels_resume_before_execution(request_context, storage):
    _seed_ready_run(storage)
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    worker = DurableLearningWorker(lambda _org: request_context)
    with patch.object(
        ExtractionResumeWorker,
        "_process_claimed_run",
        side_effect=AssertionError("must not revive erased input"),
    ):
        assert worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    run = storage.get_agent_run("run_1")
    assert run is not None and run.status == AgentRunStatus.CANCELLED
    assert run.last_error == "source_interactions_erased"
    assert worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 0


@pytest.mark.parametrize("failure_kind", ["readiness", "contention"])
@pytest.mark.parametrize("stage", ["retry", "finalization", "conversion"])
def test_readiness_defers_committed_finalization_without_spending_budget(
    storage, request_context, monkeypatch, stage, failure_kind
):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker
    from reflexio.server.services.storage.error import (
        ExecutionInputBusyError,
        ReadinessUnavailableError,
    )

    refusal = (
        ReadinessUnavailableError
        if failure_kind == "readiness"
        else ExecutionInputBusyError
    )

    _seed_interactions(storage)
    output = {
        "profiles": [{"content": "Deploy to AWS ECS.", "time_to_live": "infinity"}]
    }
    initial_resume = stage != "retry"
    _seed_ready_run(storage)
    worker = DurableLearningWorker(lambda _org: request_context)
    qualified = False

    def embedding(*_args, **_kwargs):
        if not qualified:
            raise refusal("audit pending")
        return [0.0] * 512

    monkeypatch.setattr(storage, "_get_embedding", embedding)
    monkeypatch.setattr(
        "reflexio.server.services.profile.components.consolidator.ProfileConsolidator.deduplicate",
        lambda _self, profiles, _user_id, _request_id: (profiles, [], []),
    )

    def resume(active_worker, run, _calls):
        stored = _commit_resumed_output(active_worker, run, output)
        if stage == "conversion":
            raise refusal("conversion audit pending")
        return stored

    resume_call = MagicMock(side_effect=resume)
    monkeypatch.setattr(
        ExtractionResumeWorker,
        "_resume_run",
        lambda active_worker, run, calls: resume_call(active_worker, run, calls),
    )
    starting_attempts = 0 if initial_resume else 2
    if not initial_resume:
        # Obtain a real modern output through the agent writer before testing
        # retries. Historical raw outputs have a separate qualification gate.
        completed_turns = worker.drain_org(
            storage.org_id, batch_size=6, lease_seconds=300
        )
        assert (
            completed_turns == 1
            if failure_kind == "readiness"
            else 1 <= completed_turns <= 2
        )
        first = storage.get_agent_run("run_1")
        assert first is not None and first.status == AgentRunStatus.FINALIZATION_FAILED
        for _ in range(starting_attempts):
            storage.update_agent_run_status(
                first.id, first.status, increment_finalization_attempts=True
            )
        storage.update_agent_run_status(
            first.id, first.status, next_resume_at=datetime(2000, 1, 1, tzinfo=UTC)
        )
    retained_output = None
    for _ in range(8):
        before = datetime.now(UTC)
        # One candidate can require an empty sweep-reset visit plus the
        # next visit in its class: at most two complete class rotations.
        completed_turns = worker.drain_org(
            storage.org_id, batch_size=6, lease_seconds=300
        )
        assert (
            completed_turns == 1
            if failure_kind == "readiness"
            else 1 <= completed_turns <= 2
        )
        deferred = storage.get_agent_run("run_1")
        assert deferred is not None
        assert deferred.status == AgentRunStatus.FINALIZATION_FAILED
        assert deferred.finalization_attempts == starting_attempts
        assert deferred.resume_attempts == 1
        assert deferred.committed_output is not None
        assert (
            deferred.committed_output["output"]["profiles"][0]["content"]
            == "Deploy to AWS ECS."
        )
        if retained_output is None:
            retained_output = deferred.committed_output
        assert deferred.committed_output == retained_output
        assert deferred.next_resume_at is not None
        if failure_kind == "readiness":
            assert 29 <= (deferred.next_resume_at - before).total_seconds() <= 31
        else:
            assert before <= deferred.next_resume_at <= datetime.now(UTC)
        assert (
            storage.get_agent_run_finalization_receipt(
                run_id="run_1", entity_type="profile"
            )
            is None
        )
        assert storage.get_user_profile("user_1") == []
        storage.update_agent_run_status(
            "run_1",
            AgentRunStatus.FINALIZATION_FAILED,
            next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
        )
    qualified = True
    assert worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    finalized = storage.get_agent_run("run_1")
    assert finalized is not None
    assert finalized.status == AgentRunStatus.FINALIZED
    assert finalized.finalization_attempts == starting_attempts
    assert len(storage.get_user_profile("user_1")) == 1
    assert retained_output is not None
    with storage._stream_sql() as db:
        receipt = storage._get_output_publication_receipt(
            db, "run_1", retained_output["publication"]["id"]
        )
    assert receipt is not None and len(receipt.learning_ids) == 1
    assert receipt.delivery_state == "accepted"
    assert resume_call.call_count == 1
    assert worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 0


@pytest.mark.parametrize("phase", ["preparation", "model"])
def test_readiness_resume_deferral_refunds_attempt_and_recovers(
    storage, request_context, monkeypatch, phase
):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker
    from reflexio.server.services.storage.error import ReadinessUnavailableError

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = DurableLearningWorker(lambda _org: request_context)

    def unavailable(active_worker, current, _calls):
        if phase == "model":
            reserve_compute_attempt()
            charged = active_worker._active_claim
            assert charged is not None
            assert charged.resume_attempts == current.resume_attempts + 1
        raise ReadinessUnavailableError("inference pending")

    resume = MagicMock(side_effect=unavailable)
    monkeypatch.setattr(
        ExtractionResumeWorker,
        "_resume_run",
        lambda active_worker, run, calls: resume(active_worker, run, calls),
    )
    for _ in range(8):
        before = datetime.now(UTC)
        # One candidate can require an empty sweep-reset visit plus the
        # next visit in its class: at most two complete class rotations.
        assert worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
        deferred = storage.get_agent_run("run_1")
        assert deferred is not None
        assert deferred.status == AgentRunStatus.RESUME_READY
        assert deferred.resume_attempts == 0
        assert deferred.next_resume_at is not None
        assert 29 <= (deferred.next_resume_at - before).total_seconds() <= 31
        storage.update_agent_run_status(
            "run_1",
            AgentRunStatus.RESUME_READY,
            next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
        )

    def resumed(active_worker, current, _calls):
        return _commit_resumed_output(active_worker, current, {"profiles": []})

    resume.side_effect = resumed
    assert worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    finalized = storage.get_agent_run("run_1")
    assert finalized is not None
    assert finalized.status == AgentRunStatus.FINALIZED
    assert finalized.resume_attempts == 1


@pytest.mark.parametrize("outage", [True, False])
def test_provider_outage_refunds_resume_attempts_until_recovery(
    storage, request_context, monkeypatch, outage
):
    """An outage never exhausts the run's allowance; its own failures still do."""
    from reflexio.server.llm._litellm_types import LiteLLMClientError
    from reflexio.server.llm.provider_circuit import (
        OPEN_AFTER_FAILURES,
        circuit_route,
        provider_circuit,
    )
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = DurableLearningWorker(lambda _org: request_context)
    route = circuit_route({"model": "test/model"})
    if outage:
        for _ in range(OPEN_AFTER_FAILURES):
            provider_circuit.admit(route, model="test/model").failed()

    def timing_out(active_worker, current, _calls):
        reserve_compute_attempt()
        failure = LiteLLMClientError("API call failed: provider stalled")
        failure.circuit_route = route
        raise failure from TimeoutError("provider stalled")

    resume = MagicMock(side_effect=timing_out)
    monkeypatch.setattr(
        ExtractionResumeWorker,
        "_resume_run",
        lambda active_worker, run, calls: resume(active_worker, run, calls),
    )
    limit = request_context.configurator.get_config().pending_tool_call_config
    for _ in range(limit.max_resume_attempts + 3):
        worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300)
        current = storage.get_agent_run("run_1")
        assert current is not None
        if current.status == AgentRunStatus.FAILED:
            break
        storage.update_agent_run_status(
            "run_1",
            AgentRunStatus.RESUME_READY,
            next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
        )
    current = storage.get_agent_run("run_1")
    assert current is not None
    if not outage:
        assert current.status == AgentRunStatus.FAILED
        return
    assert current.status == AgentRunStatus.RESUME_READY
    assert current.resume_attempts == 0
    assert current.last_error == "provider_unavailable"
    assert resume.call_count == limit.max_resume_attempts + 3

    provider_circuit.reset()

    def resumed(active_worker, current, _calls):
        return _commit_resumed_output(active_worker, current, {"profiles": []})

    resume.side_effect = resumed
    assert worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    finalized = storage.get_agent_run("run_1")
    assert finalized is not None
    assert finalized.status == AgentRunStatus.FINALIZED


def test_readiness_resume_refund_cannot_mutate_new_owner_or_refund_twice(storage):
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        owned_agent_run_write,
        user_extraction_lease,
    )

    _seed_interactions(storage)
    _seed_ready_run(storage)
    ready = storage.get_agent_run("run_1")
    assert ready is not None
    with user_extraction_lease(storage, "user_1"):
        adopted = adopt_agent_run_for_execution(storage, ready)
        with owned_agent_run_write(storage, adopted):
            claim = storage.update_agent_run_status(
                adopted.id,
                AgentRunStatus.RESUMING,
                expected_claimed_by=adopted.claimed_by,
                expected_claimed_at=adopted.claimed_at,
            )
        assert (
            claim is not None
            and claim.claimed_by is not None
            and claim.claimed_at is not None
        )
        with owned_agent_run_write(storage, claim):
            attempts = storage.begin_extraction_resume(
                claim.id, claim.claimed_by, claim.claimed_at.isoformat()
            )
            assert attempts == 1
            retry_at = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()
            assert not storage.defer_extraction_resume(
                claim.id,
                "different_owner",
                claim.claimed_at.isoformat(),
                attempts,
                retry_at,
            )
            assert storage.get_agent_run(claim.id).resume_attempts == 1
            assert storage.defer_extraction_resume(
                claim.id,
                claim.claimed_by,
                claim.claimed_at.isoformat(),
                attempts,
                retry_at,
            )
            assert not storage.defer_extraction_resume(
                claim.id,
                claim.claimed_by,
                claim.claimed_at.isoformat(),
                attempts,
                retry_at,
            )
            assert storage.get_agent_run(claim.id).resume_attempts == 0
    storage.update_agent_run_status(
        claim.id,
        AgentRunStatus.RESUME_READY,
        next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
    )
    ready_again = storage.get_agent_run(claim.id)
    assert ready_again is not None
    with user_extraction_lease(storage, "user_1"):
        adopted = adopt_agent_run_for_execution(storage, ready_again)
        with owned_agent_run_write(storage, adopted):
            replacement = storage.update_agent_run_status(
                adopted.id,
                AgentRunStatus.RESUMING,
                expected_claimed_by=adopted.claimed_by,
                expected_claimed_at=adopted.claimed_at,
            )
        assert (
            replacement is not None
            and replacement.claimed_by is not None
            and replacement.claimed_at is not None
        )
        assert (replacement.claimed_by, replacement.claimed_at) != (
            claim.claimed_by,
            claim.claimed_at,
        )
        with owned_agent_run_write(storage, replacement):
            assert (
                storage.begin_extraction_resume(
                    replacement.id,
                    replacement.claimed_by,
                    replacement.claimed_at.isoformat(),
                )
                == 1
            )
            assert not storage.defer_extraction_resume(
                claim.id,
                claim.claimed_by,
                claim.claimed_at.isoformat(),
                attempts,
                retry_at,
            )
            assert storage.get_agent_run(claim.id).resume_attempts == 1
            assert storage.get_agent_run(claim.id).status == AgentRunStatus.RESUMING


@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.RESUMING,
        AgentRunStatus.FINALIZING,
        AgentRunStatus.AGENT_COMPLETED,
    ],
)
@pytest.mark.parametrize("age_seconds", [1, 86400])
@pytest.mark.parametrize("parent_state", ["live", "released", "expired"])
def test_actual_user_parent_controls_adoption_independent_of_logical_age(
    storage, request_context, status, age_seconds, parent_state
):
    from uuid import UUID

    from reflexio.server.services.durable_learning.resume_selection import (
        ResumeSelection,
    )
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        owned_agent_run_write,
        owned_user_work_context,
    )
    from reflexio.server.services.storage.storage_base import LeaseLostError

    if status == AgentRunStatus.RESUMING:
        _seed_ready_run(storage)
    else:
        storage.create_agent_run(
            replace(
                _finalizing_run(
                    run_id="run_1", extractor_kind="profile", request_id="request_1"
                ),
                status=status,
                committed_output={"profiles": []},
            )
        )
    run = storage.get_agent_run("run_1")
    assert run is not None
    previous_token = storage.claim_user_extraction("user_1", "previous-owner", 300)
    assert previous_token is not None
    with owned_user_work_context(storage, "user_1", previous_token, 300):
        adopted = adopt_agent_run_for_execution(storage, run)
        # Historical row setup is separate from the actual parent's clock.
        old = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
        storage.conn.execute(
            "UPDATE _agent_runs SET status=?,claimed_at=?,updated_at=? WHERE id=?",
            (status.value, old, old, run.id),
        )
        storage.conn.commit()
    if parent_state == "released":
        storage.release_user_extraction("user_1", previous_token)
    elif parent_state == "expired":
        with storage._stream_sql() as db:
            db.query(
                "UPDATE learning_work SET lease_until=? WHERE org_id=? AND user_id=? "
                "AND lease_token=?",
                (db.now() - 1, storage.org_id, "user_1", previous_token),
            )
    before = storage.get_agent_run(run.id)
    assert before is not None and before.claimed_by == adopted.claimed_by
    selection = ResumeSelection()
    claimed = selection.claim_user(
        request_context, worker_id="new-owner", lease_seconds=300
    )
    # Fleet inspection and user admission never mutate the logical run.
    assert storage.get_agent_run(run.id) == before
    if parent_state == "live":
        try:
            assert claimed is None
            assert storage.claim_user_extraction("user_1", "competitor", 300) is None
        finally:
            storage.release_user_extraction("user_1", previous_token)
        return
    assert claimed is not None and claimed[0] == "user_1"
    user, token = claimed
    try:
        assert token != previous_token
        with owned_user_work_context(storage, user, token, 300):
            kind, current = selection.select_owned(
                request_context, user_id=user, token=token, worker_id="new-owner"
            )
            assert kind == (
                "continuation" if status == AgentRunStatus.RESUMING else "finalization"
            )
            assert current is not None and current.id == before.id
            assert current.binding == before.binding
            assert current.claimed_by is not None
            prefix, _, nonce = current.claimed_by.partition(":")
            assert prefix == "new-owner" and UUID(hex=nonce).hex == nonce
            assert current.claimed_by != before.claimed_by
            assert current.claimed_by != token
            with owned_agent_run_write(storage, current):
                pass
            with pytest.raises(LeaseLostError), owned_agent_run_write(storage, before):
                pass
            assert current.status == (
                AgentRunStatus.RESUMING
                if status == AgentRunStatus.RESUMING
                else AgentRunStatus.FINALIZING
            )
            assert current.resume_attempts == before.resume_attempts
            assert current.finalization_attempts == before.finalization_attempts
            assert current.committed_output == before.committed_output
            assert storage.claim_user_extraction("user_1", "competitor", 300) is None
    finally:
        storage.release_user_extraction(user, token)


def test_policy_migration_error_prevents_claiming_work(request_context, storage):
    from reflexio.config_migration import ConfigurationMigrationError
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_ready_run(storage)
    before = storage.get_agent_run("run_1")
    request_context.configurator.get_config.side_effect = ConfigurationMigrationError(
        "migration required"
    )
    worker = DurableLearningWorker(lambda _org: request_context)
    with (
        patch.object(
            storage, "claim_user_extraction", wraps=storage.claim_user_extraction
        ) as claim,
        patch.object(
            storage,
            "_commit_owned_work_selection",
            wraps=storage._commit_owned_work_selection,
        ) as adopt,
    ):
        assert worker.drain_org(storage.org_id, batch_size=3, lease_seconds=300) == 0
    claim.assert_not_called()
    adopt.assert_not_called()
    assert storage.get_agent_run("run_1") == before


@pytest.mark.parametrize("retired_ttl", [1, 60])
def test_retired_claim_ttl_cannot_change_shared_busy_admission(
    storage, request_context, monkeypatch, retired_ttl
):
    from reflexio.server.services.durable_learning import user_lease
    from reflexio.server.services.durable_learning import worker as worker_module

    monkeypatch.setenv("REFLEXIO_RESUME_CLAIM_TTL_SECONDS", str(retired_ttl))
    _seed_interactions(storage)
    _seed_ready_run(storage)
    storage.update_agent_run_status(
        "run_1", AgentRunStatus.RESUME_READY, max_steps_remaining=0
    )
    before = storage.get_agent_run("run_1")
    parent = storage.claim_user_extraction("user_1", "busy-owner", 300)
    assert parent is not None
    worker = worker_module.DurableLearningWorker(lambda _org: request_context)
    try:
        with (
            patch.object(worker_module, "_reserve", return_value=False),
            patch.object(
                storage, "claim_user_extraction", wraps=storage.claim_user_extraction
            ) as claim,
            patch.object(
                storage, "claim_extraction", wraps=storage.claim_extraction
            ) as ordinary_claim,
        ):
            assert worker.drain_org(storage.org_id, batch_size=3, lease_seconds=37) == 0
        claim.assert_not_called()
        ordinary_claim.assert_not_called()
        assert storage.get_agent_run("run_1") == before
        assert worker.drain_org(storage.org_id, batch_size=3, lease_seconds=37) == 0
        assert storage.get_agent_run("run_1") == before
    finally:
        storage.release_user_extraction("user_1", parent)
    with (
        patch.object(
            user_lease,
            "owned_user_work_context",
            wraps=user_lease.owned_user_work_context,
        ) as owned,
        patch(
            "litellm.completion",
            side_effect=AssertionError("exhausted budget cannot call model"),
        ),
    ):
        assert worker.drain_org(storage.org_id, batch_size=6, lease_seconds=37) == 1
    assert owned.called and all(call.args[3] == 37 for call in owned.call_args_list)
    saved = storage.get_agent_run("run_1")
    assert saved is not None and saved.status == AgentRunStatus.FAILED
    assert saved.resume_attempts == 0


@pytest.mark.parametrize("extractor_kind", ["profile", "playbook"])
def test_direct_continuation_cannot_reopen_pending_tools_after_org_opt_out(
    request_context, extractor_kind
):
    config = request_context.configurator.get_config()
    config.pending_tool_call_config.pending_ttl_seconds = 0
    worker = ExtractionResumeWorker(
        request_context=request_context, llm_client=MagicMock()
    )
    run = _finalizing_run(
        run_id="opted-out-continuation",
        extractor_kind=extractor_kind,
        request_id="request-opted-out-continuation",
    )
    tools, tool_context = worker._extra_tools_for_run(run)
    assert tools == []
    assert tool_context is None


@pytest.mark.parametrize("recover_before_exit", [False, True])
def test_transient_renewal_failure_preserves_valid_user_owner(
    storage, monkeypatch, recover_before_exit
):
    from reflexio.server.services.durable_learning import user_lease as module

    failed = threading.Event()
    recovered = threading.Event()
    renew = storage.renew_extraction
    calls = 0

    def intermittent_renewal(user_id, token, seconds):
        nonlocal calls
        calls += 1
        if calls == 1 or not recover_before_exit:
            failed.set()
            raise OSError("synthetic coordination interruption")
        result = renew(user_id, token, seconds)
        recovered.set()
        return result

    monkeypatch.setattr(storage, "renew_extraction", intermittent_renewal)
    monkeypatch.setattr(
        module.durable_learning_policy, "renewal_interval_seconds", lambda _: 0.01
    )
    with module.user_extraction_lease(storage, "user_1"):
        assert failed.wait(1), "heartbeat never exercised the transient failure"
        row = storage.conn.execute(
            "SELECT lease_token FROM learning_work WHERE org_id=? AND user_id=?",
            (storage.org_id, "user_1"),
        ).fetchone()
        # The actual database still accepts this owner, despite the renewal I/O.
        storage.fence_user_extraction("user_1", row["lease_token"])
        if recover_before_exit:
            assert recovered.wait(1), "a transient error permanently stopped renewal"


def test_explicit_fence_is_bound_to_storage_identity(storage, tmp_path, monkeypatch):
    from reflexio.server.services.durable_learning.user_lease import (
        fence_explicit_extraction,
        user_extraction_lease,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
    )

    other = SQLiteStorage(org_id=storage.org_id, db_path=str(tmp_path / "other.db"))
    fence = MagicMock(wraps=other.fence_user_extraction)
    monkeypatch.setattr(other, "fence_user_extraction", fence)
    try:
        with user_extraction_lease(storage, "user_1"):
            with pytest.raises(LeaseLostError, match="different storage"):
                fence_explicit_extraction(other)
            fence.assert_not_called()
    finally:
        other.conn.close()


@pytest.mark.integration
@pytest.mark.parametrize("extractor_kind", ["profile", "playbook"])
@pytest.mark.parametrize("open_followup", [False, True])
def test_normal_finalization_uses_actual_committed_output_witness(
    storage, request_context, monkeypatch, extractor_kind, open_followup
):
    from dataclasses import asdict, replace

    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.llm.tools import ToolLoopTrace
    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
    )
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker
    from reflexio.server.services.extraction.resumable_agent import (
        AgentRunResult,
        ResumableExtractionAgent,
    )
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredPlaybookList,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    _seed_interactions(storage)
    _seed_ready_run(storage)
    if extractor_kind == "playbook":
        storage._execute(
            "UPDATE _agent_runs SET extractor_kind='playbook' WHERE id='run_1'"
        )
    if open_followup:
        question = "Which additional approval is required?"
        prior = storage.get_pending_tool_call("ptc_1")
        assert prior is not None
        storage.create_pending_tool_call(
            replace(
                prior,
                id="ptc_next",
                status=PendingToolCallStatus.PENDING,
                result=None,
                resolved_at=None,
                valid_until=None,
                question_text=question,
                args={"question": question},
                dedup_key=build_pending_tool_call_dedup_key(
                    tool_name="ask_human", question_text=question
                ),
            )
        )
    worker = ExtractionResumeWorker(
        request_context=request_context, llm_client=MagicMock()
    )
    monkeypatch.setattr(
        ExtractionResumeWorker,
        "_messages_with_prior_knowledge",
        lambda _self, **kw: kw["messages"],
    )
    stored_provenance = ModelProvenance(
        model_name="committed_model", provider="committed_provider"
    )
    returned_provenance = ModelProvenance(
        model_name="parallel_model", provider="parallel_provider"
    )
    field = "profiles" if extractor_kind == "profile" else "playbooks"
    shape = (
        StructuredProfilesOutput
        if extractor_kind == "profile"
        else StructuredPlaybookList
    )
    stored_entry = (
        {"content": "Committed content", "time_to_live": "infinity"}
        if field == "profiles"
        else {
            "content": "Committed content",
            "trigger": "when deploying",
            "rationale": "the agreed procedure",
        }
    )
    returned_entry = {**stored_entry, "content": "Parallel uncommitted content"}
    payload = {
        "_reflexio_envelope_version": 1,
        "output": {field: [stored_entry]},
        "model_provenance": asdict(stored_provenance),
    }

    def resume(_agent, *, run, before_model, **_kwargs):
        saved = _commit_resumed_output(worker, run, payload, before_model=before_model)
        if open_followup:
            with owned_agent_run_write(storage, saved):
                storage.attach_run_tool_dependency(
                    RunToolDependencyRecord(
                        run_id=run.id, pending_tool_call_id="ptc_next"
                    )
                )
                saved = storage.update_agent_run_status(
                    saved.id,
                    saved.status,
                    pending_tool_call_ids=["ptc_next"],
                    expected_claimed_by=saved.claimed_by,
                    expected_claimed_at=saved.claimed_at,
                )
            assert saved is not None
        return AgentRunResult(
            run_id=run.id,
            output=shape.model_validate({field: [returned_entry]}),
            pending_tool_call_ids=["not_in_committed_output"],
            messages=[],
            trace=ToolLoopTrace(finished=True),
            finished_reason="structured_output",
            model_provenance=returned_provenance,
            execution_run=saved,
        )

    monkeypatch.setattr(ResumableExtractionAgent, "resume", resume)
    observed = {}

    decode = ExtractionResumeWorker._items_from_committed_output

    def inspect_decode(current_worker, run):
        items, pending, provenance = decode(current_worker, run)
        observed.update(
            run=run, contents=[item.content for item in items], provenance=provenance
        )
        return items, pending, provenance

    monkeypatch.setattr(
        ExtractionResumeWorker, "_items_from_committed_output", inspect_decode
    )
    service = (
        ProfileGenerationService
        if extractor_kind == "profile"
        else PlaybookGenerationService
    )
    monkeypatch.setattr(service, "_resolve_write_plan", lambda *_args: None)
    shared_worker = DurableLearningWorker(lambda _org: request_context)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert result is not None
    assert observed["contents"] == ["Committed content"]
    assert observed["provenance"] == stored_provenance
    assert observed["run"].status == AgentRunStatus.FINALIZING
    assert (
        observed["run"].committed_output["output"]
        == shape.model_validate(payload["output"]).model_dump()
    )
    assert len(observed["run"].committed_output["publication"]["id"]) == 32
    assert result.status == (
        AgentRunStatus.FINALIZED_PENDING_TOOL
        if open_followup
        else AgentRunStatus.FINALIZED
    )
    assert result.pending_tool_call_ids == (["ptc_next"] if open_followup else [])
    if open_followup:
        dependencies = {
            d.pending_tool_call_id: d
            for d in storage.list_run_tool_dependencies(result.id)
        }
        assert dependencies["ptc_1"].consumed_at is not None
        assert dependencies["ptc_next"].consumed_at is None


@pytest.mark.integration
def test_conversion_failure_retries_saved_output_without_another_agent_call(
    storage, request_context, monkeypatch
):
    from reflexio.server.llm.tools import ToolLoopTrace
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker
    from reflexio.server.services.extraction.resumable_agent import (
        AgentRunResult,
        ResumableExtractionAgent,
    )
    from reflexio.server.services.profile.components.extractor import ProfileExtractor
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = ExtractionResumeWorker(
        request_context=request_context, llm_client=MagicMock()
    )
    monkeypatch.setattr(
        ExtractionResumeWorker,
        "_messages_with_prior_knowledge",
        lambda _self, **kw: kw["messages"],
    )
    calls = []
    payload = {"profiles": [{"content": "Saved content", "time_to_live": "infinity"}]}

    def resume(_agent, *, run, before_model, **_kwargs):
        calls.append(run.id)
        saved = _commit_resumed_output(worker, run, payload, before_model=before_model)
        return AgentRunResult(
            run_id=run.id,
            output=StructuredProfilesOutput.model_validate(payload),
            pending_tool_call_ids=[],
            messages=[],
            trace=ToolLoopTrace(finished=True),
            finished_reason="structured_output",
            execution_run=saved,
        )

    monkeypatch.setattr(ResumableExtractionAgent, "resume", resume)
    monkeypatch.setattr(
        ProfileExtractor,
        "_convert_raw_to_user_profiles",
        MagicMock(side_effect=ValueError("controlled conversion failure")),
    )
    shared_worker = DurableLearningWorker(lambda _org: request_context)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    first = storage.get_agent_run("run_1")
    assert first is not None
    assert first.status == AgentRunStatus.FINALIZATION_FAILED
    assert first.committed_output is not None
    assert (
        first.committed_output["output"]
        == StructuredProfilesOutput.model_validate(payload).model_dump()
    )
    assert first.resume_attempts == 1 and first.finalization_attempts == 1
    storage.update_agent_run_status(
        first.id, first.status, next_resume_at=datetime(2000, 1, 1, tzinfo=UTC)
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=3, lease_seconds=300) == 1
    second = storage.get_agent_run(first.id)
    assert second is not None
    assert second.status == AgentRunStatus.FINALIZATION_FAILED
    assert second.committed_output == first.committed_output
    assert second.resume_attempts == 1 and second.finalization_attempts == 2
    assert calls == [first.id]
    assert storage.list_run_tool_dependencies(first.id)[0].consumed_at is None


@pytest.mark.integration
def test_profile_output_conversion_does_not_construct_runtime_extractor(
    storage, request_context, monkeypatch
):
    from reflexio.server.services.profile.components.extractor import ProfileExtractor

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = ExtractionResumeWorker(
        request_context=request_context, llm_client=MagicMock()
    )
    run = storage.get_agent_run("run_1")
    monkeypatch.setattr(
        ProfileExtractor,
        "__init__",
        lambda *_a, **_k: pytest.fail("pure conversion constructed an extractor"),
    )
    monkeypatch.setattr(
        storage,
        "get_user_profile",
        lambda *_a, **_k: pytest.fail("pure conversion read existing profiles"),
    )
    request_context.configurator.get_agent_context.side_effect = AssertionError(
        "pure conversion loaded agent context"
    )
    items, pending = worker._profile_items_from_output(
        run,
        request_context.configurator.get_config().profile_extractor_config,
        {"profiles": [{"content": "Saved content", "time_to_live": "infinity"}]},
    )
    assert [item.content for item in items] == ["Saved content"]
    assert items[0].user_id == run.binding.user_id
    assert items[0].generated_from_request_id == run.binding.request_id
    assert items[0].source_interaction_ids == run.binding.source_interaction_ids
    assert pending == run.pending_tool_call_ids
