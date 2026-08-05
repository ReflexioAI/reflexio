from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.billing_meter import ReceiptBillingDeliveryError
from reflexio.server.services.deferred_learning_plan import FinalizationResult
from reflexio.server.services.extraction.resume_worker import (
    ExtractionResumeWorker,
    _finalization_failure_status,
    _run_playbook_contract_selection,
    _run_uses_strict_playbook_evidence,
)
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidationOutput,
    UnifyDecision,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.profile.service import (
    ProfileGenerationService,
    ProfileGenerationServiceConfig,
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
            enabled=True,
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
        pending_tool_call_config=PendingToolCallConfig(enabled=True),
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


def _gate_initial_receipt_reads(
    storages: tuple[SQLiteStorage, SQLiteStorage],
) -> None:
    barrier = threading.Barrier(len(storages))

    def install_gate(storage: SQLiteStorage) -> None:
        original = storage.get_agent_run_finalization_receipt
        first_read = True

        def gated_read(*, run_id: str, entity_type: str) -> list[str] | None:
            nonlocal first_read
            receipt = original(run_id=run_id, entity_type=entity_type)
            if first_read:
                first_read = False
                barrier.wait(timeout=5)
            return receipt

        storage.get_agent_run_finalization_receipt = MagicMock(  # type: ignore[method-assign]
            side_effect=gated_read
        )

    for storage in storages:
        install_gate(storage)


def _hide_next_receipt_reads(storage: SQLiteStorage, *, count: int = 2) -> None:
    original = storage.get_agent_run_finalization_receipt
    remaining = count

    def stale_read(*, run_id: str, entity_type: str) -> list[str] | None:
        nonlocal remaining
        if remaining > 0:
            remaining -= 1
            return None
        return original(run_id=run_id, entity_type=entity_type)

    storage.get_agent_run_finalization_receipt = MagicMock(  # type: ignore[method-assign]
        side_effect=stale_read
    )


def _profile_service(
    context: RequestContext, *, request_id: str
) -> ProfileGenerationService:
    service = ProfileGenerationService(llm_client=MagicMock(), request_context=context)
    service.service_config = ProfileGenerationServiceConfig(
        user_id="user_1",
        request_id=request_id,
        source="api",
        auto_run=False,
        force_extraction=True,
    )
    return service


def _playbook_service(
    context: RequestContext, *, request_id: str
) -> PlaybookGenerationService:
    service = PlaybookGenerationService(llm_client=MagicMock(), request_context=context)
    service.service_config = PlaybookGenerationServiceConfig(
        request_id=request_id,
        agent_version="v1",
        user_id="user_1",
        source="api",
        auto_run=False,
        force_extraction=True,
    )
    return service


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

    def __call__(self, event: UsageEvent) -> UsageEventDeliveryStatus:
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


def test_resume_worker_resumes_profile_run_and_consumes_dependency(
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
    worker = ExtractionResumeWorker(request_context=request_context)

    with (
        patch("litellm.completion", side_effect=[response]),
        patch(
            "reflexio.server.services.profile.components.consolidator.ProfileConsolidator",
        ) as mock_consolidator_cls,
    ):
        mock_consolidator_cls.return_value.deduplicate.side_effect = (
            lambda profiles, _user_id, _request_id: (profiles, [], [])
        )
        resumed = worker.drain(max_runs=1)

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
    worker = ExtractionResumeWorker(request_context=request_context)

    with (
        patch("litellm.completion", side_effect=[response]),
        patch(
            "reflexio.server.services.profile.service."
            "ProfileGenerationService._finalize_extracted_items_with_outcome",
            side_effect=RuntimeError("storage write failed"),
        ),
    ):
        resumed = worker.drain(max_runs=1)

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
            "ProfileGenerationService._finalize_extracted_items_with_outcome",
            return_value=FinalizationResult([], won_receipt=False),
        ) as finalize,
    ):
        resumed = worker.drain(max_runs=1)

    assert resumed == 1
    finalize.assert_called_once()
    retried = storage.get_agent_run("run_1")
    assert retried is not None
    assert retried.status == AgentRunStatus.FINALIZED
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is not None


@pytest.mark.parametrize(
    ("extractor_kind", "finalize_path", "entity_type"),
    [
        (
            "profile",
            "reflexio.server.services.profile.service."
            "ProfileGenerationService._finalize_extracted_items_with_outcome",
            "profile",
        ),
        (
            "playbook",
            "reflexio.server.services.playbook.service."
            "PlaybookGenerationService._finalize_extracted_items_with_outcome",
            "user_playbook",
        ),
    ],
)
def test_resume_bills_only_items_that_survive_finalization(
    request_context, extractor_kind, finalize_path, entity_type
):
    run = AgentRunRecord(
        id="survivor-run",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind=extractor_kind,
            user_id="user_1",
            request_id="request_1",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZING,
        generation_request_snapshot={},
    )
    dropped = object()
    survivor = object()
    survivor_id = "durable-survivor-id"
    worker = ExtractionResumeWorker(request_context=request_context)

    with (
        patch(
            finalize_path,
            return_value=FinalizationResult([survivor_id], won_receipt=True),
        ) as finalize,
        patch.object(worker, "_record_finalized_learnings") as record,
    ):
        worker._finalize_items(run, [dropped, survivor])

    record.assert_called_once_with(run, [survivor_id], entity_type=entity_type)
    if extractor_kind == "playbook":
        assert finalize.call_args.kwargs["extraction_run"] is run
    else:
        assert "extraction_run" not in finalize.call_args.kwargs


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
    worker = ExtractionResumeWorker(request_context=request_context)

    with (
        patch.object(
            worker.storage,
            "get_agent_run",
            side_effect=AssertionError("billing must not re-read the run"),
        ),
        patch(
            "reflexio.server.billing_meter.record_usage_event_strict",
            return_value=UsageEventDeliveryStatus.APPENDED,
        ) as record_event,
    ):
        worker._record_finalized_learnings(run, learning_ids, entity_type="profile")
        worker._record_finalized_learnings(run, learning_ids, entity_type="profile")

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
    _seed_interactions(storage)
    request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="Extract durable user facts.",
        ),
        user_playbook_extractor_config=PlaybookConfig(
            extraction_definition_prompt="Extract durable operating rules.",
        ),
        pending_tool_call_config=PendingToolCallConfig(enabled=True),
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
    agent_completed_at = datetime(2026, 8, 31, 23, 59, tzinfo=UTC)
    storage.create_agent_run(
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
            status=AgentRunStatus.FINALIZATION_FAILED,
            generation_request_snapshot={
                "request_id": request_id,
                "output_schema_name": "StructuredPlaybookList",
            },
            committed_output=committed_output,
            next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            finalization_attempts=2,
            agent_completed_at=agent_completed_at,
            created_at=run_created_at,
        )
    )
    stored_run = storage.get_agent_run(run_id)
    assert stored_run is not None
    assert stored_run.created_at is not None
    durable_run_created_at = stored_run.created_at
    worker = ExtractionResumeWorker(
        request_context=request_context,
        llm_client=MagicMock(),
    )
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
            schedule_tagging = stack.enter_context(
                patch.object(worker, "_schedule_finalized_tagging")
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

            failed = worker.run_once()
            assert failed is not None
            assert failed.status == AgentRunStatus.FINALIZATION_FAILED
            assert failed.finalization_attempts == 3
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

            storage.update_agent_run_status(
                run_id,
                AgentRunStatus.FINALIZATION_FAILED,
                next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            )
            retry = worker.run_once()

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
        schedule_tagging.assert_not_called()
        if optimize is not None and aggregate is not None:
            optimize.assert_called_once()
            aggregate.assert_called_once()
    finally:
        configure_usage_event_recorder(None)


@pytest.mark.parametrize("next_attempt_count", [1, 3, 4])
@pytest.mark.parametrize(
    ("delivery_status", "expected_status"),
    [
        (UsageEventDeliveryStatus.FAILED, AgentRunStatus.FINALIZATION_FAILED),
        (UsageEventDeliveryStatus.UNKNOWN, AgentRunStatus.FINALIZATION_FAILED),
        (UsageEventDeliveryStatus.REJECTED, AgentRunStatus.FAILED),
    ],
)
def test_receipt_delivery_failure_status_distinguishes_transient_and_permanent(
    delivery_status,
    next_attempt_count,
    expected_status,
):
    error = ReceiptBillingDeliveryError(delivery_status)

    assert (
        _finalization_failure_status(
            error,
            next_attempt_count=next_attempt_count,
            max_finalization_attempts=3,
        )
        is expected_status
    )


@pytest.mark.parametrize("next_attempt_count", [3, 4])
def test_ordinary_finalization_failure_stops_at_attempt_ceiling(next_attempt_count):
    assert (
        _finalization_failure_status(
            RuntimeError("ordinary finalization failed"),
            next_attempt_count=next_attempt_count,
            max_finalization_attempts=3,
        )
        is AgentRunStatus.FAILED
    )


def test_retry_after_billing_reuses_ids_without_replaying_playbook_schedulers(
    request_context,
    storage,
    monkeypatch,
):
    """A post-billing retry reuses durable IDs and skips derived schedulers."""
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")
    _seed_interactions(storage)
    request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="Extract durable user facts.",
        ),
        user_playbook_extractor_config=PlaybookConfig(
            extraction_definition_prompt="Extract durable operating rules.",
        ),
        pending_tool_call_config=PendingToolCallConfig(enabled=True),
    )
    events: list[UsageEvent] = []
    delivery_attempts: list[UsageEvent] = []
    accepted_keys: set[str] = set()

    def deduplicating_recorder(event: UsageEvent) -> UsageEventDeliveryStatus:
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
        worker = ExtractionResumeWorker(
            request_context=request_context,
            llm_client=MagicMock(),
        )
        with (
            patch.object(worker, "_schedule_finalized_tagging"),
            patch.object(
                storage,
                "consume_run_tool_dependencies",
                side_effect=[RuntimeError("failed after billing"), 0],
            ),
        ):
            first_attempt = worker.run_once()
            if assert_scheduler_calls is not None:
                assert_scheduler_calls()
            assert first_attempt is not None
            assert first_attempt.status == AgentRunStatus.FINALIZATION_FAILED
            storage.update_agent_run_status(
                run_id,
                AgentRunStatus.FINALIZATION_FAILED,
                next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            )
            retry = worker.run_once()
            if assert_scheduler_calls is not None:
                assert_scheduler_calls()
            assert retry is not None
            assert retry.status == AgentRunStatus.FINALIZED

    try:
        profile_request_id = "request_profile_retry"
        storage.create_agent_run(
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
                status=AgentRunStatus.FINALIZATION_FAILED,
                generation_request_snapshot={"request_id": profile_request_id},
                committed_output={
                    "profiles": [
                        {
                            "content": "User deployment target is AWS ECS.",
                            "time_to_live": "infinity",
                        }
                    ]
                },
                next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            )
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
        storage.create_agent_run(
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
                status=AgentRunStatus.FINALIZATION_FAILED,
                generation_request_snapshot={
                    "request_id": playbook_request_id,
                    "output_schema_name": "StructuredPlaybookList",
                },
                committed_output={
                    "playbooks": [
                        {
                            "content": "Prefer AWS ECS as the deployment target.",
                            "trigger": "when selecting a deployment target",
                            "rationale": "The team standardizes on AWS.",
                        }
                    ]
                },
                next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            )
        )
        consolidation = PlaybookConsolidationOutput(
            decisions=[
                UnifyDecision(
                    new_id="NEW-0",
                    archive_existing_ids=[0],
                    content="Prefer AWS ECS as the deployment target.",
                    trigger="when selecting a deployment target",
                    rationale="The team standardizes on AWS.",
                )
            ]
        )
        with (
            patch.object(
                PlaybookGenerationService,
                "_configured_playbook_config",
                return_value=None,
            ),
            patch(
                "reflexio.server.services.playbook.components.consolidator."
                "PlaybookConsolidator.retrieve_existing_playbooks",
                side_effect=lambda _new, **_kwargs: storage.get_user_playbooks(
                    user_id="user_1",
                    agent_version="v1",
                ),
            ),
            patch(
                "reflexio.server.services.playbook.components.consolidator."
                "PlaybookConsolidator._consolidation_decisions",
                return_value=consolidation,
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
    playbook_lineage_ids = [
        event.event_id
        for event in storage.get_lineage_events(request_id=playbook_request_id)
    ]
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
        "playbook_lineage_events": 1,
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


@pytest.mark.parametrize("failing_scheduler", ["optimizer", "aggregation"])
def test_playbook_scheduler_failure_preserves_billing_and_isolated_retry(
    storage,
    failing_scheduler,
):
    run = _finalizing_run(
        run_id=f"run_scheduler_failure_{failing_scheduler}",
        extractor_kind="playbook",
        request_id=f"request_scheduler_failure_{failing_scheduler}",
    )
    run = _persist_and_load_run(storage, run)
    worker = ExtractionResumeWorker(
        request_context=_finalization_context(storage),
        llm_client=MagicMock(),
    )
    candidates = _playbook_candidates(
        request_id=run.binding.request_id,
        prefix=failing_scheduler,
    )
    recorder = _DeduplicatingUsageRecorder()
    configure_usage_event_recorder(recorder)
    try:
        with (
            patch.object(
                PlaybookGenerationService,
                "_configured_playbook_config",
                return_value=None,
            ),
            patch(
                "reflexio.server.services.playbook.components.consolidator."
                "PlaybookConsolidator.deduplicate",
                side_effect=lambda results, *_args, **_kwargs: (
                    [playbook for result in results for playbook in result],
                    [],
                    [],
                ),
            ) as deduplicate,
            patch.object(
                PlaybookGenerationService,
                "_enqueue_user_playbook_optimization",
                side_effect=(
                    RuntimeError("optimizer unavailable")
                    if failing_scheduler == "optimizer"
                    else None
                ),
            ) as optimize,
            patch.object(
                PlaybookGenerationService,
                "_trigger_playbook_aggregation",
                side_effect=(
                    RuntimeError("aggregation unavailable")
                    if failing_scheduler == "aggregation"
                    else None
                ),
            ) as aggregate,
        ):
            winner = worker._finalize_items(run, candidates)
            retry = worker._finalize_items(run, candidates)
    finally:
        configure_usage_event_recorder(None)

    assert winner.won_receipt is True
    assert retry == FinalizationResult(winner.learning_ids, won_receipt=False)
    assert deduplicate.call_count == 1
    optimize.assert_called_once()
    aggregate.assert_called_once()
    billing_ids = [
        event.entity_id
        for event in recorder.events
        if event.event_name == "learnings_generated"
        and event.entity_type == "user_playbook"
    ]
    assert billing_ids == winner.learning_ids
    assert [event.entity_id for event in recorder.attempts] == [
        *winner.learning_ids,
        *winner.learning_ids,
    ]


def test_empty_profile_receipt_wins_once_without_billing_or_recompute(storage):
    run = _finalizing_run(
        run_id="run_empty_profile",
        extractor_kind="profile",
        request_id="request_empty_profile",
    )
    run = _persist_and_load_run(storage, run)
    context = _finalization_context(storage)
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    recorder = _DeduplicatingUsageRecorder()
    configure_usage_event_recorder(recorder)
    try:
        with patch.object(
            ProfileGenerationService,
            "_resolve_write_plan",
            return_value=None,
        ) as resolve:
            winner = worker._finalize_items(run, [])
            retry = worker._finalize_items(run, [])
            wrapper_ids = _profile_service(
                context, request_id=run.binding.request_id
            )._finalize_extracted_items([], finalization_run_id=run.id)
    finally:
        configure_usage_event_recorder(None)

    assert winner == FinalizationResult([], won_receipt=True)
    assert retry == FinalizationResult([], won_receipt=False)
    assert type(wrapper_ids) is list
    assert wrapper_ids == []
    resolve.assert_called_once()
    assert (
        storage.get_agent_run_finalization_receipt(run_id=run.id, entity_type="profile")
        == []
    )
    assert [
        event for event in recorder.events if event.event_name == "learnings_generated"
    ] == []


def test_empty_playbook_receipt_retries_without_redispatch(storage):
    run = AgentRunRecord(
        id="run_empty_playbook",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="playbook",
            user_id="user_1",
            request_id="request_empty_playbook",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.RESUME_READY,
        generation_request_snapshot={"output_schema_name": "StructuredPlaybookList"},
        committed_output={"playbooks": []},
        next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
    )
    run = _persist_and_load_run(storage, run)
    context = _finalization_context(storage)
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    original_finalize = worker._finalize_items
    outcomes: list[FinalizationResult] = []

    def tracked_finalize(*args, **kwargs) -> FinalizationResult:
        outcome = original_finalize(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    recorder = _DeduplicatingUsageRecorder()
    configure_usage_event_recorder(recorder)
    try:
        with (
            patch.object(storage, "claim_ready_agent_run", return_value=run),
            patch.object(
                worker, "_load_resolved_tool_calls", return_value=[MagicMock()]
            ),
            patch.object(worker, "_resume_run", return_value=([], [], None)),
            patch.object(
                worker, "_items_from_committed_output", return_value=([], [], None)
            ),
            patch.object(worker, "_finalize_items", side_effect=tracked_finalize),
            patch.object(
                PlaybookGenerationService,
                "_resolve_write_plan",
                return_value=None,
            ) as resolve,
            patch.object(
                PlaybookGenerationService,
                "_enqueue_user_playbook_optimization",
            ) as optimize,
            patch.object(
                PlaybookGenerationService,
                "_trigger_playbook_aggregation",
            ) as aggregate,
            patch(
                "reflexio.server.services.extraction.resume_worker.schedule_tagging"
            ) as schedule_tagging,
            patch.object(
                storage,
                "consume_run_tool_dependencies",
                side_effect=[RuntimeError("failed after finalization"), 0],
            ),
        ):
            failed = worker.run_once()
            assert failed is not None
            assert failed.status == AgentRunStatus.FINALIZATION_FAILED
            schedule_tagging.assert_called_once()
            storage.update_agent_run_status(
                run.id,
                AgentRunStatus.FINALIZATION_FAILED,
                next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
            )
            retried = worker.run_once()
            schedule_tagging.assert_called_once()
            wrapper_ids = _playbook_service(
                context, request_id=run.binding.request_id
            )._finalize_extracted_items([], finalization_run_id=run.id)
            schedule_tagging.assert_called_once()

        assert retried is not None
        assert retried.status == AgentRunStatus.FINALIZED
        assert outcomes == [
            FinalizationResult([], won_receipt=True),
            FinalizationResult([], won_receipt=False),
        ]
        assert type(wrapper_ids) is list
        assert wrapper_ids == []
        resolve.assert_called_once()
        optimize.assert_not_called()
        aggregate.assert_not_called()
        assert (
            storage.get_agent_run_finalization_receipt(
                run_id=run.id, entity_type="user_playbook"
            )
            == []
        )
        assert [
            event
            for event in recorder.events
            if event.event_name == "learnings_generated"
        ] == []
    finally:
        configure_usage_event_recorder(None)


def test_identical_profile_ids_use_atomic_receipt_owner(tmp_path):
    db_path = str(tmp_path / "identical-profile-receipt.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage_a = SQLiteStorage(org_id="org_1", db_path=db_path)
        storage_b = SQLiteStorage(org_id="org_1", db_path=db_path)
    run = _finalizing_run(
        run_id="run_identical_profile",
        extractor_kind="profile",
        request_id="request_identical_profile",
    )
    run = _persist_and_load_run(storage_a, run)
    contexts = [_finalization_context(item) for item in (storage_a, storage_b)]
    workers = [
        ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
        for context in contexts
    ]
    candidate_batches = [
        [
            UserProfile(
                profile_id="profile-shared",
                user_id="user_1",
                content=f"Profile content from attempt {index}.",
                last_modified_timestamp=1_000 + index,
                generated_from_request_id=run.binding.request_id,
            )
        ]
        for index in range(2)
    ]
    recorder = _DeduplicatingUsageRecorder()
    configure_usage_event_recorder(recorder)
    try:
        with patch(
            "reflexio.server.services.profile.components.consolidator."
            "ProfileConsolidator.deduplicate",
            side_effect=lambda profiles, _user_id, _request_id: (profiles, [], []),
        ):
            winner = workers[0]._finalize_items(run, candidate_batches[0])
            _hide_next_receipt_reads(storage_b)
            loser = workers[1]._finalize_items(run, candidate_batches[1])
    finally:
        configure_usage_event_recorder(None)

    assert winner == FinalizationResult(["profile-shared"], won_receipt=True)
    assert loser == FinalizationResult(["profile-shared"], won_receipt=False)
    assert [profile.content for profile in storage_a.get_user_profile("user_1")] == [
        "Profile content from attempt 0."
    ]
    assert [
        event.entity_id
        for event in recorder.events
        if event.event_name == "learnings_generated" and event.entity_type == "profile"
    ] == ["profile-shared"]
    assert [event.entity_id for event in recorder.attempts] == [
        "profile-shared",
        "profile-shared",
    ]


def test_identical_playbook_ids_use_atomic_receipt_owner(tmp_path):
    db_path = str(tmp_path / "identical-playbook-receipt.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage_a = SQLiteStorage(org_id="org_1", db_path=db_path)
        storage_b = SQLiteStorage(org_id="org_1", db_path=db_path)
    storage_a.conn.execute(
        "CREATE TABLE receipt_race_writes (attempt INTEGER NOT NULL, content TEXT NOT NULL)"
    )
    storage_a.conn.commit()
    run = _finalizing_run(
        run_id="run_identical_playbook",
        extractor_kind="playbook",
        request_id="request_identical_playbook",
    )
    run = _persist_and_load_run(storage_a, run)
    contexts = [_finalization_context(item) for item in (storage_a, storage_b)]
    workers = [
        ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
        for context in contexts
    ]
    candidate_batches = [
        _playbook_candidates(
            request_id=run.binding.request_id, prefix=f"attempt-{index}"
        )
        for index in range(2)
    ]
    persist_attempts = iter(range(2))

    def persist_fixed_ids(service, plan) -> None:
        attempt = next(persist_attempts)
        for index, playbook in enumerate(plan.new_playbooks):
            playbook.user_playbook_id = 88 + index
            service.storage.conn.execute(
                "INSERT INTO receipt_race_writes (attempt, content) VALUES (?, ?)",
                (attempt, playbook.content),
            )

    recorder = _DeduplicatingUsageRecorder()
    configure_usage_event_recorder(recorder)
    try:
        with (
            patch.object(
                PlaybookGenerationService,
                "_configured_playbook_config",
                return_value=None,
            ),
            patch(
                "reflexio.server.services.playbook.components.consolidator."
                "PlaybookConsolidator.deduplicate",
                side_effect=lambda results, *_args, **_kwargs: (
                    [playbook for result in results for playbook in result],
                    [],
                    [],
                ),
            ),
            patch.object(
                PlaybookGenerationService,
                "_persist_write_plan",
                autospec=True,
                side_effect=persist_fixed_ids,
            ),
            patch.object(
                PlaybookGenerationService,
                "_enqueue_user_playbook_optimization",
            ) as optimize,
            patch.object(
                PlaybookGenerationService,
                "_trigger_playbook_aggregation",
            ) as aggregate,
        ):
            winner = workers[0]._finalize_items(run, candidate_batches[0])
            _hide_next_receipt_reads(storage_b)
            loser = workers[1]._finalize_items(run, candidate_batches[1])
    finally:
        configure_usage_event_recorder(None)

    assert winner == FinalizationResult(["88", "89"], won_receipt=True)
    assert loser == FinalizationResult(["88", "89"], won_receipt=False)
    persisted_writes = storage_a.conn.execute(
        "SELECT attempt, content FROM receipt_race_writes ORDER BY content"
    ).fetchall()
    assert [(row["attempt"], row["content"]) for row in persisted_writes] == [
        (0, "Use deployment procedure attempt-0-0."),
        (0, "Use deployment procedure attempt-0-1."),
    ]
    optimize.assert_called_once()
    aggregate.assert_called_once()
    assert [
        event.entity_id
        for event in recorder.events
        if event.event_name == "learnings_generated"
        and event.entity_type == "user_playbook"
    ] == ["88", "89"]
    assert [event.entity_id for event in recorder.attempts] == ["88", "89", "88", "89"]


def test_two_stale_profile_workers_preserve_order_and_bill_only_winner(tmp_path):
    db_path = str(tmp_path / "stale-finalizers.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage_a = SQLiteStorage(org_id="org_1", db_path=db_path)
        storage_b = SQLiteStorage(org_id="org_1", db_path=db_path)
    run = _finalizing_run(
        run_id="run_profile_race",
        extractor_kind="profile",
        request_id="request_profile_race",
    )
    run = _persist_and_load_run(storage_a, run)
    _gate_initial_receipt_reads((storage_a, storage_b))
    contexts = [_finalization_context(storage) for storage in (storage_a, storage_b)]
    resume_workers = [
        ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
        for context in contexts
    ]
    candidate_batches = [
        [
            UserProfile(
                profile_id=f"profile-{worker_index}-{item_index}",
                user_id="user_1",
                content=f"Profile candidate {worker_index}-{item_index}.",
                last_modified_timestamp=1_000 + item_index,
                generated_from_request_id="request_profile_race",
            )
            for item_index in range(2)
        ]
        for worker_index in range(2)
    ]
    outcomes: list[FinalizationResult] = []
    errors: list[BaseException] = []

    def finalize(index: int) -> None:
        try:
            outcomes.append(
                resume_workers[index]._finalize_items(run, candidate_batches[index])
            )
        except BaseException as exc:  # noqa: BLE001 - intentional thread error capture
            errors.append(exc)

    recorder = _DeduplicatingUsageRecorder()
    configure_usage_event_recorder(recorder)
    try:
        with patch(
            "reflexio.server.services.profile.components.consolidator."
            "ProfileConsolidator.deduplicate",
            side_effect=lambda profiles, _user_id, _request_id: (profiles, [], []),
        ):
            workers = [
                threading.Thread(target=finalize, args=(index,)) for index in range(2)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10)
    finally:
        configure_usage_event_recorder(None)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    persisted_ids = [
        profile.profile_id for profile in storage_a.get_user_profile("user_1")
    ]
    receipt_ids = storage_a.get_agent_run_finalization_receipt(
        run_id=run.id, entity_type="profile"
    )
    assert len(persisted_ids) == 2
    assert receipt_ids == persisted_ids
    assert [outcome.won_receipt for outcome in outcomes].count(True) == 1
    assert [outcome.won_receipt for outcome in outcomes].count(False) == 1
    assert all(outcome.learning_ids == receipt_ids for outcome in outcomes)
    assert receipt_ids in [
        [profile.profile_id for profile in batch] for batch in candidate_batches
    ]
    wrapper_ids = _profile_service(
        contexts[0], request_id=run.binding.request_id
    )._finalize_extracted_items(
        candidate_batches[1],
        finalization_run_id=run.id,
    )
    assert type(wrapper_ids) is list
    assert wrapper_ids == receipt_ids
    billing_events = [
        event
        for event in recorder.events
        if event.event_name == "learnings_generated" and event.entity_type == "profile"
    ]
    assert [event.entity_id for event in billing_events] == persisted_ids
    assert [event.entity_id for event in recorder.attempts] == [
        *persisted_ids,
        *persisted_ids,
    ]


def test_two_stale_playbook_workers_preserve_order_and_dispatch_once(tmp_path):
    db_path = str(tmp_path / "stale-playbook-finalizers.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage_a = SQLiteStorage(org_id="org_1", db_path=db_path)
        storage_b = SQLiteStorage(org_id="org_1", db_path=db_path)
    run = _finalizing_run(
        run_id="run_playbook_race",
        extractor_kind="playbook",
        request_id="request_playbook_race",
    )
    run = _persist_and_load_run(storage_a, run)
    _gate_initial_receipt_reads((storage_a, storage_b))
    contexts = [_finalization_context(storage) for storage in (storage_a, storage_b)]
    resume_workers = [
        ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
        for context in contexts
    ]
    candidate_batches = [
        _playbook_candidates(
            request_id=run.binding.request_id,
            prefix=f"worker-{worker_index}",
        )
        for worker_index in range(2)
    ]
    outcomes: list[FinalizationResult] = []
    errors: list[BaseException] = []

    def finalize(index: int) -> None:
        try:
            outcomes.append(
                resume_workers[index]._finalize_items(run, candidate_batches[index])
            )
        except BaseException as exc:  # noqa: BLE001 - intentional thread error capture
            errors.append(exc)

    recorder = _DeduplicatingUsageRecorder()
    configure_usage_event_recorder(recorder)
    try:
        with (
            patch.object(
                PlaybookGenerationService,
                "_configured_playbook_config",
                return_value=None,
            ),
            patch(
                "reflexio.server.services.playbook.components.consolidator."
                "PlaybookConsolidator.deduplicate",
                side_effect=lambda results, *_args, **_kwargs: (
                    [playbook for result in results for playbook in result],
                    [],
                    [],
                ),
            ),
            patch.object(
                PlaybookGenerationService,
                "_enqueue_user_playbook_optimization",
            ) as optimize,
            patch.object(
                PlaybookGenerationService,
                "_trigger_playbook_aggregation",
            ) as aggregate,
        ):
            threads = [
                threading.Thread(target=finalize, args=(index,)) for index in range(2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            assert all(not thread.is_alive() for thread in threads)
            assert errors == []
            receipt_ids = storage_a.get_agent_run_finalization_receipt(
                run_id=run.id,
                entity_type="user_playbook",
            )
            persisted_ids = [
                str(row[0])
                for row in storage_a.conn.execute(
                    "SELECT user_playbook_id FROM user_playbooks "
                    "WHERE request_id = ? ORDER BY user_playbook_id ASC",
                    (run.binding.request_id,),
                ).fetchall()
            ]
            wrapper_ids = _playbook_service(
                contexts[0], request_id=run.binding.request_id
            )._finalize_extracted_items(
                candidate_batches[1],
                finalization_run_id=run.id,
            )

            assert len(persisted_ids) == 2
            assert receipt_ids == persisted_ids
            assert [outcome.won_receipt for outcome in outcomes].count(True) == 1
            assert [outcome.won_receipt for outcome in outcomes].count(False) == 1
            assert all(outcome.learning_ids == receipt_ids for outcome in outcomes)
            assert type(wrapper_ids) is list
            assert wrapper_ids == receipt_ids
            optimize.assert_called_once()
            aggregate.assert_called_once()
            billing_ids = [
                event.entity_id
                for event in recorder.events
                if event.event_name == "learnings_generated"
                and event.entity_type == "user_playbook"
            ]
            assert billing_ids == receipt_ids
            assert [event.entity_id for event in recorder.attempts] == [
                *receipt_ids,
                *receipt_ids,
            ]
    finally:
        configure_usage_event_recorder(None)


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
    worker = ExtractionResumeWorker(request_context=request_context)

    with patch(
        "litellm.completion",
        side_effect=AssertionError("exhausted budget must not call LLM"),
    ):
        resumed = worker.drain(max_runs=1)

    assert resumed == 1
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FAILED
    assert run.last_error == "Resumable extraction max-step budget exhausted"
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is None
