from __future__ import annotations

import json
import tempfile
import threading
from collections.abc import Callable
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
from reflexio.server.services.deferred_learning_plan import FinalizationResult
from reflexio.server.services.extraction.resume_worker import (
    ExtractionResumeWorker,
    _run_playbook_contract_selection,
    _run_uses_strict_playbook_evidence,
)
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidationOutput,
    UnifyDecision,
)
from reflexio.server.services.playbook.service import PlaybookGenerationService
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
    configure_usage_event_recorder,
)


@pytest.fixture
def storage():
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id="org_1", db_path=f"{temp_dir}/reflexio.db")


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
        pending_tool_call_config=PendingToolCallConfig(enabled=True),
    )
    ctx.configurator.get_agent_context.return_value = "Test agent context"
    ctx.prompt_manager = MagicMock()
    ctx.prompt_manager.render_prompt.side_effect = lambda prompt_id, variables: (
        f"{prompt_id}: {variables}"
    )
    return ctx


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
    )
    learning_ids = ["profile_1"]
    worker = ExtractionResumeWorker(request_context=request_context)

    with patch("reflexio.server.billing_meter.record_usage_event") as record_event:
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


def test_retry_after_billing_reuses_ids_without_replaying_playbook_schedulers(
    request_context,
    storage,
):
    """A post-billing retry reuses durable IDs and skips derived schedulers."""
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
    configure_usage_event_recorder(events.append)

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


def test_two_stale_sqlite_workers_bill_only_the_receipt_winner(tmp_path):
    db_path = str(tmp_path / "stale-finalizers.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage_a = SQLiteStorage(org_id="org_1", db_path=db_path)
        storage_b = SQLiteStorage(org_id="org_1", db_path=db_path)
    storage_a.create_agent_run(
        AgentRunRecord(
            id="run_race",
            binding=AgentBinding(
                org_id="org_1",
                extractor_kind="profile",
                user_id="user_1",
                request_id="request_race",
                agent_version="v1",
                source="api",
            ),
            status=AgentRunStatus.FINALIZING,
            generation_request_snapshot={"request_id": "request_race"},
        )
    )
    barrier = threading.Barrier(2)

    def _context(storage_instance: SQLiteStorage) -> RequestContext:
        context = RequestContext.__new__(RequestContext)
        context.org_id = "org_1"
        context.storage = storage_instance
        context.configurator = MagicMock()
        context.configurator.get_config.return_value = Config(
            storage_config=StorageConfigSQLite(),
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="Extract durable user facts.",
            ),
        )
        return context

    def _gate_first_receipt_read(storage_instance: SQLiteStorage) -> None:
        original = storage_instance.get_agent_run_finalization_receipt
        first_read = True

        def gated_read(*, run_id: str, entity_type: str) -> list[str] | None:
            nonlocal first_read
            receipt = original(run_id=run_id, entity_type=entity_type)
            if first_read:
                first_read = False
                barrier.wait(timeout=5)
            return receipt

        storage_instance.get_agent_run_finalization_receipt = MagicMock(  # type: ignore[method-assign]
            side_effect=gated_read
        )

    _gate_first_receipt_read(storage_a)
    _gate_first_receipt_read(storage_b)
    resume_workers = [
        ExtractionResumeWorker(
            request_context=_context(storage), llm_client=MagicMock()
        )
        for storage in (storage_a, storage_b)
    ]
    run = AgentRunRecord(
        id="run_race",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="profile",
            user_id="user_1",
            request_id="request_race",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZING,
        generation_request_snapshot={"request_id": "request_race"},
    )
    candidates = [
        UserProfile(
            profile_id=f"profile-race-{index}",
            user_id="user_1",
            content="User deployment target is AWS ECS.",
            last_modified_timestamp=1_000,
            generated_from_request_id="request_race",
        )
        for index in range(2)
    ]
    errors: list[BaseException] = []

    def finalize(index: int) -> None:
        try:
            resume_workers[index]._finalize_items(run, [candidates[index]])
        except BaseException as exc:  # noqa: BLE001 - intentional thread error capture
            errors.append(exc)

    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
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
        run_id="run_race", entity_type="profile"
    )
    assert len(persisted_ids) == 1
    assert receipt_ids == persisted_ids
    billing_events = [
        event
        for event in events
        if event.event_name == "learnings_generated" and event.entity_type == "profile"
    ]
    assert [event.entity_id for event in billing_events] == persisted_ids


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
