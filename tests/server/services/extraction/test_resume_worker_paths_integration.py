"""Resume-worker paths against real SQLite storage with a mocked LLM.

Covers the playbook resume path end to end, the retryable failure exits of a
resume, finalization retries without committed output, recovery refusals, and
the receipt/tagging guards. Each test owns its database and org id.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    ToolUseConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.tools import ToolLoopTrace
from reflexio.server.metering_events import ReceiptDeliveryError
from reflexio.server.services.deferred_learning_plan import FinalizationResult
from reflexio.server.services.extraction.recovery import RecoveryRefusedError
from reflexio.server.services.extraction.resumable_agent import AgentRunResult
from reflexio.server.services.extraction.resume_worker import (
    ExtractionResumeWorker,
    _finalization_failure_status,
    _run_playbook_contract_selection,
)
from reflexio.server.services.storage.error import ReadinessUnavailableError
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
    UsageEventDeliveryStatus,
    configure_usage_event_recorder,
    exempt_usage_event_recorder,
)

pytestmark = pytest.mark.integration

PLAYBOOK_FINALIZE = (
    "reflexio.server.services.playbook.service."
    "PlaybookGenerationService._finalize_extracted_items_with_outcome"
)
PROFILE_FINALIZE = (
    "reflexio.server.services.profile.service."
    "ProfileGenerationService._finalize_extracted_items_with_outcome"
)


@pytest.fixture(autouse=True)
def recorded_tagging(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resume_worker.schedule_tagging",
        lambda **kwargs: calls.append(kwargs),
    )
    return calls


@pytest.fixture(autouse=True)
def explicit_test_billing_exemption() -> Iterator[None]:
    configure_usage_event_recorder(exempt_usage_event_recorder)
    try:
        yield
    finally:
        configure_usage_event_recorder(None)


@pytest.fixture
def org_id() -> str:
    return f"org_resume_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def storage(tmp_path, org_id) -> Iterator[SQLiteStorage]:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id=org_id, db_path=str(tmp_path / "reflexio.db"))


def _context(
    storage: SQLiteStorage,
    org_id: str,
    *,
    profile: bool = True,
    playbook: bool = True,
    tool_can_use: list[ToolUseConfig] | None = None,
    max_resume_attempts: int | None = None,
) -> RequestContext:
    pending = PendingToolCallConfig(enabled=True, max_finalization_attempts=3)
    if max_resume_attempts is not None:
        pending = pending.model_copy(
            update={"max_resume_attempts": max_resume_attempts}
        )
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = org_id
    ctx.storage = storage
    ctx.storage_base_dir = None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="Extract durable user deployment facts.",
        )
        if profile
        else None,
        user_playbook_extractor_config=PlaybookConfig(
            extraction_definition_prompt="Extract durable deployment rules.",
        )
        if playbook
        else None,
        tool_can_use=tool_can_use or [],
        pending_tool_call_config=pending,
    )
    ctx.configurator.get_agent_context.return_value = "Test agent context"
    ctx.prompt_manager = MagicMock()
    ctx.prompt_manager.render_prompt.side_effect = lambda prompt_id, variables: (
        f"{prompt_id}: {variables}"
    )
    ctx.prompt_manager.get_active_version.return_value = None
    return ctx


def _seed_interactions(
    storage: SQLiteStorage, *, with_request: bool = True, expert: bool = False
) -> None:
    if with_request:
        storage.add_request(
            Request(
                request_id="request_1",
                user_id="user_1",
                created_at=1_000,
                source="api",
                agent_version="v1",
                session_id="session_1",
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
            expert_content="Always deploy to ECS after a staging rehearsal."
            if expert
            else "",
        )
    )


def _seed_ready_run(
    storage: SQLiteStorage,
    org_id: str,
    *,
    kind: str,
    snapshot: dict | None = None,
    source_ids: tuple[int, ...] = (1, 2),
    user_id: str | None = "user_1",
    config_hash: str = "",
) -> str:
    run_id = f"run_{kind}"
    storage.create_agent_run(
        AgentRunRecord(
            id=run_id,
            binding=AgentBinding(
                org_id=org_id,
                extractor_kind=kind,
                user_id=user_id,
                request_id="request_1",
                agent_version="v1",
                source="api",
                source_interaction_ids=list(source_ids),
                window_start_interaction_id=min(source_ids, default=None),
                window_end_interaction_id=max(source_ids, default=None),
                extractor_config_hash=config_hash,
            ),
            status=AgentRunStatus.FINALIZED_PENDING_TOOL,
            generation_request_snapshot=snapshot or {"request_id": "request_1"},
        )
    )
    now = datetime.now(UTC)
    question = f"What is the deployment target for {kind}?"
    scope = human_feedback_scope(org_id)
    call_id = f"ptc_{kind}"
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id=call_id,
            org_id=org_id,
            user_id="user_1",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human", question_text=question
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=question,
            args={"question": question},
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(minutes=5),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id=run_id, pending_tool_call_id=call_id)
    )
    storage.resolve_pending_tool_call(
        call_id,
        result={"answer": "Use AWS ECS."},
        resolved_at=now,
        valid_for_seconds=3600,
    )
    return run_id


def _playbook_answer(make_stop) -> object:
    return make_stop(
        json.dumps(
            {
                "playbooks": [
                    {
                        "rationale": "The user named the deployment standard.",
                        "trigger": "When choosing a deployment target",
                        "content": "Deploy to AWS ECS.",
                    }
                ]
            }
        )
    )


def _binding(
    org_id: str, kind: str, user_id: str | None, **extra: object
) -> AgentBinding:
    return AgentBinding(
        org_id=org_id,
        extractor_kind=kind,
        user_id=user_id,
        request_id="request_1",
        agent_version="v1",
        source="api",
        **extra,  # type: ignore[arg-type]
    )


def _unfinished(reason: str) -> AgentRunResult:
    return AgentRunResult(
        run_id="unused",
        output=None,
        pending_tool_call_ids=[],
        messages=[],
        trace=ToolLoopTrace(),
        finished_reason=reason,
    )


@pytest.fixture(autouse=True)
def provider_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)


# ---------------------------------------------------------------------------
# Playbook resume end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expert", [False, True])
def test_playbook_resume_finalizes_extracted_rule_and_consumes_dependency(
    storage, org_id, tool_call_completion, recorded_tagging, expert
):
    _seed_interactions(storage, expert=expert)
    run_id = _seed_ready_run(
        storage,
        org_id,
        kind="playbook",
        snapshot={
            "request_id": "request_1",
            "output_schema_name": "StructuredPlaybookList",
        },
    )
    ctx = _context(
        storage,
        org_id,
        tool_can_use=[
            ToolUseConfig(tool_name="deploy", tool_description="Ships a release")
        ],
    )
    _, make_stop = tool_call_completion
    worker = ExtractionResumeWorker(request_context=ctx)

    with (
        patch("litellm.completion", side_effect=[_playbook_answer(make_stop)]) as llm,
        patch(
            PLAYBOOK_FINALIZE,
            return_value=FinalizationResult(["pb-1"], won_receipt=True),
        ) as finalize,
    ):
        assert worker.drain(max_runs=5) == 1

    prompt = json.dumps(llm.call_args.kwargs["messages"])
    if expert:
        assert "playbook_extraction_context_expert" in prompt
    else:
        assert "deploy: Ships a release" in prompt
    items = finalize.call_args.args[0]
    assert [item.content for item in items] == ["Deploy to AWS ECS."]
    assert items[0].trigger == "When choosing a deployment target"
    assert finalize.call_args.kwargs["finalization_run_id"] == run_id
    run = storage.get_agent_run(run_id)
    assert run is not None and run.status == AgentRunStatus.FINALIZED
    assert storage.list_run_tool_dependencies(run_id)[0].consumed_at is not None
    assert [call["user_id"] for call in recorded_tagging] == ["user_1"]


def test_strict_playbook_resume_keeps_cited_source_turns(
    storage, org_id, tool_call_completion
):
    _seed_interactions(storage)
    run_id = _seed_ready_run(
        storage,
        org_id,
        kind="playbook",
        snapshot={"output_schema_name": "StructuredReferencedExtractedPlaybookList"},
    )
    _, make_stop = tool_call_completion
    answer = make_stop(
        json.dumps(
            {
                "playbooks": [
                    {
                        "rationale": "The user asked which target to use.",
                        "evidence_kind": "preference",
                        "trigger": "When choosing a deployment target",
                        "content": "Ask for the deployment standard first.",
                        "evidence_refs": ["T1"],
                    }
                ]
            }
        )
    )
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))

    with (
        patch("litellm.completion", side_effect=[answer]) as llm,
        patch(
            PLAYBOOK_FINALIZE, return_value=FinalizationResult([], won_receipt=False)
        ) as finalize,
    ):
        assert worker.drain(max_runs=1) == 1

    assert "[T1]" in json.dumps(llm.call_args.kwargs["messages"])
    (item,) = finalize.call_args.args[0]
    assert item.content == "Ask for the deployment standard first."
    assert item.source_interaction_ids == [1]
    run = storage.get_agent_run(run_id)
    assert run is not None and run.status == AgentRunStatus.FINALIZED


def test_playbook_resume_that_does_not_finish_returns_run_to_retry_queue(
    storage, org_id
):
    _seed_interactions(storage)
    run_id = _seed_ready_run(
        storage,
        org_id,
        kind="playbook",
        snapshot={"output_schema_name": "StructuredPlaybookList"},
    )
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))
    unfinished = _unfinished("max_steps")

    with (
        patch(
            "reflexio.server.services.extraction.resume_worker."
            "ResumableExtractionAgent.resume",
            return_value=unfinished,
        ),
        patch(PLAYBOOK_FINALIZE) as finalize,
    ):
        assert worker.drain(max_runs=1) == 1

    finalize.assert_not_called()
    run = storage.get_agent_run(run_id)
    assert run is not None
    assert run.status == AgentRunStatus.RESUME_READY
    assert run.last_error == "Playbook resume did not finish: max_steps"
    assert run.next_resume_at is not None
    assert storage.list_run_tool_dependencies(run_id)[0].consumed_at is None


def test_profile_resume_that_does_not_finish_fails_at_attempt_ceiling(storage, org_id):
    _seed_interactions(storage)
    run_id = _seed_ready_run(storage, org_id, kind="profile")
    worker = ExtractionResumeWorker(
        request_context=_context(storage, org_id, max_resume_attempts=1)
    )
    unfinished = _unfinished("refused")

    with patch(
        "reflexio.server.services.extraction.resume_worker."
        "ResumableExtractionAgent.resume",
        return_value=unfinished,
    ):
        assert worker.drain(max_runs=1) == 1

    run = storage.get_agent_run(run_id)
    assert run is not None
    assert run.status == AgentRunStatus.FAILED
    assert run.last_error == "Profile resume did not finish: refused"


# ---------------------------------------------------------------------------
# Resume failures that keep the run retryable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile_configured", "source_ids", "expected_error"),
    [
        (False, (1, 2), "Current extractor config not found for profile"),
        (True, (), "No source interactions found for resumed run"),
    ],
)
def test_resume_without_config_or_sources_is_retried_without_llm(
    storage, org_id, profile_configured, source_ids, expected_error
):
    _seed_interactions(storage)
    run_id = _seed_ready_run(storage, org_id, kind="profile", source_ids=source_ids)
    worker = ExtractionResumeWorker(
        request_context=_context(storage, org_id, profile=profile_configured)
    )

    with patch(
        "litellm.completion", side_effect=AssertionError("must not call the LLM")
    ):
        assert worker.drain(max_runs=1) == 1

    run = storage.get_agent_run(run_id)
    assert run is not None
    assert run.status == AgentRunStatus.RESUME_READY
    assert run.last_error == expected_error


def test_resume_of_unsupported_extractor_kind_is_retried_without_llm(storage, org_id):
    _seed_interactions(storage)
    run_id = _seed_ready_run(storage, org_id, kind="custom")
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))

    with patch(
        "litellm.completion", side_effect=AssertionError("must not call the LLM")
    ):
        assert worker.drain(max_runs=1) == 1

    run = storage.get_agent_run(run_id)
    assert run is not None
    assert run.status == AgentRunStatus.RESUME_READY
    assert run.last_error == "Unsupported extractor kind 'custom'"


def test_resume_feeds_only_resolved_answers_and_ignores_open_questions(
    storage, org_id, tool_call_completion
):
    _seed_interactions(storage)
    run_id = _seed_ready_run(storage, org_id, kind="profile")
    now = datetime.now(UTC)
    scope = human_feedback_scope(org_id)
    open_question = "Which region should we prefer?"
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_open",
            org_id=org_id,
            user_id="user_1",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human", question_text=open_question
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=open_question,
            args={"question": open_question},
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(minutes=5),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id=run_id, pending_tool_call_id="ptc_open")
    )
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))
    with patch.object(
        worker,
        "_resume_run",
        return_value=([], [], None),
    ) as resume:
        assert worker.drain(max_runs=1) == 1

    resolved_calls = resume.call_args.args[1]
    assert [call.id for call in resolved_calls] == ["ptc_profile"]
    assert resolved_calls[0].result == {"answer": "Use AWS ECS."}


def test_resume_reconstructs_request_when_request_row_is_missing(
    storage, org_id, tool_call_completion
):
    _seed_interactions(storage, with_request=False)
    run_id = _seed_ready_run(storage, org_id, kind="profile", config_hash="stale")
    _, make_stop = tool_call_completion
    answer = make_stop(
        json.dumps(
            {"profiles": [{"content": "Deploys to ECS.", "time_to_live": "infinity"}]}
        )
    )
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))

    with (
        patch("litellm.completion", side_effect=[answer]) as llm,
        patch(
            PROFILE_FINALIZE, return_value=FinalizationResult([], won_receipt=False)
        ) as finalize,
    ):
        assert worker.drain(max_runs=1) == 1

    # The fallback request carries the run's source and the request id as the
    # session, so both interactions still reach the prompt.
    prompt = json.dumps(llm.call_args.kwargs["messages"])
    assert "Which deployment target should we use?" in prompt
    assert "I need the deployment standard." in prompt
    assert [p.content for p in finalize.call_args.args[0]] == ["Deploys to ECS."]
    run = storage.get_agent_run(run_id)
    assert run is not None and run.status == AgentRunStatus.FINALIZED


def test_resume_start_lost_to_another_owner_leaves_run_untouched(storage, org_id):
    _seed_interactions(storage)
    run_id = _seed_ready_run(storage, org_id, kind="profile")
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))

    with (
        patch.object(storage, "begin_extraction_resume", return_value=None),
        patch(
            "litellm.completion", side_effect=AssertionError("must not call the LLM")
        ),
    ):
        returned = worker.run_once()

    assert returned is not None and returned.id == run_id
    run = storage.get_agent_run(run_id)
    assert run is not None
    assert run.status == AgentRunStatus.RESUMING
    assert run.resume_attempts == 0
    assert storage.list_run_tool_dependencies(run_id)[0].consumed_at is None


def test_claim_lost_before_finalizing_skips_finalization(storage, org_id):
    _seed_interactions(storage)
    run_id = _seed_ready_run(storage, org_id, kind="profile")
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))

    def resume_then_lose_claim(run, _calls):
        # Another worker reclaims the run while this one is still resuming.
        with storage._lock:
            storage.conn.execute(
                "UPDATE _agent_runs SET claimed_by = 'other-worker' WHERE id = ?",
                (run.id,),
            )
            storage.conn.commit()
        return [], [], None

    with (
        patch.object(worker, "_resume_run", side_effect=resume_then_lose_claim),
        patch(PROFILE_FINALIZE) as finalize,
    ):
        assert worker.run_once() is not None

    finalize.assert_not_called()
    run = storage.get_agent_run(run_id)
    assert run is not None
    assert run.status == AgentRunStatus.RESUMING
    assert run.claimed_by == "other-worker"
    assert storage.list_run_tool_dependencies(run_id)[0].consumed_at is None


def test_drain_stops_when_no_run_is_ready(storage, org_id):
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))
    assert worker.drain(max_runs=3) == 0


# ---------------------------------------------------------------------------
# Finalization retries
# ---------------------------------------------------------------------------


def test_finalization_failure_status_treats_readiness_as_retryable():
    status = _finalization_failure_status(
        ReadinessUnavailableError("conversion pending"),
        next_attempt_count=99,
        max_finalization_attempts=3,
    )
    assert status == AgentRunStatus.FINALIZATION_FAILED


# ---------------------------------------------------------------------------
# Selected recovery refusals
# ---------------------------------------------------------------------------


def _seed_recoverable_run(storage: SQLiteStorage, org_id: str) -> AgentRunRecord:
    storage.create_agent_run(
        AgentRunRecord(
            id="run_recover",
            binding=_binding(
                org_id, "profile", "user_1", source_interaction_ids=[1, 2]
            ),
            status=AgentRunStatus.AGENT_COMPLETED,
            generation_request_snapshot={"request_id": "request_1"},
        )
    )
    failed = storage.update_agent_run_status(
        "run_recover",
        AgentRunStatus.FINALIZATION_FAILED,
        committed_output={"profiles": []},
        last_error="previous failure",
    )
    assert failed is not None
    return failed


def _selection(org_id: str, run_id: str, updated_at=None) -> SimpleNamespace:
    return SimpleNamespace(
        org_id=org_id, project_id=None, run_id=run_id, updated_at=updated_at
    )


def test_recovery_refuses_another_organization(storage, org_id):
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))
    with pytest.raises(RecoveryRefusedError, match="organization_mismatch"):
        worker.recover_selected(_selection("other-org", "run_x"), operation_id="op")


def test_recovery_refuses_unknown_run(storage, org_id):
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))
    with pytest.raises(RecoveryRefusedError, match="run_not_found"):
        worker.recover_selected(_selection(org_id, "missing"), operation_id="op")


def test_recovery_refuses_when_the_previewed_row_changed(storage, org_id):
    _seed_interactions(storage)
    run_id = _seed_recoverable_run(storage, org_id).id
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))
    stale = datetime(2000, 1, 1, tzinfo=UTC)

    with (
        patch(
            "reflexio.server.services.extraction.recovery.validate_selection",
            return_value=None,
        ),
        pytest.raises(RecoveryRefusedError, match="claim_conflict"),
    ):
        worker.recover_selected(_selection(org_id, run_id, stale), operation_id="op")

    run = storage.get_agent_run(run_id)
    assert run is not None
    assert run.recovery_operation_id is None
    assert run.status == AgentRunStatus.FINALIZATION_FAILED
    assert run.claimed_by is None


def test_recovery_that_does_not_reach_finalized_is_not_verified(storage, org_id):
    _seed_interactions(storage)
    current = _seed_recoverable_run(storage, org_id)
    run_id = current.id
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))

    def unverified(claimed, **_kwargs):
        return replace(claimed, status=AgentRunStatus.FINALIZATION_FAILED)

    with (
        patch(
            "reflexio.server.services.extraction.recovery.validate_selection",
            return_value=None,
        ),
        patch.object(worker, "_execute_claimed_run", side_effect=unverified),
        pytest.raises(RecoveryRefusedError, match="recovery_not_verified"),
    ):
        worker.recover_selected(
            _selection(org_id, run_id, current.updated_at), operation_id="op-1"
        )

    run = storage.get_agent_run(run_id)
    assert run is not None
    assert run.recovery_operation_id == "op-1"
    assert run.status == AgentRunStatus.FINALIZING


# ---------------------------------------------------------------------------
# Contract selection, receipts and tagging guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("active_version", "expert", "expected"),
    [
        (None, False, (False, False)),
        ("4.6.0", False, (True, True)),
        ("3.6.0", True, (True, False)),
    ],
)
def test_legacy_run_without_schema_snapshot_uses_active_prompt_version(
    storage, org_id, active_version, expert, expected
):
    ctx = _context(storage, org_id)
    prompt_manager = MagicMock()
    prompt_manager.get_active_version.return_value = active_version
    ctx.prompt_manager = prompt_manager
    run = AgentRunRecord(
        id="legacy",
        binding=_binding(org_id, "playbook", "user_1"),
        status=AgentRunStatus.RESUMING,
        generation_request_snapshot={},
    )

    assert _run_playbook_contract_selection(ctx, run, expert=expert) == expected
    prompt_manager.get_active_version.assert_called_once_with(
        "playbook_extraction_context_expert"
        if expert
        else "playbook_extraction_context"
    )


def test_learning_receipt_without_durable_timestamp_is_not_delivered(storage, org_id):
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))
    run = AgentRunRecord(
        id="no_timestamp",
        binding=_binding(org_id, "profile", "u"),
        status=AgentRunStatus.FINALIZING,
        generation_request_snapshot={},
    )

    with patch(
        "reflexio.server.services.extraction.resume_worker."
        "emit_learnings_generated_records_strict"
    ) as emit:
        with pytest.raises(ReceiptDeliveryError) as raised:
            worker._record_finalized_learnings(run, ["p-1"], entity_type="profile")
        worker._record_finalized_learnings(run, [], entity_type="profile")

    assert raised.value.status is UsageEventDeliveryStatus.UNKNOWN
    emit.assert_not_called()


@pytest.mark.parametrize(
    ("user_id", "kind"), [(None, "profile"), ("", "playbook"), ("u", "custom")]
)
def test_tagging_is_scheduled_only_for_owned_profile_or_playbook_runs(
    storage, org_id, recorded_tagging, user_id, kind
):
    worker = ExtractionResumeWorker(request_context=_context(storage, org_id))
    run = AgentRunRecord(
        id="tagging",
        binding=_binding(org_id, kind, user_id),
        status=AgentRunStatus.FINALIZED,
        generation_request_snapshot={},
    )
    worker._schedule_finalized_tagging(run)
    assert recorded_tagging == []
