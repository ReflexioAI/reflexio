"""Real HTTP/storage/executor boundaries with controlled external model calls."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api import create_app
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.profile.service import ProfileGenerationService

pytestmark = pytest.mark.integration


def _prepared_presentation_text() -> str:
    from reflexio.server.services.extraction.overflow_presentation import (
        current_overflow_presentation,
    )

    presentation = current_overflow_presentation.get()
    assert presentation is not None
    return presentation.text


@pytest.mark.parametrize("configuration_error", [False, True])
def test_input_budget_failure_reports_failed_without_immediate_replay_or_losing_inputs(
    pipeline, monkeypatch, configuration_error
):
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError
    from reflexio.server.llm.input_budget import (
        InputBudgetConfigurationError,
        InputBudgetExceededError,
    )
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    engine, worker, client = pipeline
    storage = engine.get_storage()
    with storage._stream_sql(read_only=True) as db:
        started = db.now()
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: started)
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    dispatch = Mock(
        side_effect=AssertionError("Guard failure must not reach a provider")
    )
    monkeypatch.setattr(
        "reflexio.server.llm.litellm_client.LiteLLMClient._completion_with_hard_timeout",
        dispatch,
    )
    reason = (
        "input_budget_configuration_error"
        if configuration_error
        else "input_budget_exceeded"
    )
    budget = (
        InputBudgetConfigurationError("Sensitive details are not a status reason")
        if configuration_error
        else InputBudgetExceededError("Sensitive details are not a status reason")
    )
    wrapped = ProviderRequestGuardError("Guard rejected the operation")
    wrapped.__cause__ = budget
    compute = Mock(side_effect=wrapped)
    monkeypatch.setattr(ProfileGenerationService, "compute_generation", compute)
    assert client.post(
        "/api/publish_interaction", json=payload("budget-failure", force=True)
    ).json()["success"]
    assert worker.drain_org("stream-test", 1, 300) == 0
    assert client.get(
        "/api/learning_status", params={"request_id": "budget-failure"}
    ).json() == {"status": "failed", "reason": reason}
    storage = engine.get_storage()
    assert storage.count_all_profiles() == 0
    assert len(storage.get_user_interaction("u")) == 10
    with storage._stream_sql() as db:
        rows = db.query("SELECT id FROM _agent_runs WHERE user_id=? LIMIT 2", ("u",))
    assert len(rows) == 1
    run = storage.get_agent_run(rows[0]["id"])
    assert run.status.value == "failed"
    assert run.last_error == reason
    assert run.generation_request_snapshot["overflow"]["attempts"] == []
    worker.drain_org("stream-test", 2, 300)
    assert compute.call_count == 1
    dispatch.assert_not_called()
    assert storage.pending_extraction_effects() == []


def _overflow_coordinator(pipeline, *, attempts=(), actual_source=False, legacy=False):
    from reflexio.server.services.storage.storage_base import (
        AgentBinding,
        AgentRunRecord,
        AgentRunStatus,
    )

    engine, _, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("overflow-checkpoint", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("checkpoint-test", 300)
    window = storage.prepare_extraction(user, token)
    source_digest = "captured-source"
    source_digests = {}
    if actual_source:
        from reflexio.server.services.durable_learning.window_executor import (
            load_window_inputs,
        )
        from reflexio.server.services.extraction.agent_run_records import (
            source_interaction_digest,
        )
        from reflexio.server.services.extraction.input_preparation import (
            prepare_extraction_input,
        )

        groups = load_window_inputs(storage, window)
        source_digest = prepare_extraction_input(groups, user_id=user).source_digest
        source_digests = {
            str(item.interaction_id): source_interaction_digest(item)
            for group in groups
            for item in group.interactions
        }
    storage.create_agent_run(
        AgentRunRecord(
            id=f"window:{window.window_id}",
            binding=AgentBinding(
                org_id=storage.org_id,
                user_id=user,
                extractor_kind=window.kind,
                request_id="overflow-checkpoint",
                agent_version=None,
                source="test",
            ),
            status=AgentRunStatus.AGENT_COMPLETED if legacy else AgentRunStatus.RUNNING,
            committed_output={"profiles": []} if legacy else None,
            generation_request_snapshot={
                "source_interaction_ids": [
                    item["interaction_id"] for item in window.manifest
                ],
                "source_interaction_digests": source_digests,
                **(
                    {}
                    if legacy
                    else {
                        "overflow": {
                            "version": 1,
                            "revision": 0,
                            "source_digest": source_digest,
                            "attempt_limit": 128,
                            "attempts": list(attempts),
                            "checkpoint": {"stage": "prepare", "completed_steps": {}},
                        }
                    }
                ),
            },
        )
    )
    return storage, window, token


@pytest.mark.parametrize(
    "change", ["result", "question", "withdraw", "supersede", "expire", "delete"]
)
def test_overflow_prior_answer_is_complete_pinned_and_fenced(
    pipeline, monkeypatch, change
):
    from datetime import UTC, datetime, timedelta

    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.extraction.prior_answer_search import (
        append_prior_knowledge_context,
    )
    from reflexio.server.services.storage.storage_base import (
        PendingToolCallRecord,
        PendingToolCallStatus,
        PriorAnswerMatch,
        build_scope_hash,
        human_feedback_scope,
    )

    storage, window, token = _overflow_coordinator(pipeline)
    run = storage.get_agent_run(f"window:{window.window_id}")
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        run.generation_request_snapshot["overflow"],
        source_digest="captured-source",
        context_limits={},
    )
    now = datetime.now(UTC)
    answer = {
        "answer": "Evidence " * 300
        + "Only prepare; do not deploy until separately approved."
    }
    scope = human_feedback_scope(storage.org_id)
    record = storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_overflow_answer",
            org_id=storage.org_id,
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key="overflow-answer",
            status=PendingToolCallStatus.RESOLVED,
            question_text="May I deploy?",
            result=answer,
            resolved_at=now,
            expires_at=now + timedelta(days=1),
            cache_until=now + timedelta(hours=1),
            valid_until=now + timedelta(hours=1),
        )
    )
    matches = Mock(
        return_value=[
            PriorAnswerMatch(
                pending_tool_call_id=record.id,
                status=record.status,
                question_text=record.question_text,
                result=record.result,
                resolved_at=record.resolved_at,
                valid_until=record.valid_until,
            )
        ]
    )
    monkeypatch.setattr(
        "reflexio.server.services.extraction.prior_answer_search.find_prior_answer_matches",
        matches,
    )
    kwargs = {
        "messages": [{"role": "user", "content": "original source"}],
        "storage": storage,
        "org_id": storage.org_id,
        "extractor_kind": "profile",
        "extractor_config": None,
        "source": None,
        "agent_version": None,
    }
    with coordinator.provider_scope():
        first = append_prior_knowledge_context(**kwargs)
    assert (
        "Only prepare; do not deploy until separately approved." in first[-1]["content"]
    )
    witnesses = coordinator.checkpoint["answer_witnesses"]
    with storage.commit_scope():
        storage.validate_extraction_answers(witnesses)
    mutations = {
        "result": (
            "UPDATE _pending_tool_calls SET result=? WHERE id=?",
            ('{"answer":"Deploy now"}', record.id),
        ),
        "question": (
            "UPDATE _pending_tool_calls SET question_text=? WHERE id=?",
            ("Changed scope", record.id),
        ),
        "withdraw": (
            "UPDATE _pending_tool_calls SET status=? WHERE id=?",
            ("cancelled", record.id),
        ),
        "supersede": (
            "UPDATE _pending_tool_calls SET superseded_by=? WHERE id=?",
            ("new-answer", record.id),
        ),
        "expire": (
            "UPDATE _pending_tool_calls SET valid_until=? WHERE id=?",
            ((now - timedelta(seconds=1)).isoformat(), record.id),
        ),
        "delete": ("DELETE FROM _pending_tool_calls WHERE id=?", (record.id,)),
    }
    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(*mutations[change])
    with coordinator.provider_scope():
        second = append_prior_knowledge_context(**kwargs)
    assert second == first
    matches.assert_called_once()
    with (
        storage.commit_scope(),
        pytest.raises(InputBudgetConfigurationError, match="answer changed"),
    ):
        storage.validate_extraction_answers(witnesses)
    assert coordinator.checkpoint["answer_witnesses"] == witnesses
    assert coordinator.checkpoint["stage"] == "prepare"
    assert storage.count_all_profiles() == 0


@pytest.mark.parametrize("answer_before_checkpoint", [False, True])
@pytest.mark.parametrize("checkpoint_fault", [False, True])
def test_waiting_overflow_child_resumes_saved_messages_and_consumes_only_its_answer(
    pipeline, monkeypatch, answer_before_checkpoint, checkpoint_fault
):
    from datetime import UTC, datetime, timedelta

    from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        prepare_extraction_input,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
        OverflowSuspendedError,
    )
    from reflexio.server.services.extraction.overflow_presentation import (
        present_extraction_chunk,
    )
    from reflexio.server.services.extraction.resumable_agent import (
        ResumableExtractionAgent,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )
    from reflexio.server.services.storage.storage_base import (
        PendingToolCallRecord,
        PendingToolCallStatus,
        RunToolDependencyRecord,
        build_scope_hash,
        human_feedback_scope,
    )

    storage, window, token = _overflow_coordinator(pipeline, actual_source=True)
    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    run = storage.get_agent_run(f"window:{window.window_id}")
    now = datetime.now(UTC)
    scope = human_feedback_scope(storage.org_id)
    for call_id in ("child-answer", "unrelated-answer"):
        storage.create_pending_tool_call(
            PendingToolCallRecord(
                id=call_id,
                org_id=storage.org_id,
                scope=scope,
                scope_hash=build_scope_hash(scope),
                tool_name="ask_human",
                dedup_key=call_id,
                status=PendingToolCallStatus.PENDING,
                question_text="What is approved?",
                expires_at=now + timedelta(days=1),
                cache_until=now + timedelta(hours=1),
            )
        )
        storage.attach_run_tool_dependency(
            RunToolDependencyRecord(run_id=run.id, pending_tool_call_id=call_id)
        )
    qualifier = "Only prepare; production deployment needs separate approval."

    def resolve(call_id):
        storage.resolve_pending_tool_call(
            call_id, result={"answer": qualifier}, valid_for_seconds=3600
        )

    resolve("unrelated-answer")
    if answer_before_checkpoint:
        resolve("child-answer")

    def coordinator_from_storage():
        parent = storage.get_agent_run(run.id)
        return OverflowCoordinator(
            storage,
            window,
            token,
            parent.generation_request_snapshot["overflow"],
            source_digest=prepared.source_digest,
            context_limits={},
            prepared=prepared,
        )

    coordinator = coordinator_from_storage()
    agent = ResumableExtractionAgent(client=Mock(), storage=storage)
    progress = []

    def wait_loop(**kwargs):
        progress.extend(
            [
                *kwargs["messages"],
                {
                    "role": "tool",
                    "content": "saved tool output",
                    "tool_call_id": "lookup",
                },
            ]
        )
        return ToolLoopResult(
            ctx=None,
            trace=ToolLoopTrace(),
            finished_reason="max_steps",
            messages=progress,
            pending_tool_call_ids=["child-answer"],
        )

    loop = Mock(side_effect=wait_loop)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", loop
    )
    with (
        coordinator.provider_scope(),
        present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record),
    ):
        seed = [{"role": "user", "content": _prepared_presentation_text()}]
        with pytest.raises(OverflowSuspendedError):
            agent.start(run=run, messages=seed, output_schema=StructuredProfilesOutput)
    assert coordinator.checkpoint["active_child"]["max_steps_remaining"] == 7
    assert storage.get_agent_run(run.id).committed_output is None
    if not answer_before_checkpoint:
        with (
            coordinator.provider_scope(),
            present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record),
            pytest.raises(OverflowSuspendedError),
        ):
            agent.start(run=run, messages=seed, output_schema=StructuredProfilesOutput)
        assert loop.call_count == 1
        resolve("child-answer")

    if checkpoint_fault:
        original_save = storage.checkpoint_extraction_overflow

        def fail_after_write(*args, **kwargs):
            result = original_save(*args, **kwargs)
            if kwargs.get("checkpoint", {}).get("answer_dependencies"):
                raise RuntimeError("Injected checkpoint failure")
            return result

        monkeypatch.setattr(storage, "checkpoint_extraction_overflow", fail_after_write)
        with (
            coordinator.provider_scope(),
            present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record),
            pytest.raises(RuntimeError, match="Injected checkpoint failure"),
        ):
            agent.start(run=run, messages=seed, output_schema=StructuredProfilesOutput)
        monkeypatch.setattr(storage, "checkpoint_extraction_overflow", original_save)
        with storage._stream_sql() as db:
            assert (
                db.query(
                    "SELECT consumed_at FROM _run_tool_dependencies WHERE pending_tool_call_id=?",
                    ("child-answer",),
                )[0]["consumed_at"]
                is None
            )
        assert "waiting_child" in coordinator_from_storage().checkpoint
        coordinator = coordinator_from_storage()

    completion = ToolLoopResult(
        ctx=None,
        trace=ToolLoopTrace(finished=True),
        finished_reason="structured_output",
        messages=[],
        structured_output=StructuredProfilesOutput(profiles=[]),
    )
    loop.side_effect = None
    loop.return_value = completion
    with (
        coordinator.provider_scope(),
        present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record),
    ):
        result = agent.start(
            run=run, messages=seed, output_schema=StructuredProfilesOutput
        )
        replay = agent.start(
            run=run, messages=seed, output_schema=StructuredProfilesOutput
        )
    assert result.output == replay.output == StructuredProfilesOutput(profiles=[])
    assert loop.call_count == 2
    resumed = loop.call_args.kwargs["messages"]
    assert resumed[:-1] == progress
    assert qualifier in resumed[-1]["content"]
    assert "waiting_child" not in coordinator.checkpoint
    with storage.commit_scope():
        storage.validate_extraction_answers(coordinator.checkpoint["answer_witnesses"])
        storage.validate_extraction_answer_dependencies(
            window, coordinator.checkpoint["answer_dependencies"]
        )
    with storage._stream_sql() as db:
        deps = {
            row["pending_tool_call_id"]: row["consumed_at"]
            for row in db.query(
                "SELECT pending_tool_call_id,consumed_at FROM _run_tool_dependencies WHERE run_id=?",
                (run.id,),
            )
        }
    assert deps["child-answer"] is not None
    assert deps["unrelated-answer"] is None
    assert storage.get_agent_run(run.id).committed_output is None
    assert storage.count_all_profiles() == 0
    assert storage.pending_extraction_effects() == []


def test_overflow_human_tools_wait_for_pending_and_reuse_complete_resolved_answers(
    pipeline,
):
    from reflexio.server.llm.tools import AsyncAccepted, Completed, ToolRegistry
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.extraction.pending_tool_call_dispatch import (
        PendingToolCallToolContext,
        create_ask_human_tool,
        create_attach_pending_info_request_tool,
    )

    storage, window, token = _overflow_coordinator(pipeline)
    run = storage.get_agent_run(f"window:{window.window_id}")
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        run.generation_request_snapshot["overflow"],
        source_digest="captured-source",
        context_limits={},
    )
    ctx = PendingToolCallToolContext(
        storage=storage,
        run_id=run.id,
        org_id=storage.org_id,
        extractor_kind="profile",
        user_id=window.user_id,
    )
    registry = ToolRegistry(
        [create_ask_human_tool(), create_attach_pending_info_request_tool()]
    )
    with coordinator.provider_scope():
        requested = registry.handle_outcome(
            "ask_human", '{"question":"What is approved?"}', ctx
        )
        assert isinstance(requested, AsyncAccepted)
        call_id = requested.pending_tool_call_id
        attached = registry.handle_outcome(
            "attach_pending_info_request",
            '{"pending_tool_call_id":"' + call_id + '"}',
            ctx,
        )
        assert isinstance(attached, AsyncAccepted)
        assert attached.pending_tool_call_id == call_id
        answer = {
            "answer": "Reasoning " * 300
            + "Only prepare, never deploy without approval."
        }
        storage.resolve_pending_tool_call(
            call_id, result=answer, valid_for_seconds=3600
        )
        reused = registry.handle_outcome(
            "attach_pending_info_request",
            '{"pending_tool_call_id":"' + call_id + '"}',
            ctx,
        )
        assert isinstance(reused, Completed)
        assert reused.result["result"] == answer
        assert coordinator.checkpoint["answer_witnesses"][call_id]["result"] == answer
    assert storage.get_agent_run(run.id).committed_output is None
    assert storage.count_all_profiles() == 0
    assert storage.pending_extraction_effects() == []


@pytest.mark.parametrize("cached_gate", [False, True])
@pytest.mark.parametrize(
    "cached_children,reserved,reconcile_fault",
    [(31, 79, False), (48, 127, False), (48, 127, True)],
)
def test_48_chunk_restart_counts_only_uncached_children(
    pipeline, monkeypatch, cached_gate, cached_children, reserved, reconcile_fault
):
    from dataclasses import replace

    from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
    from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.chunk_gate import judge_extraction_chunks
    from reflexio.server.services.extraction.input_preparation import (
        PreparedChunk,
        PreparedExtractionInput,
        prepare_extraction_input,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.extraction.overflow_pipeline import (
        OverflowExtractionCall,
        extract_overflow_window,
    )
    from reflexio.server.services.extraction.overflow_presentation import (
        present_extraction_chunk,
    )
    from reflexio.server.services.extraction.resumable_agent import (
        ResumableExtractionAgent,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    attempts = tuple(
        {"id": f"earlier-{i}", "usage": None, "model_provenance": None}
        for i in range(reserved)
    )
    storage, window, token = _overflow_coordinator(
        pipeline, actual_source=True, attempts=attempts
    )
    original = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    spans = [span for chunk in original.chunks for span in chunk.spans]
    assert len(spans) == 10
    slices = []
    for index, span in enumerate(spans):
        count = 5 if index < 8 else 4
        for part in range(count):
            start = len(span.text) * part // count
            end = len(span.text) * (part + 1) // count
            slices.append(
                replace(span, start=start, end=end, text=span.text[start:end])
            )
    prepared = PreparedExtractionInput(
        original.source_digest,
        tuple(PreparedChunk(i, (span,)) for i, span in enumerate(slices)),
    )
    assert len(prepared.chunks) == 48
    parent = storage.get_agent_run(f"window:{window.window_id}")
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest=prepared.source_digest,
        context_limits={},
        prepared=prepared,
    )
    policy = coordinator.step_key(
        "extraction-policy-v1",
        {
            "messages": [
                {"role": "system", "content": "contract"},
                {"role": "user", "content": "<prepared-source>"},
            ],
            "schema": StructuredProfilesOutput.model_json_schema(),
        },
    )
    checkpoint = coordinator.checkpoint
    checkpoint["extraction_policy"] = policy
    coordinator.save_checkpoint(checkpoint)
    provenance = ModelProvenance("observed", "fixture")

    def build_call():
        return OverflowExtractionCall(
            messages=[
                {"role": "system", "content": "contract"},
                {"role": "user", "content": _prepared_presentation_text()},
            ],
            schema=StructuredProfilesOutput,
        )

    for chunk in prepared.chunks[:cached_children]:
        with present_extraction_chunk(prepared, chunk, coordinator.record):
            call = build_call()
        key = coordinator.step_key(
            "extract-v1",
            {
                "messages": call.messages,
                "schema": StructuredProfilesOutput.model_json_schema(),
                "tools": [],
            },
        )
        coordinator.register_extraction_child(chunk.text, key)
        coordinator.complete_step(
            key,
            {
                "output": StructuredProfilesOutput.model_validate(
                    {
                        "profiles": [
                            {
                                "content": "shared lesson",
                                "time_to_live": "infinity",
                                "source_span": chunk.spans[0].text,
                            }
                        ]
                        if reconcile_fault
                        else [],
                    }
                ).model_dump(mode="json"),
                "trace": {"turns": [], "finished": True},
            },
            provenance,
        )
    if cached_gate:
        with present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record):
            gate_prompt = _prepared_presentation_text()
        gate_key = coordinator.step_key(
            "gate-v1",
            {
                "model": "fixture",
                "messages": [{"role": "user", "content": gate_prompt}],
            },
        )
        coordinator.complete_step(gate_key, True, provenance)
    parent = storage.get_agent_run(parent.id)
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest=prepared.source_digest,
        context_limits={},
        prepared=prepared,
    )
    agent = ResumableExtractionAgent(client=Mock(), storage=storage)

    def complete_uncached(**kwargs):
        coordinator._advance(reserve_attempt_id=f"continued-{loop.call_count}")
        return ToolLoopResult(
            ctx=None,
            trace=ToolLoopTrace(finished=True),
            finished_reason="structured_output",
            messages=kwargs["messages"],
            structured_output=StructuredProfilesOutput(profiles=[]),
            provenance=provenance,
        )

    loop = Mock(side_effect=complete_uncached)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", loop
    )
    gate_client = Mock()
    reconciliation_client = Mock()

    def reconcile(**kwargs):
        coordinator._advance(reserve_attempt_id="reconciliation-after-127")
        return CompletionResult(
            StructuredProfilesOutput.model_validate(
                {
                    "profiles": [
                        {
                            "content": "shared lesson",
                            "time_to_live": "infinity",
                            "source_span": prepared.chunks[0].spans[0].text,
                        }
                    ],
                }
            ),
            provenance,
        )

    reconciliation_client.generate_chat_response_with_provenance.side_effect = reconcile

    def extract():
        return extract_overflow_window(
            coordinator=coordinator,
            client=reconciliation_client,
            build_call=build_call,
            execute_call=lambda call: agent.start(
                run=parent, messages=call.messages, output_schema=call.schema
            ),
            final_validator=None,
        )

    if reconcile_fault:
        original_update = storage.update_agent_run_status

        def fault_after_parent_write(*args, **kwargs):
            original_update(*args, **kwargs)
            raise RuntimeError("Injected parent completion failure")

        monkeypatch.setattr(
            storage, "update_agent_run_status", fault_after_parent_write
        )
        with (
            coordinator.provider_scope(),
            pytest.raises(RuntimeError, match="parent completion failure"),
        ):
            extract()
        monkeypatch.setattr(storage, "update_agent_run_status", original_update)
        parent = storage.get_agent_run(parent.id)
        assert parent.committed_output is None
        assert len(parent.generation_request_snapshot["overflow"]["attempts"]) == 128
        coordinator = OverflowCoordinator(
            storage,
            window,
            token,
            parent.generation_request_snapshot["overflow"],
            source_digest=prepared.source_digest,
            context_limits={},
            prepared=prepared,
        )
    with coordinator.provider_scope():
        if cached_gate:
            assert judge_extraction_chunks(
                coordinator=coordinator,
                prepared=prepared,
                record=coordinator.record,
                client=gate_client,
                model="fixture",
                build_prompt=lambda: _prepared_presentation_text(),
                minimum_after_positive=51,
            )
        result = extract()
    assert result.output == StructuredProfilesOutput.model_validate(
        {
            "profiles": [
                {
                    "content": "shared lesson",
                    "time_to_live": "infinity",
                    "source_span": prepared.chunks[0].spans[0].text,
                }
            ]
            if reconcile_fault
            else [],
        }
    )
    assert (
        reconciliation_client.generate_chat_response_with_provenance.call_count
        == int(reconcile_fault)
    )
    assert loop.call_count == 48 - cached_children
    gate_client.generate_chat_response_with_provenance.assert_not_called()
    journal = storage.get_agent_run(parent.id).generation_request_snapshot["overflow"]
    assert len(journal["attempts"]) == reserved + 48 - cached_children + int(
        reconcile_fault
    )
    assert len(journal["checkpoint"]["source_coverage"]) == 48
    assert coordinator.unfinished_extraction_chunks(prepared) == 0
    assert storage.count_all_profiles() == 0


@pytest.mark.parametrize(
    "change",
    [
        "content",
        "source_span",
        "source_ids",
        "notes",
        "status",
        "user",
        "version",
        "retire",
        "delete",
    ],
)
def test_historical_playbook_fence_keeps_full_evidence_and_rejects_stale_rows(
    pipeline, change
):
    import json
    from datetime import UTC, datetime

    from reflexio.models.api_schema.service_schemas import UserPlaybook
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )

    storage, window, token = _overflow_coordinator(pipeline)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest="captured-source",
        context_limits={},
    )
    now = datetime.now(UTC)
    span = (
        "Stored historical tool evidence " * 4000
        + "Prepare only; separate approval is needed to deploy."
    )
    source_ids = [window.manifest[0]["interaction_id"]]
    with storage.commit_scope(), storage._stream_sql() as db:
        playbook_id = db.query(
            "INSERT INTO user_playbooks (user_id,playbook_name,created_at,request_id,agent_version,content,trigger,rationale,source,source_interaction_ids,source_span,notes,reader_angle,tags,expanded_terms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) RETURNING user_playbook_id",
            (
                window.user_id,
                "deployment",
                now.isoformat(),
                "overflow-checkpoint",
                "v1",
                "Prepare only",
                "when planning",
                "Require explicit approval",
                "test",
                json.dumps(source_ids),
                span,
                "Original note",
                "instruction",
                '["control"]',
                "planning approval",
            ),
        )[0]["user_playbook_id"]
    historical = UserPlaybook(
        user_playbook_id=playbook_id,
        user_id=window.user_id,
        playbook_name="deployment",
        created_at=int(now.timestamp()),
        request_id="overflow-checkpoint",
        agent_version="v1",
        content="Prepare only",
        trigger="when planning",
        rationale="Require explicit approval",
        source="test",
        source_interaction_ids=source_ids,
        source_span=span,
        notes="Original note",
        reader_angle="instruction",
        tags=["control"],
        expanded_terms="planning approval",
    )
    coordinator.remember_historical_playbooks([historical])
    witnesses = coordinator.checkpoint["historical_playbooks"]
    assert len(witnesses[str(playbook_id)]) == 64
    assert len(json.dumps(coordinator.checkpoint)) < 2048
    assert historical.source_span == span
    with storage.commit_scope():
        storage.validate_extraction_historical_playbooks(window, witnesses)
    mutations = {
        "content": (
            "UPDATE user_playbooks SET content=? WHERE user_playbook_id=?",
            ("Deploy now", playbook_id),
        ),
        "source_span": (
            "UPDATE user_playbooks SET source_span=? WHERE user_playbook_id=?",
            (span + " Changed final condition", playbook_id),
        ),
        "source_ids": (
            "UPDATE user_playbooks SET source_interaction_ids=? WHERE user_playbook_id=?",
            ("[]", playbook_id),
        ),
        "notes": (
            "UPDATE user_playbooks SET notes=? WHERE user_playbook_id=?",
            ("Changed control note", playbook_id),
        ),
        "status": (
            "UPDATE user_playbooks SET status=? WHERE user_playbook_id=?",
            ("archived", playbook_id),
        ),
        "user": (
            "UPDATE user_playbooks SET user_id=? WHERE user_playbook_id=?",
            ("another-user", playbook_id),
        ),
        "version": (
            "UPDATE user_playbooks SET agent_version=? WHERE user_playbook_id=?",
            ("v2", playbook_id),
        ),
        "retire": (
            "UPDATE user_playbooks SET retired_at=? WHERE user_playbook_id=?",
            (int(now.timestamp()), playbook_id),
        ),
        "delete": (
            "DELETE FROM user_playbooks WHERE user_playbook_id=?",
            (playbook_id,),
        ),
    }
    with storage.commit_scope(), storage._stream_sql() as db:
        db.query(*mutations[change])
    with (
        storage.commit_scope(),
        pytest.raises(
            InputBudgetConfigurationError, match="Historical playbook changed"
        ),
    ):
        storage.validate_extraction_historical_playbooks(window, witnesses)
    assert coordinator.checkpoint["historical_playbooks"] == witnesses
    assert historical.source_span == span
    assert storage.count_all_profiles() == 0
    assert storage.pending_extraction_effects() == []


@pytest.mark.parametrize("verdict", ["verified", "unsafe", "undetermined"])
def test_historical_consolidation_scans_complete_source_and_replays_without_calls(
    pipeline, monkeypatch, verdict
):
    import json

    from reflexio.models.api_schema.service_schemas import UserPlaybook
    from reflexio.models.config_schema import DeduplicationConfig
    from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.llm.token_accounting import RunTokenTotals
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        prepare_extraction_input,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.extraction.protected_record import (
        ProtectedEvidence,
        ProtectedRecord,
        ProtectedRecordUpdate,
    )
    from reflexio.server.services.playbook.components.consolidator import (
        PlaybookConsolidationOutput,
        PlaybookConsolidator,
        UnifyDecision,
    )
    from reflexio.server.services.playbook.components.overflow_consolidation import (
        ConsolidationVerification,
        HistoricalScanSelection,
        verified_consolidation_decisions,
    )

    storage, window, token = _overflow_coordinator(pipeline, actual_source=True)
    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    span = prepared.chunks[0].spans[0]
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(span.evidence_ref,),
                    quote=span.text,
                    role=span.role,
                    source_order=span.source_order,
                )
            ]
        ),
        visible_spans=(span,),
    )
    old_text = (
        "tool result with no new policy. " * 12000
        + "Deployment needs separate approval."
    )
    old = UserPlaybook(
        user_playbook_id=51,
        user_id=window.user_id,
        agent_version="v1",
        request_id="older",
        content="Prepare only. Deployment needs separate approval.",
        trigger="after preparation",
        rationale="Preserve approval scope",
        source_span=old_text,
        source_interaction_ids=[9001],
        notes="Original note",
    )
    new = old.model_copy(
        update={
            "user_playbook_id": 0,
            "request_id": "current",
            "source_interaction_ids": [span.interaction_id],
            "source_span": span.text,
        }
    )
    output = PlaybookConsolidationOutput(
        decisions=[
            UnifyDecision(
                new_id="NEW-0",
                archive_existing_ids=[0],
                content=old.content,
                trigger=old.trigger or "",
                rationale=old.rationale or "",
            )
        ]
    )
    owner = PlaybookConsolidator.__new__(PlaybookConsolidator)
    owner.model_name = "gpt-4o"
    owner.model_provenance = None
    owner._dedup_config = DeduplicationConfig()
    owner.consolidated_output_indices = set()
    owner.request_context = pipeline[0].request_context
    calls = []
    primary_ranges = []
    active = None
    provenance = ModelProvenance(model_name="gpt-4o", provider="openai")

    def dispatch(**kwargs):
        assert active is not None
        calls.append(kwargs["response_format"])
        active._guard(
            {
                "model": "gpt-4o",
                "messages": kwargs["messages"],
                "max_tokens": 4096,
                "response_format": kwargs["response_format"],
            },
            1,
            (),
            True,
        )
        active._observe(
            provenance, RunTokenTotals(prompt_tokens=100, completion_tokens=10)
        )
        payload_data = json.loads(kwargs["messages"][-1]["content"])
        if kwargs["response_format"] is HistoricalScanSelection:
            chunk = payload_data["chunk"]
            primary_ranges.append((chunk["primary_start"], chunk["primary_end"]))
            assert (
                "".join(item["text"] for item in chunk["snippets"])
                == old_text[chunk["start"] : chunk["end"]]
            )
            assert payload_data["author_role"] == "unknown"
            quote = "Deployment needs separate approval."
            result = HistoricalScanSelection(
                coverage="complete",
                snippet_ids=[
                    item["snippet_id"]
                    for item in chunk["snippets"]
                    if quote in item["text"]
                ],
            )
            assert kwargs["structured_output_validator"](result) == []
        else:
            assert kwargs["response_format"] is ConsolidationVerification
            assert payload_data["historical_evidence"][0]["quotes"][-1][
                "text"
            ].endswith("Deployment needs separate approval.")
            assert old_text not in kwargs["messages"][-1]["content"]
            result = ConsolidationVerification(
                verdict=verdict, reason="fixture verdict"
            )
        return CompletionResult(value=result, provenance=provenance)

    owner.client = Mock(
        generate_chat_response_with_provenance=Mock(side_effect=dispatch)
    )
    monkeypatch.setattr(
        owner, "_render_consolidation_prompt", lambda *_: "metadata-only proposal"
    )

    def propose(*_):
        assert active is not None
        active._guard(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "metadata-only proposal"}],
                "max_tokens": 4096,
            },
            1,
            (),
            True,
        )
        active._observe(
            provenance, RunTokenTotals(prompt_tokens=100, completion_tokens=10)
        )
        owner.model_provenance = provenance
        calls.append("proposal")
        return output

    monkeypatch.setattr(owner, "_consolidation_decisions", propose)

    def replay():
        nonlocal active
        parent = storage.get_agent_run(f"window:{window.window_id}")
        active = OverflowCoordinator(
            storage,
            window,
            token,
            parent.generation_request_snapshot["overflow"],
            source_digest=prepared.source_digest,
            context_limits={"gpt-4o": 65536},
            prepared=prepared,
        )
        active.candidate_records[active.playbook_key(new)] = record
        active.remember_historical_playbooks([old])
        with active.provider_scope():
            return verified_consolidation_decisions(owner, active, [new], [old])

    if verdict == "verified":
        assert replay() == output
    else:
        with pytest.raises(
            InputBudgetConfigurationError, match="unsafe or undetermined"
        ):
            replay()
    assert len(primary_ranges) > 1
    assert primary_ranges[0][0] == 0 and primary_ranges[-1][1] == len(old_text)
    assert all(
        a[1] == b[0] for a, b in zip(primary_ranges, primary_ranges[1:], strict=False)
    )
    first_calls = len(calls)
    assert active is not None
    first_usage = active.token_totals.prompt_tokens
    # Fully completed history and verification remain reusable even when all
    # provider capacity has since been reserved. No speculative call is needed.
    current = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    for index in range(len(current["attempts"]), 128):
        current = storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=current["revision"],
            source_digest=prepared.source_digest,
            reserve_attempt_id=f"reserved-{index}",
        )
    if verdict == "verified":
        assert replay() == output
        rows, archived, groups = owner._build_deduplicated_results(
            new_playbooks=[new],
            existing_playbooks=[old],
            dedup_output=output,
            generation_request_id="current",
            agent_version="v1",
        )
        assert archived == [51] and groups == [(0, [51])]
        assert rows[0].source_span is not None
        assert old_text in rows[0].source_span
        assert 9001 in rows[0].source_interaction_ids
    else:
        with pytest.raises(InputBudgetConfigurationError):
            replay()
    assert len(calls) == first_calls
    assert active is not None
    assert active.token_totals.prompt_tokens == first_usage
    assert len(active.checkpoint["historical_chunks"]) == 1
    assert old.source_span == old_text and old.notes == "Original note"
    assert storage.count_all_profiles() == 0
    assert storage.pending_extraction_effects() == []


@pytest.mark.parametrize("observed", [False, True])
def test_saved_legacy_usage_is_fenced_and_added_once_without_fake_old_attempts(
    pipeline, observed
):
    import json
    from dataclasses import asdict

    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
        legacy_observed_usage,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        OverflowCheckpointConflictError,
    )

    storage, window, token = _overflow_coordinator(
        pipeline, actual_source=True, legacy=True
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    payload_data = dict(parent.committed_output)
    payload_data["trace"] = {
        "finished": True,
        "turns": [
            {
                "tool_name": "structured_output",
                "args": {},
                "result": {},
                "latency_ms": 1,
                "prompt_tokens": 100 if observed else None,
                "completion_tokens": 20 if observed else None,
            }
        ],
    }
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET committed_output=? WHERE id=?",
            (json.dumps(payload_data), parent.id),
        )
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        prepare_extraction_input,
    )

    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    manifest = OverflowCoordinator.initial_manifest(prepared.source_digest)
    witness = legacy_observed_usage(payload_data)
    if witness is not None:
        manifest["legacy_observed_usage"] = witness
    incorrect = dict(
        manifest,
        legacy_observed_usage={
            "prompt_tokens": 999,
            "completion_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_write_input_tokens": 0,
        },
    )
    with (
        storage.commit_scope(),
        pytest.raises(OverflowCheckpointConflictError, match="usage changed"),
    ):
        storage.adopt_extraction_overflow(
            window,
            token,
            source_digest=prepared.source_digest,
            source_digests=parent.generation_request_snapshot[
                "source_interaction_digests"
            ],
            manifest=incorrect,
        )
    with storage.commit_scope():
        adopted = storage.adopt_extraction_overflow(
            window,
            token,
            source_digest=prepared.source_digest,
            source_digests=parent.generation_request_snapshot[
                "source_interaction_digests"
            ],
            manifest=manifest,
        )
    assert adopted["attempts"] == []
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        adopted,
        source_digest=prepared.source_digest,
        context_limits={},
    )
    assert coordinator.token_totals.prompt_tokens == (100 if observed else 0)
    coordinator._advance(reserve_attempt_id="new-observation")
    coordinator._advance(
        observed_attempt={
            "id": "new-observation",
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "cache_read_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
            "model_provenance": {"model_name": "test"},
        }
    )
    saved = storage.get_agent_run(parent.id).generation_request_snapshot["overflow"]
    resumed = OverflowCoordinator(
        storage,
        window,
        token,
        saved,
        source_digest=prepared.source_digest,
        context_limits={},
    )
    assert len(saved["attempts"]) == 1
    assert asdict(resumed.token_totals) == asdict(coordinator.token_totals)
    assert resumed.token_totals.prompt_tokens == (107 if observed else 7)
    assert resumed.token_totals.completion_tokens == (23 if observed else 3)


@pytest.mark.parametrize("changed_policy", [False, True])
def test_cached_overflow_output_preserves_child_models_and_prompt_contract(
    pipeline, changed_policy
):
    from dataclasses import asdict, replace

    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        PreparedChunk,
        PreparedExtractionInput,
        prepare_extraction_input,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.extraction.overflow_pipeline import (
        OverflowExtractionCall,
        extract_overflow_window,
    )
    from reflexio.server.services.extraction.overflow_presentation import (
        current_overflow_presentation,
        present_extraction_chunk,
    )
    from reflexio.server.services.extraction.resumable_agent import (
        encode_committed_output,
    )
    from reflexio.server.services.profile.components.extractor import (
        _profile_evidence_validation_errors,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    storage, window, token = _overflow_coordinator(pipeline, actual_source=True)
    original = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    spans = tuple(span for chunk in original.chunks for span in chunk.spans)
    midpoint = len(spans) // 2
    prepared = PreparedExtractionInput(
        original.source_digest,
        (
            PreparedChunk(0, spans[:midpoint]),
            PreparedChunk(1, spans[midpoint:]),
        ),
    )
    run = storage.get_agent_run(f"window:{window.window_id}")
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        run.generation_request_snapshot["overflow"],
        source_digest=prepared.source_digest,
        context_limits={},
        prepared=prepared,
    )
    output = StructuredProfilesOutput.model_validate(
        {
            "profiles": [
                {
                    "content": "first supported lesson",
                    "time_to_live": "infinity",
                    "source_span": spans[0].text,
                },
                {
                    "content": "second supported lesson",
                    "time_to_live": "infinity",
                    "source_span": spans[-1].text,
                },
            ]
        }
    )
    assert output.profiles is not None
    child_models = {
        coordinator.candidate_key(candidate): asdict(ModelProvenance(model, "provider"))
        for candidate, model in zip(
            output.profiles, ["child-X", "child-Y"], strict=True
        )
    }
    policy = coordinator.step_key(
        "extraction-policy-v1",
        {
            "messages": [
                {"role": "system", "content": "original contract"},
                {"role": "user", "content": "<prepared-source>"},
            ],
            "schema": StructuredProfilesOutput.model_json_schema(),
        },
    )
    checkpoint = coordinator.checkpoint
    checkpoint["extraction_policy"] = policy
    coordinator.save_checkpoint(checkpoint)
    coordinator.saved_run = replace(
        run,
        generation_request_snapshot={
            **run.generation_request_snapshot,
            "output_schema_name": StructuredProfilesOutput.__name__,
        },
        committed_output=encode_committed_output(
            output, ModelProvenance("reconciler-Z", "provider")
        ),
    )

    def build_call():
        presentation = current_overflow_presentation.get()
        assert presentation is not None
        return OverflowExtractionCall(
            messages=[
                {
                    "role": "system",
                    "content": "changed contract"
                    if changed_policy
                    else "original contract",
                },
                {"role": "user", "content": presentation.text},
            ],
            schema=StructuredProfilesOutput,
            validator=lambda value: _profile_evidence_validation_errors(
                value, presentation.visible_spans
            ),
        )

    for chunk, candidate, model in zip(
        prepared.chunks, output.profiles, ["child-X", "child-Y"], strict=True
    ):
        with present_extraction_chunk(prepared, chunk, coordinator.record):
            call = build_call()
        # Freeze the original call contract even in the changed-policy case;
        # the cached final parent must still reject the changed prompt below.
        original_messages = [
            {"role": "system", "content": "original contract"},
            call.messages[1],
        ]
        key = coordinator.step_key(
            "extract-v1",
            {
                "messages": original_messages,
                "schema": StructuredProfilesOutput.model_json_schema(),
                "tools": [],
            },
        )
        coordinator.register_extraction_child(chunk.text, key)
        coordinator.complete_step(
            key,
            {
                "output": StructuredProfilesOutput(profiles=[candidate]).model_dump(
                    mode="json"
                ),
                "trace": {"turns": [], "finished": True},
            },
            ModelProvenance(model, "provider"),
        )

    checkpoint = coordinator.checkpoint
    checkpoint.update(
        candidate_models=child_models,
        source_coverage=[coordinator.candidate_key(c.text) for c in prepared.chunks],
        stage="extracted",
    )
    coordinator.save_checkpoint(checkpoint)

    execute = Mock(side_effect=AssertionError("Saved output must not dispatch models"))
    if changed_policy:
        with pytest.raises(InputBudgetConfigurationError, match="policy changed"):
            extract_overflow_window(
                coordinator=coordinator,
                client=Mock(),
                build_call=build_call,
                execute_call=execute,
                final_validator=None,
            )
        assert coordinator.checkpoint["stage"] == "extracted"
    else:
        result = extract_overflow_window(
            coordinator=coordinator,
            client=Mock(),
            build_call=build_call,
            execute_call=execute,
            final_validator=None,
        )
        assert result.output == output
        assert coordinator.checkpoint["stage"] == "recovered"
        assert coordinator.carried_source_spans is not None
        assert {
            (span.interaction_id, span.text)
            for span in coordinator.carried_source_spans
        } == {
            (spans[0].interaction_id, spans[0].text),
            (spans[-1].interaction_id, spans[-1].text),
        }
    assert coordinator.checkpoint["candidate_models"] == child_models
    execute.assert_not_called()


def test_saved_output_adoption_requires_original_witnesses_and_retains_output(pipeline):
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        prepare_extraction_input,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        OverflowCheckpointConflictError,
    )

    storage, window, token = _overflow_coordinator(
        pipeline, actual_source=True, legacy=True
    )
    run = storage.get_agent_run(f"window:{window.window_id}")
    original = run.generation_request_snapshot["source_interaction_digests"]
    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    manifest = OverflowCoordinator.initial_manifest(prepared.source_digest)
    for invalid in ({}, dict.fromkeys(original, "changed")):
        with storage.commit_scope(), pytest.raises(OverflowCheckpointConflictError):
            storage.validate_extraction_source_digest(window, prepared.source_digest)
            storage.adopt_extraction_overflow(
                window,
                token,
                source_digest=prepared.source_digest,
                source_digests=invalid,
                manifest=manifest,
            )
        assert (
            "overflow" not in storage.get_agent_run(run.id).generation_request_snapshot
        )
    with storage.commit_scope():
        storage.validate_extraction_source_digest(window, prepared.source_digest)
        adopted = storage.adopt_extraction_overflow(
            window,
            token,
            source_digest=prepared.source_digest,
            source_digests=original,
            manifest=manifest,
        )
    saved = storage.get_agent_run(run.id)
    assert adopted["attempts"] == []
    assert saved.committed_output == run.committed_output
    assert saved.generation_request_snapshot["source_interaction_digests"] == original
    assert (
        storage.extraction_status(window.user_id, "overflow-checkpoint")["status"]
        != "done"
    )
    assert storage.count_all_profiles() == 0


def test_overflow_publication_requires_complete_coverage_even_for_cached_plans(
    pipeline,
):
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        prepare_extraction_input,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )

    storage, window, token = _overflow_coordinator(pipeline, actual_source=True)
    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ],
        source_digest=prepared.source_digest,
        context_limits={},
        prepared=prepared,
    )
    with pytest.raises(ProviderRequestGuardError, match="full source coverage"):
        coordinator.require_full_source_coverage()
    checkpoint = coordinator.checkpoint
    checkpoint["stage"] = "extracted"
    checkpoint["source_coverage"] = ["another-source"]
    coordinator.save_checkpoint(checkpoint)
    with pytest.raises(ProviderRequestGuardError, match="full source coverage"):
        coordinator.require_full_source_coverage()
    checkpoint["source_coverage"] = [
        coordinator.candidate_key(chunk.text) for chunk in prepared.chunks
    ]
    coordinator.save_checkpoint(checkpoint)
    coordinator.require_full_source_coverage()
    assert storage.count_all_profiles() == 0
    assert (
        storage.extraction_status(window.user_id, "overflow-checkpoint")["status"]
        != "done"
    )


def test_overflow_checkpoint_quota_is_atomic_and_usage_is_immutable(pipeline):
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        OverflowCheckpointConflictError,
    )

    storage, window, token = _overflow_coordinator(pipeline)
    manifest = storage.checkpoint_extraction_overflow(
        window,
        token,
        expected_revision=0,
        source_digest="captured-source",
        reserve_attempt_id="attempt-1",
    )
    assert manifest["attempts"] == [
        {"id": "attempt-1", "usage": None, "model_provenance": None}
    ]
    with pytest.raises(OverflowCheckpointConflictError):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest="captured-source",
            reserve_attempt_id="attempt-stale",
        )
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "cache_read_input_tokens": 10,
        "cache_write_input_tokens": 20,
    }
    manifest = storage.checkpoint_extraction_overflow(
        window,
        token,
        expected_revision=1,
        source_digest="captured-source",
        observed_attempt={
            "id": "attempt-1",
            "usage": usage,
            "model_provenance": {"model_name": "fixture", "provider": None},
        },
        checkpoint={"stage": "protected-record", "completed_steps": {}},
    )
    assert manifest["revision"] == 2 and len(manifest["attempts"]) == 1
    with pytest.raises(OverflowCheckpointConflictError, match="immutable"):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=2,
            source_digest="captured-source",
            observed_attempt={
                "id": "attempt-1",
                "usage": {**usage, "prompt_tokens": 0},
                "model_provenance": {"model_name": "fixture", "provider": None},
            },
        )
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert saved == manifest
    checkpoint = {
        **manifest["checkpoint"],
        "output_schema_name": "StructuredProfilesOutput",
    }
    manifest = storage.checkpoint_extraction_overflow(
        window,
        token,
        expected_revision=2,
        source_digest="captured-source",
        checkpoint=checkpoint,
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert (
        parent.generation_request_snapshot["output_schema_name"]
        == "StructuredProfilesOutput"
    )
    for invalid in (
        {**checkpoint, "output_schema_name": "StructuredPlaybookList"},
        {"stage": "protected-record", "completed_steps": {}},
    ):
        with pytest.raises(
            OverflowCheckpointConflictError, match="schema is immutable"
        ):
            storage.checkpoint_extraction_overflow(
                window,
                token,
                expected_revision=3,
                source_digest="captured-source",
                checkpoint=invalid,
            )
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ]
        == manifest
    )
    assert storage.count_all_profiles() == 0
    assert storage.pending_extraction_effects() == []
    assert storage.extraction_status("u", "overflow-checkpoint")["status"] != "done"


def test_overflow_controller_reloads_history_without_refunding_or_double_counting(
    pipeline, monkeypatch
):
    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.llm.input_budget import (
        extraction_provider_guard,
        extraction_usage_observer,
    )
    from reflexio.server.llm.token_accounting import RunTokenTotals
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
        current_overflow_coordinator,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        OverflowAttemptBudgetExceededError,
        OverflowCheckpointConflictError,
    )

    storage, window, token = _overflow_coordinator(pipeline)
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )

    def reload():
        return OverflowCoordinator(
            storage,
            window,
            token,
            storage.get_agent_run(
                f"window:{window.window_id}"
            ).generation_request_snapshot["overflow"],
            source_digest="captured-source",
            context_limits={"fixture": 32768},
        )

    params = {
        "model": "fixture",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "source"}],
    }
    first = reload()
    provenance = ModelProvenance(model_name="fixture", provider="fixture-provider")
    key = first.step_key("extract", {"chunk": 0, "record": [], "prompt": "v1"})
    with first.provider_scope():
        assert current_overflow_coordinator.get() is first
        guard, observe = (
            extraction_provider_guard.get(),
            extraction_usage_observer.get(),
        )
        assert guard is not None and observe is not None
        # First transport result is uncertain. Its reservation stays consumed.
        guard(params, 30, (), True)
        guard(params, 30, (), True)
        observe(provenance, RunTokenTotals(101, 11, 20, 30))
        first.complete_step(key, {"candidate": "verified evidence"}, provenance)
    assert current_overflow_coordinator.get() is None
    assert extraction_provider_guard.get() is None
    assert extraction_usage_observer.get() is None

    resumed = reload()
    assert resumed.token_totals == RunTokenTotals(101, 11, 20, 30)
    receipt = resumed.completed_step(key)
    assert receipt is not None
    assert receipt["output"] == {"candidate": "verified evidence"}
    assert (
        resumed.step_key("extract", {"chunk": 0, "record": [], "prompt": "v2"}) != key
    )
    resumed.require_remaining_work(126)
    with pytest.raises(OverflowAttemptBudgetExceededError):
        resumed.require_remaining_work(127)
    with resumed.provider_scope():
        guard, observe = (
            extraction_provider_guard.get(),
            extraction_usage_observer.get(),
        )
        assert guard is not None and observe is not None
        guard(params, 30, (), True)
        observe(provenance, RunTokenTotals(202, 22, 40, 60))
    assert reload().token_totals == RunTokenTotals(303, 33, 60, 90)
    with pytest.raises(OverflowCheckpointConflictError):
        resumed.complete_step(key, {"candidate": "changed"}, provenance)
    # The SQL transition also rejects a caller bypassing complete_step().
    with pytest.raises(OverflowCheckpointConflictError):
        resumed.save_checkpoint({"stage": "reconcile", "completed_steps": {}})
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status("u", "overflow-checkpoint")["status"] != "done"


@pytest.mark.parametrize("suspend_midway", [False, True])
def test_overflow_agent_child_stages_output_without_completing_parent(
    pipeline, monkeypatch, suspend_midway
):
    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        prepare_extraction_input,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
        OverflowSuspendedError,
    )
    from reflexio.server.services.extraction.overflow_presentation import (
        OverflowPresentation,
        present_extraction_chunk,
    )
    from reflexio.server.services.extraction.protected_record import ProtectedRecord
    from reflexio.server.services.extraction.resumable_agent import (
        ResumableExtractionAgent,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )

    storage, window, token = _overflow_coordinator(pipeline, actual_source=True)
    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    run = storage.get_agent_run(f"window:{window.window_id}")
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        run.generation_request_snapshot["overflow"],
        source_digest=prepared.source_digest,
        context_limits={},
    )
    completion = ToolLoopResult(
        ctx=None,
        trace=ToolLoopTrace(finished=True),
        finished_reason="structured_output",
        messages=[],
        structured_output=StructuredProfilesOutput(profiles=[]),
        provenance=ModelProvenance(model_name="served-fixture", provider="fixture"),
    )
    seed = [
        {
            "role": "user",
            "content": OverflowPresentation(
                prepared, prepared.chunks[0], ProtectedRecord()
            ).text,
        }
    ]
    progress_messages = [
        *seed,
        {"role": "tool", "content": "saved lookup result", "tool_call_id": "lookup-1"},
    ]
    loop = Mock(return_value=completion)
    if suspend_midway:
        loop.side_effect = [
            ToolLoopResult(
                ctx=None,
                trace=ToolLoopTrace(),
                finished_reason="max_steps",
                messages=progress_messages,
            ),
            OverflowSuspendedError("next worker turn"),
            completion,
        ]
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", loop
    )
    agent = ResumableExtractionAgent(client=Mock(), storage=storage)
    if suspend_midway:
        with (
            coordinator.provider_scope(),
            present_extraction_chunk(prepared, prepared.chunks[0], ProtectedRecord()),
            pytest.raises(OverflowSuspendedError),
        ):
            agent.start(run=run, messages=seed, output_schema=StructuredProfilesOutput)
        parent = storage.get_agent_run(run.id)
        assert (
            parent.generation_request_snapshot["overflow"]["checkpoint"][
                "active_child"
            ]["max_steps_remaining"]
            == 7
        )
        coordinator = OverflowCoordinator(
            storage,
            window,
            token,
            parent.generation_request_snapshot["overflow"],
            source_digest=prepared.source_digest,
            context_limits={},
        )
    with (
        coordinator.provider_scope(),
        present_extraction_chunk(prepared, prepared.chunks[0], ProtectedRecord()),
    ):
        first = agent.start(
            run=run,
            messages=seed,
            output_schema=StructuredProfilesOutput,
        )
        with pytest.raises(
            InputBudgetConfigurationError, match="omits prepared source"
        ):
            agent.start(
                run=run,
                messages=[{"role": "user", "content": "criteria without input"}],
                output_schema=StructuredProfilesOutput,
            )
        cached = agent.start(
            run=run,
            messages=seed,
            output_schema=StructuredProfilesOutput,
        )
    assert first.output == cached.output
    assert first.model_provenance == cached.model_provenance
    assert loop.call_count == (3 if suspend_midway else 1)
    if suspend_midway:
        assert loop.call_args.kwargs["messages"] == progress_messages
    parent = storage.get_agent_run(run.id)
    assert parent.status.value == "running"
    assert parent.committed_output is None
    assert (
        "active_child"
        not in parent.generation_request_snapshot["overflow"]["checkpoint"]
    )
    assert (
        len(
            parent.generation_request_snapshot["overflow"]["checkpoint"][
                "completed_steps"
            ]
        )
        == 1
    )
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status("u", "overflow-checkpoint")["status"] != "done"


def test_invalidating_overflow_scrubs_child_evidence_and_fences_future_checkpoints(
    pipeline,
):
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
        WindowInputsDeletedError,
    )

    storage, window, token = _overflow_coordinator(pipeline)
    storage.checkpoint_extraction_overflow(
        window,
        token,
        expected_revision=0,
        source_digest="captured-source",
        reserve_attempt_id="spent-before-erasure",
        checkpoint={
            "stage": "extract",
            "completed_steps": {
                "first": {
                    "output": "private original quotation",
                    "model_provenance": {"model_name": None, "provider": None},
                }
            },
        },
    )
    storage.invalidate_extraction(window, token)
    run = storage.get_agent_run(f"window:{window.window_id}")
    assert run.status.value == "cancelled"
    assert run.generation_request_snapshot["overflow"]["checkpoint"] == {
        "stage": "invalidated"
    }
    assert len(run.generation_request_snapshot["overflow"]["attempts"]) == 1
    assert run.service_config_snapshot == {}
    assert run.committed_output is None
    with pytest.raises((LeaseLostError, WindowInputsDeletedError)):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=1,
            source_digest="captured-source",
            reserve_attempt_id="late-orphan",
        )
    assert storage.count_all_profiles() == 0


def test_overflow_controller_yields_before_dispatch_without_consuming_quota(
    pipeline, monkeypatch
):
    from reflexio.server.llm.input_budget import extraction_provider_guard
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
        OverflowSuspendedError,
    )

    storage, window, token = _overflow_coordinator(pipeline)
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ],
        source_digest="captured-source",
        context_limits={"fixture": 32768},
        turn_seconds=10,
    )
    with coordinator.provider_scope():
        guard = extraction_provider_guard.get()
        assert guard is not None
        with pytest.raises(OverflowSuspendedError):
            guard(
                {"model": "fixture", "max_tokens": 8192, "messages": []},
                30,
                (),
                True,
            )
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert saved["attempts"] == []
    assert saved["revision"] == 0


def test_overflow_suspension_never_completes_or_bills_partial_work(
    pipeline, monkeypatch
):
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowSuspendedError,
    )
    from reflexio.server.services.profile.components.extractor import ProfileExtractor

    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("suspend", force=True)
    ).json()["success"]

    def suspend(*_args, **_kwargs):
        raise OverflowSuspendedError("continue saved work")

    monkeypatch.setattr(
        ProfileExtractor, "_generate_raw_updates_from_sessions", suspend
    )
    with patch("reflexio.server.metering_events.record_extraction_tokens") as bill:
        assert worker.drain_org("stream-test", 1, 300) == 0
        bill.assert_not_called()
    storage = engine.get_storage()
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status("u", "suspend")["status"] == "pending"
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT attempts,completed_seq,last_error FROM extraction_cursors WHERE user_id=? AND kind=?",
            ("u", "profile"),
        )[0]
        saved_window = db.query(
            "SELECT completed,outcome,effects FROM extraction_windows WHERE user_id=? AND kind=?",
            ("u", "profile"),
        )[0]
    assert cursor["attempts"] == 0
    assert cursor["completed_seq"] == 0
    assert cursor["last_error"] is None
    assert not saved_window["completed"]
    assert saved_window["outcome"] is None
    assert saved_window["effects"] is None


@pytest.mark.parametrize(
    "votes,expected",
    [
        (["false", "false"], False),
        (["false", "true"], True),
        (["false", "unknown"], None),
    ],
)
@pytest.mark.parametrize("reserved_before", [0, 125])
def test_chunk_gate_requires_complete_valid_negative_coverage(
    pipeline, monkeypatch, votes, expected, reserved_before
):
    from types import SimpleNamespace

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.services.extraction.chunk_gate import judge_extraction_chunks
    from reflexio.server.services.extraction.input_preparation import (
        PreparedChunk,
        PreparedExtractionInput,
        SourceSpan,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.extraction.overflow_presentation import (
        current_overflow_presentation,
    )
    from reflexio.server.services.extraction.protected_record import ProtectedRecord

    storage, window, token = _overflow_coordinator(
        pipeline,
        attempts=tuple(
            {"id": f"spent-{index}", "usage": None, "model_provenance": None}
            for index in range(reserved_before)
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=8192, fallback_models=[])
    )
    prepared = PreparedExtractionInput(
        "captured-source",
        (
            PreparedChunk(
                0, (SourceSpan(1, "r", "user", "content", 0, 13, "No correction", 1),)
            ),
            PreparedChunk(
                1, (SourceSpan(2, "r", "user", "content", 0, 14, "Do not deploy.", 2),)
            ),
        ),
    )
    dispatch = Mock(
        side_effect=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=vote, tool_calls=None, refusal=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
                model="gpt-4o",
                _hidden_params={"custom_llm_provider": "openai"},
            )
            for vote in votes
        ]
    )

    def dispatch_with_source(*args, **kwargs):
        presentation = current_overflow_presentation.get()
        assert presentation is not None
        assert presentation.chunk in prepared.chunks
        return dispatch(*args, **kwargs)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch_with_source)
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ],
        source_digest="captured-source",
        context_limits={"gpt-4o": 32768},
    )

    def prompt():
        presentation = current_overflow_presentation.get()
        assert presentation is not None
        return presentation.text

    def judge():
        return judge_extraction_chunks(
            coordinator=coordinator,
            prepared=prepared,
            record=ProtectedRecord(),
            client=client,
            model="gpt-4o",
            build_prompt=prompt,
            minimum_after_positive=3,
        )

    if reserved_before:
        from reflexio.server.services.storage.storage_base._extraction_stream import (
            OverflowAttemptBudgetExceededError,
        )

        with pytest.raises(OverflowAttemptBudgetExceededError):
            judge()
        assert dispatch.call_count == 0
        assert not coordinator.checkpoint.get("completed_steps")
        assert storage.count_all_profiles() == 0
        return
    if expected is None:
        with pytest.raises(ValueError, match="valid boolean"):
            judge()
        assert len(coordinator.checkpoint["completed_steps"]) == 1
    else:
        assert judge() is expected
        assert judge() is expected
    assert dispatch.call_count == 2
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status("u", "overflow-checkpoint")["status"] != "done"


def test_protected_pass_keeps_scoped_approval_and_adverse_outcomes_on_replay(
    pipeline, monkeypatch
):
    import json
    from types import SimpleNamespace

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.services.extraction.input_preparation import (
        PreparedChunk,
        PreparedExtractionInput,
        SourceSpan,
    )
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.extraction.protected_pass import (
        build_protected_record,
    )

    storage, window, token = _overflow_coordinator(pipeline)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda **_: "gpt-4o",
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    spans = (
        SourceSpan(
            1,
            "original-request",
            "user",
            "content",
            0,
            len("Approve implementing the plan; do not deploy yet."),
            "Approve implementing the plan; do not deploy yet.",
            0,
        ),
        SourceSpan(
            2,
            "original-request",
            "assistant",
            "content",
            0,
            len(
                "Preparation is complete. Deployment still fails with permission denied."
            ),
            "Preparation is complete. Deployment still fails with permission denied.",
            1,
        ),
    )
    prepared = PreparedExtractionInput(
        "captured-source",
        tuple(PreparedChunk(i, (span,)) for i, span in enumerate(spans)),
    )
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=8192, fallback_models=[])
    )
    responses = []
    for index, span in enumerate(spans):
        responses.append(
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "evidence": [
                                        {
                                            "kind": "instruction"
                                            if index == 0
                                            else "uncertainty",
                                            "evidence_refs": [span.evidence_ref],
                                        }
                                    ]
                                }
                            ),
                            tool_calls=None,
                            refusal=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=100 + index,
                    completion_tokens=10,
                    total_tokens=110 + index,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
                model="gpt-4o",
                _hidden_params={"custom_llm_provider": "openai"},
            )
        )
    dispatch = Mock(side_effect=responses)
    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch)

    def coordinator():
        return OverflowCoordinator(
            storage,
            window,
            token,
            storage.get_agent_run(
                f"window:{window.window_id}"
            ).generation_request_snapshot["overflow"],
            source_digest="captured-source",
            context_limits={"gpt-4o": 32768},
        )

    first = coordinator()
    record = build_protected_record(
        coordinator=first,
        prepared=prepared,
        client=client,
        extraction_criteria="Preserve scope of approvals and observed failures.",
        agent_context="Deployment assistant",
        minimum_following_attempts=3,
    )
    assert dispatch.call_count == 2
    assert [entry.quote for entry in record.evidence] == [span.text for span in spans]
    assert [entry.role for entry in record.evidence] == ["user", "assistant"]
    assert [entry.source_order for entry in record.evidence] == [0, 1]
    assert first.token_totals.prompt_tokens == 201
    resumed = coordinator()
    assert (
        build_protected_record(
            coordinator=resumed,
            prepared=prepared,
            client=client,
            extraction_criteria="Preserve scope of approvals and observed failures.",
            agent_context="Deployment assistant",
            minimum_following_attempts=3,
        )
        == record
    )
    assert dispatch.call_count == 2
    assert resumed.token_totals == first.token_totals
    assert storage.count_all_profiles() == 0


def test_overflow_checkpoint_cannot_refund_quota_or_write_after_lease_loss(pipeline):
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
        OverflowAttemptBudgetExceededError,
        OverflowCheckpointConflictError,
    )

    attempts = tuple(
        {"id": f"attempt-{index}", "usage": None, "model_provenance": None}
        for index in range(128)
    )
    storage, window, token = _overflow_coordinator(pipeline, attempts=attempts)
    with pytest.raises(OverflowAttemptBudgetExceededError):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest="captured-source",
            reserve_attempt_id="attempt-129",
        )
    with pytest.raises(LeaseLostError):
        storage.checkpoint_extraction_overflow(
            window,
            "superseded",
            expected_revision=0,
            source_digest="captured-source",
            checkpoint={"stage": "complete", "completed_steps": {}},
        )
    with pytest.raises(OverflowCheckpointConflictError):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest="replaced-source",
            checkpoint={"stage": "complete", "completed_steps": {}},
        )
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert len(saved["attempts"]) == 128 and saved["revision"] == 0
    assert saved["checkpoint"]["stage"] == "prepare"


@pytest.mark.parametrize("replacement", ["interaction", "request"])
def test_overflow_source_witness_rejects_replacement_with_unchanged_sequence(
    pipeline, replacement
):
    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        prepare_extraction_input,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        WindowInputsDeletedError,
    )

    storage, window, _ = _overflow_coordinator(pipeline)
    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    with storage.commit_scope():
        storage.validate_extraction_source_digest(window, prepared.source_digest)
    with storage._stream_sql() as db:
        if replacement == "interaction":
            db.query(
                "UPDATE interactions SET content=? WHERE interaction_id=?",
                ("Replacement correction", window.manifest[0]["interaction_id"]),
            )
        else:
            db.query(
                "UPDATE requests SET source=? WHERE request_id=?",
                ("replacement-source", "overflow-checkpoint"),
            )
    with storage.commit_scope():
        # The old identity-only check passes; the source-content witness must
        # detect the change before a cached result could publish.
        storage.validate_extraction_inputs(window)
        with pytest.raises(WindowInputsDeletedError, match="content was replaced"):
            storage.validate_extraction_source_digest(window, prepared.source_digest)
    assert storage.count_all_profiles() == 0


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    # Drive turns explicitly so crashes and overlapping publishes are reproducible.
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_scheduler.schedule_tagging",
        lambda **_: None,
    )
    config = DefaultConfigurator(org_id="stream-test", base_dir=str(tmp_path))
    config.set_config(
        Config(
            storage_config=StorageConfigSQLite(db_path=str(tmp_path / "stream.db")),
            window_size=10,
            stride_size=5,
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="Extract user preferences"
            ),
            user_playbook_extractor_config=None,
            agent_success_config=None,
        )
    )
    engine = Reflexio(
        org_id="stream-test", storage_base_dir=str(tmp_path), configurator=config
    )
    monkeypatch.setattr(
        "reflexio.server.cache.reflexio_cache.get_reflexio", lambda **_: engine
    )
    monkeypatch.setattr(
        "reflexio.server.api_endpoints.publisher_api.get_reflexio", lambda **_: engine
    )
    worker = DurableLearningWorker(lambda _: engine.request_context)
    client = TestClient(create_app(get_org_id=lambda: "stream-test"))
    return engine, worker, client


def payload(request_id, count=10, user_id="u", force=False):
    return {
        "user_id": user_id,
        "request_id": request_id,
        "session_id": "s",
        "source": "test",
        "force_extraction": force,
        "interaction_data_list": [
            {
                "role": "User",
                "content": f"I prefer concise answers with details {i}",
                "created_at": 100 - i,
            }
            for i in range(count)
        ],
    }


@pytest.mark.parametrize(
    "replay",
    [
        "clean",
        "empty",
        "commit_failure",
        "altered_cache",
        "tagged_cache",
        "old_cache",
        "vector_cache",
        "dimension_cache",
        "existing_id_cache",
        "active_cache",
        "waiting_cache",
        "clock_cache",
    ],
)
def test_source_quoted_overflow_http_commit_replay_and_cached_plan_fence(
    pipeline, monkeypatch, replay
):
    import json

    from reflexio.models.api_schema.service_schemas import UserPlaybook
    from reflexio.models.config_schema import UserPlaybookExtractorConfig
    from reflexio.server import metering_events
    from reflexio.server.metering_signals import count_input_tokens
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor
    from reflexio.server.services.extraction.protected_record import (
        ProtectedEvidence,
        ProtectedRecord,
    )
    from reflexio.server.services.playbook.service import PlaybookGenerationService
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    engine, worker, client = pipeline
    configurator = engine.request_context.configurator
    configurator.set_config(
        configurator.get_config().model_copy(
            update={
                "profile_extractor_config": None,
                "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                    extraction_definition_prompt="Learn deployment instructions"
                ),
            }
        )
    )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    storage = engine.get_storage()
    old = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="old",
        content="An existing rule remains unchanged.",
    )
    storage.save_user_playbooks([old], skip_embedding=True)
    before = storage.get_user_playbook_by_id(old.user_playbook_id).model_dump(
        mode="json"
    )

    # Only the semantic selection is controlled; real HTTP admission, original
    # witnesses, coordinator journal, cached plan and fenced commit remain real.
    def prepare(_self, coordinator, _config):
        prepared = coordinator.require_prepared()
        assert len(prepared.chunks) > 1
        selected = next(s for c in prepared.chunks for s in c.spans if s.role == "User")
        coordinator.record = ProtectedRecord(
            evidence=(
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(selected.evidence_ref,),
                    quote=selected.text,
                ),
            )
        )
        if replay == "empty":
            coordinator.record = ProtectedRecord()
        checkpoint = coordinator.checkpoint
        checkpoint["protected_record"] = coordinator.record.model_dump(mode="json")
        checkpoint["protected_coverage"] = [
            coordinator.candidate_key(c.text) for c in prepared.chunks
        ]
        coordinator.save_checkpoint(checkpoint)

    monkeypatch.setattr(WindowExecutor, "_prepare_overflow_context", prepare)
    monkeypatch.setattr(storage, "_get_embedding", lambda *_: [0.1] * 512)
    storage._enable_document_expansion = True
    expansion = Mock(
        side_effect=AssertionError(
            "Source-quoted rules must not generate expansion text"
        )
    )
    monkeypatch.setattr(storage, "_expand_document", expansion)
    monkeypatch.setattr(
        PlaybookGenerationService, "_dispatch_playbook_schedulers", lambda *_: None
    )
    generative = Mock(
        side_effect=AssertionError(
            "Overflow must not call a prose generator or consolidator"
        )
    )
    monkeypatch.setattr(PlaybookGenerationService, "compute_generation", generative)
    instruction = "Verify the named destination before presenting the plan. Keep STAGING_APPROVAL_REQUIRED. File inspection is not execution."
    body = payload("source-quote", count=3, force=True)
    body["agent_version"] = "v"
    body["interaction_data_list"] = [
        {"role": "User", "content": instruction},
        {"role": "Assistant", "content": "Diagnostic output. " * 18000},
        {"role": "Assistant", "content": "No deployment performed."},
    ]
    started_at = int(time.time())
    assert client.post("/api/publish_interaction", json=body).json()["success"]
    token_bill = Mock(wraps=metering_events.record_extraction_tokens)
    learning_bill = Mock(
        wraps=metering_events.record_learnings_generated_records_strict
    )
    monkeypatch.setattr(metering_events, "record_extraction_tokens", token_bill)
    monkeypatch.setattr(
        metering_events, "record_learnings_generated_records_strict", learning_bill
    )
    complete = storage.complete_extraction
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1 and replay not in {"clean", "empty"}:
            raise RuntimeError("Crash before commit")
        return complete(*args, **kwargs)

    monkeypatch.setattr(storage, "complete_extraction", fail_once)
    assert worker.drain_org("stream-test", 1, 300) == (
        1 if replay in {"clean", "empty"} else 0
    )
    if replay not in {"clean", "empty"}:
        assert len(storage.get_user_playbooks(user_id="u")) == 1
        token_bill.assert_not_called()
        learning_bill.assert_not_called()
        with storage._stream_sql() as db:
            row = db.query(
                "SELECT window_id,outcome FROM extraction_windows WHERE user_id=? AND completed=0 LIMIT 1",
                ("u",),
            )[0]
            outcome = json.loads(row["outcome"])
            if replay == "altered_cache":
                outcome["write_plan"]["new_playbooks"][0]["content"] = (
                    "Deploy to production without approval."
                )
            if replay == "tagged_cache":
                outcome["write_plan"]["new_playbooks"][0]["tags"] = ["unapproved"]
            if replay == "vector_cache":
                outcome["write_plan"]["new_playbooks"][0]["embedding"][0] = 0.9
            if replay == "dimension_cache":
                outcome["write_plan"]["new_playbooks"][0]["embedding"] = [0.9]
            if replay == "clock_cache":
                outcome["write_plan"]["new_playbooks"][0]["created_at"] = 1
            if replay == "existing_id_cache":
                outcome["write_plan"]["new_playbooks"][0]["user_playbook_id"] = (
                    old.user_playbook_id
                )
            if replay in {"active_cache", "waiting_cache"}:
                run = storage.get_agent_run("window:" + row["window_id"])
                key = "active_child" if replay == "active_cache" else "waiting_child"
                run.generation_request_snapshot["overflow"]["checkpoint"][key] = {
                    "key": "unconsumed-child"
                }
                db.query(
                    "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
                    (json.dumps(run.generation_request_snapshot), run.id),
                )
            if replay == "old_cache":
                run = storage.get_agent_run("window:" + row["window_id"])
                run.generation_request_snapshot["overflow"]["checkpoint"].pop(
                    "source_quoted_publication"
                )
                db.query(
                    "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
                    (json.dumps(run.generation_request_snapshot), run.id),
                )
            db.query(
                "UPDATE extraction_windows SET outcome=? WHERE window_id=?",
                (json.dumps(outcome), row["window_id"]),
            )
            now = db.now()
        monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: now + 600)
        assert worker.drain_org("stream-test", 1, 300) == (
            1 if replay in {"commit_failure", "clock_cache"} else 0
        )
    expected = 2 if replay in {"clean", "commit_failure", "clock_cache"} else 1
    rows = storage.get_user_playbooks(user_id="u", include_embedding=True)
    assert len(rows) == expected
    assert (
        storage.get_user_playbook_by_id(old.user_playbook_id).model_dump(mode="json")
        == before
    )
    if expected == 2:
        new = next(row for row in rows if row.user_playbook_id != old.user_playbook_id)
        assert new.content == new.source_span == instruction
        assert new.trigger == instruction and new.rationale is None
        assert new.embedding and new.expanded_terms is None
        assert started_at <= new.created_at <= int(time.time())
        assert "User corrected" not in new.notes
    if replay in {"clean", "empty", "commit_failure", "clock_cache"}:
        # Commit and its durable effects are separate worker turns.
        token_bill.assert_not_called()
        learning_bill.assert_not_called()
        assert worker.drain_org("stream-test", 1, 300) == 1
        status = client.get(
            "/api/learning_status", params={"request_id": "source-quote"}
        ).json()
        assert status["status"] == "done"
        with storage._stream_sql(read_only=True) as db:
            window = db.query(
                "SELECT window_id,effects_done FROM extraction_windows WHERE user_id=? AND kind=? LIMIT 1",
                ("u", "playbook"),
            )[0]
        assert window["effects_done"] == 1
        run_id = "window:" + window["window_id"]
        assert storage.get_agent_run(run_id).status.value == "finalized"
        ids = storage.get_agent_run_finalization_receipt(
            run_id=run_id, entity_type="user_playbook"
        )
        assert ids == [
            str(row.user_playbook_id)
            for row in rows
            if row.user_playbook_id != old.user_playbook_id
        ]
        token_bill.assert_called_once()
        learning_bill.assert_called_once()
        assert learning_bill.call_args.kwargs["learning_ids"] == ids
        assert token_bill.call_args.kwargs["strict"] is True
        assert token_bill.call_args.kwargs["event_key"] == "tok:" + window["window_id"]
        assert token_bill.call_args.kwargs[
            "billing_input_tokens"
        ] == count_input_tokens(
            "\n".join(item["content"] for item in body["interaction_data_list"])
        )
        worker.drain_org("stream-test", 1, 300)  # May claim one final empty turn.
        assert len(storage.get_user_playbooks(user_id="u")) == expected
        token_bill.assert_called_once()
        learning_bill.assert_called_once()
    else:
        status = client.get(
            "/api/learning_status", params={"request_id": "source-quote"}
        ).json()
        assert status == {
            "status": "failed",
            "reason": "input_budget_configuration_error",
        }
        token_bill.assert_not_called()
        learning_bill.assert_not_called()
    assert len(storage.get_user_interaction("u")) == 3
    generative.assert_not_called()
    expansion.assert_not_called()


def test_http_ack_is_durable_and_backlog_does_not_skip(pipeline, monkeypatch):
    engine, worker, client = pipeline
    entered, released = threading.Event(), threading.Event()
    seen = []

    def gate(service, _):
        seen.append(
            [
                i.interaction_id
                for group in service.service_config.window_interactions
                for i in group.interactions
            ]
        )
        entered.set()
        assert released.wait(5)
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    response = client.post("/api/publish_interaction", json=payload("first"))
    assert response.status_code == 200 and response.json()["success"]
    assert response.json()["learning_status"] == "deferred"
    assert engine.get_storage().get_request("first") is not None
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(worker.drain_org, "stream-test", 1, 3)
        assert entered.wait(3)
        for n in range(3):
            response = client.post(
                "/api/publish_interaction", json=payload(f"later{n}", 5)
            )
            assert response.json()["success"]
        assert len(seen) == 1
        released.set()
        assert active.result(timeout=5) == 1
    worker.drain_org("stream-test", 10, 3)
    assert [x[-1] for x in seen] == [10, 15, 20, 25]
    assert (
        client.get("/api/learning_status", params={"request_id": "later2"}).json()[
            "status"
        ]
        == "done"
    )


def test_ingestion_failure_not_acknowledged(pipeline, monkeypatch):
    engine, _, client = pipeline

    def fail(*_, **__):
        raise RuntimeError("write failed")

    monkeypatch.setattr(engine.get_storage(), "admit_extraction", fail)
    response = client.post("/api/publish_interaction", json=payload("failed"))
    assert not response.json()["success"]
    assert engine.get_storage().get_request("failed") is None
    assert engine.get_storage().get_user_interaction("u") == []


def test_sync_request_completes_while_later_tail_remains(pipeline, monkeypatch):
    engine, worker, client = pipeline
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", lambda *_: False
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(
            client.post,
            "/api/publish_interaction?wait_for_response=true",
            json=payload("sync"),
        )
        deadline = time.monotonic() + 3
        while (
            engine.get_storage().get_request("sync") is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert not waiting.done()
        assert client.post("/api/publish_interaction", json=payload("tail", 1)).json()[
            "success"
        ]
        worker.drain_org("stream-test", 1, 3)
        assert waiting.result(timeout=3).json()["learning_status"] == "done"
        assert (
            engine.get_storage().extraction_status("u", "tail")["status"] == "pending"
        )


def test_real_extractor_persist_and_outcome_replay(pipeline):
    engine, worker, client = pipeline
    response = client.post(
        "/api/publish_interaction", json=payload("extract", 10, force=True)
    )
    assert response.json()["success"]
    # The suite's external LLM double still exercises actual extraction,
    # deduplication, writes and receipt-backed finalization.
    worker.drain_org("stream-test", 1, 300)
    storage = engine.get_storage()
    assert storage.extraction_status("u", "extract")["status"] == "done"
    assert storage.count_all_profiles() > 0
    before = storage.count_all_profiles()
    with patch.object(
        ProfileGenerationService,
        "compute_generation",
        side_effect=AssertionError("must not extract twice"),
    ):
        worker.drain_org("stream-test", 2, 300)
    assert storage.count_all_profiles() == before


@pytest.mark.parametrize("gate_negative", [False, True])
@pytest.mark.parametrize("fail_commit", [False, True])
def test_enabled_overflow_profile_uses_real_window_commit_once(
    pipeline, monkeypatch, gate_negative, fail_commit
):
    import json
    from types import SimpleNamespace

    from reflexio.server.llm.litellm_client import LiteLLMClient

    engine, worker, client = pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "profiles": [
                                {
                                    "content": "The user prefers concise answers.",
                                    "time_to_live": "infinity",
                                    "source_span": "I prefer concise answers with details 0",
                                }
                            ]
                        }
                    ),
                    tool_calls=None,
                    refusal=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model="minimax/MiniMax-M3",
        _hidden_params={"custom_llm_provider": "minimax"},
    )

    def respond(params, hard_timeout):
        from copy import deepcopy

        response = deepcopy(completion)
        response_format = params.get("response_format")
        schema_name = getattr(response_format, "__name__", "") or json.dumps(
            response_format, default=str
        )
        if gate_negative and response_format is None:
            response.choices[0].message.content = "false"
        if "ProfileDeduplicationOutput" in schema_name or any(
            "unique_ids" in str(message.get("content", ""))
            for message in params["messages"]
        ):
            response.choices[0].message.content = json.dumps(
                {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
            )
        return response

    dispatch = Mock(side_effect=respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    assert client.post(
        "/api/publish_interaction",
        json=payload("overflow-enabled", force=not gate_negative),
    ).json()["success"]
    storage = engine.get_storage()
    if fail_commit:
        complete = storage.complete_extraction
        attempts = 0

        def fail_first_commit(*args):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError(
                    "Injected failure after learning writes, before coverage commit"
                )
            return complete(*args)

        monkeypatch.setattr(storage, "complete_extraction", fail_first_commit)
    worker.drain_org("stream-test", 1, 300)
    if fail_commit:
        assert storage.count_all_profiles() == 0
        assert storage.extraction_status("u", "overflow-enabled")["status"] != "done"
        with storage._stream_sql() as db:
            db.query(
                "UPDATE learning_work SET due_at=0 WHERE org_id=? AND user_id=?",
                (storage.org_id, "u"),
            )
            db.query("UPDATE extraction_cursors SET retry_at=0 WHERE user_id=?", ("u",))
        worker.drain_org("stream-test", 1, 300)
        assert attempts == 2
    assert storage.extraction_status("u", "overflow-enabled")["status"] == "done"
    assert storage.count_all_profiles() == (0 if gate_negative else 1)
    expected_calls = 1 if gate_negative else 2
    assert dispatch.call_count == expected_calls
    with storage._stream_sql() as db:
        rows = db.query("SELECT id FROM _agent_runs WHERE user_id=? LIMIT 2", ("u",))
    assert len(rows) == 1
    run = storage.get_agent_run(rows[0]["id"])
    assert run.status.value == ("finalized" if gate_negative else "agent_completed")
    attempts = run.generation_request_snapshot["overflow"]["attempts"]
    assert len(attempts) == expected_calls
    assert (
        sum(attempt["usage"]["prompt_tokens"] for attempt in attempts)
        == 100 * expected_calls
    )
    worker.drain_org("stream-test", 1, 300)
    assert storage.count_all_profiles() == (0 if gate_negative else 1)
    assert dispatch.call_count == expected_calls
    assert storage.get_agent_run(run.id).status.value == "finalized"


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("fail_commit", [False, True])
def test_http_overflow_human_answer_resumes_same_window_and_commits_once(
    pipeline, monkeypatch, kind, fail_commit
):
    import json
    from types import SimpleNamespace

    from reflexio.models.config_schema import (
        PendingToolCallConfig,
        UserPlaybookExtractorConfig,
    )
    from reflexio.server.llm.litellm_client import LiteLLMClient

    engine, worker, client = pipeline
    config = engine.request_context.configurator
    root = config.get_config()
    updates: dict[str, Any] = {
        "pending_tool_call_config": PendingToolCallConfig(enabled=True)
    }
    if kind == "playbook":
        updates.update(
            profile_extractor_config=None,
            user_playbook_extractor_config=UserPlaybookExtractorConfig(
                extraction_definition_prompt="Extract scoped approval practices."
            ),
        )
    config.set_config(root.model_copy(update=updates))
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
        lambda _: True,
    )
    requested = False
    answered_messages = []
    source_text = "Prepare a draft only; obtain separate approval before deploying."
    answer = (
        "Preparation is approved. "
        + "Background context. " * 100
        + "Deployment is not approved; ask separately."
    )
    if kind == "profile":
        from datetime import UTC, datetime, timedelta

        from reflexio.server.services.storage.storage_base import (
            PendingToolCallRecord,
            PendingToolCallStatus,
            build_scope_hash,
            human_feedback_scope,
        )

        scope = human_feedback_scope(engine.get_storage().org_id)
        now = datetime.now(UTC)
        engine.get_storage().create_pending_tool_call(
            PendingToolCallRecord(
                id="existing-human-scope",
                org_id=engine.get_storage().org_id,
                scope=scope,
                scope_hash=build_scope_hash(scope),
                tool_name="ask_human",
                dedup_key="existing-human-scope",
                status=PendingToolCallStatus.PENDING,
                question_text="What actions are approved?",
                expires_at=now + timedelta(days=1),
                cache_until=now + timedelta(hours=1),
            )
        )

    def respond(params, hard_timeout):
        nonlocal requested
        schema = params.get("response_format")
        name = getattr(schema, "__name__", "") or json.dumps(schema, default=str)
        tool_calls = None
        finish = "stop"
        if params.get("tools") and not requested:
            requested = True
            result = None
            tool_calls = [
                SimpleNamespace(
                    id="ask-scope",
                    type="function",
                    function=SimpleNamespace(
                        name="ask_human"
                        if kind == "playbook"
                        else "attach_pending_info_request",
                        arguments=json.dumps(
                            {"question": "What actions are approved?"}
                            if kind == "playbook"
                            else {"pending_tool_call_id": "existing-human-scope"}
                        ),
                    ),
                )
            ]
            finish = "tool_calls"
        elif (
            "Review" in name
            or "reason_code" in name
            or any(
                "reason_code" in str(message.get("content", ""))
                for message in params["messages"]
            )
        ):
            result = {
                "decisions": [
                    {
                        "id": "C1",
                        "decision": "accept",
                        "reason_code": "grounded_useful",
                        "evidence_ids": ["C1-E1"],
                    }
                ]
            }
        elif "ProfileDeduplicationOutput" in name or any(
            "unique_ids" in str(message.get("content", ""))
            for message in params["messages"]
        ):
            result = {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
        else:
            answered_messages.extend(params["messages"])
            assert answer in json.dumps(params["messages"], ensure_ascii=False)
            if kind == "profile":
                result = {
                    "profiles": [
                        {
                            "content": "The user requires separate deployment approval.",
                            "time_to_live": "infinity",
                            "source_span": source_text,
                        }
                    ]
                }
            else:
                result = {
                    "playbooks": [
                        {
                            "content": "Prepare a draft and ask separately before deploying.",
                            "trigger": "When preparing a release",
                            "rationale": "The user restricts approval to preparation.",
                            "evidence_kind": "correction",
                            "evidence_refs": [f"  i1:content:0:{len(source_text)}  "],
                        }
                    ]
                }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(result) if result is not None else None,
                        tool_calls=tool_calls,
                        refusal=None,
                    ),
                    finish_reason=finish,
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    dispatch = Mock(side_effect=respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    data = payload("human-overflow", force=True)
    data["agent_version"] = "v1"
    for item in data["interaction_data_list"]:
        item["content"] = source_text
    assert client.post("/api/publish_interaction", json=data).json()["success"]
    storage = engine.get_storage()
    worker.drain_org("stream-test", 1, 300)
    assert requested and dispatch.call_count == 1
    assert storage.extraction_status("u", "human-overflow")["status"] != "done"
    assert storage.count_all_profiles() == 0
    with storage._stream_sql() as db:
        pending = db.query("SELECT id,status FROM _pending_tool_calls LIMIT 2")
        assert len(pending) == 1 and pending[0]["status"] == "pending"
        parent_id = db.query("SELECT id FROM _agent_runs LIMIT 2")[0]["id"]
    parent = storage.get_agent_run(parent_id)
    assert parent.generation_request_snapshot["output_schema_name"] == (
        "StructuredReferencedExtractedPlaybookList"
        if kind == "playbook"
        else "StructuredProfilesOutput"
    )
    assert parent.generation_request_snapshot["overflow"]["checkpoint"][
        "waiting_child"
    ]["pending_tool_call_ids"] == [pending[0]["id"]]
    storage.resolve_pending_tool_call(
        pending[0]["id"], result={"answer": answer}, valid_for_seconds=3600
    )
    with storage._stream_sql() as db:
        db.query("UPDATE learning_work SET due_at=0")
        db.query("UPDATE extraction_cursors SET retry_at=0")
    if fail_commit:
        complete = storage.complete_extraction
        first_commit = True

        def fail_once(*args):
            nonlocal first_commit
            if first_commit:
                first_commit = False
                raise RuntimeError("Injected answer-window commit rollback")
            return complete(*args)

        monkeypatch.setattr(storage, "complete_extraction", fail_once)
    worker.drain_org("stream-test", 1, 300)
    calls_after_compute = dispatch.call_count
    if fail_commit:
        assert storage.count_all_profiles() == 0
        assert storage.extraction_status("u", "human-overflow")["status"] != "done"
        with storage._stream_sql() as db:
            db.query("UPDATE learning_work SET due_at=0")
            db.query("UPDATE extraction_cursors SET retry_at=0")
        worker.drain_org("stream-test", 1, 300)
        assert dispatch.call_count == calls_after_compute
    assert storage.extraction_status("u", "human-overflow")["status"] == "done"
    assert answer in json.dumps(answered_messages, ensure_ascii=False)
    with storage._stream_sql() as db:
        rows = db.query(
            "SELECT user_playbook_id,source_interaction_ids,source_span FROM user_playbooks WHERE user_id='u' LIMIT 2"
        )
        deps = db.query(
            "SELECT consumed_at FROM _run_tool_dependencies WHERE run_id=? LIMIT 2",
            (parent_id,),
        )
    assert storage.count_all_profiles() == (1 if kind == "profile" else 0)
    assert len(rows) == (1 if kind == "playbook" else 0)
    if rows:
        assert json.loads(rows[0]["source_interaction_ids"]) == [1]
        assert rows[0]["source_span"] == source_text
    assert len(deps) == 1 and deps[0]["consumed_at"] is not None
    final = storage.get_agent_run(parent_id)
    attempts = final.generation_request_snapshot["overflow"]["attempts"]
    assert len(attempts) == calls_after_compute
    assert (
        sum(item["usage"]["prompt_tokens"] for item in attempts)
        == 100 * calls_after_compute
    )
    worker.drain_org("stream-test", 1, 300)
    assert dispatch.call_count == calls_after_compute


@pytest.mark.parametrize("reject_again", [False, True])
def test_http_capacity_rejection_freezes_one_smaller_plan_without_refunding_work(
    pipeline, monkeypatch, reject_again
):
    import json
    from types import SimpleNamespace

    from reflexio.server.llm.input_budget import (
        ProviderInputOverflowError,
        extraction_request_identity,
    )
    from reflexio.server.llm.litellm_client import LiteLLMClient

    engine, worker, client = pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    rejected = []
    original_attempts = []
    accepted = []

    def respond(params, hard_timeout):
        identity = extraction_request_identity.get()
        assert identity is not None
        prompt = json.dumps(params["messages"], ensure_ascii=False)
        is_protection = "Select original evidence needed" in prompt
        if not is_protection and (not rejected or reject_again):
            rejected.append(identity["request_fingerprint"])
            raise ProviderInputOverflowError(identity)
        accepted.append(identity["request_fingerprint"])
        result = (
            {"evidence": [], "candidate_evidence": []}
            if is_protection
            else {"profiles": []}
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(result), tool_calls=None, refusal=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    dispatch = Mock(side_effect=respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    data = payload("repartition", force=True)
    data["interaction_data_list"][0]["content"] = "Synthetic tool progress. " * 6000
    assert client.post("/api/publish_interaction", json=data).json()["success"]
    storage = engine.get_storage()
    assert worker.drain_org("stream-test", 1, 300) == 0
    with storage._stream_sql() as db:
        parent_id = db.query("SELECT id FROM _agent_runs LIMIT 1")[0]["id"]
        cursor = db.query(
            "SELECT attempts FROM extraction_cursors WHERE kind='profile' LIMIT 1"
        )[0]
    parent = storage.get_agent_run(parent_id)
    manifest = parent.generation_request_snapshot["overflow"]
    plan = manifest["checkpoint"]["repartition"]
    assert (
        plan["count"] == 1
        and len(plan["chunk_digests"]) == len(manifest["chunk_digests"]) + 1
    )
    assert cursor["attempts"] == 0
    original_attempts = list(manifest["attempts"])
    original_receipts = dict(manifest["checkpoint"]["completed_steps"])
    assert original_attempts[-1]["usage"] is None
    assert (
        storage.count_all_profiles() == 0 and storage.pending_extraction_effects() == []
    )
    # A new worker reconstructs the already-frozen source plan after the yield.
    with storage._stream_sql() as db:
        db.query("UPDATE learning_work SET due_at=0")
        db.query("UPDATE extraction_cursors SET retry_at=0")
    worker.drain_org("stream-test", 1, 300)
    after = storage.get_agent_run(parent_id).generation_request_snapshot["overflow"]
    assert after["checkpoint"]["repartition"] == plan
    assert after["attempts"][: len(original_attempts)] == original_attempts
    assert all(
        after["checkpoint"]["completed_steps"][key] == value
        for key, value in original_receipts.items()
    )
    assert len(after["attempts"]) == dispatch.call_count <= 128
    assert accepted.count(rejected[0]) == 0
    assert len(rejected) == len(set(rejected))
    if reject_again:
        assert storage.extraction_status("u", "repartition") == {
            "status": "failed",
            "reason": "input_budget_exceeded",
        }
        before = dispatch.call_count
        worker.drain_org("stream-test", 1, 300)
        assert dispatch.call_count == before
    else:
        assert storage.extraction_status("u", "repartition")["status"] == "done"
        assert after["checkpoint"]["source_coverage"] == plan["chunk_digests"]
    assert len(storage.get_user_interaction("u")) == 10


@pytest.mark.parametrize("enable_before_retry", [False, True])
def test_computed_outcome_survives_failed_commit_without_model_replay(
    pipeline, monkeypatch, enable_before_retry
):
    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("recover", force=True)
    ).json()["success"]
    original = ProfileGenerationService.persist_generation
    monkeypatch.setattr(
        ProfileGenerationService,
        "persist_generation",
        lambda *_: (_ for _ in ()).throw(RuntimeError("commit unavailable")),
    )
    worker.drain_org("stream-test", 1, 300)
    storage = engine.get_storage()
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status("u", "recover")["reason"] == "retrying"
    monkeypatch.setattr(ProfileGenerationService, "persist_generation", original)
    if enable_before_retry:
        monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
        monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    dispatch = Mock(
        side_effect=AssertionError("Cached outcome must not reach a provider")
    )
    monkeypatch.setattr(
        "reflexio.server.llm.litellm_client.LiteLLMClient._completion_with_hard_timeout",
        dispatch,
    )
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("computed output must be reused")
        ),
    )
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_cursors SET retry_at=0")
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert storage.extraction_status("u", "recover")["status"] == "done"
    assert storage.count_all_profiles() > 0
    dispatch.assert_not_called()


def test_billing_retry_does_not_repeat_extraction(pipeline, monkeypatch):
    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
    engine, worker, client = pipeline
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor

    assert client.post(
        "/api/publish_interaction", json=payload("billing", force=True)
    ).json()["success"]
    original = WindowExecutor._bill
    monkeypatch.setattr(
        WindowExecutor,
        "_bill",
        lambda *_: (_ for _ in ()).throw(RuntimeError("billing unavailable")),
    )
    worker.drain_org("stream-test", 2, 300)
    storage = engine.get_storage()
    assert storage.extraction_status("u", "billing")["status"] == "done"
    before = storage.count_all_profiles()
    assert before > 0
    monkeypatch.setattr(WindowExecutor, "_bill", original)
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda *_: (_ for _ in ()).throw(AssertionError("no model replay for billing")),
    )
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_windows SET effects_retry_at=0")
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert storage.pending_extraction_effects() == []
    assert storage.count_all_profiles() == before
    commits = [
        call for call in sink.record.call_args_list if call.args[0] == "worker.commits"
    ]
    attempts = [
        call.kwargs["attributes"]
        for call in sink.record.call_args_list
        if call.args[0] == "worker.attempts"
    ]
    assert len(commits) == 1
    assert attempts == [
        {"phase": "window", "outcome": "success"},
        {"phase": "effects", "outcome": "failure"},
        {"phase": "effects", "outcome": "success"},
    ]


def test_transient_failures_retry_beyond_three_attempts(pipeline, monkeypatch):
    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("retry")).json()[
        "success"
    ]
    attempts = []

    def gate(*_):
        attempts.append(1)
        if len(attempts) <= 4:
            raise RuntimeError("provider temporarily unavailable")
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    storage = engine.get_storage()
    for _ in range(5):
        worker.drain_org("stream-test", 1, 300)
        with storage._stream_sql() as db:
            db.query("UPDATE extraction_cursors SET retry_at=0")
    assert len(attempts) == 5
    assert storage.extraction_status("u", "retry")["status"] == "done"


def test_worker_heartbeat_keeps_long_window_exclusive(pipeline, monkeypatch):
    engine, worker, client = pipeline
    entered, release = threading.Event(), threading.Event()

    def gate(*_):
        entered.set()
        assert release.wait(8)
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    assert client.post("/api/publish_interaction", json=payload("long")).json()[
        "success"
    ]
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(worker.drain_org, "stream-test", 1, 3)
        assert entered.wait(3)
        time.sleep(3.2)
        assert engine.get_storage().claim_extraction("racing-worker", 3) is None
        release.set()
        assert running.result(timeout=3) == 1


def test_compute_has_no_learning_writes_and_commit_has_no_model_calls(
    pipeline, monkeypatch
):
    engine, _, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("pure", force=True)
    ).json()["success"]
    from reflexio.lib._base import create_generation_litellm_client
    from reflexio.server.services.durable_learning.window_codec import encode_plan
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor

    storage = engine.get_storage()
    user, token = storage.claim_extraction("test", 300)
    window = storage.prepare_extraction(user, token)
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    service, request, _ = executor.service(window)
    with monkeypatch.context() as patches:
        for name in (
            "add_user_profile",
            "supersede_profiles_by_ids",
            "delete_profiles_by_ids",
            "save_user_playbooks",
            "supersede_user_playbooks_by_ids",
            "delete_user_playbooks_by_ids",
            "merge_records",
            "supersede_record",
            "update_operation_state",
            "upsert_operation_state",
        ):
            patches.setattr(
                storage,
                name,
                lambda *_, **__: pytest.fail("compute wrote a learning or bookmark"),
            )
        plan = service.compute_generation(request)
    assert plan is not None
    storage.save_extraction_outcome(window, token, encode_plan(plan, service))
    replay = storage.prepare_extraction(user, token)
    with monkeypatch.context() as patches:
        patches.setattr(
            ProfileGenerationService,
            "compute_generation",
            lambda *_: pytest.fail("must reuse compute"),
        )
        for name in ("_get_embedding", "_expand_document"):
            patches.setattr(
                storage,
                name,
                lambda *_, **__: pytest.fail("network work during commit"),
            )
        patches.setattr(
            "litellm.completion", lambda *_, **__: pytest.fail("model during commit")
        )
        executor.execute(replay, token)
    assert storage.count_all_profiles() > 0
    assert storage.extraction_status(user, "pure")["status"] == "done"


def test_superseded_worker_rolls_back_artifacts_cursor_and_billing(
    pipeline, monkeypatch
):
    engine, worker, client = pipeline
    storage = engine.get_storage()
    assert client.post(
        "/api/publish_interaction", json=payload("fenced", force=True)
    ).json()["success"]
    original = storage.complete_extraction

    def superseded(window, token, effects):
        return original(window, "superseded", effects)

    monkeypatch.setattr(storage, "complete_extraction", superseded)
    worker.drain_org("stream-test", 1, 300)
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status("u", "fenced")["status"] != "done"
    assert storage.pending_extraction_effects() == []


def test_best_effort_tagging_failure_does_not_replay_winner_dispatch(
    pipeline, monkeypatch
):
    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("tag", force=True)
    ).json()["success"]
    calls = []

    def tagging(**kwargs):
        calls.append(kwargs["user_id"])
        if len(calls) == 1:
            raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_scheduler.schedule_tagging", tagging
    )
    worker.drain_org("stream-test", 2, 300)
    assert calls == ["u"]
    storage = engine.get_storage()
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda *_: pytest.fail("no extraction replay"),
    )
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_windows SET effects_retry_at=0")
    worker.drain_org("stream-test", 1, 300)
    assert calls == ["u"]
    assert storage.pending_extraction_effects() == []


def test_occupied_budget_leaves_durable_work_unclaimed(pipeline, monkeypatch):
    from reflexio.server.services.durable_learning import worker as module

    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("capacity")).json()[
        "success"
    ]
    budget = threading.BoundedSemaphore(1)
    monkeypatch.setattr(module, "_budget", budget)
    assert budget.acquire(False)
    assert not worker.start_org("stream-test", 300)
    storage = engine.get_storage()
    with storage._stream_sql() as db:
        assert (
            db.query("SELECT lease_token FROM learning_work")[0]["lease_token"] is None
        )
    budget.release()
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", lambda *_: False
    )
    assert worker.drain_org("stream-test", 1, 300) == 1


def test_two_users_compute_concurrently_and_same_user_waits(pipeline, monkeypatch):
    _, worker, client = pipeline
    from reflexio.server.services.durable_learning import worker as module

    monkeypatch.setattr(module, "_budget", threading.BoundedSemaphore(2))
    entered = threading.Barrier(2)
    release = threading.Event()

    def gate(*_):
        entered.wait(timeout=3)
        assert release.wait(3)
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    for user in ["u", "v"]:
        assert client.post(
            "/api/publish_interaction", json=payload(user, user_id=user)
        ).json()["success"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [pool.submit(worker.drain_org, "stream-test", 1, 300) for _ in range(2)]
        deadline = time.monotonic() + 3
        while entered.n_waiting != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        assert [run.result(timeout=5) for run in runs] == [1, 1]


def test_http_budget_includes_ingestion_and_never_acknowledges_uncommitted_work(
    pipeline, monkeypatch
):
    engine, _, client = pipeline
    from reflexio.server.routes import interactions as routes

    monkeypatch.setattr(routes, "PUBLISH_REQUEST_TIMEOUT_SECONDS", 0.1)
    released = threading.Event()

    def slow_embedding(*_):
        assert released.wait(2)

    monkeypatch.setattr(
        engine.get_storage(), "prepare_interaction_embeddings", slow_embedding
    )
    import asyncio

    import httpx

    async def invoke():
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client.app), base_url="http://test"
            ) as session:
                response = await session.post(
                    "/api/publish_interaction", json=payload("slow")
                )
            assert response.status_code == 504
            assert response.json()["detail"]["request_id"] == "slow"
            assert time.monotonic() - started < 1
            assert engine.get_storage().get_request("slow") is None
        finally:
            released.set()

    asyncio.run(invoke())


def test_sync_tail_timeout_keeps_durable_pending_work(pipeline, monkeypatch):
    """A sync tail that runs out of clock must not discard the durable work.

    Isolates the timeout branch rather than racing for it. Two earlier shapes
    both failed to reach it reliably: publishing 1 interaction now returns
    early on ``waiting_for_window``, and publishing a full window made the
    0.1s deadline expire during ADMISSION instead (a flaky 504 under parallel
    load, since admitting 10 interactions with embeddings is not instant).

    So: admit a single interaction against a deadline generous enough to
    commit, then hold coverage at ``queued`` -- extraction genuinely in
    progress, which is the state the wait is meant to sit through -- and let
    the clock run out. The stub is undone before the final assertion so the
    durable state is read from real storage.
    """
    engine, _, client = pipeline
    storage = engine.get_storage()
    monkeypatch.setattr(
        "reflexio.server.routes.interactions.PUBLISH_REQUEST_TIMEOUT_SECONDS", 2.0
    )
    monkeypatch.setattr(
        storage,
        "extraction_status",
        lambda *_args, **_kwargs: {"status": "pending", "reason": "queued"},
    )

    started = time.monotonic()
    response = client.post(
        "/api/publish_interaction?wait_for_response=true", json=payload("tailwait", 1)
    )
    elapsed = time.monotonic() - started

    assert response.status_code == 200 and response.json()["success"]
    assert response.json()["learning_reason"] == "wait_timeout"
    # It waited rather than bailing out immediately -- otherwise this would
    # pass without the timeout branch ever running.
    assert elapsed >= 1.0, f"returned after {elapsed:.2f}s; the wait did not run"

    monkeypatch.undo()
    assert storage.extraction_status("u", "tailwait")["status"] == "pending"


def test_sync_tail_returns_early_on_incomplete_window(pipeline):
    """A partial window ends the wait at once, and keeps the work pending.

    The fixture's ``window_size`` is 10, so a single interaction cannot close a
    window and no amount of waiting will change that. The full
    ``PUBLISH_REQUEST_TIMEOUT_SECONDS`` (240s) is left in place deliberately:
    the point is that the route does not consume it. The reason must stay
    distinguishable from a real timeout -- ``wait_timeout`` on a healthy stream
    would send an operator looking for a stall that is not there.
    """
    engine, _, client = pipeline
    started = time.monotonic()
    response = client.post(
        "/api/publish_interaction?wait_for_response=true", json=payload("partial", 1)
    )
    elapsed = time.monotonic() - started

    assert response.status_code == 200 and response.json()["success"]
    assert response.json()["learning_status"] == "deferred"
    assert response.json()["learning_reason"] == "waiting_for_window"
    assert elapsed < 30, f"route consumed {elapsed:.1f}s of the publish deadline"
    # The input is admitted and still owed extraction -- returning early is not
    # the same as dropping the work.
    assert engine.get_storage().extraction_status("u", "partial")["status"] == "pending"


def test_waiter_capacity_does_not_block_admission(pipeline):
    engine, _, client = pipeline
    from reflexio.server.services.durable_learning.waiting import (
        acquire_waiter,
        release_waiter,
    )

    for _ in range(8):
        assert acquire_waiter("stream-test")
    try:
        response = client.post(
            "/api/publish_interaction?wait_for_response=true", json=payload("waiters")
        )
        assert response.json()["success"]
        assert response.json()["learning_reason"] == "waiter_capacity"
        assert engine.get_storage().get_request("waiters") is not None
    finally:
        for _ in range(8):
            release_waiter("stream-test")


def test_outbox_ack_failure_does_not_repeat_derived_dispatch(pipeline, monkeypatch):
    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("ack", force=True)
    ).json()["success"]
    dispatches = []
    monkeypatch.setattr(
        ProfileGenerationService,
        "emit_generation_side_effects",
        lambda *_: dispatches.append("profile"),
    )
    storage = engine.get_storage()
    ack = storage.ack_extraction_effects
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ack unavailable")
        return ack(*args, **kwargs)

    monkeypatch.setattr(storage, "ack_extraction_effects", fail_first)
    worker.drain_org("stream-test", 2, 300)
    assert dispatches == ["profile"]
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_windows SET effects_retry_at=0")
        db.query("UPDATE learning_work SET due_at=0")
    worker.drain_org("stream-test", 2, 300)
    assert dispatches == ["profile"]
    assert storage.pending_extraction_effects() == []


def test_erasure_cancels_uncommitted_window_run_instead_of_resuming_it(
    pipeline, monkeypatch
):
    from datetime import UTC, datetime, timedelta

    from reflexio.server.services.storage.storage_base import (
        AgentBinding,
        AgentRunRecord,
        AgentRunStatus,
    )

    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("erased", force=True)
    ).json()["success"]
    monkeypatch.setattr(
        ProfileGenerationService,
        "persist_generation",
        lambda *_: (_ for _ in ()).throw(RuntimeError("commit unavailable")),
    )
    worker.drain_org("stream-test", 1, 300)
    storage = engine.get_storage()
    with storage._stream_sql() as db:
        window_id = db.query("SELECT window_id FROM extraction_windows")[0]["window_id"]
    # The external model double skips the resumable loop, so seed the same
    # durable agent-completed state that a crash after a real model call leaves.
    storage.create_agent_run(
        AgentRunRecord(
            id=f"window:{window_id}",
            binding=AgentBinding(
                org_id="stream-test",
                user_id="u",
                extractor_kind="profile",
                request_id="erased",
                agent_version=None,
                source="test",
            ),
            status=AgentRunStatus.AGENT_COMPLETED,
            generation_request_snapshot={},
            committed_output={"profiles": []},
        )
    )
    # Even an old completed model call belongs to the window engine until its
    # learning writes and cursor commit together.
    assert (
        storage.claim_finalization_failed_agent_run(
            org_id="stream-test",
            worker_id="resume",
            now=datetime.now(UTC) + timedelta(hours=1),
        )
        is None
    )
    storage.delete_all_interactions()
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_cursors SET retry_at=0")
        db.query("UPDATE learning_work SET due_at=0")
    worker.drain_org("stream-test", 1, 300)
    assert (
        storage.get_agent_run(f"window:{window_id}").status == AgentRunStatus.CANCELLED
    )
    assert storage.count_all_profiles() == 0


def _saved_legacy_window(pipeline, monkeypatch, *, kind, empty=False, schema=None):
    """Create a real source-bound completed run before overflow admission."""
    from dataclasses import replace

    from reflexio.lib._base import create_generation_litellm_client
    from reflexio.models.config_schema import UserPlaybookExtractorConfig
    from reflexio.server.services.durable_learning.window_executor import (
        WindowExecutor,
        load_window_inputs,
    )
    from reflexio.server.services.extraction.agent_run_records import (
        build_extractor_agent_run_record,
    )
    from reflexio.server.services.extraction.resumable_agent import (
        encode_committed_output,
    )
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredPlaybookList,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )
    from reflexio.server.services.storage.storage_base import AgentRunStatus

    engine, _, http = pipeline
    if kind == "playbook":
        configurator = engine.request_context.configurator
        configurator.set_config(
            configurator.get_config().model_copy(
                update={
                    "profile_extractor_config": None,
                    "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                        extraction_definition_prompt="Extract scoped approval practices."
                    ),
                }
            )
        )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    source_text = "Prepare a draft only; obtain separate approval before deploying."
    data = payload("legacy-saved", force=True)
    data["agent_version"] = "v1"
    for interaction in data["interaction_data_list"]:
        interaction["content"] = source_text
    assert http.post("/api/publish_interaction", json=data).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("legacy-test", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None
    assert window.kind == kind
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    service, request, config = executor.service(window)
    groups = load_window_inputs(storage, window)
    output = (
        StructuredPlaybookList.model_validate(
            {
                "playbooks": []
                if empty
                else [
                    {
                        "content": "Prepare a draft and obtain separate deployment approval.",
                        "trigger": "When preparing a release",
                    }
                ]
            }
        )
        if kind == "playbook"
        else StructuredProfilesOutput.model_validate(
            {
                "profiles": []
                if empty
                else [
                    {
                        "content": "The user requires separate deployment approval.",
                        "time_to_live": "infinity",
                    }
                ]
            }
        )
    )
    run = build_extractor_agent_run_record(
        org_id=storage.org_id,
        extractor_kind=kind,
        user_id=user,
        generation_request_id=request.request_id,
        agent_version=getattr(request, "agent_version", None),
        source=request.source,
        request_interaction_data_models=groups,
        extractor_config=config,
        service_config=service.service_config,
        agent_context=engine.request_context.configurator.get_agent_context(),
        output_schema_name=schema or type(output).__name__,
    )
    run = replace(
        run,
        id=f"window:{window.window_id}",
        status=AgentRunStatus.AGENT_COMPLETED,
        committed_output=encode_committed_output(output, None),
    )
    storage.create_agent_run(run)
    return (
        engine,
        storage,
        window,
        token,
        executor,
        service,
        request,
        config,
        groups,
        run,
    )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_single_chunk_adopted_empty_output_scans_source_and_publishes(
    pipeline, monkeypatch, kind
):
    from types import SimpleNamespace

    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.services.extraction.protected_record import (
        ProtectedRecordSelection,
    )

    _, storage, window, token, executor, _, _, _, _, run = _saved_legacy_window(
        pipeline, monkeypatch, kind=kind, empty=True
    )
    scan = Mock(
        return_value=SimpleNamespace(
            value=ProtectedRecordSelection(), provenance=ModelProvenance()
        )
    )
    monkeypatch.setattr(executor.client, "generate_chat_response_with_provenance", scan)
    monkeypatch.setattr(
        executor.client,
        "_completion_with_hard_timeout",
        Mock(side_effect=AssertionError("Saved extraction must not redispatch")),
    )
    executor.execute(window, token)
    assert scan.call_count == 1
    parent = storage.get_agent_run(run.id)
    checkpoint = parent.generation_request_snapshot["overflow"]["checkpoint"]
    assert len(checkpoint["protected_coverage"]) == 1
    assert checkpoint["source_coverage"] == checkpoint["protected_coverage"]
    assert parent.committed_output == run.committed_output
    assert storage.extraction_status(window.user_id, "legacy-saved")["status"] == "done"
    assert storage.count_all_profiles() == 0
    assert storage.get_user_playbooks(user_id=window.user_id) == []


@pytest.mark.parametrize(
    ("kind", "failure"),
    [
        (kind, failure)
        for kind in ("profile", "playbook")
        for failure in (None, "unsupported_schema", "nonquote", "out_of_source")
    ]
    + [("playbook", "rewrite")],
)
def test_adopted_legacy_output_reaches_complete_publication_with_verified_evidence(
    pipeline, monkeypatch, kind, failure
):
    import json
    from types import SimpleNamespace

    from reflexio.server.services.durable_learning.window_executor import (
        load_window_inputs,
    )
    from reflexio.server.services.extraction.input_preparation import (
        prepare_extraction_input,
    )
    from reflexio.server.services.playbook.components.reviewer import (
        PlaybookCandidateReviewer,
    )
    from reflexio.server.services.playbook.playbook_service_utils import (
        uses_evidence_grounded_extraction,
    )

    engine, storage, window, token, executor, _, _, _, _, run = _saved_legacy_window(
        pipeline,
        monkeypatch,
        kind=kind,
        schema="UntrustedSchema" if failure == "unsupported_schema" else None,
    )
    assert run.committed_output is not None
    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    assert len(prepared.chunks) == 1
    span = prepared.chunks[0].spans[0]
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        storage, "_get_embedding", lambda *_args, **_kwargs: [0.1] * 512
    )
    monkeypatch.setattr(storage, "_expand_document", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(PlaybookCandidateReviewer, "is_enabled", lambda _: True)
    if kind == "playbook":
        assert uses_evidence_grounded_extraction(
            engine.request_context.prompt_manager, expert=False
        )
    schemas = []

    def respond(params, hard_timeout):
        name = getattr(params.get("response_format"), "__name__", "") or json.dumps(
            params.get("response_format"), default=str
        )
        prompt = json.dumps(params["messages"], ensure_ascii=False)
        # MiniMax requests JSON-object mode; identify the actual template stage.
        if "Select original evidence needed" in prompt:
            name = "ProtectedRecordSelection"
        elif "reason_code" in prompt:
            name = (
                "EvidenceFirstReviewOutput"
                if "supporting_evidence" in prompt
                else "ReviewOutput"
            )
        elif "unique_ids" in prompt:
            name = "ProfileDeduplicationOutput"
        schemas.append(name)
        if "ProtectedRecordSelection" in name:
            result = {
                "candidate_evidence": [
                    {
                        "candidate_ref": "C1",
                        "kind": "instruction",
                        "evidence_refs": [
                            "i999999:content:0:10"
                            if failure == "out_of_source"
                            else span.evidence_ref
                        ],
                        **(
                            {"quote": "not an original quotation"}
                            if failure == "nonquote"
                            else {}
                        ),
                    }
                ]
            }
        elif "ReviewOutput" in name:
            result = {
                "decisions": [
                    {
                        "id": "C1",
                        "decision": "accept",
                        "reason_code": "grounded_useful",
                        "evidence_ids": ["C1-E1"],
                    }
                ],
            }
            if failure == "rewrite":
                result["decisions"][0].update(
                    decision="revise",
                    reason_code="unsupported_evidence",
                    revision={
                        "content": "Deploy without requesting approval.",
                        "trigger": "Before deploying",
                        "rationale": "The model inferred a new approval rule.",
                    },
                )
            if "EvidenceFirstReviewOutput" in name:
                result["supporting_evidence"] = [
                    {
                        "id": "C1",
                        "excerpts": [{"evidence_id": "C1-E1", "text": span.text}],
                    }
                ]
        elif "ProfileDeduplicationOutput" in name or any(
            "unique_ids" in str(message.get("content", ""))
            for message in params["messages"]
        ):
            result = {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
        else:
            raise AssertionError(f"Saved extraction must not redispatch: {name}")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(result),
                        tool_calls=None,
                        refusal=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    monkeypatch.setattr(executor.client, "_completion_with_hard_timeout", respond)
    if failure and failure != "rewrite":
        from reflexio.server.llm._litellm_types import StructuredOutputRepairError
        from reflexio.server.llm.input_budget import InputBudgetConfigurationError

        expected_error = (
            InputBudgetConfigurationError
            if failure == "unsupported_schema"
            else StructuredOutputRepairError
        )
        with pytest.raises(expected_error):
            executor.execute(window, token)
        assert (
            storage.extraction_status(window.user_id, "legacy-saved")["status"]
            != "done"
        )
        assert storage.count_all_profiles() == 0
        assert storage.get_user_playbooks(user_id=window.user_id) == []
        assert not any(
            "ReviewOutput" in name or "DeduplicationOutput" in name for name in schemas
        )
    else:
        executor.execute(window, token)
        assert (
            storage.extraction_status(window.user_id, "legacy-saved")["status"]
            == "done"
        )
        parent = storage.get_agent_run(run.id)
        checkpoint = parent.generation_request_snapshot["overflow"]["checkpoint"]
        assert checkpoint["source_coverage"] == checkpoint["protected_coverage"]
        assert (
            checkpoint["candidate_evidence"]["C1"]["evidence"][0]["quote"] == span.text
        )
        if kind == "playbook":
            persisted = storage.get_user_playbooks(user_id=window.user_id)
            if failure == "rewrite":
                assert persisted == []
                assert storage.count_all_profiles() == 0
            else:
                assert len(persisted) == 1
                candidate = persisted[0]
                assert (
                    candidate.content
                    == run.committed_output["output"]["playbooks"][0]["content"]
                )
                assert (
                    candidate.source_interaction_ids
                    == run.binding.source_interaction_ids
                )
                assert (
                    candidate.source_span == span.text
                    and candidate.reader_angle is None
                    and candidate.rationale is None
                )
            assert any("ReviewOutput" in name for name in schemas)
        else:
            persisted_profiles = storage.get_user_profile(window.user_id)
            assert len(persisted_profiles) == 1
            assert (
                persisted_profiles[0].content
                == run.committed_output["output"]["profiles"][0]["content"]
            )
            assert (
                persisted_profiles[0].source_interaction_ids
                == run.binding.source_interaction_ids
            )
            assert (persisted_profiles[0].custom_features or {}).get(
                "source_span"
            ) is None
            assert any("ProfileDeduplicationOutput" in name for name in schemas)
    assert storage.get_agent_run(run.id).committed_output == run.committed_output


@pytest.mark.parametrize(
    "failure", ["guard", "budget", "quota", "suspend", "checkpoint", "lease"]
)
def test_profile_consolidation_control_failure_preserves_extracted_window_without_publication(
    pipeline, monkeypatch, failure
):
    import json
    from types import SimpleNamespace

    from reflexio.lib._base import create_generation_litellm_client
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError
    from reflexio.server.llm.input_budget import InputBudgetExceededError
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowSuspendedError,
        current_overflow_coordinator,
    )
    from reflexio.server.services.profile.components.consolidator import (
        ProfileDeduplicationOutput,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
        OverflowAttemptBudgetExceededError,
        OverflowCheckpointConflictError,
    )

    engine, _, http = pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    data = payload("profile-consolidation-stop", force=True)
    quote = "The user prefers concise explanations with clear examples."
    for item in data["interaction_data_list"]:
        item["content"] = quote
    assert http.post("/api/publish_interaction", json=data).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("guard-consolidation", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    error = {
        "guard": ProviderRequestGuardError,
        "budget": InputBudgetExceededError,
        "quota": OverflowAttemptBudgetExceededError,
        "suspend": OverflowSuspendedError,
        "checkpoint": OverflowCheckpointConflictError,
        "lease": LeaseLostError,
    }[failure]("Injected stop after full source coverage")
    monkeypatch.setattr(
        storage, "_get_embedding", lambda *_args, **_kwargs: [0.1] * 512
    )
    monkeypatch.setattr(storage, "_expand_document", lambda *_args, **_kwargs: None)

    def respond(params, hard_timeout):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "profiles": [
                                    {
                                        "content": "The user prefers concise explanations.",
                                        "time_to_live": "infinity",
                                        "source_span": quote,
                                    }
                                ]
                            }
                        ),
                        tool_calls=None,
                        refusal=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    dispatch = Mock(side_effect=respond)
    monkeypatch.setattr(executor.client, "_completion_with_hard_timeout", dispatch)
    generate = executor.client.generate_chat_response_with_provenance
    checkpoints = []

    def fail_consolidation(*args, **kwargs):
        if kwargs.get("response_format") is ProfileDeduplicationOutput:
            coordinator = current_overflow_coordinator.get()
            assert coordinator is not None
            coordinator.require_full_source_coverage()
            checkpoints.append(coordinator.checkpoint)
            raise error
        return generate(*args, **kwargs)

    monkeypatch.setattr(
        executor.client, "generate_chat_response_with_provenance", fail_consolidation
    )
    writes = {}
    for name in (
        "add_user_profile",
        "save_extraction_outcome",
        "complete_extraction",
        "update_operation_state",
        "upsert_operation_state",
    ):
        spy = Mock(wraps=getattr(storage, name))
        monkeypatch.setattr(storage, name, spy)
        writes[name] = spy
    billing = Mock(wraps=executor._billing_snapshot)
    monkeypatch.setattr(executor, "_billing_snapshot", billing)
    with patch("reflexio.server.metering_events.record_extraction_tokens") as bill:
        with pytest.raises(type(error)) as caught:
            executor.execute(window, token)
        assert caught.value is error
        bill.assert_not_called()
    billing.assert_not_called()
    for write in writes.values():
        write.assert_not_called()
    assert storage.count_all_profiles() == 0
    assert (
        storage.extraction_status(window.user_id, "profile-consolidation-stop")[
            "status"
        ]
        != "done"
    )
    assert len(checkpoints) == 1 and checkpoints[0]["stage"] == "extracted"
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent.committed_output is not None
    assert (
        parent.generation_request_snapshot["overflow"]["checkpoint"] == checkpoints[0]
    )
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT completed_seq FROM extraction_cursors WHERE user_id=? AND kind=?",
            (user, "profile"),
        )[0]
        saved = db.query(
            "SELECT completed,outcome,effects FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        )[0]
    assert cursor["completed_seq"] == 0
    assert (
        not saved["completed"] and saved["outcome"] is None and saved["effects"] is None
    )
    # The durable raw result remains readable for a later turn, not abandoned.
    assert parent.committed_output["output"]["profiles"][0]["source_span"] == quote
    assert dispatch.call_count == 1


@pytest.mark.parametrize(
    "failure", ["guard", "quota", "checkpoint", "lease", "operational"]
)
def test_dedup_embedding_fallback_preserves_control_errors(failure):
    from types import SimpleNamespace

    from reflexio.server.llm._litellm_types import ProviderRequestGuardError
    from reflexio.server.services.deduplication_utils import (
        resolve_dedup_query_embeddings,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
        OverflowAttemptBudgetExceededError,
        OverflowCheckpointConflictError,
    )

    error = {
        "guard": ProviderRequestGuardError,
        "quota": OverflowAttemptBudgetExceededError,
        "checkpoint": OverflowCheckpointConflictError,
        "lease": LeaseLostError,
        "operational": RuntimeError,
    }[failure]("embedding failure")
    storage = SimpleNamespace(
        embedding_model_name="fixture", _get_embedding=Mock(side_effect=error)
    )
    if failure == "operational":
        assert resolve_dedup_query_embeddings(
            storage, Mock(), ["query"], entity_label="Profile"
        ) == [None]
    else:
        with pytest.raises(type(error)) as caught:
            resolve_dedup_query_embeddings(
                storage, Mock(), ["query"], entity_label="Profile"
            )
        assert caught.value is error
