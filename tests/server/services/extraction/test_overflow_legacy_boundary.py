"""Protocol parents never acquire legacy provider or publication ownership."""

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock

import pytest

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction.recovery import (
    RecoveryRefusedError,
    inspect_run,
    selection_from_report,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_scope_hash,
    human_feedback_scope,
)
from reflexio.test_support.skip_decorators import skip_in_precommit
from tests.server.services.extraction.test_selected_recovery_integration import (
    context as context,
)
from tests.server.services.extraction.test_selected_recovery_integration import (
    seed,
)
from tests.server.services.test_durable_window_pipeline import (
    _saved_legacy_window,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as pipeline,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("protocol", [False, True])
@skip_in_precommit
def test_public_corrected_consumed_answer_keeps_protocol_parent_out_of_legacy_workers(
    pipeline, monkeypatch, kind, protocol
):
    engine, storage, window, _, _, _, _, _, _, run = _saved_legacy_window(
        pipeline, monkeypatch, kind=kind, empty=True
    )
    configurator = engine.request_context.configurator
    storage_factory = type(configurator).create_storage

    def scoped_storage(owner, storage_config):
        if owner.org_id == storage.org_id:
            return storage
        return storage_factory(owner, storage_config)

    # Keep the public route's real context and org scope on this owned DB.
    monkeypatch.setattr(type(configurator), "create_storage", scoped_storage)
    configurator.set_config(
        configurator.get_config().model_copy(
            update={
                "pending_tool_call_config": PendingToolCallConfig(enabled=True),
            }
        )
    )
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resume_scheduler.wake_resume_schedulers",
        lambda: None,
    )
    if not protocol:
        # The baseline case proves claim eligibility without executing its LLM.
        monkeypatch.setattr(
            "reflexio.server.api_endpoints.pending_tool_call_api._drain_resumable_followups",
            lambda _: None,
        )
    now = datetime.now(UTC)
    scope = human_feedback_scope(storage.org_id)
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="consumed-answer",
            org_id=storage.org_id,
            user_id="u",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key="consumed-answer",
            question_text="What action is approved?",
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "Draft only"},
            resolved_at=now - timedelta(minutes=10),
            valid_until=now + timedelta(hours=1),
            cache_until=now + timedelta(hours=1),
            expires_at=now + timedelta(hours=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id=run.id,
            pending_tool_call_id="consumed-answer",
            resolved_at=now - timedelta(minutes=10),
            consumed_at=now - timedelta(minutes=5),
        )
    )
    snapshot = copy.deepcopy(run.generation_request_snapshot)
    if protocol:
        snapshot["overflow"] = {
            "version": 1,
            "attempt_limit": 128,
            "attempts": [
                {
                    "id": "spent",
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 10,
                        "cache_read_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                    },
                    "model_provenance": None,
                }
            ],
            "checkpoint": {
                "stage": "extracted",
                "completed_steps": {
                    "saved": {
                        "output": {"profiles": []},
                        "model_provenance": {"model_name": None, "provider": None},
                    }
                },
            },
        }
    with storage._stream_sql() as db:
        db.query(
            "UPDATE extraction_windows SET completed=1 WHERE window_id=?",
            (window.window_id,),
        )
        db.query(
            "UPDATE _agent_runs SET status='finalized', generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), run.id),
        )
    before = storage.get_agent_run(run.id)
    # Seeding completed=1 exposes the fixture's preexisting empty effects slot;
    # correction must leave that slot unchanged rather than create new effects.
    effects_before = copy.deepcopy(storage.pending_extraction_effects())
    dispatch = Mock(
        side_effect=AssertionError(
            "Correction must not dispatch a protocol parent through legacy code"
        )
    )
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    response = pipeline[2].patch(
        "/api/pending_tool_calls/consumed-answer/answer",
        json={"answer": "Corrected draft approval", "valid_for_seconds": 3600},
    )
    assert response.status_code == 200
    corrected = storage.get_agent_run(run.id)
    assert corrected.status == AgentRunStatus.RESUME_READY
    assert corrected.committed_output == before.committed_output
    dependency = storage.list_run_tool_dependencies(run.id)[0]
    assert dependency.consumed_at is None
    assert storage.get_pending_tool_call("consumed-answer").result == {
        "answer": "Corrected draft approval"
    }
    if protocol:
        assert (
            ExtractionResumeWorker(request_context=engine.request_context).run_once()
            is None
        )
        assert (
            storage.claim_ready_agent_run(org_id=storage.org_id, worker_id="legacy")
            is None
        )
        assert storage.org_id not in storage.list_resumable_work_org_ids()
        assert (
            "overflow_protocol_recovery_unsupported"
            in inspect_run(engine.request_context, corrected)["blockers"]
        )
        assert storage.get_agent_run(run.id).generation_request_snapshot == snapshot
        assert storage.get_agent_run(run.id).committed_output == before.committed_output
        assert storage.list_run_tool_dependencies(run.id)[0].consumed_at is None
        assert storage.count_all_profiles() == 0
        with storage._stream_sql() as db:
            assert (
                db.query(
                    "SELECT user_playbook_id FROM user_playbooks WHERE user_id=? LIMIT 1",
                    ("u",),
                )
                == []
            )
        assert storage.pending_extraction_effects() == effects_before
    else:
        assert (
            storage.claim_ready_agent_run(org_id=storage.org_id, worker_id="legacy").id
            == run.id
        )
    dispatch.assert_not_called()


@pytest.mark.parametrize("marker", [None, {}, "invalid"])
@pytest.mark.parametrize(
    "status",
    [
        AgentRunStatus.AGENT_COMPLETED,
        AgentRunStatus.FINALIZATION_FAILED,
        AgentRunStatus.FINALIZING,
    ],
)
def test_all_protocol_markers_block_legacy_finalization_and_selected_claims(
    context, marker, status
):
    store = context.storage
    run = seed(context)
    with store._stream_sql() as db:
        snapshot = {**run.generation_request_snapshot, "overflow": marker}
        old = datetime.now(UTC) - timedelta(days=1)
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=?,status=?,claimed_at=?,updated_at=? WHERE id=?",
            (
                json.dumps(snapshot),
                status.value,
                old.isoformat(),
                old.isoformat(),
                run.id,
            ),
        )
    run = store.get_agent_run(run.id)
    assert (
        store.claim_finalization_failed_agent_run(
            org_id=context.org_id, worker_id="legacy"
        )
        is None
    )
    assert (
        store.claim_agent_run_for_recovery(
            run_id=run.id,
            org_id=context.org_id,
            worker_id="selected",
            expected_updated_at=run.updated_at,
            operation_id="operation",
        )
        is None
    )
    assert store.get_agent_run(run.id).status == status
    assert context.org_id not in store.list_resumable_work_org_ids()


@pytest.mark.parametrize("finalization", [False, True])
def test_stale_claimed_protocol_parent_is_refused_before_any_legacy_effect(
    context, finalization
):
    run = seed(context)
    protocol = replace(
        run,
        generation_request_snapshot={
            **run.generation_request_snapshot,
            "overflow": None,
        },
    )
    client = MagicMock()
    worker = ExtractionResumeWorker(request_context=context, llm_client=client)
    with pytest.raises(
        RecoveryRefusedError, match="overflow_protocol_recovery_unsupported"
    ):
        worker._execute_claimed_run(protocol, finalization_retry=finalization)
    with pytest.raises(RecoveryRefusedError):
        worker._resume_run(protocol, [])
    with pytest.raises(RecoveryRefusedError):
        worker._retry_finalization(protocol)
    with pytest.raises(RecoveryRefusedError):
        worker._finalize_items(protocol, [])
    client.generate_chat_response_with_provenance.assert_not_called()
    assert context.storage.get_agent_run(run.id).status == run.status


def test_forced_selected_preview_cannot_recover_a_protocol_parent(context):
    store = context.storage
    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    with store._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps({**run.generation_request_snapshot, "overflow": {}}), run.id),
        )
    client = MagicMock()
    worker = ExtractionResumeWorker(request_context=context, llm_client=client)
    with pytest.raises(
        RecoveryRefusedError, match="overflow_protocol_recovery_unsupported"
    ):
        worker.recover_selected(selection, operation_id="forced-operation")
    assert store.get_agent_run(run.id).recovery_operation_id is None
    client.generate_chat_response_with_provenance.assert_not_called()
