"""Resume transitions retain the execution witness instead of borrowing by ID."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.llm.tools import ToolLoopTrace
from reflexio.server.services.durable_learning.user_lease import (
    create_owned_agent_run,
    owned_agent_run_write,
    owned_user_work_context,
)
from reflexio.server.services.extraction.history_inputs import HistoricalInputs
from reflexio.server.services.extraction.resumable_agent import (
    AgentRunResult,
    ResumableExtractionAgent,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus, LeaseLostError
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


@pytest.fixture
def execution(pipeline):
    storage, window, token = durable._overflow_coordinator(pipeline)
    engine, _, _ = pipeline
    try:
        with owned_user_work_context(storage, window.user_id, token, 300):
            template = storage.get_agent_run(f"window:{window.window_id}")
            assert template is not None
            run = create_owned_agent_run(
                storage,
                replace(
                    template,
                    id="owned_resume",
                    binding=replace(
                        template.binding,
                        source_interaction_ids=[
                            item["interaction_id"] for item in window.manifest
                        ],
                    ),
                    status=AgentRunStatus.RESUMING,
                    claimed_by=None,
                    claimed_at=None,
                    generation_request_snapshot={},
                ),
            )
            worker = ExtractionResumeWorker(
                request_context=engine.request_context, llm_client=MagicMock()
            )
            worker._active_claim = run
            yield storage, run, token, worker
    finally:
        storage.release_user_extraction(window.user_id, token)


@pytest.mark.parametrize(
    "withdrawal", ["missing_handle", "parent", "status", "during_write"]
)
def test_status_transition_requires_retained_owner_through_commit(
    execution, monkeypatch, withdrawal
):
    storage, run, token, worker = execution
    if withdrawal == "missing_handle":
        worker._active_claim = None
    elif withdrawal == "parent":
        storage._execute(
            "UPDATE _agent_runs SET user_work_token='withdrawn' WHERE id=?", (run.id,)
        )
    elif withdrawal == "status":
        storage.update_agent_run_status(run.id, AgentRunStatus.AGENT_COMPLETED)
    else:
        update = storage.update_agent_run_status

        def expire_after_update(*args, **kwargs):
            result = update(*args, **kwargs)
            with storage._stream_sql() as db:
                db.query(
                    "UPDATE learning_work SET lease_until=0 WHERE org_id=? AND user_id=?",
                    (storage.org_id, run.binding.user_id),
                )
            return result

        monkeypatch.setattr(storage, "update_agent_run_status", expire_after_update)
    before = storage.get_agent_run(run.id)
    with pytest.raises(LeaseLostError):
        worker._update_claimed_status(run.id, AgentRunStatus.FINALIZING)
    assert storage.get_agent_run(run.id) == before
    # The forced expiry is part of the rejected write and must roll back too.
    storage.fence_user_extraction(run.binding.user_id, token)


def _result(run, *, witness):
    return AgentRunResult(
        run_id=run.id,
        output=StructuredProfilesOutput(profiles=[]),
        pending_tool_call_ids=[],
        messages=[],
        trace=ToolLoopTrace(finished=True),
        finished_reason="structured_output",
        execution_run=witness,
    )


def _source_inputs(worker, run):
    from reflexio.server.services.extraction.resume_worker import (
        _request_interaction_models_from_ids,
    )
    from reflexio.server.services.extraction.source_inputs import capture_source_inputs

    models = _request_interaction_models_from_ids(
        worker.storage,
        run.binding.source_interaction_ids,
        fallback_source=run.binding.source,
        fallback_agent_version=run.binding.agent_version,
    )
    return capture_source_inputs(worker.storage, run, models)


def test_agent_result_handoff_keeps_exact_committed_execution(execution, monkeypatch):
    storage, run, _, worker = execution

    def finish(_agent, **_kwargs):
        with owned_agent_run_write(storage, run):
            stored = storage.update_agent_run_status(
                run.id,
                AgentRunStatus.AGENT_COMPLETED,
                committed_output={"profiles": []},
                expected_claimed_by=run.claimed_by,
                expected_claimed_at=run.claimed_at,
            )
        assert stored is not None
        return _result(run, witness=stored)

    monkeypatch.setattr(ResumableExtractionAgent, "resume", finish)
    result = worker._resume_agent(
        run=run,
        messages=[],
        output_schema=StructuredProfilesOutput,
        history_inputs=HistoricalInputs(),
        source_inputs=_source_inputs(worker, run),
        resolved_calls=[],
        log_label="synthetic owned resume",
    )
    assert result.status == AgentRunStatus.AGENT_COMPLETED
    assert result.committed_output == {"profiles": []}
    assert worker._active_claim == result
    updated = worker._update_claimed_status(run.id, AgentRunStatus.FINALIZING)
    assert updated is not None and updated.status == AgentRunStatus.FINALIZING
    assert worker._active_claim == updated


@pytest.mark.parametrize("invalid", ["missing", "another_run", "another_claim"])
def test_agent_result_cannot_supply_unrelated_execution_authority(
    execution, monkeypatch, invalid
):
    _, run, _, worker = execution
    witness = (
        None
        if invalid == "missing"
        else replace(run, id="different_run")
        if invalid == "another_run"
        else replace(run, claimed_by="different_claim")
    )
    monkeypatch.setattr(
        ResumableExtractionAgent,
        "resume",
        lambda *_a, **_k: _result(run, witness=witness),
    )
    with pytest.raises(LeaseLostError):
        worker._resume_agent(
            run=run,
            messages=[],
            output_schema=StructuredProfilesOutput,
            history_inputs=HistoricalInputs(),
            source_inputs=_source_inputs(worker, run),
            resolved_calls=[],
            log_label="synthetic invalid handoff",
        )
    assert worker._active_claim == run


def test_resume_attempt_charge_rolls_back_if_owner_expires_during_write(
    execution, monkeypatch
):
    storage, run, token, worker = execution
    begin = storage.begin_extraction_resume

    def expire_after_charge(*args):
        result = begin(*args)
        assert result == 1
        with storage._stream_sql() as db:
            db.query(
                "UPDATE learning_work SET lease_until=0 WHERE org_id=? AND user_id=?",
                (storage.org_id, run.binding.user_id),
            )
        return result

    monkeypatch.setattr(storage, "begin_extraction_resume", expire_after_charge)
    model = MagicMock()

    def enter_model(current, _config):
        worker._begin_model_attempt(current)
        model()

    monkeypatch.setattr(worker, "_process_claimed_run", enter_model)
    with pytest.raises(LeaseLostError):
        worker._execute_owned_claim(
            run,
            finalization_retry=False,
            pending_config=PendingToolCallConfig(),
            selection=None,
        )
    model.assert_not_called()
    assert storage.get_agent_run(run.id) == run
    storage.fence_user_extraction(run.binding.user_id, token)


def _charged(storage, run):
    with owned_agent_run_write(storage, run):
        assert (
            storage.begin_extraction_resume(
                run.id, run.claimed_by, run.claimed_at.isoformat()
            )
            == 1
        )
        current = storage.get_agent_run(run.id)
    assert current is not None
    return current


def test_readiness_refund_is_fenced_through_commit(execution, monkeypatch):
    from reflexio.server.services.storage.error import ReadinessUnavailableError

    storage, run, token, worker = execution
    charged = []
    monkeypatch.setattr(worker, "_load_resolved_tool_calls", lambda _: [object()])

    def entered_model(current, _calls):
        active = worker._begin_model_attempt(current)
        charged.append(active)
        raise ReadinessUnavailableError("controlled readiness")

    monkeypatch.setattr(worker, "_resume_run", entered_model)
    defer = storage.defer_extraction_resume

    def expire_after_refund(*args):
        assert defer(*args)
        with storage._stream_sql() as db:
            db.query(
                "UPDATE learning_work SET lease_until=0 WHERE org_id=? AND user_id=?",
                (storage.org_id, run.binding.user_id),
            )
        return True

    monkeypatch.setattr(storage, "defer_extraction_resume", expire_after_refund)
    with pytest.raises(LeaseLostError):
        worker._process_claimed_run(run, MagicMock())
    assert len(charged) == 1
    assert storage.get_agent_run(run.id) == charged[0]
    storage.fence_user_extraction(run.binding.user_id, token)


@pytest.mark.parametrize("handoff", [True, False])
def test_readiness_after_output_uses_only_returned_witness(
    execution, monkeypatch, handoff
):
    from reflexio.server.services.storage.error import ReadinessUnavailableError

    storage, run, _, worker = execution
    run = _charged(storage, run)
    worker._active_claim = run
    monkeypatch.setattr(worker, "_load_resolved_tool_calls", lambda _: [object()])
    written = []

    def stored_then_not_ready(*_args):
        with owned_agent_run_write(storage, run):
            result = storage.update_agent_run_status(
                run.id,
                AgentRunStatus.AGENT_COMPLETED,
                committed_output={"profiles": []},
            )
        assert result is not None
        written.append(result)
        if handoff:
            worker._active_claim = result
        raise ReadinessUnavailableError("controlled conversion readiness")

    monkeypatch.setattr(worker, "_resume_run", stored_then_not_ready)
    if handoff:
        result = worker._process_claimed_run(run, MagicMock())
        assert result.status == AgentRunStatus.FINALIZATION_FAILED
        assert result.committed_output == written[0].committed_output
        assert worker._active_claim == result
    else:
        with pytest.raises(LeaseLostError):
            worker._process_claimed_run(run, MagicMock())
        assert storage.get_agent_run(run.id) == written[0]
