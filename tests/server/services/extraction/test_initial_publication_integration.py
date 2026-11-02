"""Initial learning and replay use one output-scoped financial owner."""

import json

import pytest

from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
)
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
)
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction import (
    test_interrupted_finalization_budget as accounting,
)
from tests.server.services.extraction import test_resume_publication_integration as base

storage = base.storage
request_context = base.request_context
explicit_test_billing_exemption = base.explicit_test_billing_exemption
no_background_tagging = base.no_background_tagging
publication = base.publication
shared_worker = base.shared_worker
actual_embedding_dispatch = accounting.actual_embedding_dispatch
pytestmark = pytest.mark.integration
_REAL_PLAYBOOK_DISPATCH = PlaybookGenerationService._dispatch_playbook_schedulers


def _initial(publication, monkeypatch, response=None):
    storage, worker, kind, _, events = publication
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    calls = []

    def model(**kwargs):
        calls.append(1)
        return (
            response(kwargs)
            if response is not None
            else base._loop_result(kwargs, kind, storage, nonempty=True)
        )

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    service = (
        ProfileGenerationService(worker.client, worker.request_context)
        if kind == "profile"
        else PlaybookGenerationService(worker.client, worker.request_context)
    )
    if kind == "profile":
        request = ProfileGenerationRequest(
            user_id="user_1",
            request_id="request_1",
            source="api",
            auto_run=False,
            force_extraction=True,
        )
    else:
        request = PlaybookGenerationRequest(
            user_id="user_1",
            request_id="request_1",
            source="api",
            agent_version="v1",
            auto_run=False,
            force_extraction=True,
        )
    return service, request, calls


def _run_initial(service, request):
    if isinstance(service, ProfileGenerationService):
        assert isinstance(request, ProfileGenerationRequest)
        service.run(request)
    else:
        assert isinstance(service, PlaybookGenerationService)
        assert isinstance(request, PlaybookGenerationRequest)
        service.run(request)


def test_real_initial_output_has_one_learning_receipt_and_preserves_token_effect(
    publication, monkeypatch
):
    storage, _, kind, _, events = publication
    service, request, calls = _initial(publication, monkeypatch)
    _run_initial(service, request)
    run = service._last_extraction_run
    assert calls == [1]
    assert run is not None and run.status == AgentRunStatus.FINALIZED
    assert run.committed_output is not None
    output_id = run.committed_output["publication"]["id"]
    with storage._stream_sql(read_only=True) as db:
        receipt = storage._get_output_publication_receipt(db, run.id, output_id)
    assert receipt is not None, "Initial output must retain a retry-safe obligation"
    assert receipt.delivery_state == "accepted"
    assert list(receipt.learning_ids) == [
        str(row.profile_id if kind == "profile" else row.user_playbook_id)
        for row in base._learnings(storage, kind)
    ]
    learning_events = [e for e in events if e.event_name == "learnings_generated"]
    assert len(learning_events) == 1
    assert len([e for e in events if e.event_name == "extraction_tokens"]) == 1
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM _agent_run_finalization_receipts WHERE run_id=?",
            (run.id,),
        ).fetchone()[0]
        == 0
    ), "Current output must not add run-only history"


def test_initial_writer_rollback_replays_original_plan_origin_and_bookmark(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    from reflexio.server.services.operation_state_utils import OperationStateManager

    storage, worker, kind, _, events = publication
    service, request, calls = _initial(publication, monkeypatch)
    resolve = type(service)._resolve_write_plan
    resolves = []

    def count_resolve(self, groups):
        resolves.append(1)
        return resolve(self, groups)

    monkeypatch.setattr(type(service), "_resolve_write_plan", count_resolve)
    create = storage._create_agent_output_receipt
    failures = []

    def rollback(*args, **kwargs):
        if not failures:
            failures.append(1)
            raise RuntimeError("Initial writer rollback after learning rows")
        return create(*args, **kwargs)

    monkeypatch.setattr(storage, "_create_agent_output_receipt", rollback)
    _run_initial(service, request)
    first = service._last_extraction_run
    assert first is not None and first.status == AgentRunStatus.FINALIZATION_FAILED
    assert first.finalization_attempts == 1
    assert not base._learnings(storage, kind)
    assert not [
        e
        for e in events
        if e.event_name in {"learnings_generated", "extraction_tokens"}
    ]
    original = first.committed_output
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (first.id,)
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    second = storage.get_agent_run(first.id)
    assert (
        second is not None
        and second.id == first.id
        and second.status == AgentRunStatus.FINALIZED
    )
    assert second.committed_output == original
    assert calls == [1] and resolves == [1]
    assert len(base._learnings(storage, kind)) == 1
    learning_events = [e for e in events if e.event_name == "learnings_generated"]
    assert len(learning_events) == 1 and learning_events[0].source == "api"
    # Inline tokens retain their existing fail-open policy; recovery must not
    # invent a new token charge from mutable current configuration.
    assert not [e for e in events if e.event_name == "extraction_tokens"]
    manager = OperationStateManager(
        storage, storage.org_id, service._get_extractor_state_service_name()
    )
    value = storage.get_operation_state(manager._bookmark_key(kind, scope_id="user_1"))
    assert value is not None
    position = value.get("operation_state", value)
    assert set(position["last_processed_interaction_ids"]) == {1, 2, 3}
    assert position["last_processed_timestamp"] == 1002


def test_initial_pending_question_commits_empty_receipt_and_remains_resumable(
    publication, monkeypatch
):
    storage, _, kind, _, events = publication
    base._question(storage)

    def response(kwargs):
        kwargs["registry"].handle_outcome(
            "attach_pending_info_request" if kind == "profile" else "ask_human",
            json.dumps(
                {"pending_tool_call_id": "ptc_2"}
                if kind == "profile"
                else {"question": "Which approval is required?"}
            ),
            kwargs["ctx"],
        )
        return base._loop_result(
            kwargs, kind, storage, nonempty=False, pending=["ptc_2"]
        )

    service, request, calls = _initial(publication, monkeypatch, response)
    _run_initial(service, request)
    run = service._last_extraction_run
    assert calls == [1]
    assert run is not None and run.status == AgentRunStatus.FINALIZED_PENDING_TOOL
    assert run.pending_tool_call_ids == ["ptc_2"]
    assert (
        storage.get_agent_run_finalization_receipt(
            run_id=run.id,
            entity_type="profile" if kind == "profile" else "user_playbook",
        )
        == []
    )
    assert not base._learnings(storage, kind)
    assert not [e for e in events if e.event_name == "learnings_generated"]
    dependencies = storage.list_run_tool_dependencies(run.id)
    assert len(dependencies) == 1 and dependencies[0].consumed_at is None


def test_initial_receipt_replay_does_not_repeat_tokens_or_learning(
    publication, monkeypatch
):
    from reflexio.server.services.durable_learning.user_lease import (
        user_extraction_lease,
    )

    storage, _, kind, _, events = publication
    service, request, calls = _initial(publication, monkeypatch)
    with user_extraction_lease(storage, "user_1"):
        if isinstance(service, ProfileGenerationService):
            assert isinstance(request, ProfileGenerationRequest)
            plan = service.compute_generation(request)
        else:
            assert isinstance(service, PlaybookGenerationService)
            assert isinstance(request, PlaybookGenerationRequest)
            plan = service.compute_generation(request)
        assert plan is not None
        service.persist_generation(plan)
        service.emit_generation_side_effects(plan)
        first = plan.execution_run
        assert first is not None and first.status == AgentRunStatus.FINALIZED
        service.persist_generation(plan)
        assert (
            plan.finalization_result is not None
            and not plan.finalization_result.won_receipt
        )
        service.emit_generation_side_effects(plan)
    assert calls == [1] and len(base._learnings(storage, kind)) == 1
    assert len([e for e in events if e.event_name == "learnings_generated"]) == 1
    assert len([e for e in events if e.event_name == "extraction_tokens"]) == 1


def test_initial_delivery_outage_keeps_terminal_execution_and_original_facts(
    publication, monkeypatch
):
    from reflexio.server.services.extraction.publication import _deliver
    from reflexio.server.usage_metrics import (
        UsageEventDeliveryStatus,
        configure_usage_event_recorder,
    )

    storage, _, kind, _, events = publication
    service, request, calls = _initial(publication, monkeypatch)

    def reject_learning(event):
        if event.event_name == "learnings_generated":
            return UsageEventDeliveryStatus.FAILED
        events.append(event)
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(reject_learning)
    _run_initial(service, request)
    run = service._last_extraction_run
    assert run is not None and run.status == AgentRunStatus.FINALIZED
    assert run.finalization_attempts == 0 and calls == [1]
    assert len(base._learnings(storage, kind)) == 1
    assert run.committed_output is not None
    output_id = run.committed_output["publication"]["id"]
    with storage._stream_sql(read_only=True) as db:
        receipt = storage._get_output_publication_receipt(db, run.id, output_id)
    assert receipt is not None and receipt.delivery_state == "pending"
    facts = receipt.financial_facts
    assert facts is not None and facts["source"] == "api"

    def accept(event):
        events.append(event)
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(accept)

    def forbidden_config():
        raise AssertionError("No mutable model config during delivery")

    monkeypatch.setattr(
        service.request_context.configurator, "get_config", forbidden_config
    )
    _deliver(storage, receipt)
    delivered = [e for e in events if e.event_name == "learnings_generated"]
    assert len(delivered) == 1 and delivered[0].created_at == facts["created_at"]
    assert len([e for e in events if e.event_name == "extraction_tokens"]) == 1
    assert calls == [1]


@pytest.mark.parametrize(
    "interruption", ["exit_before_secondary", "known_failure", "exit_after_secondary"]
)
def test_initial_uncached_recovery_preserves_original_bookmark(
    shared_worker: DurableLearningWorker, publication, monkeypatch, interruption
):
    from reflexio.server.services.operation_state_utils import OperationStateManager

    storage, worker, kind, _, events = publication
    service, request, calls = _initial(publication, monkeypatch)
    resolve = type(service)._resolve_write_plan
    resolves = []

    def interrupt(self, groups):
        resolves.append(1)
        first = len(resolves) == 1
        if first and interruption == "known_failure":
            raise RuntimeError("Initial secondary preparation failed before completion")
        if first and interruption == "exit_before_secondary":
            raise accounting.SimulatedProcessExit("Exit before secondary compute")
        result = resolve(self, groups)
        if first and interruption == "exit_after_secondary":
            raise accounting.SimulatedProcessExit("Exit before preparation save")
        return result

    monkeypatch.setattr(type(service), "_resolve_write_plan", interrupt)
    if interruption == "known_failure":
        _run_initial(service, request)
    else:
        with pytest.raises(accounting.SimulatedProcessExit):
            _run_initial(service, request)
    saved = service._last_extraction_run
    assert saved is not None and saved.committed_output is not None
    original = saved.committed_output
    assert not base._learnings(storage, kind)
    # The exited actual owner is already released; only business retry due
    # is adjusted. Logical claim age no longer controls adoption.
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (saved.id,)
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run(saved.id)
    assert (
        result is not None
        and result.id == saved.id
        and result.status == AgentRunStatus.FINALIZED
    )
    assert result.committed_output == original and calls == [1]
    assert result.finalization_attempts == (
        0 if interruption == "exit_before_secondary" else 1
    )
    assert resolves == [1, 1] and len(base._learnings(storage, kind)) == 1
    manager = OperationStateManager(
        storage, storage.org_id, service._get_extractor_state_service_name()
    )
    value = storage.get_operation_state(manager._bookmark_key(kind, scope_id="user_1"))
    assert value is not None, (
        "Cold recovery must retain the initial bookmark before preparation exists"
    )
    position = value.get("operation_state", value)
    assert set(position["last_processed_interaction_ids"]) == {1, 2, 3}
    assert position["last_processed_timestamp"] == 1002
    assert (
        len(
            [
                e
                for e in events
                if e.event_name == "learnings_generated" and e.source == "api"
            ]
        )
        == 1
    )


@pytest.mark.parametrize("publication", ["playbook"], indirect=True)
def test_initial_playbook_dispatch_uses_database_assigned_ids(publication, monkeypatch):
    from types import MethodType

    storage, _, kind, _, _ = publication
    service, request, _ = _initial(publication, monkeypatch)
    monkeypatch.setattr(
        service,
        "_dispatch_playbook_schedulers",
        MethodType(_REAL_PLAYBOOK_DISPATCH, service),
    )
    seen = []
    monkeypatch.setattr(
        service,
        "_enqueue_user_playbook_optimization",
        lambda rows: seen.extend(row.user_playbook_id for row in rows),
    )
    monkeypatch.setattr(service, "_trigger_playbook_aggregation", lambda: None)
    _run_initial(service, request)
    ids = [row.user_playbook_id for row in base._learnings(storage, kind)]
    assert len(ids) == 1 and ids[0] > 0
    assert seen == ids, (
        "Dispatch must see the actual persisted plan, not pre-insert ID zero"
    )


@pytest.mark.parametrize("failure_stage", ["dispatch", "delivery"])
def test_postcommit_failures_are_value_free_and_preserve_publication(
    publication, monkeypatch, caplog, failure_stage
):
    from reflexio.server.services.extraction import publication as publishing

    storage, _, kind, _, _ = publication
    service, request, calls = _initial(publication, monkeypatch)
    private = "SENTINEL_PRIVATE_PUBLICATION_AND_CREDENTIAL"

    def unavailable(*_args, **_kwargs):
        try:
            raise ValueError(private)
        except ValueError as cause:
            raise RuntimeError(private) from cause

    publish = publishing._publish_agent_output

    def failed_callback(*args, **kwargs):
        kwargs["on_published"] = unavailable
        return publish(*args, **kwargs)

    with monkeypatch.context() as patcher:
        if failure_stage == "dispatch":
            patcher.setattr(publishing, "_publish_agent_output", failed_callback)
        else:
            patcher.setattr(publishing, "_deliver", unavailable)
        _run_initial(service, request)
    run = service._last_extraction_run
    assert run is not None and run.status == AgentRunStatus.FINALIZED
    assert run.committed_output is not None and calls == [1]
    with storage._stream_sql(read_only=True) as db:
        receipt = storage._get_output_publication_receipt(
            db, run.id, run.committed_output["publication"]["id"]
        )
    assert receipt is not None
    assert len(base._learnings(storage, kind)) == len(receipt.learning_ids) == 1
    assert receipt.delivery_state == (
        "pending" if failure_stage == "delivery" else "accepted"
    )
    records = [
        record for record in caplog.records if record.name == publishing.__name__
    ]
    assert records and all(record.exc_info is None for record in records)
    assert any("error=RuntimeError at=" in record.message for record in records)
    assert private not in caplog.text
    if failure_stage == "delivery":
        publishing._deliver(storage, receipt)
        assert (
            storage._get_publication_receipt(
                receipt.receipt_id, project_id=receipt.project_id
            ).delivery_state
            == "accepted"
        )
    assert calls == [1] and len(base._learnings(storage, kind)) == 1


@pytest.mark.parametrize(
    "stage", ["before_preparation", "after_preparation", "between_compute_and_save"]
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_initial_input_contention_preserves_output_preparation_and_allowance(
    shared_worker: DurableLearningWorker, publication, monkeypatch, stage, wrapped
):
    from datetime import UTC, datetime

    from reflexio.server.services.storage.error import ExecutionInputBusyError

    storage, _, kind, _, _ = publication
    service, request, calls = _initial(publication, monkeypatch)
    service.request_context.configurator.get_config().pending_tool_call_config.max_finalization_attempts = 1
    validate = storage.validate_agent_output_sources
    resolve = type(service)._resolve_write_plan
    state = {"busy": True, "computed": False, "completion_refused": False}
    resolutions = []

    def count_resolve(self, groups):
        resolutions.append(1)
        plan = resolve(self, groups)
        state["computed"] = True
        return plan

    def input_lock(run):
        prepared = storage.agent_output_preparation(run)
        completion_busy = (
            stage == "between_compute_and_save"
            and state["computed"]
            and prepared is None
            and not state["completion_refused"]
        )
        if completion_busy:
            state["completion_refused"] = True
        if state["busy"] and (
            stage == "before_preparation" or prepared is not None or completion_busy
        ):
            if not wrapped:
                raise ExecutionInputBusyError("Synthetic input contention")
            try:
                raise ExecutionInputBusyError("Synthetic input contention")
            except ExecutionInputBusyError as cause:
                raise RuntimeError("Wrapped input refusal") from cause
        return validate(run)

    monkeypatch.setattr(type(service), "_resolve_write_plan", count_resolve)
    monkeypatch.setattr(storage, "validate_agent_output_sources", input_lock)
    before = datetime.now(UTC)
    _run_initial(service, request)
    saved = service._last_extraction_run
    assert saved is not None and saved.status == AgentRunStatus.FINALIZATION_FAILED
    assert saved.committed_output is not None and calls == [1]
    assert saved.finalization_attempts == 0 and saved.resume_attempts == 0
    assert (
        saved.next_resume_at is not None
        and before <= saved.next_resume_at <= datetime.now(UTC)
    )
    assert not base._learnings(storage, kind)
    # The initial owner has released. Inspect metadata without borrowing its
    # old witness as fresh writer authority.
    with storage._stream_sql(read_only=True) as db:
        preparation_rows = db.query(
            "SELECT preparation,active_attempt FROM _agent_output_preparations "
            "WHERE org_id=? AND run_id=? LIMIT 1",
            (storage.org_id, saved.id),
        )
    preparation = preparation_rows[0]["preparation"] if preparation_rows else None
    assert (preparation is not None) is (stage != "before_preparation")
    assert not preparation_rows or preparation_rows[0]["active_attempt"] is None
    state["busy"] = False
    # Immediate eligibility was checked above. Normalize only the business due
    # time so SQL/Python submillisecond clock precision cannot choose this tick.
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (saved.id,)
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    completed = storage.get_agent_run(saved.id)
    assert completed is not None and completed.status == AgentRunStatus.FINALIZED
    assert completed.committed_output == saved.committed_output
    assert completed.finalization_attempts == 0 and completed.resume_attempts == 0
    assert calls == [1] and resolutions == [1]
    assert len(base._learnings(storage, kind)) == 1
