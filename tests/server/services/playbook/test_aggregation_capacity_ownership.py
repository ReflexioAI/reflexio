"""Capacity belongs to the whole durable claim, never to a waiting attempt."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, call

import pytest

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.operation_limiter import (
    operation_limit,
    reset_operation_limiters_for_tests,
)
from reflexio.server.services.playbook import (
    aggregation_operations,
    aggregation_scheduler,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationClaim,
)


@pytest.fixture(autouse=True)
def capacity(monkeypatch):
    monkeypatch.setenv("REFLEXIO_AGGREGATION_CONCURRENCY_LIMIT", "1")
    monkeypatch.setenv("REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS", "0.001")
    reset_operation_limiters_for_tests()
    yield
    reset_operation_limiters_for_tests()


@pytest.fixture
def storage(tmp_path):
    instance = SQLiteStorage(
        org_id="capacity-owner", db_path=str(tmp_path / "state.db")
    )
    yield instance
    instance.conn.close()


def context_for(storage):
    config = SimpleNamespace(
        user_playbook_extractor_config=SimpleNamespace(
            aggregation_config=SimpleNamespace(min_cluster_size=2)
        )
    )
    return cast(
        RequestContext,
        SimpleNamespace(
            org_id=storage.org_id,
            storage=storage,
            configurator=SimpleNamespace(get_config=lambda: config),
        ),
    )


def stub_execution(monkeypatch):
    model = Mock()
    model.run.return_value = {"skipped": "empty corpus"}
    factory = Mock(return_value=model)
    client = Mock()
    monkeypatch.setattr(aggregation_operations, "PlaybookAggregator", factory)
    monkeypatch.setattr(
        aggregation_operations, "create_generation_litellm_client", client
    )
    return factory, client


def test_repeated_capacity_pressure_never_spends_a_real_operation_attempt(
    storage, monkeypatch
):
    operation = storage.submit_playbook_aggregation_operation("capacity", "v1")
    context = context_for(storage)
    factory, client = stub_execution(monkeypatch)
    heartbeat = Mock(wraps=aggregation_scheduler.AggregationLeaseHeartbeat)
    monkeypatch.setattr(aggregation_scheduler, "AggregationLeaseHeartbeat", heartbeat)
    observations = []
    with operation_limit(storage.org_id, "aggregation"):
        for _ in range(6):
            aggregation_operations.run_explicit_operation(context, owner="worker")
            current = storage.get_playbook_aggregation_operation(operation.operation_id)
            observations.append((current.status, current.attempts))
            # Only a failed baseline attempt needs its next retry made due.
            if current.status == "retrying":
                storage.conn.execute(
                    "UPDATE playbook_aggregation_operations SET next_attempt_at=0 WHERE operation_id=?",
                    (operation.operation_id,),
                )
                storage.conn.commit()
    assert observations == [("queued", 0)] * 6
    factory.assert_not_called()
    client.assert_not_called()
    heartbeat.assert_not_called()
    assert aggregation_operations.run_explicit_operation(context, owner="worker")
    current = storage.get_playbook_aggregation_operation(operation.operation_id)
    assert (current.status, current.attempts) == ("succeeded", 1)


@pytest.mark.parametrize("finish_fails", [False, True])
def test_explicit_permit_is_held_through_finalization_and_released_afterward(
    storage, monkeypatch, finish_fails
):
    storage.submit_playbook_aggregation_operation("finalize", "v1")
    context = context_for(storage)
    stub_execution(monkeypatch)
    finish = storage.finish_playbook_aggregation_claim
    checked = []

    def finalize(*args, **kwargs):
        with (
            pytest.raises(TimeoutError),
            operation_limit(storage.org_id, "aggregation", timeout_seconds=0),
        ):
            pytest.fail("finalization lost its physical capacity owner")
        checked.append(True)
        if finish_fails:
            raise RuntimeError("synthetic finalization failure")
        return finish(*args, **kwargs)

    monkeypatch.setattr(storage, "finish_playbook_aggregation_claim", finalize)
    if finish_fails:
        with pytest.raises(aggregation_scheduler._FinalizationReportedError):
            aggregation_operations.run_explicit_operation(context, owner="worker")
    else:
        assert aggregation_operations.run_explicit_operation(context, owner="worker")
    assert checked == [True]
    with operation_limit(storage.org_id, "aggregation", timeout_seconds=0):
        pass


def test_incremental_busy_path_preserves_unclaimed_work_and_sparse_retry_visibility(
    monkeypatch,
):
    storage = Mock(
        org_id="incremental",
        supports_incremental_playbook_aggregation=True,
        supports_durable_playbook_aggregation_operations=False,
    )
    storage.repair_playbook_aggregation_pending_state.return_value = []
    storage.claim_due_playbook_aggregation.return_value = PlaybookAggregationClaim(
        "v1", "owner", 1, 1, 9999999999
    )
    storage.get_playbook_aggregation_invalidations.return_value = []
    storage.finish_playbook_aggregation_claim.return_value = True
    monkeypatch.setattr(
        "reflexio.lib.generation_client.create_generation_litellm_client", Mock()
    )
    monkeypatch.setattr(aggregation_scheduler, "PlaybookAggregator", Mock())
    context = context_for(storage)
    deferred = Mock()
    success = Mock()
    heartbeat = Mock()
    monkeypatch.setattr(aggregation_scheduler, "report_background_success", success)
    monkeypatch.setattr(aggregation_scheduler, "AggregationLeaseHeartbeat", heartbeat)
    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: [], on_scope_deferred=deferred
    )
    with operation_limit(context.org_id, "aggregation"):
        scheduler._run_context(context)
    storage.claim_due_playbook_aggregation.assert_not_called()
    storage.get_playbook_aggregation_invalidations.assert_not_called()
    storage.finish_playbook_aggregation_claim.assert_not_called()
    heartbeat.assert_not_called()
    assert success.call_args_list == [
        call("playbook-aggregation-defer:incremental:None")
    ]
    deferred.assert_called_once_with(context, scheduler._poll_interval_seconds)


def test_explicit_claim_timeout_is_not_misreported_as_capacity_pressure(
    storage, monkeypatch
):
    storage.submit_playbook_aggregation_operation("claim-error", "v1")
    context = context_for(storage)
    monkeypatch.setattr(
        storage,
        "claim_due_playbook_aggregation",
        Mock(side_effect=TimeoutError("claim timed out")),
    )
    with pytest.raises(TimeoutError, match="claim timed out"):
        aggregation_operations.run_explicit_operation(context, owner="worker")
    with operation_limit(storage.org_id, "aggregation", timeout_seconds=0):
        pass


def test_explicit_queue_is_read_after_admission_so_a_waited_on_receipt_is_fresh(
    storage, monkeypatch
):
    operation = storage.submit_playbook_aggregation_operation(
        "concurrent-completion", "v1"
    )
    factory, _ = stub_execution(monkeypatch)

    @contextmanager
    def finish_before_admitting(*args, **kwargs):
        # Another worker finishes while this caller waits for local capacity.
        with operation_limit(*args, **kwargs):
            claim = storage.claim_due_playbook_aggregation(
                owner="other", lease_seconds=300, agent_version="v1"
            )
            assert claim is not None
            storage.begin_playbook_aggregation_operation(operation.operation_id, claim)
            storage.complete_playbook_aggregation_operation(
                operation.operation_id, claim, {}
            )
            storage.finish_playbook_aggregation_claim(
                claim,
                success=True,
                retry_after_seconds=30,
                backlog_retry_after_seconds=0,
                min_interval_seconds=3600,
            )
            yield

    monkeypatch.setattr(
        aggregation_operations, "operation_limit", finish_before_admitting
    )
    assert (
        aggregation_operations.run_explicit_operation(
            context_for(storage), owner="late"
        )
        is False
    )
    assert (
        storage.get_playbook_aggregation_operation(operation.operation_id).attempts == 1
    )
    factory.assert_not_called()


def test_incremental_execution_timeout_is_a_run_failure_and_keeps_capacity_through_cleanup(
    storage, monkeypatch, caplog
):
    storage.schedule_playbook_aggregation("v1")
    context = context_for(storage)
    model = Mock()
    model.run.side_effect = TimeoutError("synthetic execution timeout")
    monkeypatch.setattr(
        aggregation_scheduler, "PlaybookAggregator", Mock(return_value=model)
    )
    monkeypatch.setattr(
        "reflexio.lib.generation_client.create_generation_litellm_client", Mock()
    )
    failure = Mock()
    monkeypatch.setattr(aggregation_scheduler, "report_background_failure", failure)
    finish = storage.finish_playbook_aggregation_claim
    finalizations = []

    def finalize(*args, **kwargs):
        with (
            pytest.raises(TimeoutError),
            operation_limit(storage.org_id, "aggregation", timeout_seconds=0),
        ):
            pytest.fail("capacity released before incremental cleanup")
        finalizations.append(kwargs["success"])
        return finish(*args, **kwargs)

    monkeypatch.setattr(storage, "finish_playbook_aggregation_claim", finalize)
    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: []
    )
    scheduler._run_context(context)
    assert finalizations == [False]
    failure.assert_called_once()
    assert isinstance(failure.call_args.args[2], TimeoutError)
    assert (
        failure.call_args.kwargs["scope"]
        == f"playbook-aggregation-run:{storage.org_id}:None:v1"
    )
    assert "limiter_saturated" not in caplog.text
    with operation_limit(storage.org_id, "aggregation", timeout_seconds=0):
        pass


def test_capacity_pressure_on_one_org_does_not_stop_another_org(
    storage, monkeypatch, tmp_path
):
    other = SQLiteStorage(org_id="other-org", db_path=str(tmp_path / "other.db"))
    stub_execution(monkeypatch)
    try:
        operation = other.submit_playbook_aggregation_operation("other", "v1")
        with operation_limit(storage.org_id, "aggregation"):
            assert aggregation_operations.run_explicit_operation(
                context_for(other), owner="worker"
            )
        current = other.get_playbook_aggregation_operation(operation.operation_id)
        assert current is not None
        assert (current.status, current.attempts) == ("succeeded", 1)
    finally:
        other.conn.close()
