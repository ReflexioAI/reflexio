"""Receipts survive restart and share the aggregation output transaction."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from reflexio.models.api_schema.aggregation_operations import (
    AggregationOperationConflictError,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    instance = SQLiteStorage(
        org_id="operations", db_path=str(tmp_path / "operations.db")
    )
    yield instance
    instance.conn.close()


def claim_operation(storage, operation):
    claim = storage.claim_due_playbook_aggregation(
        owner="test", lease_seconds=300, agent_version=operation.agent_version
    )
    assert claim is not None
    storage.begin_playbook_aggregation_operation(operation.operation_id, claim)
    return claim


def test_idempotent_submission_and_active_conflicts(storage):
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(
            pool.map(
                lambda _: storage.submit_playbook_aggregation_operation("same", "v1"),
                range(8),
            )
        )
    assert len({row.operation_id for row in rows}) == 1
    with pytest.raises(AggregationOperationConflictError):
        storage.submit_playbook_aggregation_operation("same", "changed")
    with pytest.raises(AggregationOperationConflictError):
        storage.submit_playbook_aggregation_operation("another", "v1")
    assert (
        storage.submit_playbook_aggregation_operation("other", "v2").operation_id
        != rows[0].operation_id
    )


def test_completion_rolls_back_with_outputs_and_survives_restart(storage):
    operation = storage.submit_playbook_aggregation_operation("atomic", "v1")
    claim = claim_operation(storage, operation)
    storage.conn.execute("CREATE TABLE test_output (value TEXT)")
    storage.conn.commit()
    with pytest.raises(RuntimeError, match="crash"), storage.commit_scope():
        storage.conn.execute("INSERT INTO test_output VALUES ('first')")
        storage.complete_playbook_aggregation_operation(
            operation.operation_id, claim, {"playbooks_generated": 1}
        )
        raise RuntimeError("crash")
    assert storage.conn.execute("SELECT count(*) FROM test_output").fetchone()[0] == 0
    assert (
        storage.get_playbook_aggregation_operation(operation.operation_id).status
        == "running"
    )
    with storage.commit_scope():
        storage.conn.execute("INSERT INTO test_output VALUES ('second')")
        storage.complete_playbook_aggregation_operation(
            operation.operation_id, claim, {"playbooks_generated": 1}
        )
    reopened = SQLiteStorage(org_id="operations", db_path=storage.db_path)
    try:
        receipt = reopened.get_playbook_aggregation_operation(operation.operation_id)
        assert receipt is not None and receipt.status == "succeeded"
        assert reopened.next_playbook_aggregation_operation() is None
    finally:
        reopened.conn.close()
    assert storage.next_playbook_aggregation_operation() is None
    assert (
        storage.submit_playbook_aggregation_operation("atomic", "v1").status
        == "succeeded"
    )
    assert (
        storage.get_playbook_aggregation_operation(
            operation.operation_id
        ).result.playbooks_generated
        == 1
    )


def test_stale_lease_cannot_complete_or_fail_success(storage):
    operation = storage.submit_playbook_aggregation_operation("fenced", "v1")
    claim = claim_operation(storage, operation)
    with pytest.raises(RuntimeError, match="lease lost"):
        storage.complete_playbook_aggregation_operation(
            operation.operation_id, replace(claim, fence=claim.fence + 1), {}
        )
    storage.complete_playbook_aggregation_operation(operation.operation_id, claim, {})
    storage.fail_playbook_aggregation_operation(
        operation.operation_id, claim, error="late", retry_seconds=None
    )
    assert (
        storage.get_playbook_aggregation_operation(operation.operation_id).status
        == "succeeded"
    )


def test_retry_and_expired_lease_resume(storage):
    operation = storage.submit_playbook_aggregation_operation("resume", "v1")
    first = claim_operation(storage, operation)
    storage.fail_playbook_aggregation_operation(
        operation.operation_id, first, error="execution_failed", retry_seconds=30
    )
    assert storage.next_playbook_aggregation_operation() is None
    storage.conn.execute("UPDATE playbook_aggregation_lease SET claim_expires_at=0")
    storage.conn.execute("UPDATE playbook_aggregation_operations SET next_attempt_at=0")
    storage.conn.commit()
    second = claim_operation(storage, operation)
    assert second.fence > first.fence
    assert (
        storage.get_playbook_aggregation_operation(operation.operation_id).attempts == 2
    )
    storage.complete_playbook_aggregation_operation(
        operation.operation_id, second, {"skipped": "empty corpus"}
    )
    assert (
        storage.get_playbook_aggregation_operation(
            operation.operation_id
        ).result.skipped
        == "empty corpus"
    )


@pytest.mark.parametrize("outcome", ["success", "transient", "invalid"])
def test_executor_terminal_zero_retry_exhaustion(storage, monkeypatch, outcome):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from reflexio.server.services.playbook import aggregation_operations as executor

    operation = storage.submit_playbook_aggregation_operation("worker", "v1")
    context = MagicMock(storage=storage, org_id="operations", configurator=MagicMock())
    context.configurator.get_config.return_value.user_playbook_extractor_config = (
        SimpleNamespace(aggregation_config=SimpleNamespace(min_cluster_size=2))
    )
    aggregator = MagicMock()
    aggregator.run.return_value = {"skipped": "empty corpus"}
    if outcome != "success":
        aggregator.run.side_effect = (
            ValueError("invalid config")
            if outcome == "invalid"
            else RuntimeError("secret provider error")
        )
    monkeypatch.setattr(executor, "PlaybookAggregator", lambda **_kwargs: aggregator)
    monkeypatch.setattr(
        executor, "create_generation_litellm_client", lambda _ctx: MagicMock()
    )
    for attempt in range(1, 6 if outcome == "transient" else 2):
        assert executor.run_explicit_operation(context, owner="worker")
        current = storage.get_playbook_aggregation_operation(operation.operation_id)
        assert current.attempts == attempt
        if outcome == "transient" and attempt < 5:
            assert current.status == "retrying"
            assert current.next_attempt_at >= current.updated_at + 60
            storage.conn.execute(
                "UPDATE playbook_aggregation_operations SET next_attempt_at=0"
            )
            storage.conn.commit()
    assert current.status == ("succeeded" if outcome == "success" else "failed")
    assert current.error is None or "secret" not in current.error
    assert storage.next_playbook_aggregation_operation() is None
