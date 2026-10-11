"""Receipts survive restart and share the aggregation output transaction."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

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


def _seed_erasure_source(storage):
    from unittest.mock import patch

    from reflexio.models.api_schema.service_schemas import UserPlaybook

    with patch.object(storage, "_get_embedding", return_value=[0.1] * 512):
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    request_id="private-source",
                    user_id="user",
                    agent_version="v1",
                    playbook_name="test",
                    content="Private input",
                    trigger="Private workflow",
                )
            ]
        )
    return storage.get_user_playbooks(user_id="user")[0]


def test_trigger_replacement_keeps_concurrent_erasure_fenced(storage):
    import sqlite3

    from reflexio.server.services.storage.sqlite_storage.playbook._aggregation import (
        init_playbook_aggregation_tables,
    )

    source = _seed_erasure_source(storage)
    operation = storage.submit_playbook_aggregation_operation("during-init", "v1")
    claim = claim_operation(storage, operation)
    reopened = SQLiteStorage(org_id=storage.org_id, db_path=storage.db_path)
    paused = Event()
    resume = Event()
    timed_out = Event()

    def pause_after_drop(statement):
        if statement.strip().startswith(
            "CREATE TRIGGER cancel_erased_aggregation_operations"
        ):
            paused.set()
            if not resume.wait(10):
                timed_out.set()

    reopened.conn.set_trace_callback(pause_after_drop)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            refresh = executor.submit(init_playbook_aggregation_tables, reopened.conn)
            try:
                assert paused.wait(5)
                # At the exact DROP/CREATE boundary, another writer must be
                # blocked by the transaction rather than erase an unfenced row.
                storage.conn.execute("PRAGMA busy_timeout=0")
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    storage.conn.execute(
                        "DELETE FROM user_playbooks WHERE user_playbook_id=?",
                        (source.user_playbook_id,),
                    )
                storage.conn.rollback()
            finally:
                resume.set()
            refresh.result(timeout=10)
        assert not timed_out.is_set()
        storage.conn.execute("PRAGMA busy_timeout=5000")
        storage.conn.execute(
            "DELETE FROM user_playbooks WHERE user_playbook_id=?",
            (source.user_playbook_id,),
        )
        storage.conn.commit()
        receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
        assert receipt.status == "failed" and receipt.error == "input_erased"
        assert not storage.validate_playbook_aggregation_claim(claim)
    finally:
        resume.set()
        reopened.conn.set_trace_callback(None)
        reopened.conn.close()


def test_failed_trigger_replacement_rolls_back_all_fences(storage, monkeypatch):
    import sqlite3

    from reflexio.server.services.storage.sqlite_storage.playbook import _aggregation

    source = _seed_erasure_source(storage)
    operation = storage.submit_playbook_aggregation_operation("failed-init", "v1")
    claim = claim_operation(storage, operation)

    def fence_definitions():
        return {
            row["name"]: row["sql"]
            for row in storage.conn.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
                "AND name IN ('cancel_erased_aggregation_operations',"
                "'cancel_purged_aggregation_operations','fence_retired_aggregation_inputs')"
            )
        }

    before = fence_definitions()
    monkeypatch.setattr(
        _aggregation,
        "AGGREGATION_OPERATIONS_DDL",
        _aggregation.AGGREGATION_OPERATIONS_DDL.replace(
            "CREATE TRIGGER fence_retired_aggregation_inputs",
            "CREATE TRIGGER fence_retired_aggregation_inputs BROKEN",
        ),
    )
    with pytest.raises(sqlite3.OperationalError, match="BROKEN"):
        _aggregation.init_playbook_aggregation_tables(storage.conn)
    assert not storage.conn.in_transaction
    assert fence_definitions() == before and len(before) == 3
    assert storage.purge_content(
        entity_type="user_playbook", entity_id=str(source.user_playbook_id)
    )
    receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
    assert receipt.status == "failed" and receipt.error == "input_erased"
    assert not storage.validate_playbook_aggregation_claim(claim)


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


def test_running_operation_is_recovered_after_process_restart(storage):
    operation = storage.submit_playbook_aggregation_operation("crashed", "v1")
    first = claim_operation(storage, operation)
    storage.conn.execute("UPDATE playbook_aggregation_lease SET claim_expires_at=0")
    storage.conn.commit()
    reopened = SQLiteStorage(org_id="operations", db_path=storage.db_path)
    try:
        ready = reopened.next_playbook_aggregation_operation()
        assert ready is not None and ready.status == "running"
        second = claim_operation(reopened, ready)
        assert second.fence > first.fence
        receipt = reopened.get_playbook_aggregation_operation(operation.operation_id)
        assert receipt is not None and receipt.attempts == 2
        with pytest.raises(RuntimeError, match="lease lost"):
            storage.complete_playbook_aggregation_operation(
                operation.operation_id, first, {}
            )
        reopened.complete_playbook_aggregation_operation(
            operation.operation_id, second, {}
        )
        receipt = reopened.get_playbook_aggregation_operation(operation.operation_id)
        assert receipt is not None and receipt.status == "succeeded"
    finally:
        reopened.conn.close()


def test_erasure_cancellation_rolls_back_a_late_output_transaction(storage):
    # The delete trigger must invalidate a run computed from pre-erasure inputs.
    from unittest.mock import patch

    from reflexio.models.api_schema.service_schemas import UserPlaybook

    with patch.object(storage, "_get_embedding", return_value=[0.0] * 512):
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    request_id="source-request",
                    user_id="user",
                    agent_version="v1",
                    playbook_name="test",
                    content="input",
                )
            ]
        )
    item = storage.get_user_playbooks(user_id="user")[0]
    operation = storage.submit_playbook_aggregation_operation("erased", "v1")
    claim = claim_operation(storage, operation)
    storage.conn.execute(
        "DELETE FROM user_playbooks WHERE user_playbook_id=?", (item.user_playbook_id,)
    )
    storage.conn.commit()
    storage.conn.execute("CREATE TABLE test_output (value TEXT)")
    storage.conn.commit()
    with pytest.raises(RuntimeError), storage.commit_scope():
        storage.conn.execute(
            "INSERT INTO test_output VALUES ('private derived output')"
        )
        storage.complete_playbook_aggregation_operation(
            operation.operation_id, claim, {}
        )
    assert storage.conn.execute("SELECT count(*) FROM test_output").fetchone()[0] == 0
    receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
    assert receipt.status == "failed" and receipt.error == "input_erased"


@pytest.mark.parametrize("retire_after_snapshot", [False, True])
@pytest.mark.parametrize("empty_content", [False, True])
def test_lineage_preserving_erasure_fences_captured_private_inputs(
    storage, retire_after_snapshot, empty_content
):
    from unittest.mock import patch

    from reflexio.models.api_schema.domain.entities import LineageContext
    from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserPlaybook

    with patch.object(storage, "_get_embedding", return_value=[0.1] * 512):
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    request_id=f"source-{user}",
                    user_id=user,
                    agent_version="v1",
                    playbook_name="test",
                    content="" if empty_content else f"Private input from {user}",
                    rationale=f"Private rationale from {user}",
                )
                for user in ("erased-user", "other-user")
            ]
        )
    erased = storage.get_user_playbooks(user_id="erased-user")[0]
    other = storage.get_user_playbooks(user_id="other-user")[0]
    lineage = LineageContext(op_kind="merge", request_id="link", actor="test")
    if not retire_after_snapshot:
        # An inbound reference makes clear_user_data preserve the current
        # survivor's skeleton and erase its body through UPDATE, not DELETE.
        storage.merge_records(
            entity_type="user_playbook",
            survivor_id=str(erased.user_playbook_id),
            source_ids=[str(other.user_playbook_id)],
            context=lineage,
        )
    operation = storage.submit_playbook_aggregation_operation("captured-private", "v1")
    claim = claim_operation(storage, operation)
    snapshot = storage.capture_playbook_aggregation_rerun_snapshot("v1", limit=10)
    if empty_content:
        # Empty bodies are excluded by the same predicate as incremental
        # intake; their remaining identity/metadata still must be erased.
        assert not any(
            item.user_playbook_id == erased.user_playbook_id
            for item in snapshot.user_playbooks
        )
    else:
        captured = next(
            item
            for item in snapshot.user_playbooks
            if item.user_playbook_id == erased.user_playbook_id
        )
        assert captured.rationale == erased.rationale
    if retire_after_snapshot:
        # A row captured while current remains relevant even if a later
        # revision excludes it from subsequent snapshots before erasure.
        assert storage.supersede_record(
            entity_type="user_playbook",
            incumbent_id=str(erased.user_playbook_id),
            successor_id=str(other.user_playbook_id),
            context=lineage,
        )
    counts = storage.clear_user_data("erased-user")
    assert counts["purged_user_playbooks"] == 1
    assert (
        storage.conn.execute(
            "SELECT content FROM user_playbooks WHERE user_playbook_id=?",
            (erased.user_playbook_id,),
        ).fetchone()[0]
        == ""
    )
    with (
        patch.object(storage, "_get_embedding", return_value=[0.1] * 512),
        pytest.raises(RuntimeError),
        storage.commit_scope(),
    ):
        storage.save_agent_playbooks(
            [
                AgentPlaybook(
                    agent_version="v1",
                    playbook_name="test",
                    content=erased.rationale,
                    embedding=[0.1] * 512,
                )
            ]
        )
        storage.complete_playbook_aggregation_operation(
            operation.operation_id, claim, {}
        )
    assert storage.get_agent_playbooks() == []
    receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
    if retire_after_snapshot:
        assert receipt.status == "running"
        assert not storage.validate_playbook_aggregation_claim(claim)
        storage.conn.execute(
            "UPDATE playbook_aggregation_lease SET claim_expires_at=unixepoch()-1"
        )
        storage.conn.commit()
        replacement = claim_operation(storage, operation)
        storage.complete_playbook_aggregation_operation(
            operation.operation_id, replacement, {}
        )
        assert (
            storage.get_playbook_aggregation_operation(operation.operation_id).status
            == "succeeded"
        )
    else:
        assert receipt.status == "failed" and receipt.error == "input_erased"


@pytest.mark.parametrize("running", [False, True])
def test_expired_excluded_tombstone_gc_does_not_cancel_operation(storage, running):
    from unittest.mock import patch

    from reflexio.models.api_schema.service_schemas import UserPlaybook

    with patch.object(storage, "_get_embedding", return_value=[0.1] * 512):
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    request_id="old-source",
                    user_id="old-user",
                    agent_version="v1",
                    playbook_name="test",
                    content="Excluded old input",
                )
            ]
        )
    storage.conn.execute("UPDATE user_playbooks SET status='superseded', retired_at=1")
    storage.conn.commit()
    operation = storage.submit_playbook_aggregation_operation("unrelated-gc", "v1")
    claim = claim_operation(storage, operation) if running else None
    assert (
        storage.capture_playbook_aggregation_rerun_snapshot(
            "v1", limit=10
        ).user_playbooks
        == ()
    )
    assert (
        storage.gc_expired_tombstones(entity_type="user_playbook", older_than_epoch=2)
        == 1
    )
    receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
    assert receipt.status == ("running" if running else "queued")
    if claim is None:
        claim = claim_operation(storage, operation)
    storage.complete_playbook_aggregation_operation(operation.operation_id, claim, {})
    assert (
        storage.get_playbook_aggregation_operation(operation.operation_id).status
        == "succeeded"
    )


def test_partial_results_are_not_success_and_claim_notification_is_scoped(
    storage, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from reflexio.server.services.playbook import aggregation_operations as executor
    from reflexio.server.services.playbook import aggregation_scheduler

    operation = storage.submit_playbook_aggregation_operation("partial", "v1")
    storage.supports_incremental_playbook_aggregation = False
    context = MagicMock(storage=storage, org_id="operations", configurator=MagicMock())
    context.configurator.get_config.return_value.user_playbook_extractor_config = (
        SimpleNamespace(aggregation_config=SimpleNamespace(min_cluster_size=2))
    )
    aggregator = MagicMock()
    aggregator.run.return_value = {"retryable_failures": 1}
    monkeypatch.setattr(executor, "PlaybookAggregator", lambda **_kwargs: aggregator)
    monkeypatch.setattr(
        executor, "create_generation_litellm_client", lambda _context: MagicMock()
    )
    heartbeat = MagicMock(wraps=aggregation_scheduler.AggregationLeaseHeartbeat)
    monkeypatch.setattr(aggregation_scheduler, "AggregationLeaseHeartbeat", heartbeat)
    claimed = MagicMock()
    assert executor.run_explicit_operation(
        context, owner="worker", project_id="project", on_work_claimed=claimed
    )
    claimed.assert_called_once_with(context)
    assert heartbeat.call_args.kwargs == {
        "org_id": "operations",
        "project_id": "project",
    }
    receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
    assert receipt.status == "retrying" and receipt.result is None


@pytest.mark.parametrize(
    "claim_outcome", ["contended", "claimed", "failed", "finalization_failed"]
)
def test_explicit_claim_resets_org_scope_only_after_claim_returns(
    storage, monkeypatch, claim_outcome
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from reflexio.server import background_work
    from reflexio.server.services.playbook import aggregation_scheduler

    operation = storage.submit_playbook_aggregation_operation("org-recovery", "v1")
    if claim_outcome == "contended":
        assert (
            storage.claim_due_playbook_aggregation(
                owner="other-worker", lease_seconds=300, agent_version="v1"
            )
            is not None
        )
    elif claim_outcome == "failed":
        monkeypatch.setattr(
            storage,
            "claim_due_playbook_aggregation",
            MagicMock(side_effect=RuntimeError("claim unavailable")),
        )
    context = MagicMock(storage=storage, org_id="operations", configurator=MagicMock())
    # Claimed work fails configuration validation after organization recovery.
    context.configurator.get_config.return_value.user_playbook_extractor_config = (
        SimpleNamespace(aggregation_config=SimpleNamespace(min_cluster_size=1))
    )
    org_scope = "playbook-aggregation-org:operations:project"
    other_project = "playbook-aggregation-org:operations:other-project"
    other_org = "playbook-aggregation-org:other-org:project"
    operation_scope = (
        f"playbook-aggregation-operation:operations:project:{operation.operation_id}"
    )
    finalize_scope = "playbook-aggregation-operation-finalize:operations:project:v1"
    streak = (1, 1.0, 1.0)
    monkeypatch.setattr(
        background_work,
        "_streaks",
        dict.fromkeys(
            (org_scope, other_project, other_org, operation_scope, finalize_scope),
            streak,
        ),
    )
    monkeypatch.setattr(background_work, "_classifier", lambda _exc: True)
    monkeypatch.setattr(background_work, "_monotonic", lambda: 2.0)
    monkeypatch.setattr(aggregation_scheduler, "current_project_id", lambda: "project")
    begin_operation = storage.begin_playbook_aggregation_operation
    before_execution = []

    def begin_after_org_recovery(*args, **kwargs):
        before_execution.append(
            (
                background_work._streaks.get(org_scope),
                background_work._streaks.get(operation_scope),
                background_work._streaks.get(finalize_scope),
            )
        )
        return begin_operation(*args, **kwargs)

    monkeypatch.setattr(
        storage, "begin_playbook_aggregation_operation", begin_after_org_recovery
    )
    if claim_outcome == "finalization_failed":
        monkeypatch.setattr(
            storage,
            "finish_playbook_aggregation_claim",
            MagicMock(side_effect=RuntimeError("finalization unavailable")),
        )
    scheduler = aggregation_scheduler.PlaybookAggregationScheduler(
        context_provider=lambda: []
    )
    if claim_outcome == "failed":
        with pytest.raises(RuntimeError, match="claim unavailable"):
            scheduler._run_context(context)
        assert background_work._streaks[org_scope] == streak
    else:
        if claim_outcome == "finalization_failed":
            with pytest.raises(aggregation_scheduler._FinalizationReportedError):
                scheduler._run_context(context)
        else:
            scheduler._run_context(context)
        assert org_scope not in background_work._streaks
        receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
        assert receipt.status == (
            "queued" if claim_outcome == "contended" else "failed"
        )
    assert background_work._streaks[other_org] == streak
    assert background_work._streaks[other_project] == streak
    assert background_work._streaks[operation_scope][0] == (
        2 if claim_outcome in {"claimed", "finalization_failed"} else 1
    )
    assert before_execution == (
        [(None, streak, streak)]
        if claim_outcome in {"claimed", "finalization_failed"}
        else []
    )
    if claim_outcome == "finalization_failed":
        assert background_work._streaks[finalize_scope][0] == 2
    elif claim_outcome == "claimed":
        assert finalize_scope not in background_work._streaks
    else:
        assert background_work._streaks[finalize_scope] == streak


def test_synchronous_rerun_cannot_bypass_explicit_lease_when_incremental_disabled(
    storage,
):
    from unittest.mock import MagicMock

    from reflexio.lib._generation import GenerationMixin

    storage.supports_incremental_playbook_aggregation = False
    assert storage.supports_durable_playbook_aggregation_operations is True
    operation = storage.submit_playbook_aggregation_operation("exclusive", "v1")
    claim_operation(storage, operation)
    generation = object.__new__(GenerationMixin)
    generation.request_context = MagicMock(storage=storage, org_id="operations")
    generation.llm_client = MagicMock()
    with pytest.raises(RuntimeError, match="already running"):
        generation.run_playbook_aggregation(agent_version="v1")


@pytest.mark.parametrize("success", [False, True])
def test_retired_snapshot_retries_and_releases_owned_lease_without_state_overwrite(
    storage, success
):
    _seed_erasure_source(storage)
    operation = storage.submit_playbook_aggregation_operation("retire-and-retry", "v1")
    claim = claim_operation(storage, operation)
    assert storage.capture_playbook_aggregation_rerun_snapshot(
        "v1", limit=10
    ).user_playbooks
    storage.conn.execute("UPDATE user_playbooks SET status='superseded', retired_at=1")
    storage.conn.execute(
        "UPDATE playbook_aggregation_state SET pending=1, next_attempt_at=123, last_success_at=7"
    )
    storage.conn.commit()
    state_sql = "SELECT state_version,pending,next_attempt_at,last_success_at FROM playbook_aggregation_state WHERE agent_version='v1'"
    advanced = tuple(storage.conn.execute(state_sql).fetchone())
    assert advanced[0] > claim.state_version
    with pytest.raises(RuntimeError, match="lease lost"):
        storage.complete_playbook_aggregation_operation(
            operation.operation_id, claim, {}
        )
    storage.fail_playbook_aggregation_operation(
        operation.operation_id, claim, error="execution_failed", retry_seconds=30
    )
    receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
    assert receipt.status == "retrying" and receipt.next_attempt_at is not None
    assert not storage.finish_playbook_aggregation_claim(
        claim,
        success=success,
        retry_after_seconds=30,
        backlog_retry_after_seconds=0,
        min_interval_seconds=0,
    )
    assert tuple(storage.conn.execute(state_sql).fetchone()) == advanced
    assert (
        storage.conn.execute(
            "SELECT claim_owner FROM playbook_aggregation_lease WHERE singleton=1"
        ).fetchone()[0]
        is None
    )
    storage.conn.execute("UPDATE playbook_aggregation_operations SET next_attempt_at=0")
    storage.conn.commit()
    replacement = claim_operation(storage, operation)
    assert replacement.fence > claim.fence
    storage.complete_playbook_aggregation_operation(
        operation.operation_id, replacement, {}
    )
    assert (
        storage.get_playbook_aggregation_operation(operation.operation_id).status
        == "succeeded"
    )


def test_old_worker_failure_and_finish_cannot_touch_replacement(storage):
    operation = storage.submit_playbook_aggregation_operation("replacement", "v1")
    old = claim_operation(storage, operation)
    storage.conn.execute("UPDATE playbook_aggregation_lease SET claim_expires_at=0")
    storage.conn.commit()
    replacement = claim_operation(storage, operation)
    storage.fail_playbook_aggregation_operation(
        operation.operation_id, old, error="late", retry_seconds=None
    )
    assert not storage.finish_playbook_aggregation_claim(
        old,
        success=False,
        retry_after_seconds=30,
        backlog_retry_after_seconds=0,
        min_interval_seconds=0,
    )
    receipt = storage.get_playbook_aggregation_operation(operation.operation_id)
    assert receipt.status == "running"
    assert storage.validate_playbook_aggregation_claim(replacement)
    storage.complete_playbook_aggregation_operation(
        operation.operation_id, replacement, {}
    )


@pytest.mark.parametrize("running", [False, True])
def test_deleting_purged_current_skeleton_preserves_fresh_operation(storage, running):
    source = _seed_erasure_source(storage)
    assert storage.purge_content(
        entity_type="user_playbook", entity_id=str(source.user_playbook_id)
    )
    operation = storage.submit_playbook_aggregation_operation(
        "after-purge-delete", "v1"
    )
    claim = claim_operation(storage, operation) if running else None
    assert not storage.capture_playbook_aggregation_rerun_snapshot(
        "v1", limit=10
    ).user_playbooks
    before = storage.get_playbook_aggregation_operation(operation.operation_id)
    state_before = storage.conn.execute(
        "SELECT state_version FROM playbook_aggregation_state WHERE agent_version='v1'"
    ).fetchone()[0]
    storage.conn.execute(
        "DELETE FROM user_playbooks WHERE user_playbook_id=?",
        (source.user_playbook_id,),
    )
    storage.conn.commit()
    assert storage.get_playbook_aggregation_operation(operation.operation_id) == before
    assert (
        storage.conn.execute(
            "SELECT state_version FROM playbook_aggregation_state WHERE agent_version='v1'"
        ).fetchone()[0]
        == state_before
    )
    if claim is None:
        claim = claim_operation(storage, operation)
    assert storage.validate_playbook_aggregation_claim(claim)
    storage.complete_playbook_aggregation_operation(operation.operation_id, claim, {})
    assert (
        storage.get_playbook_aggregation_operation(operation.operation_id).status
        == "succeeded"
    )


@pytest.mark.parametrize("finalization_fails", [False, True])
def test_explicit_finalization_does_not_reset_incremental_failure_scope(
    storage, monkeypatch, finalization_fails
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from reflexio.server import background_work
    from reflexio.server.services.playbook import aggregation_operations as executor
    from reflexio.server.services.playbook.aggregation_scheduler import (
        _FinalizationReportedError,
    )

    storage.submit_playbook_aggregation_operation("finalize-recovery", "v1")
    context = MagicMock(storage=storage, org_id="operations", configurator=MagicMock())
    context.configurator.get_config.return_value.user_playbook_extractor_config = (
        SimpleNamespace(aggregation_config=SimpleNamespace(min_cluster_size=1))
    )
    incremental = "playbook-aggregation-finalize:operations:project:v1"
    explicit = "playbook-aggregation-operation-finalize:operations:project:v1"
    streak = (1, 1.0, 1.0)
    monkeypatch.setattr(
        background_work, "_streaks", {incremental: streak, explicit: streak}
    )
    monkeypatch.setattr(background_work, "_classifier", lambda _exc: True)
    monkeypatch.setattr(background_work, "_monotonic", lambda: 2.0)
    if finalization_fails:
        monkeypatch.setattr(
            storage,
            "finish_playbook_aggregation_claim",
            MagicMock(side_effect=RuntimeError("finalization unavailable")),
        )
        with pytest.raises(_FinalizationReportedError):
            executor.run_explicit_operation(
                context, owner="worker", project_id="project"
            )
        assert background_work._streaks[explicit][0] == 2
    else:
        assert executor.run_explicit_operation(
            context, owner="worker", project_id="project"
        )
        assert explicit not in background_work._streaks
    assert background_work._streaks[incremental] == streak
