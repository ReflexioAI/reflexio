"""Agent-run mutations join the same transaction as their caller's fence."""

import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    PendingToolCallStatus,
    RunToolDependencyRecord,
)
from tests.server.services.storage.sqlite_storage.test_agent_run_storage import (
    _agent_run,
    _pending_call,
)


@pytest.fixture(name="storage")
def temporary_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(SQLiteStorage, "_get_embedding", lambda *_a, **_k: [0.0] * 512)
    store = SQLiteStorage(org_id="org_1", db_path=str(tmp_path / "agent-run.db"))
    try:
        yield store
    finally:
        store.conn.close()


def _snapshot(connection):
    queries = {
        "runs": "SELECT * FROM _agent_runs ORDER BY rowid",
        "calls": "SELECT * FROM _pending_tool_calls ORDER BY rowid",
        "dependencies": "SELECT * FROM _run_tool_dependencies ORDER BY rowid",
        "receipts": "SELECT * FROM _agent_run_finalization_receipts ORDER BY rowid",
    }
    return {
        table: [tuple(row) for row in connection.execute(query).fetchall()]
        for table, query in queries.items()
    }


@pytest.fixture
def writes(storage):
    now = datetime.now(UTC)
    storage.create_agent_run(_agent_run("running", AgentRunStatus.RUNNING))
    storage.create_agent_run(_agent_run("ready", AgentRunStatus.RESUME_READY))
    finalizing = storage.create_agent_run(
        replace(
            _agent_run("finalizing", AgentRunStatus.FINALIZATION_FAILED),
            committed_output={"profiles": []},
        )
    )
    storage.create_pending_tool_call(_pending_call("pending", now=now))
    storage.create_pending_tool_call(
        replace(
            _pending_call("resolved", now=now),
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "test"},
            resolved_at=now,
        )
    )
    storage.create_pending_tool_call(
        replace(
            _pending_call("expired", now=now),
            status=PendingToolCallStatus.EXPIRED,
            expires_at=now - timedelta(days=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="ready", pending_tool_call_id="resolved", resolved_at=now
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="running", pending_tool_call_id="pending")
    )
    return {
        "create_run": lambda: storage.create_agent_run(
            _agent_run("new", AgentRunStatus.RUNNING)
        ),
        "status": lambda: storage.update_agent_run_status(
            "running", AgentRunStatus.FAILED
        ),
        "receipt": lambda: storage.save_agent_run_finalization_receipt(
            run_id="finalizing", entity_type="profile", learning_ids=[]
        ),
        "fail_running": lambda: storage.fail_running_agent_runs_for_request(
            org_id="org_1",
            extractor_kind="profile",
            user_id="user_1",
            request_id="request_1",
            last_error="test",
        ),
        "claim_ready": lambda: storage.claim_ready_agent_run(
            org_id="org_1", worker_id="test", now=now
        ),
        "claim_finalizing": lambda: storage.claim_finalization_failed_agent_run(
            org_id="org_1", worker_id="test", now=now
        ),
        "claim_recovery": lambda: storage.claim_agent_run_for_recovery(
            run_id="finalizing",
            org_id="org_1",
            worker_id="test",
            expected_updated_at=finalizing.updated_at,
            operation_id="test",
            now=now,
        ),
        "attach": lambda: storage.attach_run_tool_dependency(
            RunToolDependencyRecord(run_id="finalizing", pending_tool_call_id="pending")
        ),
        "consume": lambda: storage.consume_run_tool_dependencies("ready"),
        "create_pending": lambda: storage.create_pending_tool_call(
            _pending_call("new_pending", now=now)
        ),
        "create_or_attach": lambda: storage.create_or_attach_pending_tool_call(
            record=_pending_call("new_pending", now=now),
            dependency=RunToolDependencyRecord(
                run_id="finalizing", pending_tool_call_id="new_pending"
            ),
            now=now,
        ),
        "cancel": lambda: storage.cancel_pending_tool_call("pending", cancelled_at=now),
        "expire": lambda: storage.expire_pending_tool_calls(
            now=now + timedelta(hours=2)
        ),
        "delete_expired": lambda: storage.delete_expired_pending_tool_calls(
            now=int(now.timestamp()), grace_seconds=0
        ),
        "resolve": lambda: storage.resolve_pending_tool_call(
            "pending", result={"answer": "test"}, resolved_at=now, valid_for_seconds=600
        ),
        "revise": lambda: storage.update_resolved_pending_tool_call_result(
            "resolved",
            result={"answer": "changed"},
            resolved_at=now,
            valid_for_seconds=600,
        ),
        "not_applicable": lambda: storage.mark_pending_tool_call_not_applicable(
            "pending", resolved_at=now, valid_for_seconds=600
        ),
        "noop_resolve": lambda: storage.resolve_pending_tool_call(
            "missing", result={}, valid_for_seconds=600
        ),
        "noop_revise": lambda: storage.update_resolved_pending_tool_call_result(
            "missing", result={}, valid_for_seconds=600
        ),
        "noop_not_applicable": lambda: storage.mark_pending_tool_call_not_applicable(
            "missing", valid_for_seconds=600
        ),
        "noop_expire": lambda: storage.expire_pending_tool_calls(
            now=now - timedelta(days=2)
        ),
        "noop_delete": lambda: storage.delete_expired_pending_tool_calls(
            now=int((now - timedelta(days=2)).timestamp()), grace_seconds=0
        ),
    }


OPERATIONS = (
    "create_run",
    "status",
    "receipt",
    "fail_running",
    "claim_ready",
    "claim_finalizing",
    "claim_recovery",
    "attach",
    "consume",
    "create_pending",
    "create_or_attach",
    "cancel",
    "expire",
    "delete_expired",
    "resolve",
    "revise",
    "not_applicable",
    "noop_resolve",
    "noop_revise",
    "noop_not_applicable",
    "noop_expire",
    "noop_delete",
)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("outcome", ("commit", "rollback", "standalone"))
def test_agent_run_write_transaction_owner(storage, writes, operation, outcome):
    with sqlite3.connect(storage.db_path) as observer:
        before = _snapshot(observer)
        if outcome == "standalone":
            writes[operation]()
            assert _snapshot(observer) == _snapshot(storage.conn)
            if not operation.startswith("noop_"):
                assert _snapshot(observer) != before
            return

        def scoped_write():
            with storage.commit_scope():
                # Even a no-op helper must not publish its caller's earlier write.
                storage.conn.execute(
                    "UPDATE _agent_runs SET last_error='sentinel' WHERE id='running'"
                )
                with storage.commit_scope():
                    writes[operation]()
                assert _snapshot(observer) == before
                if outcome == "rollback":
                    raise RuntimeError("abort outer transaction")

        if outcome == "rollback":
            with pytest.raises(RuntimeError, match="abort outer transaction"):
                scoped_write()
            assert _snapshot(observer) == before
        else:
            scoped_write()
            assert _snapshot(observer) != before
            assert _snapshot(observer) == _snapshot(storage.conn)
        assert not storage.conn.in_transaction


@pytest.mark.parametrize("outer_scope", (False, True))
def test_pending_resolution_failure_rolls_back_every_table(
    storage, writes, outer_scope
):
    # Fail AFTER the pending row changes, while cascading to its dependent run.
    storage.conn.execute("""CREATE TRIGGER refuse_dependency_resolution
        BEFORE UPDATE ON _run_tool_dependencies BEGIN
        SELECT RAISE(ABORT, 'injected dependency failure'); END""")
    storage.conn.commit()
    with sqlite3.connect(storage.db_path) as observer:
        before = _snapshot(observer)
        with pytest.raises(StorageError, match="injected dependency failure"):
            if outer_scope:
                with storage.commit_scope():
                    writes["resolve"]()
            else:
                writes["resolve"]()
        assert _snapshot(observer) == before
        assert not storage.conn.in_transaction


@pytest.mark.parametrize("revision_offset", (0, 1, -1))
def test_recovery_claim_uses_precise_observed_revision(
    storage, writes, revision_offset
):
    run = storage.get_agent_run("finalizing")
    original = run.updated_at
    claimed = storage.claim_agent_run_for_recovery(
        run_id=run.id,
        org_id="org_1",
        worker_id="recovery",
        expected_updated_at=original + timedelta(microseconds=revision_offset),
        operation_id="recover",
    )
    if revision_offset == 0:
        assert claimed is not None
        assert claimed.status == AgentRunStatus.FINALIZING
    else:
        assert claimed is None
        assert storage.get_agent_run(run.id) == run
