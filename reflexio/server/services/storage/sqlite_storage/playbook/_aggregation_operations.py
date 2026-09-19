"""SQLite receipts for explicit aggregation, fenced by the existing lease."""

import json
import sqlite3
import uuid
from typing import Any

from reflexio.models.api_schema.aggregation_operations import (
    AggregationOperationConflictError,
    PlaybookAggregationOperation,
    PlaybookAggregationResult,
)
from reflexio.server.services.storage.storage_base.playbook import (
    PlaybookAggregationClaim,
)

AGGREGATION_OPERATIONS_DDL = """
CREATE TABLE IF NOT EXISTS playbook_aggregation_operations (
    operation_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    agent_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued','running','retrying','succeeded','failed')),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    started_at INTEGER,
    completed_at INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at INTEGER,
    error TEXT,
    result TEXT,
    claim_fence INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_aggregation_operation_active
    ON playbook_aggregation_operations(agent_version)
    WHERE status IN ('queued','running','retrying');
CREATE TRIGGER IF NOT EXISTS cancel_erased_aggregation_operations BEFORE DELETE ON user_playbooks
FOR EACH ROW BEGIN
 UPDATE playbook_aggregation_operations SET status='failed',error='input_erased',
 completed_at=unixepoch(),updated_at=unixepoch(),next_attempt_at=NULL
 WHERE agent_version=OLD.agent_version AND status IN ('queued','running','retrying');
END;
"""


def operation_from_row(row: Any) -> PlaybookAggregationOperation:
    value = dict(row)
    value.pop("claim_fence", None)
    if isinstance(value.get("result"), str):
        value["result"] = json.loads(value["result"])
    return PlaybookAggregationOperation.model_validate(value)


class AggregationOperationsMixin:
    conn: sqlite3.Connection
    commit_scope: Any
    _lock: Any
    validate_playbook_aggregation_claim: Any
    schedule_playbook_aggregation: Any

    def submit_playbook_aggregation_operation(
        self, request_id: str, agent_version: str
    ) -> PlaybookAggregationOperation:
        with self.commit_scope():
            old = self.conn.execute(
                "SELECT * FROM playbook_aggregation_operations WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if old:
                operation = operation_from_row(old)
                if operation.agent_version != agent_version:
                    raise AggregationOperationConflictError(
                        operation.operation_id, "request_id parameters changed"
                    )
                return operation
            active = self.conn.execute(
                "SELECT operation_id FROM playbook_aggregation_operations WHERE agent_version=? "
                "AND status IN ('queued','running','retrying')",
                (agent_version,),
            ).fetchone()
            if active:
                raise AggregationOperationConflictError(
                    active[0], "aggregation operation already active"
                )
            operation_id = uuid.uuid4().hex
            self.conn.execute(
                "INSERT INTO playbook_aggregation_operations "
                "(operation_id,request_id,agent_version,status,created_at,updated_at,next_attempt_at) "
                "VALUES (?,?,?,'queued',unixepoch(),unixepoch(),unixepoch())",
                (operation_id, request_id, agent_version),
            )
            self.schedule_playbook_aggregation(agent_version)
            operation = self.get_playbook_aggregation_operation(operation_id)
            if operation is None:
                raise RuntimeError("aggregation operation receipt missing")
            return operation

    def get_playbook_aggregation_operation(
        self, operation_id: str
    ) -> PlaybookAggregationOperation | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM playbook_aggregation_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            return operation_from_row(row) if row else None

    def next_playbook_aggregation_operation(
        self,
    ) -> PlaybookAggregationOperation | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM playbook_aggregation_operations "
                "WHERE status IN ('queued','running','retrying') AND next_attempt_at<=unixepoch() "
                "ORDER BY created_at,operation_id LIMIT 1"
            ).fetchone()
            return operation_from_row(row) if row else None

    def begin_playbook_aggregation_operation(
        self, operation_id: str, claim: PlaybookAggregationClaim
    ) -> PlaybookAggregationOperation:
        with self.commit_scope():
            if not self.validate_playbook_aggregation_claim(claim):
                raise RuntimeError("aggregation operation lease lost")
            cursor = self.conn.execute(
                "UPDATE playbook_aggregation_operations SET status='running', attempts=min(attempts+1,5), "
                "claim_fence=?,started_at=COALESCE(started_at,unixepoch()),updated_at=unixepoch(),error=NULL "
                "WHERE operation_id=? AND agent_version=? AND status IN ('queued','running','retrying')",
                (claim.fence, operation_id, claim.agent_version),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("aggregation operation no longer runnable")
            operation = self.get_playbook_aggregation_operation(operation_id)
            if operation is None:
                raise RuntimeError("aggregation operation receipt missing")
            return operation

    def complete_playbook_aggregation_operation(
        self, operation_id: str, claim: PlaybookAggregationClaim, result: dict[str, Any]
    ) -> None:
        with self.commit_scope():
            if not self.validate_playbook_aggregation_claim(claim):
                raise RuntimeError("aggregation operation lease lost")
            cursor = self.conn.execute(
                "UPDATE playbook_aggregation_operations SET status='succeeded',result=?,error=NULL, "
                "updated_at=unixepoch(),completed_at=unixepoch(),next_attempt_at=NULL "
                "WHERE operation_id=? AND status='running' AND claim_fence=? AND agent_version=?",
                (
                    PlaybookAggregationResult.model_validate(result).model_dump_json(),
                    operation_id,
                    claim.fence,
                    claim.agent_version,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("aggregation operation completion fence lost")

    def fail_playbook_aggregation_operation(
        self,
        operation_id: str,
        claim: PlaybookAggregationClaim,
        *,
        error: str,
        retry_seconds: int | None,
    ) -> None:
        with self.commit_scope():
            if not self.validate_playbook_aggregation_claim(claim):
                return
            self.conn.execute(
                "UPDATE playbook_aggregation_operations SET status=?,error=?,updated_at=unixepoch(), "
                "completed_at=CASE WHEN ? IS NULL THEN unixepoch() ELSE NULL END, "
                "next_attempt_at=CASE WHEN ? IS NULL THEN NULL ELSE unixepoch()+? END "
                "WHERE operation_id=? AND status='running' AND claim_fence=? AND agent_version=?",
                (
                    "failed" if retry_seconds is None else "retrying",
                    error,
                    retry_seconds,
                    retry_seconds,
                    retry_seconds,
                    operation_id,
                    claim.fence,
                    claim.agent_version,
                ),
            )
