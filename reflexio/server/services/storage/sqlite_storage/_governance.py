from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Literal, Protocol, cast

from reflexio.models.api_schema.domain import AgentPlaybook, AgentPlaybookSourceWindow
from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
    PurgeOperationTarget,
    SubjectBarrierStatus,
    SubjectWriteBarrier,
)
from reflexio.server.services.storage.governance_validation import (
    _CANONICAL_DELETE_TARGET_NAMES,
    _PREPARE_PHASE,
    _SNAPSHOT_TARGET_NAME,
    _validate_governance_int_list,
)

_LEGACY_AUDIT_REQUEST_REF = "reqref_v1_legacy_unknown"

_PURGE_OPERATION_TARGETS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS purge_operation_targets (
    org_id TEXT NOT NULL,
    purge_id TEXT NOT NULL,
    target_name TEXT NOT NULL,
    target_ref TEXT NOT NULL DEFAULT '',
    phase TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    detail TEXT,
    deleted_count INTEGER NOT NULL DEFAULT 0,
    error_detail TEXT,
    started_at INTEGER,
    completed_at INTEGER,
    PRIMARY KEY (org_id, purge_id, target_name, target_ref, phase),
    FOREIGN KEY (org_id, purge_id) REFERENCES purge_operations(org_id, purge_id) ON DELETE CASCADE
);
"""

GOVERNANCE_DDL = f"""
CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    actor_type TEXT NOT NULL DEFAULT 'system',
    actor_ref TEXT,
    operation TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    subject_ref TEXT,
    request_ref TEXT NOT NULL,
    idempotency_key TEXT,
    status TEXT NOT NULL DEFAULT 'ok',
    detail TEXT,
    created_at INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_events_org_idem
    ON audit_events(org_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_audit_events_subject_created
    ON audit_events(org_id, subject_ref, created_at, event_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_org_created
    ON audit_events(org_id, created_at, event_id);

CREATE TABLE IF NOT EXISTS purge_operations (
    org_id TEXT NOT NULL,
    purge_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    subject_ref TEXT,
    request_ref TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_code TEXT,
    error_detail TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    completed_at INTEGER,
    PRIMARY KEY (org_id, purge_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_purge_operations_org_idem
    ON purge_operations(org_id, idempotency_key);

{_PURGE_OPERATION_TARGETS_TABLE_DDL}
CREATE INDEX IF NOT EXISTS idx_purge_targets_purge_phase
    ON purge_operation_targets(org_id, purge_id, phase, status);

CREATE TABLE IF NOT EXISTS subject_write_barriers (
    org_id TEXT NOT NULL,
    subject_ref TEXT NOT NULL,
    purge_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    error_detail TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (org_id, subject_ref),
    CHECK (status IN ('erasing', 'erased', 'failed'))
);
CREATE INDEX IF NOT EXISTS idx_subject_write_barriers_org_purge
    ON subject_write_barriers(org_id, purge_id);
"""


def init_governance_tables(conn: sqlite3.Connection) -> None:
    _upgrade_legacy_purge_operation_targets_table(conn)
    conn.executescript(GOVERNANCE_DDL)
    _enforce_audit_request_ref_not_null(conn)
    _ensure_governance_subject_ref_columns(conn)


def _ensure_governance_subject_ref_columns(conn: sqlite3.Connection) -> None:
    for table in (
        "requests",
        "interactions",
        "profiles",
        "user_playbooks",
        "agent_success_evaluation_result",
    ):
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if not columns:
            continue
        if "governance_subject_ref" in columns:
            pass
        else:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN governance_subject_ref TEXT")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table}_governance_subject_ref "
            f"ON {table}(governance_subject_ref)"
        )


def _json_dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, default=str)


def _json_loads(text: str | None) -> Any:
    if not text:
        return None
    return json.loads(text)


def _upgrade_legacy_purge_operation_targets_table(conn: sqlite3.Connection) -> None:
    target_columns = [
        row[1] for row in conn.execute("PRAGMA table_info(purge_operation_targets)")
    ]
    if not target_columns or "org_id" in target_columns:
        return
    conn.execute(
        "ALTER TABLE purge_operation_targets RENAME TO purge_operation_targets_legacy"
    )
    conn.executescript(_PURGE_OPERATION_TARGETS_TABLE_DDL)
    conn.execute(
        """INSERT INTO purge_operation_targets (
               org_id, purge_id, target_name, target_ref, phase, status, detail,
               deleted_count, error_detail, started_at, completed_at
           )
           SELECT uniquely_mapped_purges.org_id, legacy.purge_id, legacy.target_name,
                  legacy.target_ref, legacy.phase, legacy.status, legacy.detail,
                  legacy.deleted_count, legacy.error_detail, legacy.started_at,
                  legacy.completed_at
           FROM purge_operation_targets_legacy AS legacy
           JOIN (
               SELECT MIN(org_id) AS org_id, purge_id
               FROM purge_operations
               GROUP BY purge_id
               HAVING COUNT(*) = 1
           ) AS uniquely_mapped_purges
             ON uniquely_mapped_purges.purge_id = legacy.purge_id"""
    )
    conn.execute("DROP TABLE purge_operation_targets_legacy")


def _enforce_audit_request_ref_not_null(conn: sqlite3.Connection) -> None:
    audit_columns = [row[1] for row in conn.execute("PRAGMA table_info(audit_events)")]
    if not audit_columns:
        return
    conn.execute(
        "UPDATE audit_events SET request_ref = ? WHERE request_ref IS NULL",
        (_LEGACY_AUDIT_REQUEST_REF,),
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS audit_events_request_ref_not_null
        BEFORE INSERT ON audit_events
        WHEN NEW.request_ref IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'audit_events.request_ref is required');
        END
        """
    )


def _row_to_audit_event(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        org_id=row["org_id"],
        actor_type=row["actor_type"],
        actor_ref=row["actor_ref"],
        operation=row["operation"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        subject_ref=row["subject_ref"],
        request_ref=row["request_ref"],
        idempotency_key=row["idempotency_key"],
        status=row["status"],
        detail=_json_loads(row["detail"]),
        created_at=row["created_at"],
    )


def _row_to_purge_operation(row: sqlite3.Row) -> PurgeOperation:
    return PurgeOperation(
        purge_id=row["purge_id"],
        org_id=row["org_id"],
        operation_type=row["operation_type"],
        scope_type=row["scope_type"],
        subject_ref=row["subject_ref"],
        request_ref=row["request_ref"],
        idempotency_key=row["idempotency_key"],
        status=row["status"],
        error_code=row["error_code"],
        error_detail=row["error_detail"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _row_to_purge_target(row: sqlite3.Row) -> PurgeOperationTarget:
    return PurgeOperationTarget(
        purge_id=row["purge_id"],
        target_name=row["target_name"],
        target_ref=row["target_ref"],
        phase=row["phase"],
        status=row["status"],
        detail=_json_loads(row["detail"]),
        deleted_count=row["deleted_count"],
        error_detail=row["error_detail"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


def _row_to_subject_write_barrier(row: sqlite3.Row) -> SubjectWriteBarrier:
    return SubjectWriteBarrier(
        org_id=str(row["org_id"]),
        subject_ref=str(row["subject_ref"]),
        purge_id=str(row["purge_id"]),
        status=cast(SubjectBarrierStatus, str(row["status"])),
        error_code=row["error_code"],
        error_detail=row["error_detail"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
    )


class _SQLiteGovernanceDeps(Protocol):
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str
    _has_sqlite_vec: bool

    def _subject_ref_for_user_id(self, user_id: str) -> str: ...

    def _fetchall(
        self, sql: str, params: list[Any] | tuple[Any, ...]
    ) -> list[sqlite3.Row]: ...

    def _fetchone(
        self, sql: str, params: list[Any] | tuple[Any, ...]
    ) -> sqlite3.Row | None: ...

    def _partition_purge_vs_delete(
        self, entity_type: Literal["profile", "user_playbook"], ids: list[str]
    ) -> tuple[list[str], list[str]]: ...

    def _delete_in_chunks(
        self, table_name: str, column_name: str, values: list[Any]
    ) -> None: ...

    def _delete_source_windows_for_user_playbook_ids(
        self, user_playbook_ids: list[int]
    ) -> None: ...

    def _get_embedding(self, text: str) -> list[float]: ...

    def set_source_windows_for_agent_playbook(
        self, agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
    ) -> None: ...

    def get_source_windows_for_agent_playbook(
        self, agent_playbook_id: int
    ) -> list[AgentPlaybookSourceWindow]: ...

    def get_agent_playbook_by_id(
        self,
        agent_playbook_id: int,
        *,
        include_tombstones: bool = False,
    ) -> AgentPlaybook | None: ...

    def _index_agent_playbook_fts_vec(self, ap: AgentPlaybook) -> None: ...


class SQLiteGovernanceMixin:
    """SQLite governance storage primitives."""

    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str

    def _deps(self) -> _SQLiteGovernanceDeps:
        return cast(_SQLiteGovernanceDeps, self)

    def _validate_prepared_delete_target_matrix_locked(self, purge_id: str) -> None:
        snapshot = self.conn.execute(
            """SELECT 1 FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                 AND phase = ? AND status = 'complete'""",
            (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
        ).fetchone()
        if snapshot is None:
            raise ValueError("Cannot delete user data without target snapshot marker")
        delete_rows = self.conn.execute(
            """SELECT target_name, status FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND phase = 'delete'
                 AND target_ref = 'all'""",
            (self.org_id, purge_id),
        ).fetchall()
        delete_statuses = {
            str(row["target_name"]): str(row["status"]) for row in delete_rows
        }
        missing_delete_targets = [
            target_name
            for target_name in _CANONICAL_DELETE_TARGET_NAMES
            if delete_statuses.get(target_name) not in {"pending", "complete"}
        ]
        if missing_delete_targets:
            raise ValueError(
                "Cannot delete user data without complete delete target matrix: "
                + ", ".join(missing_delete_targets)
            )

    def _validate_hide_for_rebuild_targets_locked(self, purge_id: str) -> None:
        rebuild_rows = self.conn.execute(
            """SELECT DISTINCT target_ref
               FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                 AND phase = 'rebuild_without_erased_sources' AND target_ref != ''
               ORDER BY target_ref ASC""",
            (self.org_id, purge_id),
        ).fetchall()
        if not rebuild_rows:
            return
        hidden_refs = {
            str(row["target_ref"])
            for row in self.conn.execute(
                """SELECT target_ref
                   FROM purge_operation_targets
                   WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                     AND phase = 'hide_for_rebuild' AND status = 'complete'""",
                (self.org_id, purge_id),
            ).fetchall()
        }
        missing_hidden_refs = [
            str(row["target_ref"])
            for row in rebuild_rows
            if str(row["target_ref"]) not in hidden_refs
        ]
        if missing_hidden_refs:
            raise ValueError(
                "Cannot delete user data before hide_for_rebuild completes for "
                f"planned agent_playbooks: {', '.join(missing_hidden_refs)}"
            )

    def _planned_governance_delete_counts(
        self, user_id: str, owned_user_playbook_ids: set[int]
    ) -> dict[str, int]:
        request_row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM requests WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        interaction_row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM interactions WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        eval_result_row = self.conn.execute(
            """SELECT COUNT(*) AS cnt
               FROM agent_success_evaluation_result
               WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        rle_result_row = self.conn.execute(
            """SELECT COUNT(*) AS cnt
               FROM retrieved_learning_evaluation
               WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        session_count_row = self.conn.execute(
            "SELECT COUNT(DISTINCT session_id) AS cnt FROM requests WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        subject_ref = self._deps()._subject_ref_for_user_id(user_id)
        session_outcome_row = self.conn.execute(
            """SELECT COUNT(*) AS cnt FROM session_outcomes
               WHERE user_id = ? OR governance_subject_ref = ?""",
            (user_id, subject_ref),
        ).fetchone()
        profile_rows = self.conn.execute(
            "SELECT profile_id FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        if (
            request_row is None
            or interaction_row is None
            or eval_result_row is None
            or rle_result_row is None
            or session_count_row is None
            or session_outcome_row is None
        ):
            raise ValueError("Missing governance count rows")
        profile_ids = [str(row["profile_id"]) for row in profile_rows]
        purge_profile_ids, delete_profile_ids = self._deps()._partition_purge_vs_delete(
            "profile",
            profile_ids,
        )
        playbook_ids = [
            str(user_playbook_id)
            for user_playbook_id in sorted(owned_user_playbook_ids)
        ]
        purge_playbook_ids, delete_playbook_ids = (
            self._deps()._partition_purge_vs_delete(
                "user_playbook",
                playbook_ids,
            )
        )
        return {
            "session_outcome": int(session_outcome_row["cnt"]),
            "request": int(request_row["cnt"]),
            "interaction": int(interaction_row["cnt"]),
            "profile": len(delete_profile_ids),
            "profile_purge": len(purge_profile_ids),
            "user_playbook": len(delete_playbook_ids),
            "agent_success_evaluation_result": int(eval_result_row["cnt"]),
            # Offline tuner tables are enterprise-only (tenant stream); the
            # OSS SQLite backend has no such tables, so the planned and
            # deleted counts are structurally zero. The delete-target matrix
            # validator still requires the target rows to exist.
            "offline_tuner_reward_label": 0,
            "offline_tuner_reward_label_target_by_target_owner": 0,
            "retrieved_learning_evaluation_result": int(rle_result_row["cnt"]),
            # Planned as an upper bound: up to 3 evaluation state namespaces
            # per session (retrieved-eval state, agent-success marker,
            # grade-cache rows). The delete phase reports the exact count.
            "evaluation_operation_state": 3 * int(session_count_row["cnt"]),
            "user_playbook_purge": len(purge_playbook_ids),
        }

    def _owned_user_playbook_ids_locked(self, user_id: str) -> set[int]:
        return {
            int(row["user_playbook_id"])
            for row in self.conn.execute(
                "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        }

    def _prepared_owned_user_playbook_ids_locked(self, purge_id: str) -> set[int]:
        row = self.conn.execute(
            """SELECT detail FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = ?
                 AND target_ref = 'all' AND phase = ? AND status = 'complete'""",
            (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
        ).fetchone()
        if row is None:
            raise ValueError("Prepared target snapshot is missing")
        detail = _json_loads(row["detail"])
        if not isinstance(detail, dict):
            raise ValueError("Prepared target snapshot detail is missing")
        return set(
            _validate_governance_int_list(
                "owned_user_playbook_ids",
                detail.get("owned_user_playbook_ids"),
            )
        )
