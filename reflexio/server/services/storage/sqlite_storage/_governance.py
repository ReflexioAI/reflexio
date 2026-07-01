from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from typing import Any, Literal, Protocol, cast

from reflexio.models.api_schema.domain import AgentPlaybook, AgentPlaybookSourceWindow
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
    PurgeOperationTarget,
    SubjectBarrierStatus,
    SubjectWriteBarrier,
)
from reflexio.server.services.governance.config import (
    get_governance_ref_secret,
    governance_subject_ref,
)
from reflexio.server.services.storage.error import SubjectWriteBarrierError
from reflexio.server.services.storage.governance_validation import (
    _CANONICAL_DELETE_TARGET_NAMES,
    _PREPARE_PHASE,
    _SNAPSHOT_TARGET_NAME,
    _canonicalize_audit_event_for_persistence,
    _canonicalize_governance_windows,
    _epoch_now,
    _is_successful_erase_event,
    _parse_governance_window_list,
    _successful_erase_identity,
    _validate_governance_error_code,
    _validate_governance_error_detail,
    _validate_governance_int_list,
    _validate_governance_prefixed_ref,
    _validate_governance_purge_id,
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


def _build_agent_playbook_source_window_rows(
    agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
) -> list[tuple[int, int, str]]:
    by_id: dict[int, list[int]] = {}
    for window in windows:
        ids = by_id.setdefault(window.user_playbook_id, [])
        seen = set(ids)
        for source_id in window.source_interaction_ids:
            if source_id not in seen:
                ids.append(source_id)
                seen.add(source_id)
    return [
        (
            agent_playbook_id,
            user_playbook_id,
            _json_dumps(source_interaction_ids) or "[]",
        )
        for user_playbook_id, source_interaction_ids in by_id.items()
    ]


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

    # Provided via MRO by the co-composed AuditEventStoreMixin (audit bucket);
    # reached here by the cross-bucket purge / barrier completion methods.
    _append_audit_event_with_cursor: Callable[
        [sqlite3.Connection | sqlite3.Cursor, AuditEvent], bool
    ]

    # Provided via MRO by the co-composed PurgeOperationStoreMixin (purge bucket);
    # reached here by the cross-bucket rebuild-hide / governance-erase-execution
    # / barrier completion methods.
    get_purge_operation: Callable[[str], PurgeOperation]
    _record_purge_target_locked: Callable[..., None]

    def _deps(self) -> _SQLiteGovernanceDeps:
        return cast(_SQLiteGovernanceDeps, self)

    def _barrier_from_purge(
        self,
        purge_operation: PurgeOperation,
        *,
        subject_ref: str,
    ) -> SubjectWriteBarrier:
        if purge_operation.subject_ref != subject_ref:
            raise ValueError(
                "Purge operation subject_ref must match the barrier subject_ref"
            )
        status_by_purge_status: dict[str, SubjectBarrierStatus] = {
            "pending": "erasing",
            "running": "erasing",
            "complete": "erased",
            "failed": "failed",
        }
        return SubjectWriteBarrier(
            org_id=purge_operation.org_id,
            subject_ref=subject_ref,
            purge_id=purge_operation.purge_id,
            status=status_by_purge_status[purge_operation.status],
            error_code=purge_operation.error_code,
            error_detail=purge_operation.error_detail,
            created_at=purge_operation.created_at,
            updated_at=purge_operation.updated_at,
        )

    def _active_subject_barrier_locked(self, subject_ref: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT * FROM subject_write_barriers
               WHERE org_id = ? AND subject_ref = ? AND status IN ('erasing', 'erased')""",
            (self.org_id, subject_ref),
        ).fetchone()

    def _assert_subject_writable_locked(self, subject_ref: str) -> None:
        row = self._active_subject_barrier_locked(subject_ref)
        if row is not None:
            raise SubjectWriteBarrierError(
                f"subject {subject_ref} is blocked by erasure barrier {row['purge_id']}"
            )

    def _subject_ref_for_user_id(self, user_id: str) -> str:
        return governance_subject_ref(
            self.org_id,
            user_id,
            get_governance_ref_secret(),
        )

    def _legacy_request_ids_for_subject_locked(self, subject_ref: str) -> set[str]:
        request_ids: set[str] = set()
        for row in self.conn.execute(
            """SELECT request_id, user_id
               FROM requests
               WHERE governance_subject_ref IS NULL"""
        ):
            user_id = str(row["user_id"])
            if self._subject_ref_for_user_id(user_id) != subject_ref:
                continue
            request_ids.add(str(row["request_id"]))
        return request_ids

    def _legacy_user_id_rows_remain_locked(
        self,
        *,
        table: str,
        subject_ref: str,
        request_ids: set[str] | None = None,
        request_id_column: str | None = None,
    ) -> bool:
        sql = f"SELECT user_id{', ' + request_id_column if request_id_column else ''} FROM {table} WHERE governance_subject_ref IS NULL"  # noqa: S608
        for row in self.conn.execute(sql):
            user_id = str(row["user_id"])
            if self._subject_ref_for_user_id(user_id) == subject_ref:
                return True
            if (
                request_ids
                and request_id_column is not None
                and str(row[request_id_column]) in request_ids
            ):
                return True
        return False

    def _same_subject_rows_remain_locked(self, subject_ref: str) -> bool:
        legacy_request_ids = self._legacy_request_ids_for_subject_locked(subject_ref)
        for table in (
            "requests",
            "interactions",
            "profiles",
            "user_playbooks",
            "agent_success_evaluation_result",
        ):
            row = self.conn.execute(
                f"""SELECT 1 FROM {table}
                    WHERE governance_subject_ref = ?
                    LIMIT 1""",
                (subject_ref,),
            ).fetchone()
            if row is not None:
                return True
        if legacy_request_ids:
            return True
        if self._legacy_user_id_rows_remain_locked(
            table="interactions",
            subject_ref=subject_ref,
            request_ids=legacy_request_ids,
            request_id_column="request_id",
        ):
            return True
        if self._legacy_user_id_rows_remain_locked(
            table="profiles",
            subject_ref=subject_ref,
            request_ids=legacy_request_ids,
            request_id_column="generated_from_request_id",
        ):
            return True
        if self._legacy_user_id_rows_remain_locked(
            table="user_playbooks",
            subject_ref=subject_ref,
            request_ids=legacy_request_ids,
            request_id_column="request_id",
        ):
            return True
        return self._legacy_user_id_rows_remain_locked(
            table="agent_success_evaluation_result",
            subject_ref=subject_ref,
        )

    def _replace_agent_playbook_source_windows_locked(
        self, agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
    ) -> None:
        self.conn.execute(
            "DELETE FROM agent_playbook_source_user_playbooks WHERE agent_playbook_id = ?",
            (agent_playbook_id,),
        )
        source_window_rows = _build_agent_playbook_source_window_rows(
            agent_playbook_id, windows
        )
        if source_window_rows:
            self.conn.executemany(
                """INSERT OR IGNORE INTO agent_playbook_source_user_playbooks
                   (agent_playbook_id, user_playbook_id, source_interaction_ids)
                   VALUES (?, ?, ?)""",
                source_window_rows,
            )

    def _delete_agent_playbook_search_rows_locked(self, agent_playbook_id: int) -> None:
        self.conn.execute(
            "DELETE FROM agent_playbooks_fts WHERE rowid = ?",
            (agent_playbook_id,),
        )
        if self._deps()._has_sqlite_vec:
            self.conn.execute(
                "DELETE FROM agent_playbooks_vec WHERE rowid = ?",
                (agent_playbook_id,),
            )

    def _upsert_agent_playbook_search_rows_locked(
        self,
        *,
        agent_playbook_id: int,
        trigger: str | None,
        content: str,
        expanded_terms: str | None,
        embedding: list[float],
    ) -> None:
        self._delete_agent_playbook_search_rows_locked(agent_playbook_id)
        fts_parts = [trigger or "", content]
        if expanded_terms:
            fts_parts.append(expanded_terms)
        self.conn.execute(
            "INSERT INTO agent_playbooks_fts(rowid, search_text) VALUES (?, ?)",
            (
                agent_playbook_id,
                " ".join(part for part in fts_parts if part) or "",
            ),
        )
        if self._deps()._has_sqlite_vec and embedding:
            self.conn.execute(
                "INSERT INTO agent_playbooks_vec(rowid, embedding) VALUES (?, ?)",
                (agent_playbook_id, json.dumps(embedding)),
            )

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
        profile_rows = self.conn.execute(
            "SELECT profile_id FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        if request_row is None or interaction_row is None or eval_result_row is None:
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
            "request": int(request_row["cnt"]),
            "interaction": int(interaction_row["cnt"]),
            "profile": len(delete_profile_ids),
            "profile_purge": len(purge_profile_ids),
            "user_playbook": len(delete_playbook_ids),
            "agent_success_evaluation_result": int(eval_result_row["cnt"]),
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

    def _purge_governance_entity_content_locked(
        self,
        *,
        entity_type: Literal["profile", "user_playbook"],
        entity_id: str,
        rowid: int,
    ) -> bool:
        from ._lineage import _PURGE_SQL, _append_event_stmt

        sql = _PURGE_SQL[entity_type]
        cur = self.conn.execute(sql, (entity_id,))
        if cur.rowcount <= 0:
            return False
        _append_event_stmt(
            self.conn,
            org_id=self.org_id,
            entity_type=entity_type,
            entity_id=entity_id,
            op="purge",
            prov="wasPurged",
            source_ids=[],
            actor="erasure",
            request_id=f"purge_{entity_id}",
            reason="content_purge",
        )
        if entity_type == "profile":
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?",
                (entity_id,),
            )
            if self._deps()._has_sqlite_vec:
                self.conn.execute(
                    "DELETE FROM profiles_vec WHERE rowid = ?",
                    (rowid,),
                )
        else:
            self.conn.execute(
                "DELETE FROM user_playbooks_fts WHERE rowid = ?",
                (rowid,),
            )
            if self._deps()._has_sqlite_vec:
                self.conn.execute(
                    "DELETE FROM user_playbooks_vec WHERE rowid = ?",
                    (rowid,),
                )
        return True

    def _clear_user_data_for_governance_locked(
        self,
        user_id: str,
        *,
        expected_user_playbook_ids: set[int] | None = None,
    ) -> dict[str, int]:
        deps = self._deps()
        interaction_ids = [
            int(row["interaction_id"])
            for row in self.conn.execute(
                "SELECT interaction_id FROM interactions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        raw_upb_ids = [
            int(row["user_playbook_id"])
            for row in self.conn.execute(
                "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        if (
            expected_user_playbook_ids is not None
            and set(raw_upb_ids) != expected_user_playbook_ids
        ):
            raise ValueError(
                "Current user playbooks no longer match prepared purge snapshot"
            )
        request_ids = [
            str(row["request_id"])
            for row in self.conn.execute(
                "SELECT request_id FROM requests WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        profile_rows = self.conn.execute(
            "SELECT rowid, profile_id FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        profile_rowid_by_id = {
            str(row["profile_id"]): int(row["rowid"]) for row in profile_rows
        }
        all_profile_ids = list(profile_rowid_by_id)

        purge_profile_ids, delete_profile_ids = deps._partition_purge_vs_delete(
            "profile",
            all_profile_ids,
        )
        purge_upb_str_ids, delete_upb_str_ids = deps._partition_purge_vs_delete(
            "user_playbook",
            [str(user_playbook_id) for user_playbook_id in raw_upb_ids],
        )
        purge_upb_ids = [int(entity_id) for entity_id in purge_upb_str_ids]
        delete_upb_ids = [int(entity_id) for entity_id in delete_upb_str_ids]
        erased_entity_ids = [
            *request_ids,
            *[str(interaction_id) for interaction_id in interaction_ids],
            *all_profile_ids,
            *[str(user_playbook_id) for user_playbook_id in raw_upb_ids],
        ]
        if erased_entity_ids:
            erased_entity_id_set = set(erased_entity_ids)
            lineage_source_event_ids: list[int] = []
            for row in self.conn.execute(
                "SELECT event_id, source_ids FROM lineage_event WHERE org_id = ?",
                (self.org_id,),
            ).fetchall():
                try:
                    source_ids = json.loads(str(row["source_ids"] or "[]"))
                except json.JSONDecodeError:
                    source_ids = []
                if any(
                    str(source_id) in erased_entity_id_set for source_id in source_ids
                ):
                    lineage_source_event_ids.append(int(row["event_id"]))
            deps._delete_in_chunks(
                "lineage_event", "event_id", lineage_source_event_ids
            )
            placeholders = ",".join("?" for _ in erased_entity_ids)
            self.conn.execute(
                f"""DELETE FROM lineage_event
                    WHERE org_id = ?
                      AND (
                        request_id IN ({placeholders})
                        OR entity_id IN ({placeholders})
                      )""",  # noqa: S608
                [self.org_id, *erased_entity_ids, *erased_entity_ids],
            )
        delete_profile_rowids = [
            profile_rowid_by_id[profile_id]
            for profile_id in delete_profile_ids
            if profile_id in profile_rowid_by_id
        ]

        deps._delete_in_chunks("interactions_fts", "rowid", interaction_ids)
        deps._delete_in_chunks("user_playbooks_fts", "rowid", delete_upb_ids)
        deps._delete_in_chunks("profiles_fts", "profile_id", delete_profile_ids)
        if deps._has_sqlite_vec:
            deps._delete_in_chunks("interactions_vec", "rowid", interaction_ids)
            deps._delete_in_chunks("user_playbooks_vec", "rowid", delete_upb_ids)
            deps._delete_in_chunks("profiles_vec", "rowid", delete_profile_rowids)

        interactions_cur = self.conn.execute(
            "DELETE FROM interactions WHERE user_id = ?",
            (user_id,),
        )
        eval_results_cur = self.conn.execute(
            """DELETE FROM agent_success_evaluation_result
               WHERE user_id = ?""",
            (user_id,),
        )
        requests_cur = self.conn.execute(
            "DELETE FROM requests WHERE user_id = ?",
            (user_id,),
        )
        if raw_upb_ids:
            deps._delete_source_windows_for_user_playbook_ids(raw_upb_ids)
        if delete_upb_ids:
            deps._delete_in_chunks("user_playbooks", "user_playbook_id", delete_upb_ids)
        if delete_profile_ids:
            deps._delete_in_chunks("profiles", "profile_id", delete_profile_ids)

        purged_profiles = 0
        for profile_id in purge_profile_ids:
            rowid = profile_rowid_by_id.get(profile_id)
            if rowid is None:
                continue
            purged_profiles += int(
                self._purge_governance_entity_content_locked(
                    entity_type="profile",
                    entity_id=profile_id,
                    rowid=rowid,
                )
            )

        purged_user_playbooks = 0
        for user_playbook_id in purge_upb_ids:
            purged_user_playbooks += int(
                self._purge_governance_entity_content_locked(
                    entity_type="user_playbook",
                    entity_id=str(user_playbook_id),
                    rowid=user_playbook_id,
                )
            )

        return {
            "interactions": interactions_cur.rowcount,
            "user_playbooks": len(delete_upb_ids),
            "profiles": len(delete_profile_ids),
            "requests": requests_cur.rowcount,
            "agent_success_evaluation_results": eval_results_cur.rowcount,
            "purged_profiles": purged_profiles,
            "purged_user_playbooks": purged_user_playbooks,
        }

    def hide_governance_agent_playbooks_for_rebuild(self, purge_id: str) -> list[int]:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                target_rows = self.conn.execute(
                    """SELECT target_ref
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ?
                         AND target_name = 'agent_playbook'
                         AND phase = 'rebuild_without_erased_sources'
                         AND target_ref != ''
                         AND status != 'complete'
                       ORDER BY CAST(target_ref AS INTEGER) ASC""",
                    (self.org_id, purge_id),
                ).fetchall()
                agent_playbook_ids = [int(row["target_ref"]) for row in target_rows]
                if not agent_playbook_ids:
                    self.conn.commit()
                    return []
                placeholders = ",".join("?" for _ in agent_playbook_ids)
                self.conn.execute(
                    f"""UPDATE agent_playbooks
                        SET status = ?
                        WHERE agent_playbook_id IN ({placeholders})""",
                    [Status.ARCHIVE_IN_PROGRESS.value, *agent_playbook_ids],
                )
                for agent_playbook_id in agent_playbook_ids:
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name="agent_playbook",
                        target_ref=str(agent_playbook_id),
                        phase="hide_for_rebuild",
                        status="complete",
                        detail=None,
                        deleted_count=0,
                        error_detail=None,
                    )
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name="agent_playbook",
                        target_ref=str(agent_playbook_id),
                        phase="rebuild_without_erased_sources",
                        status="running",
                        detail=None,
                        deleted_count=0,
                        error_detail=None,
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return agent_playbook_ids

    def apply_governance_user_data_delete(
        self, purge_id: str, user_id: str
    ) -> dict[str, int]:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        name_map = {
            "interactions": "interaction",
            "user_playbooks": "user_playbook",
            "profiles": "profile",
            "requests": "request",
            "agent_success_evaluation_results": "agent_success_evaluation_result",
            "purged_profiles": "profile_purge",
            "purged_user_playbooks": "user_playbook_purge",
        }
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self._validate_prepared_delete_target_matrix_locked(purge_id)
                self._validate_hide_for_rebuild_targets_locked(purge_id)
                expected_user_playbook_ids = (
                    self._prepared_owned_user_playbook_ids_locked(purge_id)
                )
                counts = self._clear_user_data_for_governance_locked(
                    user_id,
                    expected_user_playbook_ids=expected_user_playbook_ids,
                )
                for key, value in counts.items():
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name=name_map.get(key, key),
                        target_ref="all",
                        phase="delete",
                        status="complete",
                        detail={"count": int(value)},
                        deleted_count=int(value),
                        error_detail=None,
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return counts

    def apply_governance_agent_playbook_rebuild(
        self,
        purge_id: str,
        agent_playbook_id: int,
        remaining_source_windows: list[dict[str, object]],
        content: str | None,
        trigger: str | None,
        rationale: str | None,
        blocking_issue: dict[str, object] | None,
        expanded_terms: str | None,
        tags: list[str] | None,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        windows = _parse_governance_window_list(
            "remaining_source_windows", remaining_source_windows
        )
        canonical_remaining_windows = [window.model_dump() for window in windows]
        content_value = content or ""
        trigger_value = trigger or None
        embedding_text = trigger_value or content_value
        embedding = (
            self._deps()._get_embedding(embedding_text) if embedding_text else []
        )
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                rebuild_target_row = self.conn.execute(
                    """SELECT status, detail
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                         AND target_ref = ? AND phase = 'rebuild_without_erased_sources'""",
                    (self.org_id, purge_id, str(agent_playbook_id)),
                ).fetchone()
                if rebuild_target_row is None:
                    raise ValueError("planned rebuild target does not exist")
                if rebuild_target_row["status"] == "complete":
                    raise ValueError("planned rebuild target is already complete")
                rebuild_detail = _json_loads(rebuild_target_row["detail"])
                if not isinstance(rebuild_detail, dict) or not {
                    "original_source_windows",
                    "previous_lifecycle_status",
                    "remaining_source_windows",
                }.issubset(rebuild_detail):
                    raise ValueError(
                        "planned rebuild target is missing source window detail"
                    )
                planned_remaining_windows = _canonicalize_governance_windows(
                    "planned remaining_source_windows",
                    cast(
                        list[dict[str, object]],
                        rebuild_detail["remaining_source_windows"],
                    ),
                )
                if planned_remaining_windows != canonical_remaining_windows:
                    raise ValueError(
                        "remaining_source_windows must match the planned rebuild target"
                    )
                previous_lifecycle_status = cast(
                    str | None, rebuild_detail["previous_lifecycle_status"]
                )
                hide_target_row = self.conn.execute(
                    """SELECT status
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                         AND target_ref = ? AND phase = 'hide_for_rebuild'""",
                    (self.org_id, purge_id, str(agent_playbook_id)),
                ).fetchone()
                if hide_target_row is None or hide_target_row["status"] != "complete":
                    raise ValueError("hide_for_rebuild target must be complete")
                if windows:
                    cur = self.conn.execute(
                        """UPDATE agent_playbooks
                           SET content = ?, trigger = ?, rationale = ?, blocking_issue = ?,
                               embedding = ?, expanded_terms = ?, tags = ?, status = ?
                           WHERE agent_playbook_id = ?""",
                        (
                            content_value,
                            trigger_value,
                            rationale,
                            json.dumps(blocking_issue)
                            if blocking_issue is not None
                            else None,
                            _json_dumps(embedding),
                            expanded_terms,
                            _json_dumps(tags),
                            previous_lifecycle_status,
                            agent_playbook_id,
                        ),
                    )
                    if cur.rowcount == 0:
                        raise ValueError(
                            f"Agent playbook with ID {agent_playbook_id} not found"
                        )
                    self._replace_agent_playbook_source_windows_locked(
                        agent_playbook_id, windows
                    )
                    self._upsert_agent_playbook_search_rows_locked(
                        agent_playbook_id=agent_playbook_id,
                        trigger=trigger_value,
                        content=content_value,
                        expanded_terms=expanded_terms,
                        embedding=embedding,
                    )
                else:
                    from ._playbook import _emit_hard_delete_playbook

                    self._delete_agent_playbook_search_rows_locked(agent_playbook_id)
                    self.conn.execute(
                        "DELETE FROM agent_playbook_source_user_playbooks WHERE agent_playbook_id = ?",
                        (agent_playbook_id,),
                    )
                    cur = self.conn.execute(
                        "DELETE FROM agent_playbooks WHERE agent_playbook_id = ?",
                        (agent_playbook_id,),
                    )
                    if cur.rowcount == 0:
                        raise ValueError(
                            f"Agent playbook with ID {agent_playbook_id} not found"
                        )
                    _emit_hard_delete_playbook(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="agent_playbook",
                        entity_id=str(agent_playbook_id),
                        request_id=purge_id,
                    )
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name="agent_playbook",
                    target_ref=str(agent_playbook_id),
                    phase="rebuild_without_erased_sources",
                    status="complete",
                    detail=None,
                    deleted_count=0,
                    error_detail=None,
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def complete_purge_operation_with_audit(
        self, purge_id: str, audit_event: AuditEvent
    ) -> PurgeOperation:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        if audit_event.org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        if audit_event.idempotency_key != purge_id:
            raise ValueError("Audit event idempotency key must match purge_id")
        if not _is_successful_erase_event(audit_event, purge_id=purge_id):
            raise ValueError(
                "Completion requires a successful ERASE audit event for this purge"
            )
        audit_event = _canonicalize_audit_event_for_persistence(audit_event)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                row = self.conn.execute(
                    "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
                    (purge_id, self.org_id),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Purge operation {purge_id!r} not found")
                purge_operation = _row_to_purge_operation(row)
                if purge_operation.subject_ref != audit_event.subject_ref:
                    raise ValueError(
                        "Audit event subject_ref must match purge operation subject_ref"
                    )
                if purge_operation.request_ref != audit_event.request_ref:
                    raise ValueError(
                        "Audit event request_ref must match purge operation request_ref"
                    )
                barrier_row = self.conn.execute(
                    """SELECT status FROM subject_write_barriers
                       WHERE org_id = ? AND subject_ref = ? AND purge_id = ?""",
                    (self.org_id, audit_event.subject_ref, purge_id),
                ).fetchone()
                if barrier_row is None or barrier_row["status"] != "erasing":
                    raise ValueError("subject erasure barrier is missing")
                snapshot = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                         AND phase = ? AND status = 'complete'""",
                    (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
                ).fetchone()
                if snapshot is None:
                    raise ValueError(
                        "Cannot complete purge without target snapshot marker"
                    )
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
                    if delete_statuses.get(target_name) != "complete"
                ]
                if missing_delete_targets:
                    raise ValueError(
                        "Cannot complete purge without complete delete target matrix: "
                        + ", ".join(missing_delete_targets)
                    )
                incomplete = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND status != 'complete'
                       LIMIT 1""",
                    (self.org_id, purge_id),
                ).fetchone()
                if incomplete is not None:
                    raise ValueError("Cannot complete purge with incomplete targets")
                existing_audit_row = self.conn.execute(
                    """SELECT * FROM audit_events
                       WHERE org_id = ? AND idempotency_key = ?""",
                    (self.org_id, purge_id),
                ).fetchone()
                if existing_audit_row is not None:
                    existing_event = _row_to_audit_event(existing_audit_row)
                    if not _is_successful_erase_event(
                        existing_event, purge_id=purge_id
                    ):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                    if _successful_erase_identity(
                        existing_event
                    ) != _successful_erase_identity(audit_event):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                else:
                    self._append_audit_event_with_cursor(self.conn, audit_event)
                    existing_audit_row = self.conn.execute(
                        """SELECT * FROM audit_events
                           WHERE org_id = ? AND idempotency_key = ?""",
                        (self.org_id, purge_id),
                    ).fetchone()
                if existing_audit_row is None:
                    raise ValueError(
                        "Completion requires exactly one successful ERASE audit row "
                        "for the purge_id"
                    )
                existing_event = _row_to_audit_event(existing_audit_row)
                if not _is_successful_erase_event(existing_event, purge_id=purge_id):
                    raise ValueError(
                        "Completion requires exactly one matching successful ERASE "
                        "audit row for the purge_id"
                    )
                if _successful_erase_identity(
                    existing_event
                ) != _successful_erase_identity(audit_event):
                    raise ValueError(
                        "Completion requires exactly one matching successful ERASE "
                        "audit row for the purge_id"
                    )
                self.conn.execute(
                    """UPDATE purge_operations
                       SET status = 'complete',
                           error_code = NULL,
                           error_detail = NULL,
                           updated_at = ?,
                           completed_at = ?
                       WHERE purge_id = ? AND org_id = ?""",
                    (now, now, purge_id, self.org_id),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return self.get_purge_operation(purge_id)

    def begin_subject_erasure_barrier(
        self, subject_ref: str, purge_id: str
    ) -> SubjectWriteBarrier:
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        validated_purge_id = _validate_governance_purge_id("purge_id", purge_id)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                purge_row = self.conn.execute(
                    """SELECT * FROM purge_operations
                       WHERE purge_id = ? AND org_id = ?""",
                    (validated_purge_id, self.org_id),
                ).fetchone()
                if purge_row is None:
                    raise ValueError(
                        f"Purge operation {validated_purge_id!r} not found"
                    )
                purge_operation = _row_to_purge_operation(purge_row)
                if purge_operation.subject_ref != subject_ref:
                    raise ValueError(
                        "Purge operation subject_ref must match the barrier subject_ref"
                    )
                existing_barrier = self.conn.execute(
                    """SELECT * FROM subject_write_barriers
                       WHERE org_id = ? AND subject_ref = ?""",
                    (self.org_id, subject_ref),
                ).fetchone()
                if (
                    existing_barrier is not None
                    and str(existing_barrier["purge_id"]) != validated_purge_id
                ):
                    raise ValueError(
                        "Existing barrier purge_id must match the requested purge_id"
                    )
                if (
                    existing_barrier is not None
                    and str(existing_barrier["status"]) == "erased"
                ):
                    row = existing_barrier
                    self.conn.commit()
                    return _row_to_subject_write_barrier(row)
                self.conn.execute(
                    """INSERT INTO subject_write_barriers
                       (org_id, subject_ref, purge_id, status, created_at, updated_at)
                       VALUES (?, ?, ?, 'erasing', ?, ?)
                       ON CONFLICT(org_id, subject_ref) DO UPDATE SET
                         purge_id = excluded.purge_id,
                         status = 'erasing',
                         error_code = NULL,
                         error_detail = NULL,
                         updated_at = excluded.updated_at""",
                    (self.org_id, subject_ref, validated_purge_id, now, now),
                )
                row = self.conn.execute(
                    """SELECT * FROM subject_write_barriers
                       WHERE org_id = ? AND subject_ref = ?""",
                    (self.org_id, subject_ref),
                ).fetchone()
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        if row is None:
            raise ValueError("subject erasure barrier insert failed")
        return _row_to_subject_write_barrier(row)

    def assert_subject_writable(self, subject_ref: str) -> None:
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_subject_writable_locked(subject_ref)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def complete_subject_erasure_barrier_after_empty_check(
        self, purge_id: str, audit_event: AuditEvent
    ) -> PurgeOperation:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        if audit_event.org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        if audit_event.idempotency_key != purge_id:
            raise ValueError("Audit event idempotency key must match purge_id")
        if not _is_successful_erase_event(audit_event, purge_id=purge_id):
            raise ValueError(
                "Completion requires a successful ERASE audit event for this purge"
            )
        audit_event = _canonicalize_audit_event_for_persistence(audit_event)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                row = self.conn.execute(
                    "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
                    (purge_id, self.org_id),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Purge operation {purge_id!r} not found")
                purge_operation = _row_to_purge_operation(row)
                if purge_operation.subject_ref != audit_event.subject_ref:
                    raise ValueError(
                        "Audit event subject_ref must match purge operation subject_ref"
                    )
                if purge_operation.request_ref != audit_event.request_ref:
                    raise ValueError(
                        "Audit event request_ref must match purge operation request_ref"
                    )
                snapshot = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                         AND phase = ? AND status = 'complete'""",
                    (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
                ).fetchone()
                if snapshot is None:
                    raise ValueError(
                        "Cannot complete purge without target snapshot marker"
                    )
                if self._same_subject_rows_remain_locked(audit_event.subject_ref or ""):
                    raise ValueError("same-subject rows remain")
                delete_rows = self.conn.execute(
                    """SELECT target_name, status FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND phase = 'delete'
                         AND target_ref = 'all'""",
                    (self.org_id, purge_id),
                ).fetchall()
                delete_statuses = {
                    str(target_row["target_name"]): str(target_row["status"])
                    for target_row in delete_rows
                }
                missing_delete_targets = [
                    target_name
                    for target_name in _CANONICAL_DELETE_TARGET_NAMES
                    if delete_statuses.get(target_name) != "complete"
                ]
                if missing_delete_targets:
                    raise ValueError(
                        "Cannot complete purge without complete delete target matrix: "
                        + ", ".join(missing_delete_targets)
                    )
                incomplete = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND status != 'complete'
                       LIMIT 1""",
                    (self.org_id, purge_id),
                ).fetchone()
                if incomplete is not None:
                    raise ValueError("Cannot complete purge with incomplete targets")
                existing_audit_row = self.conn.execute(
                    """SELECT * FROM audit_events
                       WHERE org_id = ? AND idempotency_key = ?""",
                    (self.org_id, purge_id),
                ).fetchone()
                if existing_audit_row is not None:
                    existing_event = _row_to_audit_event(existing_audit_row)
                    if not _is_successful_erase_event(
                        existing_event, purge_id=purge_id
                    ):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                    if _successful_erase_identity(
                        existing_event
                    ) != _successful_erase_identity(audit_event):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                else:
                    self._append_audit_event_with_cursor(self.conn, audit_event)
                    existing_audit_row = self.conn.execute(
                        """SELECT * FROM audit_events
                           WHERE org_id = ? AND idempotency_key = ?""",
                        (self.org_id, purge_id),
                    ).fetchone()
                if existing_audit_row is None:
                    raise ValueError(
                        "Completion requires exactly one successful ERASE audit row "
                        "for the purge_id"
                    )
                existing_event = _row_to_audit_event(existing_audit_row)
                if not _is_successful_erase_event(existing_event, purge_id=purge_id):
                    raise ValueError(
                        "Completion requires exactly one matching successful ERASE "
                        "audit row for the purge_id"
                    )
                if _successful_erase_identity(
                    existing_event
                ) != _successful_erase_identity(audit_event):
                    raise ValueError(
                        "Completion requires exactly one matching successful ERASE "
                        "audit row for the purge_id"
                    )
                barrier_update = self.conn.execute(
                    """UPDATE subject_write_barriers
                       SET status = 'erased', error_code = NULL, error_detail = NULL, updated_at = ?
                       WHERE org_id = ? AND subject_ref = ? AND purge_id = ? AND status = 'erasing'""",
                    (now, self.org_id, audit_event.subject_ref, purge_id),
                )
                if barrier_update.rowcount != 1:
                    raise ValueError("subject erasure barrier is missing")
                self.conn.execute(
                    """UPDATE purge_operations
                       SET status = 'complete',
                           error_code = NULL,
                           error_detail = NULL,
                           updated_at = ?,
                           completed_at = ?
                       WHERE purge_id = ? AND org_id = ?""",
                    (now, now, purge_id, self.org_id),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return self.get_purge_operation(purge_id)

    def fail_subject_erasure_barrier(
        self,
        subject_ref: str,
        purge_id: str,
        error_code: str,
        error_detail: str,
    ) -> SubjectWriteBarrier:
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        validated_purge_id = _validate_governance_purge_id("purge_id", purge_id)
        validated_error_code = _validate_governance_error_code(error_code)
        validated_error_detail = _validate_governance_error_detail(error_detail)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                update_cursor = self.conn.execute(
                    """UPDATE subject_write_barriers
                       SET status = 'failed',
                           error_code = ?,
                           error_detail = ?,
                           updated_at = ?
                       WHERE org_id = ? AND subject_ref = ? AND purge_id = ?
                         AND status = 'erasing'""",
                    (
                        validated_error_code,
                        validated_error_detail,
                        now,
                        self.org_id,
                        subject_ref,
                        validated_purge_id,
                    ),
                )
                if update_cursor.rowcount != 1:
                    raise ValueError(
                        "subject erasure barrier failure requires a matching barrier"
                    )
                purge_row = self.conn.execute(
                    "SELECT 1 FROM purge_operations WHERE purge_id = ? AND org_id = ?",
                    (validated_purge_id, self.org_id),
                ).fetchone()
                if purge_row is not None:
                    self.conn.execute(
                        """UPDATE purge_operations
                           SET status = 'failed', error_code = ?, error_detail = ?,
                               updated_at = ?, completed_at = ?
                           WHERE purge_id = ? AND org_id = ?""",
                        (
                            validated_error_code,
                            validated_error_detail,
                            now,
                            now,
                            validated_purge_id,
                            self.org_id,
                        ),
                    )
                row = self.conn.execute(
                    """SELECT * FROM subject_write_barriers
                       WHERE org_id = ? AND subject_ref = ? AND purge_id = ?""",
                    (self.org_id, subject_ref, validated_purge_id),
                ).fetchone()
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        if row is None:
            raise ValueError("subject erasure barrier update failed")
        return _row_to_subject_write_barrier(row)

    def get_subject_write_barrier(self, subject_ref: str) -> SubjectWriteBarrier | None:
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        row = self.conn.execute(
            """SELECT * FROM subject_write_barriers
               WHERE org_id = ? AND subject_ref = ?""",
            (self.org_id, subject_ref),
        ).fetchone()
        if row is None:
            return None
        return _row_to_subject_write_barrier(row)
