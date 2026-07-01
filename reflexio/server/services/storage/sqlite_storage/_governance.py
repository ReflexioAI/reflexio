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
from reflexio.server.services.storage.governance_validation import (
    _CANONICAL_DELETE_TARGET_NAMES,
    _PREPARE_PHASE,
    _SNAPSHOT_TARGET_NAME,
    _canonicalize_governance_windows,
    _parse_governance_window_list,
    _validate_governance_int_list,
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

    # Provided via MRO by the co-composed PurgeOperationStoreMixin (purge bucket);
    # reached here by the cross-bucket rebuild-hide method.
    _record_purge_target_locked: Callable[..., None]

    def _deps(self) -> _SQLiteGovernanceDeps:
        return cast(_SQLiteGovernanceDeps, self)

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
