from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any, Literal

from reflexio.models.api_schema.domain import AgentPlaybookSourceWindow
from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
    PurgeOperationTarget,
)

GOVERNANCE_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id TEXT NOT NULL,
    actor_type TEXT NOT NULL DEFAULT 'system',
    actor_ref TEXT,
    operation TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    subject_ref TEXT,
    request_ref TEXT,
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

CREATE TABLE IF NOT EXISTS purge_operations (
    purge_id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
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
    completed_at INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_purge_operations_org_idem
    ON purge_operations(org_id, idempotency_key);

CREATE TABLE IF NOT EXISTS purge_operation_targets (
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
    PRIMARY KEY (purge_id, target_name, target_ref, phase),
    FOREIGN KEY (purge_id) REFERENCES purge_operations(purge_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_purge_targets_purge_phase
    ON purge_operation_targets(purge_id, phase, status);
"""

_PREPARE_PHASE = "prepare_targets"
_SNAPSHOT_TARGET_NAME = "target_snapshot"


def init_governance_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(GOVERNANCE_DDL)


def _epoch_now() -> int:
    return int(datetime.now(UTC).timestamp())


def _json_dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, default=str)


def _json_loads(text: str | None) -> Any:
    if not text:
        return None
    return json.loads(text)


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


class SQLiteGovernanceMixin:
    """SQLite governance storage primitives."""

    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str

    def _append_audit_event_with_cursor(
        self, cur: sqlite3.Cursor, event: AuditEvent
    ) -> bool:
        inserted = cur.execute(
            """INSERT OR IGNORE INTO audit_events (
                   org_id, actor_type, actor_ref, operation, entity_type, entity_id,
                   subject_ref, request_ref, idempotency_key, status, detail, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.org_id,
                event.actor_type,
                event.actor_ref,
                event.operation,
                event.entity_type,
                event.entity_id,
                event.subject_ref,
                event.request_ref,
                event.idempotency_key,
                event.status,
                _json_dumps(event.detail),
                event.created_at,
            ),
        )
        return inserted.rowcount > 0

    def _record_purge_target_locked(
        self,
        *,
        purge_id: str,
        target_name: str,
        target_ref: str,
        phase: str,
        status: Literal["pending", "running", "failed", "complete"],
        detail: dict[str, object] | None,
        deleted_count: int,
        error_detail: str | None,
    ) -> None:
        now = _epoch_now()
        existing = self.conn.execute(
            """SELECT started_at, completed_at
               FROM purge_operation_targets
               WHERE purge_id = ? AND target_name = ? AND target_ref = ? AND phase = ?""",
            (purge_id, target_name, target_ref, phase),
        ).fetchone()
        started_at = existing["started_at"] if existing else None
        completed_at = existing["completed_at"] if existing else None
        if started_at is None and status in {"running", "failed", "complete"}:
            started_at = now
        if status in {"failed", "complete"}:
            completed_at = now
        self.conn.execute(
            """INSERT INTO purge_operation_targets (
                   purge_id, target_name, target_ref, phase, status, detail,
                   deleted_count, error_detail, started_at, completed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(purge_id, target_name, target_ref, phase) DO UPDATE SET
                   status = excluded.status,
                   detail = excluded.detail,
                   deleted_count = excluded.deleted_count,
                   error_detail = excluded.error_detail,
                   started_at = COALESCE(purge_operation_targets.started_at, excluded.started_at),
                   completed_at = excluded.completed_at""",
            (
                purge_id,
                target_name,
                target_ref,
                phase,
                status,
                _json_dumps(detail),
                deleted_count,
                error_detail,
                started_at,
                completed_at,
            ),
        )
        self.conn.execute(
            """UPDATE purge_operations
               SET status = CASE
                   WHEN status IN ('complete', 'failed') THEN status
                   WHEN ? IN ('running', 'complete') THEN 'running'
                   ELSE status
               END,
                   updated_at = ?
               WHERE purge_id = ? AND org_id = ?""",
            (status, now, purge_id, self.org_id),
        )

    def append_audit_event(self, event: AuditEvent) -> bool:
        with self._lock:
            inserted = self._append_audit_event_with_cursor(self.conn, event)
            self.conn.commit()
            return inserted

    def list_audit_events(
        self, subject_ref: str | None = None, *, org_id: str | None = None
    ) -> list[AuditEvent]:
        sql = "SELECT * FROM audit_events WHERE org_id = ?"
        params: list[Any] = [org_id or self.org_id]
        if subject_ref is not None:
            sql += " AND subject_ref = ?"
            params.append(subject_ref)
        sql += " ORDER BY created_at ASC, event_id ASC"
        rows = self._fetchall(sql, params)
        return [_row_to_audit_event(row) for row in rows]

    def begin_purge_operation(
        self,
        purge_id: str,
        idempotency_key: str,
        operation_type: Literal["user_erasure", "org_purge"],
        scope_type: Literal["user", "org"],
        subject_ref: str | None,
        request_ref: str,
    ) -> PurgeOperation:
        now = _epoch_now()
        with self._lock:
            existing = self.conn.execute(
                """SELECT * FROM purge_operations
                   WHERE org_id = ? AND idempotency_key = ?""",
                (self.org_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                return _row_to_purge_operation(existing)
            self.conn.execute(
                """INSERT INTO purge_operations (
                       purge_id, org_id, operation_type, scope_type, subject_ref,
                       request_ref, idempotency_key, status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    purge_id,
                    self.org_id,
                    operation_type,
                    scope_type,
                    subject_ref,
                    request_ref,
                    idempotency_key,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return self.get_purge_operation(purge_id)

    def record_purge_target(
        self,
        purge_id: str,
        target_name: str,
        phase: str,
        status: Literal["pending", "running", "failed", "complete"],
        target_ref: str = "",
        detail: dict[str, object] | None = None,
        deleted_count: int = 0,
        error_detail: str | None = None,
    ) -> None:
        with self._lock:
            self._record_purge_target_locked(
                purge_id=purge_id,
                target_name=target_name,
                target_ref=target_ref,
                phase=phase,
                status=status,
                detail=detail,
                deleted_count=deleted_count,
                error_detail=error_detail,
            )
            self.conn.commit()

    def list_purge_targets(
        self, purge_id: str, phase: str | None = None
    ) -> list[PurgeOperationTarget]:
        sql = "SELECT * FROM purge_operation_targets WHERE purge_id = ?"
        params: list[Any] = [purge_id]
        if phase is not None:
            sql += " AND phase = ?"
            params.append(phase)
        sql += " ORDER BY phase ASC, target_name ASC, target_ref ASC"
        rows = self._fetchall(sql, params)
        return [_row_to_purge_target(row) for row in rows]

    def purge_targets_prepared(self, purge_id: str) -> bool:
        row = self._fetchone(
            """SELECT 1 FROM purge_operation_targets
               WHERE purge_id = ? AND target_name = ? AND target_ref = 'all'
                 AND phase = ? AND status = 'complete'""",
            (purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
        )
        return row is not None

    def prepare_governance_erase_targets(
        self, purge_id: str, user_id: str, owned_user_playbook_ids: set[int]
    ) -> None:
        request_count = self._fetchone(
            "SELECT COUNT(*) AS cnt FROM requests WHERE user_id = ?",
            (user_id,),
        )["cnt"]
        interaction_count = self._fetchone(
            "SELECT COUNT(*) AS cnt FROM interactions WHERE user_id = ?",
            (user_id,),
        )["cnt"]
        profile_count = self._fetchone(
            "SELECT COUNT(*) AS cnt FROM profiles WHERE user_id = ?",
            (user_id,),
        )["cnt"]
        playbook_count = len(owned_user_playbook_ids)
        affected_agent_playbook_ids: list[int] = []
        if owned_user_playbook_ids:
            placeholders = ",".join("?" for _ in owned_user_playbook_ids)
            rows = self._fetchall(
                f"""SELECT DISTINCT agent_playbook_id
                    FROM agent_playbook_source_user_playbooks
                    WHERE user_playbook_id IN ({placeholders})
                    ORDER BY agent_playbook_id ASC""",
                sorted(owned_user_playbook_ids),
            )
            affected_agent_playbook_ids = [int(row["agent_playbook_id"]) for row in rows]
        targets = {
            "request": request_count,
            "interaction": interaction_count,
            "profile": profile_count,
            "user_playbook": playbook_count,
        }
        with self._lock:
            for target_name, count in targets.items():
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name=target_name,
                    target_ref="all",
                    phase="delete",
                    status="pending",
                    detail={"count": int(count)},
                    deleted_count=0,
                    error_detail=None,
                )
            for agent_playbook_id in affected_agent_playbook_ids:
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name="agent_playbook",
                    target_ref=str(agent_playbook_id),
                    phase="rebuild",
                    status="pending",
                    detail=None,
                    deleted_count=0,
                    error_detail=None,
                )
            self._record_purge_target_locked(
                purge_id=purge_id,
                target_name=_SNAPSHOT_TARGET_NAME,
                target_ref="all",
                phase=_PREPARE_PHASE,
                status="complete",
                detail={
                    "user_id": user_id,
                    "owned_user_playbook_ids": sorted(owned_user_playbook_ids),
                    "affected_agent_playbook_ids": affected_agent_playbook_ids,
                },
                deleted_count=0,
                error_detail=None,
            )
            self.conn.commit()

    def hide_governance_agent_playbooks_for_rebuild(self, purge_id: str) -> list[int]:
        targets = self.list_purge_targets(purge_id, phase="rebuild")
        agent_playbook_ids = [
            int(target.target_ref)
            for target in targets
            if target.target_name == "agent_playbook" and target.target_ref
        ]
        if not agent_playbook_ids:
            return []
        placeholders = ",".join("?" for _ in agent_playbook_ids)
        with self._lock:
            self.conn.execute(
                f"""UPDATE agent_playbooks
                    SET status = 'archived'
                    WHERE agent_playbook_id IN ({placeholders})
                      AND status IS NULL""",
                agent_playbook_ids,
            )
            for agent_playbook_id in agent_playbook_ids:
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name="agent_playbook",
                    target_ref=str(agent_playbook_id),
                    phase="rebuild",
                    status="running",
                    detail=None,
                    deleted_count=0,
                    error_detail=None,
                )
            self.conn.commit()
        return agent_playbook_ids

    def apply_governance_user_data_delete(
        self, purge_id: str, user_id: str
    ) -> dict[str, int]:
        counts = self.clear_user_data(user_id)
        name_map = {
            "interactions": "interaction",
            "user_playbooks": "user_playbook",
            "profiles": "profile",
            "requests": "request",
            "purged_profiles": "profile_purge",
            "purged_user_playbooks": "user_playbook_purge",
        }
        with self._lock:
            for key, value in counts.items():
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name=name_map.get(key, key),
                    target_ref="all",
                    phase="delete",
                    status="complete",
                    detail={"user_id": user_id},
                    deleted_count=int(value),
                    error_detail=None,
                )
            self.conn.commit()
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
        windows = [
            AgentPlaybookSourceWindow(
                user_playbook_id=int(window["user_playbook_id"]),
                source_interaction_ids=[
                    int(source_id)
                    for source_id in (window.get("source_interaction_ids") or [])
                ],
            )
            for window in remaining_source_windows
        ]
        with self._lock:
            cur = self.conn.execute(
                """UPDATE agent_playbooks
                   SET content = ?, trigger = ?, rationale = ?, blocking_issue = ?,
                       expanded_terms = ?, tags = ?, status = NULL
                   WHERE agent_playbook_id = ?""",
                (
                    content or "",
                    trigger,
                    rationale,
                    json.dumps(blocking_issue) if blocking_issue is not None else None,
                    expanded_terms,
                    _json_dumps(tags),
                    agent_playbook_id,
                ),
            )
            if cur.rowcount == 0:
                raise ValueError(
                    f"Agent playbook with ID {agent_playbook_id} not found"
                )
            self.conn.commit()
        self.set_source_windows_for_agent_playbook(agent_playbook_id, windows)
        self.record_purge_target(
            purge_id=purge_id,
            target_name="agent_playbook",
            target_ref=str(agent_playbook_id),
            phase="rebuild",
            status="complete",
        )

    def complete_purge_operation_with_audit(
        self, purge_id: str, audit_event: AuditEvent
    ) -> PurgeOperation:
        now = _epoch_now()
        with self._lock:
            try:
                row = self.conn.execute(
                    "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
                    (purge_id, self.org_id),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Purge operation {purge_id!r} not found")
                snapshot = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE purge_id = ? AND target_name = ? AND target_ref = 'all'
                         AND phase = ? AND status = 'complete'""",
                    (purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
                ).fetchone()
                if snapshot is None:
                    raise ValueError(
                        "Cannot complete purge without target snapshot marker"
                    )
                incomplete = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE purge_id = ? AND status != 'complete'
                       LIMIT 1""",
                    (purge_id,),
                ).fetchone()
                if incomplete is not None:
                    raise ValueError("Cannot complete purge with incomplete targets")
                self._append_audit_event_with_cursor(self.conn, audit_event)
                self.conn.execute(
                    """UPDATE purge_operations
                       SET status = 'complete',
                           error_code = NULL,
                           error_detail = NULL,
                           updated_at = ?,
                           completed_at = COALESCE(completed_at, ?)
                       WHERE purge_id = ? AND org_id = ?""",
                    (now, now, purge_id, self.org_id),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return self.get_purge_operation(purge_id)

    def fail_purge_operation(
        self, purge_id: str, error_code: str, error_detail: str
    ) -> PurgeOperation:
        now = _epoch_now()
        with self._lock:
            cur = self.conn.execute(
                """UPDATE purge_operations
                   SET status = 'failed', error_code = ?, error_detail = ?,
                       updated_at = ?, completed_at = ?
                   WHERE purge_id = ? AND org_id = ?""",
                (error_code, error_detail, now, now, purge_id, self.org_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Purge operation {purge_id!r} not found")
            self.conn.commit()
        return self.get_purge_operation(purge_id)

    def get_purge_operation(self, purge_id: str) -> PurgeOperation:
        row = self._fetchone(
            "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
            (purge_id, self.org_id),
        )
        if row is None:
            raise ValueError(f"Purge operation {purge_id!r} not found")
        return _row_to_purge_operation(row)

    def gc_governance_retention(self, *, _config: Any) -> int:
        return 0
