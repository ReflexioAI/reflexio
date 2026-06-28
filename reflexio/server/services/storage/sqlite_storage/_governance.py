from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast

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
_ALLOWED_DETAIL_KEYS = frozenset(
    {
        "affected_agent_playbook_ids",
        "agent_playbook_id",
        "count",
        "deleted_count",
        "erased_source_ids",
        "owned_user_playbook_ids",
        "original_source_windows",
        "prepared",
        "rebuilt_agent_playbook_ids",
        "remaining_source_windows",
        "route",
        "source_interaction_ids",
        "status",
        "user_playbook_id",
    }
)
_DISALLOWED_DETAIL_KEYS = frozenset(
    {
        "content",
        "email",
        "prompt",
        "request_id",
        "request_ref",
        "user_id",
    }
)
_EMAIL_RE = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")
_REQUEST_ID_RE = re.compile(
    r"\b(?:reqref_(?!v1_)|request[_-]|req[_-])[A-Za-z0-9_-]*\b",
    re.IGNORECASE,
)
_TOKEN_NAME_RE = re.compile(
    r"\b(?:api[-_ ]?token|token[-_ ]?name|bearer|secret[-_ ]?key)\b",
    re.IGNORECASE,
)
_RAW_EXCEPTION_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\s*:")
_SAFE_INTERNAL_ID_RE = re.compile(r"^[0-9]+$")
_USER_LIKE_TARGET_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_SAFE_ERROR_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_IDENTIFIERISH_ERROR_CODE_RE = re.compile(
    r"^(?:user|subject|request|req|actor|email)[-_.:]?[A-Za-z0-9_.:-]+$",
    re.IGNORECASE,
)


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


def _raise_governance_validation_error(field_name: str, reason: str) -> None:
    raise ValueError(f"Unsafe governance {field_name}: {reason}")


def _validate_governance_string(field_name: str, value: str) -> None:
    if _EMAIL_RE.search(value):
        _raise_governance_validation_error(field_name, "email")
    if _REQUEST_ID_RE.search(value):
        _raise_governance_validation_error(field_name, "request_id")
    lowered = value.lower()
    if "prompt" in lowered or "content" in lowered:
        _raise_governance_validation_error(field_name, "prompt/content")
    if _TOKEN_NAME_RE.search(value):
        _raise_governance_validation_error(field_name, "token")
    if _RAW_EXCEPTION_RE.search(value):
        _raise_governance_validation_error(field_name, "raw exception text")


def _validate_governance_prefixed_ref(
    field_name: str, value: str | None, *, prefix: str
) -> None:
    if value is None:
        return
    if not value.startswith(prefix):
        _raise_governance_validation_error(field_name, f"must start with {prefix}")


def _validate_governance_int(field_name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise_governance_validation_error(field_name, "expected int")


def _validate_governance_int_list(field_name: str, value: Any) -> None:
    if not isinstance(value, list):
        _raise_governance_validation_error(field_name, "expected list[int]")
    for item in value:
        _validate_governance_int(field_name, item)


def _normalize_governance_window_item(
    field_name: str, index: int, item: object
) -> dict[str, Any]:
    if not isinstance(item, dict):
        _raise_governance_validation_error(
            f"{field_name}[{index}]", "expected window dict"
        )
    window_item = cast(dict[Any, Any], item)
    normalized_item: dict[str, Any] = {}
    for raw_key, raw_value in window_item.items():
        normalized_key = str(raw_key).strip().lower()
        if normalized_key in normalized_item:
            _raise_governance_validation_error(
                f"{field_name}[{index}]", f"duplicate key {normalized_key}"
            )
        normalized_item[normalized_key] = raw_value
    return normalized_item


def _validate_governance_window_list(field_name: str, value: Any) -> None:
    if not isinstance(value, list):
        _raise_governance_validation_error(field_name, "expected list[window]")
    for index, item in enumerate(value):
        normalized_item = _normalize_governance_window_item(field_name, index, item)
        normalized_keys = set(normalized_item)
        unexpected_keys = normalized_keys - {"user_playbook_id", "source_interaction_ids"}
        if unexpected_keys:
            _raise_governance_validation_error(
                f"{field_name}[{index}]", sorted(unexpected_keys)[0]
            )
        if "user_playbook_id" in normalized_item:
            _validate_governance_int(
                f"{field_name}[{index}].user_playbook_id",
                normalized_item["user_playbook_id"],
            )
        if "source_interaction_ids" in normalized_item:
            _validate_governance_int_list(
                f"{field_name}[{index}].source_interaction_ids",
                normalized_item["source_interaction_ids"],
            )


def _parse_governance_window_list(
    field_name: str, value: list[dict[str, object]]
) -> list[AgentPlaybookSourceWindow]:
    _validate_governance_window_list(field_name, value)
    windows: list[AgentPlaybookSourceWindow] = []
    for index, item in enumerate(value):
        normalized_item = _normalize_governance_window_item(field_name, index, item)
        user_playbook_id = int(normalized_item["user_playbook_id"])
        source_ids = normalized_item.get("source_interaction_ids") or []
        windows.append(
            AgentPlaybookSourceWindow(
                user_playbook_id=user_playbook_id,
                source_interaction_ids=[int(source_id) for source_id in source_ids],
            )
        )
    return windows


def _validate_governance_target_ref(target_ref: str) -> None:
    if target_ref in {"", "all"}:
        return
    if _SAFE_INTERNAL_ID_RE.fullmatch(target_ref):
        return
    if target_ref.startswith(("reqref_v1_", "subref_v1_", "actref_v1_")):
        return
    _validate_governance_string("target_ref", target_ref)
    if _USER_LIKE_TARGET_REF_RE.fullmatch(target_ref):
        _raise_governance_validation_error("target_ref", "user-like identifier")
    _raise_governance_validation_error("target_ref", "must be minimized or internal")


def _validate_governance_detail_entry(field_name: str, key: str, value: Any) -> None:
    if key in _DISALLOWED_DETAIL_KEYS:
        _raise_governance_validation_error(field_name, key)
    if key not in _ALLOWED_DETAIL_KEYS:
        _raise_governance_validation_error(field_name, key)
    if key in {"count", "deleted_count", "agent_playbook_id", "user_playbook_id"}:
        _validate_governance_int(field_name, value)
        return
    if key in {
        "affected_agent_playbook_ids",
        "erased_source_ids",
        "owned_user_playbook_ids",
        "rebuilt_agent_playbook_ids",
        "source_interaction_ids",
    }:
        _validate_governance_int_list(field_name, value)
        return
    if key in {"original_source_windows", "remaining_source_windows"}:
        _validate_governance_window_list(field_name, value)
        return
    if key == "prepared":
        if not isinstance(value, bool):
            _raise_governance_validation_error(field_name, "expected bool")
        return
    if key in {"route", "status"}:
        if not isinstance(value, str):
            _raise_governance_validation_error(field_name, "expected str")
        _validate_governance_string(field_name, value)
        return
    _raise_governance_validation_error(field_name, key)


def _validate_governance_detail(
    field_name: str, detail: dict[str, object] | None
) -> dict[str, object] | None:
    if detail is None:
        return None
    if not isinstance(detail, dict):
        _raise_governance_validation_error(field_name, "expected dict")
    for key, value in detail.items():
        normalized_key = str(key).strip().lower()
        _validate_governance_detail_entry(f"{field_name}.{normalized_key}", normalized_key, value)
    return detail


def _validate_governance_error_detail(error_detail: str | None) -> str | None:
    if error_detail is None:
        return None
    _validate_governance_string("error_detail", error_detail)
    return error_detail


def _validate_governance_error_code(error_code: str) -> str:
    if not error_code:
        _raise_governance_validation_error("error_code", "required")
    _validate_governance_string("error_code", error_code)
    if error_code.startswith(("subref_v1_", "reqref_v1_", "actref_v1_")):
        _raise_governance_validation_error("error_code", "identifier")
    if _IDENTIFIERISH_ERROR_CODE_RE.fullmatch(error_code):
        _raise_governance_validation_error("error_code", "identifier")
    if not _SAFE_ERROR_CODE_RE.fullmatch(error_code):
        _raise_governance_validation_error(
            "error_code", "must be a stable diagnostic code"
        )
    return error_code


def _validate_audit_event_for_persistence(event: AuditEvent) -> None:
    _validate_governance_prefixed_ref("actor_ref", event.actor_ref, prefix="actref_v1_")
    _validate_governance_prefixed_ref(
        "subject_ref", event.subject_ref, prefix="subref_v1_"
    )
    if event.request_ref is None:
        _raise_governance_validation_error("request_ref", "required")
    _validate_governance_prefixed_ref(
        "request_ref", event.request_ref, prefix="reqref_v1_"
    )
    if event.entity_id is not None:
        _validate_governance_string("entity_id", event.entity_id)
    _validate_governance_detail("audit_event.detail", event.detail)


def _is_successful_erase_event(event: AuditEvent, *, purge_id: str | None = None) -> bool:
    return (
        event.operation == "ERASE"
        and event.status == "ok"
        and event.idempotency_key is not None
        and (purge_id is None or event.idempotency_key == purge_id)
    )


def _successful_erase_identity(event: AuditEvent) -> tuple[str, str, str, str | None, str | None, str, str | None]:
    return (
        event.org_id,
        event.operation,
        event.entity_type,
        event.subject_ref,
        event.request_ref,
        event.status,
        event.idempotency_key,
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


class _SQLiteGovernanceDeps(Protocol):
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str

    def _fetchall(self, sql: str, params: list[Any] | tuple[Any, ...]) -> list[sqlite3.Row]:
        ...

    def _fetchone(self, sql: str, params: list[Any] | tuple[Any, ...]) -> sqlite3.Row | None:
        ...

    def clear_user_data(self, user_id: str) -> dict[str, int]:
        ...

    def set_source_windows_for_agent_playbook(
        self, agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
    ) -> None:
        ...


class SQLiteGovernanceMixin:
    """SQLite governance storage primitives."""

    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str

    def _deps(self) -> _SQLiteGovernanceDeps:
        return cast(_SQLiteGovernanceDeps, self)

    def _append_audit_event_with_cursor(
        self, cur: sqlite3.Connection | sqlite3.Cursor, event: AuditEvent
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
        detail = _validate_governance_detail("detail", detail)
        error_detail = _validate_governance_error_detail(error_detail)
        _validate_governance_target_ref(target_ref)
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
        if _is_successful_erase_event(event):
            raise ValueError(
                "Successful ERASE audit rows may only be written by "
                "complete_purge_operation_with_audit()"
            )
        _validate_audit_event_for_persistence(event)
        with self._lock:
            inserted = self._append_audit_event_with_cursor(self.conn, event)
            self.conn.commit()
            return inserted

    def list_audit_events(
        self, subject_ref: str | None = None, *, org_id: str | None = None
    ) -> list[AuditEvent]:
        deps = self._deps()
        sql = "SELECT * FROM audit_events WHERE org_id = ?"
        params: list[Any] = [org_id or deps.org_id]
        if subject_ref is not None:
            sql += " AND subject_ref = ?"
            params.append(subject_ref)
        sql += " ORDER BY created_at ASC, event_id ASC"
        rows = deps._fetchall(sql, params)
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
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        _validate_governance_prefixed_ref(
            "request_ref", request_ref, prefix="reqref_v1_"
        )
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
        deps = self._deps()
        sql = "SELECT * FROM purge_operation_targets WHERE purge_id = ?"
        params: list[Any] = [purge_id]
        if phase is not None:
            sql += " AND phase = ?"
            params.append(phase)
        sql += " ORDER BY phase ASC, target_name ASC, target_ref ASC"
        rows = deps._fetchall(sql, params)
        return [_row_to_purge_target(row) for row in rows]

    def purge_targets_prepared(self, purge_id: str) -> bool:
        row = self._deps()._fetchone(
            """SELECT 1 FROM purge_operation_targets
               WHERE purge_id = ? AND target_name = ? AND target_ref = 'all'
                 AND phase = ? AND status = 'complete'""",
            (purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
        )
        return row is not None

    def prepare_governance_erase_targets(
        self, purge_id: str, user_id: str, owned_user_playbook_ids: set[int]
    ) -> None:
        deps = self._deps()
        request_row = deps._fetchone(
            "SELECT COUNT(*) AS cnt FROM requests WHERE user_id = ?",
            (user_id,),
        )
        interaction_row = deps._fetchone(
            "SELECT COUNT(*) AS cnt FROM interactions WHERE user_id = ?",
            (user_id,),
        )
        profile_row = deps._fetchone(
            "SELECT COUNT(*) AS cnt FROM profiles WHERE user_id = ?",
            (user_id,),
        )
        if request_row is None or interaction_row is None or profile_row is None:
            raise ValueError("Missing governance count rows")
        request_count = request_row["cnt"]
        interaction_count = interaction_row["cnt"]
        profile_count = profile_row["cnt"]
        playbook_count = len(owned_user_playbook_ids)
        affected_agent_playbook_ids: list[int] = []
        if owned_user_playbook_ids:
            placeholders = ",".join("?" for _ in owned_user_playbook_ids)
            rows = deps._fetchall(
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
        counts = self._deps().clear_user_data(user_id)
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
                    detail={"count": int(value)},
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
        windows = _parse_governance_window_list(
            "remaining_source_windows", remaining_source_windows
        )
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
        self._deps().set_source_windows_for_agent_playbook(agent_playbook_id, windows)
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
        if audit_event.org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        if audit_event.idempotency_key != purge_id:
            raise ValueError("Audit event idempotency key must match purge_id")
        if not _is_successful_erase_event(audit_event, purge_id=purge_id):
            raise ValueError(
                "Completion requires a successful ERASE audit event for this purge"
            )
        _validate_audit_event_for_persistence(audit_event)
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
                existing_audit_row = self.conn.execute(
                    """SELECT * FROM audit_events
                       WHERE org_id = ? AND idempotency_key = ?""",
                    (self.org_id, purge_id),
                ).fetchone()
                if existing_audit_row is not None:
                    existing_event = _row_to_audit_event(existing_audit_row)
                    if not _is_successful_erase_event(existing_event, purge_id=purge_id):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                    if _successful_erase_identity(existing_event) != _successful_erase_identity(
                        audit_event
                    ):
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
                if _successful_erase_identity(existing_event) != _successful_erase_identity(
                    audit_event
                ):
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
        validated_error_code = _validate_governance_error_code(error_code)
        validated_error_detail = _validate_governance_error_detail(error_detail)
        now = _epoch_now()
        with self._lock:
            cur = self.conn.execute(
                """UPDATE purge_operations
                   SET status = 'failed', error_code = ?, error_detail = ?,
                   updated_at = ?, completed_at = ?
                   WHERE purge_id = ? AND org_id = ?""",
                (
                    validated_error_code,
                    validated_error_detail,
                    now,
                    now,
                    purge_id,
                    self.org_id,
                ),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Purge operation {purge_id!r} not found")
            self.conn.commit()
        return self.get_purge_operation(purge_id)

    def get_purge_operation(self, purge_id: str) -> PurgeOperation:
        row = self._deps()._fetchone(
            "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
            (purge_id, self.org_id),
        )
        if row is None:
            raise ValueError(f"Purge operation {purge_id!r} not found")
        return _row_to_purge_operation(row)

    def gc_governance_retention(self, *, _config: Any) -> int:
        return 0
