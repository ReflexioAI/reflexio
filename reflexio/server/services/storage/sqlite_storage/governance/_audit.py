"""SQLite audit-event store methods.

Extracted verbatim from ``_governance.py`` (the AuditEventStore bucket): the
three public methods (``append_audit_event``, ``list_audit_events``,
``gc_governance_retention``) plus the Audit-owned private
``_append_audit_event_with_cursor`` (called cross-bucket by the residual purge /
barrier completion methods, which reach it via MRO co-composition).

The residual ``SQLiteGovernanceMixin`` stays composed alongside this mixin and
permanently holds the shared infra (``conn``, ``_lock``, ``org_id``, ``_deps()``)
and the module-level helpers (``_json_dumps``, ``_row_to_audit_event``), which
are imported here rather than duplicated.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from reflexio.models.api_schema.domain.governance import AuditEvent
from reflexio.models.config_schema import GovernanceRetentionConfig
from reflexio.server.services.storage.governance_validation import (
    _canonicalize_audit_event_for_persistence,
    _epoch_now,
    _is_successful_erase_event,
)

from .._governance import _json_dumps, _row_to_audit_event

if TYPE_CHECKING:
    from .._governance import _SQLiteGovernanceDeps


class AuditEventStoreMixin:
    """SQLite audit-event store primitives."""

    # Type hints for instance attributes/methods provided via MRO by the
    # co-composed residual SQLiteGovernanceMixin / SQLiteStorageBase.
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str
    _deps: Callable[[], _SQLiteGovernanceDeps]

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

    def append_audit_event(self, event: AuditEvent) -> bool:
        if _is_successful_erase_event(event):
            raise ValueError(
                "Successful ERASE audit rows may only be written by "
                "complete_purge_operation_with_audit()"
            )
        if event.org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        event = _canonicalize_audit_event_for_persistence(event)
        with self._lock:
            inserted = self._append_audit_event_with_cursor(self.conn, event)
            self.conn.commit()
            return inserted

    def list_audit_events(
        self, subject_ref: str | None = None, *, org_id: str | None = None
    ) -> list[AuditEvent]:
        deps = self._deps()
        if org_id is not None and org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        sql = "SELECT * FROM audit_events WHERE org_id = ?"
        params: list[Any] = [self.org_id]
        if subject_ref is not None:
            sql += " AND subject_ref = ?"
            params.append(subject_ref)
        sql += " ORDER BY created_at ASC, event_id ASC"
        rows = deps._fetchall(sql, params)
        return [_row_to_audit_event(row) for row in rows]

    def gc_governance_retention(self, *, config: GovernanceRetentionConfig) -> int:
        if not config.audit_events_retention_enabled:
            return 0
        cutoff_epoch = _epoch_now() - config.audit_events_retention_days * 24 * 60 * 60
        with self._lock:
            cur = self.conn.execute(
                """DELETE FROM audit_events
                   WHERE event_id IN (
                       SELECT event_id
                       FROM audit_events
                       WHERE org_id = ? AND created_at < ?
                       ORDER BY created_at ASC, event_id ASC
                       LIMIT ?
                   )""",
                (
                    self.org_id,
                    cutoff_epoch,
                    config.audit_events_delete_batch_limit,
                ),
            )
            deleted = int(cur.rowcount or 0)
            self.conn.commit()
        return deleted
