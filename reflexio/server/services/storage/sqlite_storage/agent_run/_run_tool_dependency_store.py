"""SQLite RunToolDependencyStore methods (the RunToolDependencyStore bucket).

Extracted verbatim from ``_agent_run.py`` (the RunToolDependencyStore bucket):
the four run-tool-dependency methods. This is the LAST bucket — the residual
``_agent_run.py`` module is now helpers-only (it no longer defines a mixin
class), and permanently holds the shared row/datetime helpers (``_dt_str``,
``_row_to_run_tool_dependency`` …), which are imported here rather than
duplicated.

``consume_run_tool_dependencies`` keeps its resolved-unconsumed ``WHERE`` guard
and ``rowcount`` return; ``count_unresolved_followup_dependencies`` keeps its
three-table JOIN. Both are moved WHOLE, byte-identically.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from reflexio.server.services.storage.storage_base import (
    PendingToolCallStatus,
    RunToolDependencyRecord,
)

from .._agent_run import _dt_str, _row_to_run_tool_dependency
from .._base import SQLiteStorageBase


class SQLiteRunToolDependencyStoreMixin:
    """SQLite-backed run-tool-dependency store primitives."""

    _lock: Any
    conn: sqlite3.Connection
    _fetchone: Any
    _fetchall: Any
    _current_timestamp: Any

    @SQLiteStorageBase.handle_exceptions
    def attach_run_tool_dependency(
        self, record: RunToolDependencyRecord
    ) -> RunToolDependencyRecord:
        with self._lock:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO _run_tool_dependencies (
                    run_id, pending_tool_call_id, dependency_kind, resolved_at,
                    consumed_at
                ) VALUES (?,?,?,?,?)
                """,
                (
                    record.run_id,
                    record.pending_tool_call_id,
                    record.dependency_kind.value,
                    _dt_str(record.resolved_at),
                    _dt_str(record.consumed_at),
                ),
            )
            self.conn.commit()
        row = self._fetchone(
            """
            SELECT * FROM _run_tool_dependencies
            WHERE run_id = ? AND pending_tool_call_id = ?
            """,
            (record.run_id, record.pending_tool_call_id),
        )
        if row is None:  # pragma: no cover
            raise RuntimeError("Failed to attach run tool dependency")
        return _row_to_run_tool_dependency(row)

    @SQLiteStorageBase.handle_exceptions
    def count_unresolved_followup_dependencies(
        self,
        *,
        org_id: str,
        extractor_kind: str,
        tool_name: str,
    ) -> int:
        row = self._fetchone(
            """
            SELECT COUNT(*) AS count
            FROM _run_tool_dependencies d
            JOIN _agent_runs r ON r.id = d.run_id
            JOIN _pending_tool_calls p ON p.id = d.pending_tool_call_id
            WHERE r.org_id = ?
              AND r.extractor_kind = ?
              AND p.tool_name = ?
              AND p.status = ?
              AND d.resolved_at IS NULL
              AND d.consumed_at IS NULL
            """,
            (
                org_id,
                extractor_kind,
                tool_name,
                PendingToolCallStatus.PENDING.value,
            ),
        )
        return int(row["count"]) if row is not None else 0

    @SQLiteStorageBase.handle_exceptions
    def list_run_tool_dependencies(self, run_id: str) -> list[RunToolDependencyRecord]:
        rows = self._fetchall(
            "SELECT * FROM _run_tool_dependencies WHERE run_id = ?",
            (run_id,),
        )
        return [_row_to_run_tool_dependency(row) for row in rows]

    @SQLiteStorageBase.handle_exceptions
    def consume_run_tool_dependencies(self, run_id: str) -> int:
        consumed_at = self._current_timestamp()
        with self._lock:
            cur = self.conn.execute(
                """
                UPDATE _run_tool_dependencies
                SET consumed_at = ?
                WHERE run_id = ?
                  AND resolved_at IS NOT NULL
                  AND consumed_at IS NULL
                """,
                (consumed_at, run_id),
            )
            self.conn.commit()
        return int(cur.rowcount)
