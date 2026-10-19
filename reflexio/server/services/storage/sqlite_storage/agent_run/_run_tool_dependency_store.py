"""SQLite attachment, listing and consumption of run-tool dependencies."""

from __future__ import annotations

import sqlite3
from typing import Any

from reflexio.server.services.storage.storage_base import (
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
