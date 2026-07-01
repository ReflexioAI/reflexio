"""SQLite PendingToolCallStore methods (the PendingToolCallStore bucket).

Extracted verbatim from ``_agent_run.py`` (the PendingToolCallStore bucket): the
twelve pending-tool-call methods plus the PTC-owned
``_insert_pending_tool_call_unlocked`` private.

Four of these methods are multi-table transactions that cascade agent-run state
by calling AgentRunStore-owned ``_..._unlocked`` privates
(``_finalize_runs_without_pending_dependencies_unlocked`` from ``cancel`` /
``expire``; ``_mark_runs_ready_with_actionable_dependencies_unlocked`` from
``update_resolved`` / ``mark_not_applicable``;
``_finalize_runs_without_actionable_dependencies_unlocked`` from
``mark_not_applicable``). Those privates live in the co-composed
``SQLiteAgentRunStoreMixin`` and are reached via MRO — kept SINGLE there, never
duplicated. Each transaction is moved WHOLE (its ``BEGIN IMMEDIATE`` /
``with self._lock`` block plus all writes and its commit/rollback).

The residual ``SQLiteAgentRunMixin`` stays composed alongside this mixin and
permanently holds the shared row/datetime helpers (``_dt_str``,
``_row_to_pending_tool_call``, ``_record_to_prior_answer_match`` …), which are
imported here rather than duplicated.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    PendingToolCallUpsertResult,
    PriorAnswerMatch,
    RunToolDependencyRecord,
    not_applicable_tool_result,
)

from .._agent_run import (
    _dt_str,
    _record_to_prior_answer_match,
    _row_to_pending_tool_call,
)
from .._base import SQLiteStorageBase, _json_dumps


class SQLitePendingToolCallStoreMixin:
    """SQLite-backed pending-tool-call store primitives."""

    _lock: Any
    conn: sqlite3.Connection
    _fetchone: Any
    _fetchall: Any
    _current_timestamp: Any
    org_id: str

    # Provided via MRO by the co-composed SQLiteAgentRunStoreMixin; reached from
    # the cross-bucket pending-tool-call methods below (cancel / expire /
    # update_resolved / mark_not_applicable).
    _finalize_runs_without_pending_dependencies_unlocked: Callable[..., None]
    _mark_runs_ready_with_actionable_dependencies_unlocked: Callable[..., None]
    _finalize_runs_without_actionable_dependencies_unlocked: Callable[..., None]

    @SQLiteStorageBase.handle_exceptions
    def create_pending_tool_call(
        self, record: PendingToolCallRecord
    ) -> PendingToolCallRecord:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO _pending_tool_calls (
                    id, org_id, user_id, scope, scope_hash, tool_name, dedup_key,
                    status, question_text, answer_format, args, tags, result,
                    embedding, superseded_by, resolved_at, expires_at, cache_until,
                    valid_until
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.id,
                    record.org_id,
                    record.user_id,
                    _json_dumps(record.scope),
                    record.scope_hash,
                    record.tool_name,
                    record.dedup_key,
                    record.status.value,
                    record.question_text,
                    record.answer_format,
                    _json_dumps(record.args),
                    _json_dumps(record.tags),
                    _json_dumps(record.result),
                    _json_dumps(record.embedding),
                    record.superseded_by,
                    _dt_str(record.resolved_at),
                    _dt_str(record.expires_at),
                    _dt_str(record.cache_until),
                    _dt_str(record.valid_until),
                ),
            )
            self.conn.commit()
        stored = self.get_pending_tool_call(record.id)
        if stored is None:  # pragma: no cover
            raise RuntimeError(f"Failed to create pending tool call {record.id}")
        return stored

    def _insert_pending_tool_call_unlocked(self, record: PendingToolCallRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO _pending_tool_calls (
                id, org_id, user_id, scope, scope_hash, tool_name, dedup_key,
                status, question_text, answer_format, args, tags, result,
                embedding, superseded_by, resolved_at, expires_at, cache_until,
                valid_until
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                record.id,
                record.org_id,
                record.user_id,
                _json_dumps(record.scope),
                record.scope_hash,
                record.tool_name,
                record.dedup_key,
                record.status.value,
                record.question_text,
                record.answer_format,
                _json_dumps(record.args),
                _json_dumps(record.tags),
                _json_dumps(record.result),
                _json_dumps(record.embedding),
                record.superseded_by,
                _dt_str(record.resolved_at),
                _dt_str(record.expires_at),
                _dt_str(record.cache_until),
                _dt_str(record.valid_until),
            ),
        )

    @SQLiteStorageBase.handle_exceptions
    def create_or_attach_pending_tool_call(
        self,
        *,
        record: PendingToolCallRecord,
        dependency: RunToolDependencyRecord,
        now: datetime | None = None,
    ) -> PendingToolCallUpsertResult:
        current = now or datetime.now(UTC)
        created = False
        pending_tool_call_id = record.id
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute(
                    """
                    SELECT * FROM _pending_tool_calls
                    WHERE org_id = ?
                      AND scope_hash = ?
                      AND tool_name = ?
                      AND dedup_key = ?
                      AND status = ?
                      AND cache_until > ?
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    (
                        record.org_id,
                        record.scope_hash,
                        record.tool_name,
                        record.dedup_key,
                        PendingToolCallStatus.PENDING.value,
                        _dt_str(current),
                    ),
                ).fetchone()
                pending_tool_call_id = row["id"] if row is not None else record.id
                if row is None:
                    self._insert_pending_tool_call_unlocked(record)
                    created = True
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO _run_tool_dependencies (
                        run_id, pending_tool_call_id, dependency_kind,
                        resolved_at, consumed_at
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        dependency.run_id,
                        pending_tool_call_id,
                        dependency.dependency_kind.value,
                        _dt_str(dependency.resolved_at),
                        _dt_str(dependency.consumed_at),
                    ),
                )
            except Exception:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()

        stored = self.get_pending_tool_call(pending_tool_call_id)
        if stored is None:  # pragma: no cover
            raise RuntimeError("Failed to create or attach pending tool call")
        return PendingToolCallUpsertResult(pending_tool_call=stored, created=created)

    @SQLiteStorageBase.handle_exceptions
    def get_pending_tool_call(self, call_id: str) -> PendingToolCallRecord | None:
        row = self._fetchone(
            "SELECT * FROM _pending_tool_calls WHERE id = ?", (call_id,)
        )
        return _row_to_pending_tool_call(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def list_pending_tool_calls(
        self,
        *,
        status: PendingToolCallStatus | None = None,
        limit: int = 100,
    ) -> list[PendingToolCallRecord]:
        bounded_limit = max(1, min(limit, 500))
        params: list[Any] = [self.org_id]
        status_clause = ""
        if status is not None:
            status_clause = "AND status = ?"
            params.append(status.value)
        params.append(bounded_limit)
        rows = self._fetchall(
            f"""
            SELECT * FROM _pending_tool_calls
            WHERE org_id = ?
              {status_clause}
            ORDER BY created_at DESC, id ASC
            LIMIT ?
            """,
            tuple(params),
        )
        return [_row_to_pending_tool_call(row) for row in rows]

    @SQLiteStorageBase.handle_exceptions
    def cancel_pending_tool_call(
        self,
        call_id: str,
        *,
        cancelled_at: datetime | None = None,
    ) -> PendingToolCallRecord | None:
        now = cancelled_at or datetime.now(UTC)
        now_s = _dt_str(now)
        with self._lock:
            self.conn.execute(
                """
                UPDATE _pending_tool_calls
                SET status = ?
                WHERE id = ?
                  AND status = ?
                """,
                (
                    PendingToolCallStatus.CANCELLED.value,
                    call_id,
                    PendingToolCallStatus.PENDING.value,
                ),
            )
            self.conn.execute(
                """
                UPDATE _run_tool_dependencies
                SET resolved_at = ?
                WHERE pending_tool_call_id = ? AND resolved_at IS NULL
                """,
                (now_s, call_id),
            )
            self._finalize_runs_without_pending_dependencies_unlocked(now_s or "")
            self.conn.commit()
        return self.get_pending_tool_call(call_id)

    @SQLiteStorageBase.handle_exceptions
    def expire_pending_tool_calls(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> int:
        current = now or datetime.now(UTC)
        now_s = _dt_str(current)
        bounded_limit = max(1, min(limit, 500))
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self.conn.execute(
                    """
                    SELECT id
                    FROM _pending_tool_calls
                    WHERE status = ?
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    ORDER BY expires_at ASC, created_at ASC, id ASC
                    LIMIT ?
                    """,
                    (
                        PendingToolCallStatus.PENDING.value,
                        now_s,
                        bounded_limit,
                    ),
                ).fetchall()
                call_ids = [row["id"] for row in rows]
                if not call_ids:
                    self.conn.commit()
                    return 0

                placeholders = ",".join("?" for _ in call_ids)
                self.conn.execute(
                    f"""
                    UPDATE _pending_tool_calls
                    SET status = ?
                    WHERE id IN ({placeholders})
                      AND status = ?
                    """,
                    (
                        PendingToolCallStatus.EXPIRED.value,
                        *call_ids,
                        PendingToolCallStatus.PENDING.value,
                    ),
                )
                self.conn.execute(
                    f"""
                    UPDATE _run_tool_dependencies
                    SET resolved_at = ?
                    WHERE pending_tool_call_id IN ({placeholders})
                      AND resolved_at IS NULL
                    """,
                    (now_s, *call_ids),
                )
                self._finalize_runs_without_pending_dependencies_unlocked(now_s or "")
            except Exception:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()
        return len(call_ids)

    @SQLiteStorageBase.handle_exceptions
    def delete_expired_pending_tool_calls(
        self, *, now: int, grace_seconds: int, limit: int = 1000
    ) -> int:
        """Delete terminal 'expired'-status rows whose expires_at is past the grace window.

        Only rows with status='expired' are deleted. RESOLVED rows are never
        touched even if their expires_at is in the past, preserving live cached
        results for resumable extraction.

        Args:
            now: Current Unix epoch seconds.
            grace_seconds: Grace buffer; cutoff = now - grace_seconds.
            limit: Max rows to delete per call.

        Returns:
            Number of rows deleted.
        """
        cutoff_iso = datetime.fromtimestamp(now - grace_seconds, UTC).isoformat()
        bounded_limit = max(1, min(limit, 10_000))
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self.conn.execute(
                    "SELECT id FROM _pending_tool_calls "
                    "WHERE org_id = ? AND status = 'expired' AND expires_at IS NOT NULL AND expires_at < ? "
                    "ORDER BY expires_at ASC LIMIT ?",
                    (self.org_id, cutoff_iso, bounded_limit),
                ).fetchall()
                if not rows:
                    self.conn.commit()
                    return 0
                ids = [r["id"] for r in rows]
                ph = ",".join("?" * len(ids))
                cur = self.conn.execute(
                    f"DELETE FROM _pending_tool_calls WHERE id IN ({ph})", ids
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def find_active_pending_tool_call(
        self,
        *,
        org_id: str,
        scope_hash: str,
        tool_name: str,
        dedup_key: str,
        now: datetime | None = None,
    ) -> PendingToolCallRecord | None:
        now_s = _dt_str(now or datetime.now(UTC))
        row = self._fetchone(
            """
            SELECT * FROM _pending_tool_calls
            WHERE org_id = ?
              AND scope_hash = ?
              AND tool_name = ?
              AND dedup_key = ?
              AND status = ?
              AND cache_until > ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (
                org_id,
                scope_hash,
                tool_name,
                dedup_key,
                PendingToolCallStatus.PENDING.value,
                now_s,
            ),
        )
        return _row_to_pending_tool_call(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def search_prior_tool_calls(
        self,
        *,
        org_id: str,
        scope_hash: str,
        tool_name: str,
        query_embedding: list[float] | None = None,
        now: datetime | None = None,
        limit: int = 8,
    ) -> list[PriorAnswerMatch]:
        current = now or datetime.now(UTC)
        bounded_limit = max(1, min(limit, 50))
        rows = self._fetchall(
            """
            SELECT * FROM _pending_tool_calls
            WHERE org_id = ?
              AND scope_hash = ?
              AND tool_name = ?
              AND (
                (
                  status = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                )
                OR (
                  status = ?
                  AND (valid_until IS NULL OR valid_until > ?)
                )
              )
            ORDER BY
              CASE status WHEN ? THEN 0 ELSE 1 END,
              COALESCE(resolved_at, created_at) DESC,
              id ASC
            """,
            (
                org_id,
                scope_hash,
                tool_name,
                PendingToolCallStatus.PENDING.value,
                _dt_str(current),
                PendingToolCallStatus.RESOLVED.value,
                _dt_str(current),
                PendingToolCallStatus.RESOLVED.value,
            ),
        )
        seen_resolved_dedup_keys: set[str] = set()
        records: list[PendingToolCallRecord] = []
        for row in rows:
            record = _row_to_pending_tool_call(row)
            if record.status == PendingToolCallStatus.RESOLVED:
                if record.dedup_key in seen_resolved_dedup_keys:
                    continue
                seen_resolved_dedup_keys.add(record.dedup_key)
            records.append(record)
        matches = [
            _record_to_prior_answer_match(record, query_embedding=query_embedding)
            for record in records
        ]
        if query_embedding:
            matches.sort(
                key=lambda match: (
                    match.similarity is not None,
                    match.similarity or -1.0,
                    match.resolved_at
                    or match.created_at
                    or datetime.min.replace(tzinfo=UTC),
                ),
                reverse=True,
            )
        return matches[:bounded_limit]

    @SQLiteStorageBase.handle_exceptions
    def resolve_pending_tool_call(
        self,
        call_id: str,
        *,
        result: dict[str, Any],
        resolved_at: datetime | None = None,
        valid_for_seconds: int,
    ) -> PendingToolCallRecord | None:
        resolved = resolved_at or datetime.now(UTC)
        valid_until = resolved + timedelta(seconds=valid_for_seconds)
        with self._lock:
            cur = self.conn.execute(
                """
                UPDATE _pending_tool_calls
                SET status = ?, result = ?, resolved_at = ?, valid_until = ?
                WHERE id = ?
                  AND status = ?
                """,
                (
                    PendingToolCallStatus.RESOLVED.value,
                    _json_dumps(result),
                    _dt_str(resolved),
                    _dt_str(valid_until),
                    call_id,
                    PendingToolCallStatus.PENDING.value,
                ),
            )
            if cur.rowcount == 0:
                self.conn.commit()
                return self.get_pending_tool_call(call_id)
            self.conn.execute(
                """
                UPDATE _pending_tool_calls
                SET status = ?,
                    superseded_by = ?
                WHERE id != ?
                  AND status = ?
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND (org_id, scope_hash, tool_name, dedup_key) = (
                    SELECT org_id, scope_hash, tool_name, dedup_key
                    FROM _pending_tool_calls
                    WHERE id = ?
                  )
                """,
                (
                    PendingToolCallStatus.SUPERSEDED.value,
                    call_id,
                    call_id,
                    PendingToolCallStatus.RESOLVED.value,
                    _dt_str(resolved),
                    call_id,
                ),
            )
            self.conn.execute(
                """
                UPDATE _run_tool_dependencies
                SET resolved_at = ?
                WHERE pending_tool_call_id = ? AND resolved_at IS NULL
                """,
                (_dt_str(resolved), call_id),
            )
            self.conn.execute(
                """
                UPDATE _agent_runs
                SET status = ?, updated_at = ?
                WHERE status IN (?, ?)
                  AND EXISTS (
                    SELECT 1
                    FROM _run_tool_dependencies d
                    WHERE d.run_id = _agent_runs.id
                      AND d.pending_tool_call_id = ?
                      AND d.resolved_at IS NOT NULL
                      AND d.consumed_at IS NULL
                  )
                """,
                (
                    AgentRunStatus.RESUME_READY.value,
                    self._current_timestamp(),
                    AgentRunStatus.FINALIZED.value,
                    AgentRunStatus.FINALIZED_PENDING_TOOL.value,
                    call_id,
                ),
            )
            self.conn.commit()
        return self.get_pending_tool_call(call_id)

    @SQLiteStorageBase.handle_exceptions
    def update_resolved_pending_tool_call_result(
        self,
        call_id: str,
        *,
        result: dict[str, Any],
        resolved_at: datetime | None = None,
        valid_for_seconds: int,
    ) -> PendingToolCallRecord | None:
        resolved = resolved_at or datetime.now(UTC)
        valid_until = resolved + timedelta(seconds=valid_for_seconds)
        now_s = _dt_str(resolved) or self._current_timestamp()
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self.conn.execute(
                    """
                    UPDATE _pending_tool_calls
                    SET result = ?, resolved_at = ?, valid_until = ?
                    WHERE id = ?
                      AND status = ?
                    """,
                    (
                        _json_dumps(result),
                        _dt_str(resolved),
                        _dt_str(valid_until),
                        call_id,
                        PendingToolCallStatus.RESOLVED.value,
                    ),
                )
                if cur.rowcount == 0:
                    self.conn.commit()
                    return self.get_pending_tool_call(call_id)
                self.conn.execute(
                    """
                    UPDATE _pending_tool_calls
                    SET status = ?,
                        superseded_by = ?
                    WHERE id != ?
                      AND status = ?
                      AND (valid_until IS NULL OR valid_until > ?)
                      AND (org_id, scope_hash, tool_name, dedup_key) = (
                        SELECT org_id, scope_hash, tool_name, dedup_key
                        FROM _pending_tool_calls
                        WHERE id = ?
                      )
                    """,
                    (
                        PendingToolCallStatus.SUPERSEDED.value,
                        call_id,
                        call_id,
                        PendingToolCallStatus.RESOLVED.value,
                        _dt_str(resolved),
                        call_id,
                    ),
                )
                self.conn.execute(
                    """
                    UPDATE _run_tool_dependencies
                    SET resolved_at = ?,
                        consumed_at = NULL
                    WHERE pending_tool_call_id = ?
                    """,
                    (_dt_str(resolved), call_id),
                )
                self._mark_runs_ready_with_actionable_dependencies_unlocked(
                    now_s, pending_tool_call_id=call_id
                )
            except Exception:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()
        return self.get_pending_tool_call(call_id)

    @SQLiteStorageBase.handle_exceptions
    def mark_pending_tool_call_not_applicable(
        self,
        call_id: str,
        *,
        resolved_at: datetime | None = None,
        valid_for_seconds: int,
    ) -> PendingToolCallRecord | None:
        resolved = resolved_at or datetime.now(UTC)
        valid_until = resolved + timedelta(seconds=valid_for_seconds)
        now_s = _dt_str(resolved) or self._current_timestamp()
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self.conn.execute(
                    """
                    UPDATE _pending_tool_calls
                    SET status = ?, result = ?, resolved_at = ?, valid_until = ?
                    WHERE id = ?
                      AND status IN (?, ?)
                    """,
                    (
                        PendingToolCallStatus.RESOLVED.value,
                        _json_dumps(not_applicable_tool_result()),
                        _dt_str(resolved),
                        _dt_str(valid_until),
                        call_id,
                        PendingToolCallStatus.PENDING.value,
                        PendingToolCallStatus.RESOLVED.value,
                    ),
                )
                if cur.rowcount == 0:
                    self.conn.commit()
                    return self.get_pending_tool_call(call_id)
                self.conn.execute(
                    """
                    UPDATE _run_tool_dependencies
                    SET resolved_at = COALESCE(resolved_at, ?),
                        consumed_at = ?
                    WHERE pending_tool_call_id = ?
                      AND consumed_at IS NULL
                    """,
                    (_dt_str(resolved), _dt_str(resolved), call_id),
                )
                self._mark_runs_ready_with_actionable_dependencies_unlocked(
                    now_s, pending_tool_call_id=call_id
                )
                self._finalize_runs_without_actionable_dependencies_unlocked(
                    now_s, pending_tool_call_id=call_id
                )
            except Exception:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()
        return self.get_pending_tool_call(call_id)
