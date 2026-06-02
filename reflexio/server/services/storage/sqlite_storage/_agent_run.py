"""SQLite storage for resumable extraction agent runs."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    PendingToolCallUpsertResult,
    PriorAnswerMatch,
    RunToolDependencyKind,
    RunToolDependencyRecord,
    embedding_similarity,
    not_applicable_tool_result,
)

from ._base import SQLiteStorageBase, _json_dumps, _json_loads


def _dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _dt_str(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _row_to_agent_run(row: sqlite3.Row) -> AgentRunRecord:
    data = dict(row)
    binding = AgentBinding(
        org_id=data["org_id"],
        extractor_kind=data["extractor_kind"],
        user_id=data.get("user_id"),
        request_id=data["request_id"],
        agent_version=data.get("agent_version"),
        source=data.get("source"),
        source_interaction_ids=_json_loads(data.get("source_interaction_ids")) or [],
        window_start_interaction_id=data.get("window_start_interaction_id"),
        window_end_interaction_id=data.get("window_end_interaction_id"),
        extractor_config_hash=data.get("extractor_config_hash"),
    )
    return AgentRunRecord(
        id=data["id"],
        binding=binding,
        status=AgentRunStatus(data["status"]),
        generation_request_snapshot=_json_loads(data.get("generation_request_snapshot"))
        or {},
        service_config_snapshot=_json_loads(data.get("service_config_snapshot")),
        agent_context_snapshot=data.get("agent_context_snapshot"),
        committed_output=_json_loads(data.get("committed_output")),
        pending_tool_call_ids=_json_loads(data.get("pending_tool_call_ids")) or [],
        max_steps_remaining=(
            int(data["max_steps_remaining"])
            if data.get("max_steps_remaining") is not None
            else None
        ),
        resume_attempts=int(data.get("resume_attempts") or 0),
        finalization_attempts=int(data.get("finalization_attempts") or 0),
        next_resume_at=_dt(data.get("next_resume_at")),
        claimed_by=data.get("claimed_by"),
        claimed_at=_dt(data.get("claimed_at")),
        agent_completed_at=_dt(data.get("agent_completed_at")),
        finalized_at=_dt(data.get("finalized_at")),
        created_at=_dt(data.get("created_at")),
        updated_at=_dt(data.get("updated_at")),
        expires_at=_dt(data.get("expires_at")),
        last_error=data.get("last_error"),
    )


def _row_to_pending_tool_call(row: sqlite3.Row) -> PendingToolCallRecord:
    data = dict(row)
    return PendingToolCallRecord(
        id=data["id"],
        org_id=data["org_id"],
        scope=_json_loads(data.get("scope")) or {},
        scope_hash=data["scope_hash"],
        tool_name=data["tool_name"],
        dedup_key=data["dedup_key"],
        status=PendingToolCallStatus(data["status"]),
        question_text=data["question_text"],
        args=_json_loads(data.get("args")) or {},
        tags=_json_loads(data.get("tags")) or [],
        user_id=data.get("user_id"),
        answer_format=data.get("answer_format"),
        result=_json_loads(data.get("result")),
        embedding=_json_loads(data.get("embedding")),
        superseded_by=data.get("superseded_by"),
        created_at=_dt(data.get("created_at")),
        resolved_at=_dt(data.get("resolved_at")),
        expires_at=_dt(data.get("expires_at")),
        cache_until=_dt(data.get("cache_until")),
        valid_until=_dt(data.get("valid_until")),
    )


def _record_to_prior_answer_match(
    record: PendingToolCallRecord,
    *,
    query_embedding: list[float] | None = None,
) -> PriorAnswerMatch:
    return PriorAnswerMatch(
        pending_tool_call_id=record.id,
        status=record.status,
        question_text=record.question_text,
        result=record.result,
        valid_until=record.valid_until,
        answer_format=record.answer_format,
        created_at=record.created_at,
        resolved_at=record.resolved_at,
        expires_at=record.expires_at,
        similarity=embedding_similarity(query_embedding, record.embedding),
    )


def _row_to_run_tool_dependency(row: sqlite3.Row) -> RunToolDependencyRecord:
    data = dict(row)
    return RunToolDependencyRecord(
        run_id=data["run_id"],
        pending_tool_call_id=data["pending_tool_call_id"],
        dependency_kind=RunToolDependencyKind(data["dependency_kind"]),
        resolved_at=_dt(data.get("resolved_at")),
        consumed_at=_dt(data.get("consumed_at")),
        created_at=_dt(data.get("created_at")),
    )


class SQLiteAgentRunMixin:
    """SQLite-backed resumable extraction run storage."""

    _lock: Any
    conn: sqlite3.Connection
    _fetchone: Any
    _fetchall: Any
    _current_timestamp: Any
    org_id: str

    def _finalize_runs_without_pending_dependencies_unlocked(self, now_s: str) -> None:
        self.conn.execute(
            """
            UPDATE _agent_runs
            SET status = ?,
                finalized_at = COALESCE(finalized_at, ?),
                updated_at = ?
            WHERE status = ?
              AND NOT EXISTS (
                SELECT 1
                FROM _run_tool_dependencies d
                JOIN _pending_tool_calls p
                  ON p.id = d.pending_tool_call_id
                WHERE d.run_id = _agent_runs.id
                  AND d.resolved_at IS NULL
                  AND d.consumed_at IS NULL
                  AND p.status = ?
              )
            """,
            (
                AgentRunStatus.FINALIZED.value,
                now_s,
                now_s,
                AgentRunStatus.FINALIZED_PENDING_TOOL.value,
                PendingToolCallStatus.PENDING.value,
            ),
        )

    def _mark_runs_ready_with_actionable_dependencies_unlocked(
        self, now_s: str, *, pending_tool_call_id: str
    ) -> None:
        self.conn.execute(
            """
            UPDATE _agent_runs
            SET status = ?,
                updated_at = ?
            WHERE status IN (?, ?)
              AND EXISTS (
                SELECT 1
                FROM _run_tool_dependencies changed
                WHERE changed.run_id = _agent_runs.id
                  AND changed.pending_tool_call_id = ?
              )
              AND EXISTS (
                SELECT 1
                FROM _run_tool_dependencies d
                JOIN _pending_tool_calls p
                  ON p.id = d.pending_tool_call_id
                WHERE d.run_id = _agent_runs.id
                  AND d.resolved_at IS NOT NULL
                  AND d.consumed_at IS NULL
                  AND p.status = ?
                  AND COALESCE(json_extract(p.result, '$.not_applicable'), 0) != 1
              )
            """,
            (
                AgentRunStatus.RESUME_READY.value,
                now_s,
                AgentRunStatus.FINALIZED.value,
                AgentRunStatus.FINALIZED_PENDING_TOOL.value,
                pending_tool_call_id,
                PendingToolCallStatus.RESOLVED.value,
            ),
        )

    def _finalize_runs_without_actionable_dependencies_unlocked(
        self, now_s: str, *, pending_tool_call_id: str
    ) -> None:
        self.conn.execute(
            """
            UPDATE _agent_runs
            SET status = ?,
                finalized_at = COALESCE(finalized_at, ?),
                updated_at = ?
            WHERE status IN (?, ?)
              AND EXISTS (
                SELECT 1
                FROM _run_tool_dependencies changed
                WHERE changed.run_id = _agent_runs.id
                  AND changed.pending_tool_call_id = ?
              )
              AND NOT EXISTS (
                SELECT 1
                FROM _run_tool_dependencies d
                JOIN _pending_tool_calls p
                  ON p.id = d.pending_tool_call_id
                WHERE d.run_id = _agent_runs.id
                  AND d.resolved_at IS NULL
                  AND d.consumed_at IS NULL
                  AND p.status = ?
              )
              AND NOT EXISTS (
                SELECT 1
                FROM _run_tool_dependencies d
                JOIN _pending_tool_calls p
                  ON p.id = d.pending_tool_call_id
                WHERE d.run_id = _agent_runs.id
                  AND d.resolved_at IS NOT NULL
                  AND d.consumed_at IS NULL
                  AND p.status = ?
                  AND COALESCE(json_extract(p.result, '$.not_applicable'), 0) != 1
              )
            """,
            (
                AgentRunStatus.FINALIZED.value,
                now_s,
                now_s,
                AgentRunStatus.FINALIZED_PENDING_TOOL.value,
                AgentRunStatus.RESUME_READY.value,
                pending_tool_call_id,
                PendingToolCallStatus.PENDING.value,
                PendingToolCallStatus.RESOLVED.value,
            ),
        )

    @SQLiteStorageBase.handle_exceptions
    def create_agent_run(self, record: AgentRunRecord) -> AgentRunRecord:
        binding = record.binding
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO _agent_runs (
                    id, org_id, extractor_kind, user_id,
                    request_id, agent_version, source, source_interaction_ids,
                    window_start_interaction_id, window_end_interaction_id,
                    extractor_config_hash, status, generation_request_snapshot,
                    service_config_snapshot, agent_context_snapshot,
                    committed_output, pending_tool_call_ids, max_steps_remaining,
                    resume_attempts, finalization_attempts, next_resume_at,
                    claimed_by, claimed_at, agent_completed_at, finalized_at,
                    expires_at, last_error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.id,
                    binding.org_id,
                    binding.extractor_kind,
                    binding.user_id,
                    binding.request_id,
                    binding.agent_version,
                    binding.source,
                    _json_dumps(binding.source_interaction_ids),
                    binding.window_start_interaction_id,
                    binding.window_end_interaction_id,
                    binding.extractor_config_hash,
                    record.status.value,
                    _json_dumps(record.generation_request_snapshot),
                    _json_dumps(record.service_config_snapshot),
                    record.agent_context_snapshot,
                    _json_dumps(record.committed_output),
                    _json_dumps(record.pending_tool_call_ids),
                    record.max_steps_remaining,
                    record.resume_attempts,
                    record.finalization_attempts,
                    _dt_str(record.next_resume_at),
                    record.claimed_by,
                    _dt_str(record.claimed_at),
                    _dt_str(record.agent_completed_at),
                    _dt_str(record.finalized_at),
                    _dt_str(record.expires_at),
                    record.last_error,
                ),
            )
            self.conn.commit()
        stored = self.get_agent_run(record.id)
        if stored is None:  # pragma: no cover
            raise RuntimeError(f"Failed to create agent run {record.id}")
        return stored

    @SQLiteStorageBase.handle_exceptions
    def get_agent_run(self, run_id: str) -> AgentRunRecord | None:
        row = self._fetchone("SELECT * FROM _agent_runs WHERE id = ?", (run_id,))
        return _row_to_agent_run(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def update_agent_run_status(
        self,
        run_id: str,
        status: AgentRunStatus,
        *,
        committed_output: dict[str, Any] | None = None,
        pending_tool_call_ids: list[str] | None = None,
        max_steps_remaining: int | None = None,
        next_resume_at: datetime | None = None,
        last_error: str | None = None,
        increment_finalization_attempts: bool = False,
    ) -> AgentRunRecord | None:
        current_timestamp = self._current_timestamp()
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status.value, current_timestamp]
        if committed_output is not None:
            assignments.append("committed_output = ?")
            params.append(_json_dumps(committed_output))
        if pending_tool_call_ids is not None:
            assignments.append("pending_tool_call_ids = ?")
            params.append(_json_dumps(pending_tool_call_ids))
        if max_steps_remaining is not None:
            assignments.append("max_steps_remaining = ?")
            params.append(max(0, max_steps_remaining))
        if next_resume_at is not None:
            assignments.append("next_resume_at = ?")
            params.append(_dt_str(next_resume_at))
        if last_error is not None:
            assignments.append("last_error = ?")
            params.append(last_error)
        if increment_finalization_attempts:
            assignments.append("finalization_attempts = finalization_attempts + 1")
        if status == AgentRunStatus.AGENT_COMPLETED:
            assignments.append("agent_completed_at = ?")
            params.append(current_timestamp)
        if status in (AgentRunStatus.FINALIZED, AgentRunStatus.FINALIZED_PENDING_TOOL):
            assignments.append("finalized_at = ?")
            params.append(current_timestamp)
        params.append(run_id)
        with self._lock:
            self.conn.execute(
                f"UPDATE _agent_runs SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            self.conn.commit()
        return self.get_agent_run(run_id)

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

    @SQLiteStorageBase.handle_exceptions
    def claim_ready_agent_run(
        self,
        *,
        org_id: str,
        worker_id: str,
        now: datetime | None = None,
        claim_ttl_seconds: int = 600,
    ) -> AgentRunRecord | None:
        current = now or datetime.now(UTC)
        stale_before = current - timedelta(seconds=claim_ttl_seconds)
        with self._lock:
            row = self.conn.execute(
                """
                SELECT r.*
                FROM _agent_runs r
                WHERE r.org_id = ?
                  AND (
                    r.status = ?
                    OR (r.status = ? AND r.claimed_at < ?)
                )
                  AND (r.next_resume_at IS NULL OR r.next_resume_at <= ?)
                  AND EXISTS (
                    SELECT 1
                    FROM _run_tool_dependencies d
                    JOIN _pending_tool_calls p
                      ON p.id = d.pending_tool_call_id
                    WHERE d.run_id = r.id
                      AND d.resolved_at IS NOT NULL
                      AND d.consumed_at IS NULL
                      AND p.status = ?
                      AND COALESCE(json_extract(p.result, '$.not_applicable'), 0) != 1
                  )
                ORDER BY
                    r.org_id ASC,
                    r.extractor_kind ASC,
                    COALESCE(r.user_id, '') ASC,
                    COALESCE(r.window_start_interaction_id, 0) ASC,
                    r.updated_at ASC
                LIMIT 1
                """,
                (
                    org_id,
                    AgentRunStatus.RESUME_READY.value,
                    AgentRunStatus.RESUMING.value,
                    _dt_str(stale_before),
                    _dt_str(current),
                    PendingToolCallStatus.RESOLVED.value,
                ),
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                """
                UPDATE _agent_runs
                SET status = ?,
                    claimed_by = ?,
                    claimed_at = ?,
                    resume_attempts = resume_attempts + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    AgentRunStatus.RESUMING.value,
                    worker_id,
                    _dt_str(current),
                    self._current_timestamp(),
                    row["id"],
                ),
            )
            self.conn.commit()
        return self.get_agent_run(row["id"])

    @SQLiteStorageBase.handle_exceptions
    def claim_finalization_failed_agent_run(
        self,
        *,
        org_id: str,
        worker_id: str,
        now: datetime | None = None,
        claim_ttl_seconds: int = 600,
    ) -> AgentRunRecord | None:
        current = now or datetime.now(UTC)
        stale_before = current - timedelta(seconds=claim_ttl_seconds)
        with self._lock:
            # Claim runs that still need their committed output finalized:
            #  - FINALIZATION_FAILED: an explicit retry is due.
            #  - stale FINALIZING: a worker crashed mid-finalize.
            #  - stale AGENT_COMPLETED: publish-time finalization never ran or
            #    crashed before flipping the status (the run row carries
            #    committed_output but was orphaned); the staleness guard ensures
            #    we never race an in-flight publish-time finalize.
            row = self.conn.execute(
                """
                SELECT *
                FROM _agent_runs
                WHERE org_id = ?
                  AND (
                    status = ?
                    OR (status = ? AND claimed_at < ?)
                    OR (status = ? AND updated_at < ?)
                )
                  AND committed_output IS NOT NULL
                  AND (next_resume_at IS NULL OR next_resume_at <= ?)
                ORDER BY
                    org_id ASC,
                    extractor_kind ASC,
                    COALESCE(user_id, '') ASC,
                    COALESCE(window_start_interaction_id, 0) ASC,
                    updated_at ASC
                LIMIT 1
                """,
                (
                    org_id,
                    AgentRunStatus.FINALIZATION_FAILED.value,
                    AgentRunStatus.FINALIZING.value,
                    _dt_str(stale_before),
                    AgentRunStatus.AGENT_COMPLETED.value,
                    _dt_str(stale_before),
                    _dt_str(current),
                ),
            ).fetchone()
            if row is None:
                return None
            self.conn.execute(
                """
                UPDATE _agent_runs
                SET status = ?,
                    claimed_by = ?,
                    claimed_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    AgentRunStatus.FINALIZING.value,
                    worker_id,
                    _dt_str(current),
                    self._current_timestamp(),
                    row["id"],
                ),
            )
            self.conn.commit()
        return self.get_agent_run(row["id"])

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

    @SQLiteStorageBase.handle_exceptions
    def list_resumable_work_org_ids(
        self,
        *,
        now: datetime | None = None,
        limit: int = 1000,
    ) -> list[str]:
        current = now or datetime.now(UTC)
        now_s = _dt_str(current)
        bounded_limit = max(1, min(limit, 10_000))
        # Cross-org maintenance query. Surfaces any org that has a run ready to
        # resume / awaiting finalization, or a pending tool call due to expire.
        rows = self._fetchall(
            """
            SELECT DISTINCT org_id FROM (
                SELECT r.org_id
                FROM _agent_runs r
                WHERE r.status IN (?, ?)
                  AND EXISTS (
                    SELECT 1
                    FROM _run_tool_dependencies d
                    JOIN _pending_tool_calls p
                      ON p.id = d.pending_tool_call_id
                    WHERE d.run_id = r.id
                      AND d.resolved_at IS NOT NULL
                      AND d.consumed_at IS NULL
                      AND p.status = ?
                      AND COALESCE(json_extract(p.result, '$.not_applicable'), 0) != 1
                  )
                UNION
                SELECT org_id FROM _agent_runs
                WHERE status IN (?, ?, ?)
                UNION
                SELECT org_id FROM _pending_tool_calls
                WHERE status = ?
                  AND expires_at IS NOT NULL
                  AND expires_at <= ?
            )
            ORDER BY org_id ASC
            LIMIT ?
            """,
            (
                AgentRunStatus.RESUME_READY.value,
                AgentRunStatus.RESUMING.value,
                PendingToolCallStatus.RESOLVED.value,
                AgentRunStatus.FINALIZATION_FAILED.value,
                AgentRunStatus.FINALIZING.value,
                AgentRunStatus.AGENT_COMPLETED.value,
                PendingToolCallStatus.PENDING.value,
                now_s,
                bounded_limit,
            ),
        )
        return [row["org_id"] for row in rows]
