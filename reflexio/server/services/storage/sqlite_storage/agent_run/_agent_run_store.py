"""SQLite AgentRunStore methods (the AgentRunStore bucket).

Extracted from ``_agent_run.py`` (the AgentRunStore bucket): the agent-run
lifecycle and provenance lookup methods plus the three ``_..._unlocked`` run-cascade
privates. The privates are consumed cross-bucket by the pending-tool-call
methods (``cancel``/``expire``/``update_resolved``/``mark_not_applicable``) in
``SQLitePendingToolCallStoreMixin``, which reach them via MRO co-composition —
kept SINGLE here, never duplicated.

The residual ``_agent_run.py`` module (helpers-only, no mixin class)
permanently holds the shared row/datetime helpers (``_dt``, ``_dt_str``,
``_row_to_agent_run`` …); ``_json_dumps`` comes from ``.._base``. Both are
imported here rather than duplicated.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallStatus,
)

from .._agent_run import _dt_str, _row_to_agent_run
from .._base import SQLiteStorageBase, _json_dumps


def _valid_finalized_learning_ids(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(learning_id, str) and bool(learning_id.strip())
        for learning_id in value
    )


class SQLiteAgentRunStoreMixin:
    """SQLite-backed resumable extraction run store primitives."""

    commit_scope: Any
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
              AND json_type(generation_request_snapshot, '$.overflow') IS NULL
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
              AND json_type(generation_request_snapshot, '$.overflow') IS NULL
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
                  AND COALESCE(json_type(p.result, '$.not_applicable'), '') != 'true'
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
              AND json_type(generation_request_snapshot, '$.overflow') IS NULL
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
                  AND COALESCE(json_type(p.result, '$.not_applicable'), '') != 'true'
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
        with self.commit_scope():
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
        stored = self.get_agent_run(record.id)
        if stored is None:  # pragma: no cover
            raise RuntimeError(f"Failed to create agent run {record.id}")
        return stored

    @SQLiteStorageBase.handle_exceptions
    def get_agent_run(self, run_id: str) -> AgentRunRecord | None:
        row = self._fetchone("SELECT * FROM _agent_runs WHERE id = ?", (run_id,))
        return _row_to_agent_run(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def save_agent_run_finalization_receipt(
        self,
        *,
        run_id: str,
        entity_type: str,
        learning_ids: list[str],
    ) -> bool:
        expected_by_extractor = {
            "profile": "profile",
            "playbook": "user_playbook",
        }
        if not _valid_finalized_learning_ids(learning_ids):
            raise ValueError(
                "agent-run finalization receipt learning ids must be non-empty strings"
            )
        encoded_ids = _json_dumps(learning_ids)
        with self.commit_scope():
            run = self.conn.execute(
                "SELECT org_id, extractor_kind FROM _agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if run is None or run["org_id"] != self.org_id:
                raise ValueError("agent-run finalization receipt owner is invalid")
            if expected_by_extractor.get(run["extractor_kind"]) != entity_type:
                raise ValueError(
                    "agent-run finalization receipt entity type is invalid"
                )
            inserted = (
                self.conn.execute(
                    """
                INSERT OR IGNORE INTO _agent_run_finalization_receipts
                    (run_id, entity_type, learning_ids)
                VALUES (?, ?, ?)
                """,
                    (run_id, entity_type, encoded_ids),
                ).rowcount
                == 1
            )
            stored = self.conn.execute(
                """
                SELECT entity_type, learning_ids
                FROM _agent_run_finalization_receipts
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if stored is None or stored["entity_type"] != entity_type:
                raise ValueError("agent-run finalization receipt is immutable")
            stored_ids = json.loads(stored["learning_ids"])
            if not _valid_finalized_learning_ids(stored_ids):
                raise ValueError("agent-run finalization receipt is corrupt")
            return inserted

    @SQLiteStorageBase.handle_exceptions
    def get_latest_finalized_agent_run_for_request(
        self,
        *,
        org_id: str,
        extractor_kind: str,
        user_id: str | None,
        request_id: str,
    ) -> AgentRunRecord | None:
        row = self._fetchone(
            """
            SELECT *
            FROM _agent_runs
            WHERE org_id = ?
              AND extractor_kind = ?
              AND user_id IS ?
              AND request_id = ?
              AND status IN (?, ?)
            ORDER BY created_at DESC, updated_at DESC, id DESC
            LIMIT 1
            """,
            (
                org_id,
                extractor_kind,
                user_id,
                request_id,
                AgentRunStatus.FINALIZED.value,
                AgentRunStatus.FINALIZED_PENDING_TOOL.value,
            ),
        )
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
        expected_statuses: tuple[AgentRunStatus, ...] | None = None,
        expected_claimed_by: str | None = None,
        expected_claimed_at: datetime | None = None,
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
        status_filter = ""
        if expected_statuses:
            placeholders = ",".join("?" for _ in expected_statuses)
            status_filter = f" AND status IN ({placeholders})"
            params.extend(expected.value for expected in expected_statuses)
        if expected_claimed_by is not None:
            status_filter += " AND claimed_by = ? AND claimed_at = ?"
            params.extend([expected_claimed_by, _dt_str(expected_claimed_at)])
        with self.commit_scope():
            changed = self.conn.execute(
                f"UPDATE _agent_runs SET {', '.join(assignments)} WHERE id = ?{status_filter}",
                params,
            ).rowcount
        return (
            self.get_agent_run(run_id)
            if changed or expected_claimed_by is None
            else None
        )

    @SQLiteStorageBase.handle_exceptions
    def fail_running_agent_runs_for_request(
        self,
        *,
        org_id: str,
        extractor_kind: str,
        user_id: str | None,
        request_id: str,
        last_error: str,
    ) -> int:
        current_timestamp = self._current_timestamp()
        with self.commit_scope():
            cursor = self.conn.execute(
                """
                UPDATE _agent_runs
                SET status = ?,
                    updated_at = ?,
                    last_error = ?
                WHERE org_id = ?
                  AND extractor_kind = ?
                  AND user_id IS ?
                  AND request_id = ?
                  AND status IN (?, ?)
                """,
                (
                    AgentRunStatus.FAILED.value,
                    current_timestamp,
                    last_error,
                    org_id,
                    extractor_kind,
                    user_id,
                    request_id,
                    AgentRunStatus.RUNNING.value,
                    AgentRunStatus.RESUMING.value,
                ),
            )
            return cursor.rowcount

    @SQLiteStorageBase.handle_exceptions
    def get_agent_run_recovery_facts(self, run_id: str) -> dict[str, Any]:
        row = self._fetchone(
            """SELECT
              (r.id NOT LIKE 'window:%' OR EXISTS (
                SELECT 1 FROM extraction_windows w WHERE w.window_id=substr(r.id,8)
                AND w.completed=1 AND w.invalidated=0)) AS window_ready,
              (SELECT count(*) FROM _run_tool_dependencies d
                WHERE d.run_id=r.id AND d.consumed_at IS NULL) AS dependencies,
              (f.receipt_id IS NOT NULL OR legacy.run_id IS NOT NULL) AS receipt_exists,
              json_array_length(COALESCE(f.learning_ids,legacy.learning_ids)) AS receipt_count,
              (SELECT count(*) FROM json_each(COALESCE(f.learning_ids,legacy.learning_ids)) ids
                WHERE NOT (
                  (COALESCE(f.entity_type,legacy.entity_type)='profile' AND EXISTS(SELECT 1 FROM profiles p WHERE CAST(p.profile_id AS TEXT)=ids.value)) OR
                  (COALESCE(f.entity_type,legacy.entity_type)='user_playbook' AND EXISTS(SELECT 1 FROM user_playbooks p WHERE CAST(p.user_playbook_id AS TEXT)=ids.value))
                )) AS receipt_missing_results,
              (SELECT count(DISTINCT i.interaction_id)
                FROM json_each(r.source_interaction_ids) src
                JOIN interactions i ON i.interaction_id=src.value
                JOIN requests q ON q.request_id=i.request_id
                WHERE i.user_id=r.user_id AND q.user_id=r.user_id
                  AND q.source IS r.source AND q.agent_version IS r.agent_version
              ) AS valid_source_count
            FROM _agent_runs r
            LEFT JOIN _agent_output_receipts f ON f.run_id=r.id AND f.org_id=r.org_id
              AND f.row_kind='output' AND f.output_id=json_extract(r.committed_output,'$.publication.id')
            LEFT JOIN _agent_run_finalization_receipts legacy ON legacy.run_id=r.id
              AND json_type(r.committed_output,'$.publication') IS NULL
            WHERE r.id=? AND r.org_id=?""",
            (run_id, self.org_id),
        )
        return dict(row) if row else {}
