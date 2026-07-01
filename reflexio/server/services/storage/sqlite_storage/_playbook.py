"""Playbook CRUD + search methods for SQLite storage."""

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

from reflexio.models.api_schema.service_schemas import (
    AgentSuccessEvaluationResult,
)

from ._base import (
    SQLiteStorageBase,
    _epoch_to_iso,
    _json_dumps,
    _row_to_eval_result,
)
from ._lineage import _append_event_stmt


def _emit_hard_delete_playbook(
    conn: sqlite3.Connection,
    *,
    org_id: str,
    entity_type: str,
    entity_id: str,
    request_id: str,
    actor: str = "api",
) -> None:
    """Emit a single hard_delete lineage event for a playbook entity."""
    _append_event_stmt(
        conn,
        org_id=org_id,
        entity_type=entity_type,
        entity_id=entity_id,
        op="hard_delete",
        prov="wasInvalidatedBy",
        source_ids=[],
        actor=actor,
        request_id=request_id,
        reason="erasure",
    )


def _build_tags_sql(alias: str, tags: list[str] | None) -> tuple[str, list[Any]]:
    if not tags:
        return "", []
    placeholders = ",".join("?" for _ in tags)
    return (
        f"EXISTS (SELECT 1 FROM json_each({alias}.tags) WHERE value IN ({placeholders}))",
        list(tags),
    )


class PlaybookMixin:
    """Mixin providing user playbook, agent playbook, and evaluation CRUD + search."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    org_id: str
    _execute: Any
    _fetchone: Any
    _fetchall: Any
    _get_embedding: Any
    _should_expand_documents: Any
    _expand_document: Any
    _fts_upsert: Any
    _fts_delete: Any
    _vec_upsert: Any
    _vec_delete: Any
    _delete_playbook_search_rows: Any
    _has_sqlite_vec: bool
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any

    # ------------------------------------------------------------------
    # Agent Success Evaluation methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def save_agent_success_evaluation_results(
        self, results: list[AgentSuccessEvaluationResult]
    ) -> None:
        for result in results:
            embedding_text = f"{result.failure_type} {result.failure_reason}"
            if embedding_text.strip():
                result.embedding = self._get_embedding(embedding_text)
            else:
                result.embedding = []

            created_at_iso = _epoch_to_iso(result.created_at)
            subject_ref = self._subject_ref_for_user_id(result.user_id)
            with self._lock:
                try:
                    self.conn.execute("BEGIN IMMEDIATE")
                    self._assert_subject_writable_locked(subject_ref)
                    self.conn.execute(
                        """INSERT INTO agent_success_evaluation_result
                           (user_id, session_id, agent_version, evaluation_name, is_success,
                            failure_type, failure_reason, regular_vs_shadow,
                            number_of_correction_per_session, user_turns_to_resolution,
                            is_escalated, embedding, created_at, governance_subject_ref)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            result.user_id,
                            result.session_id,
                            result.agent_version,
                            result.evaluation_name,
                            int(result.is_success),
                            result.failure_type,
                            result.failure_reason,
                            result.regular_vs_shadow.value
                            if result.regular_vs_shadow
                            else None,
                            result.number_of_correction_per_session,
                            result.user_turns_to_resolution,
                            int(result.is_escalated),
                            _json_dumps(result.embedding) if result.embedding else None,
                            created_at_iso,
                            subject_ref,
                        ),
                    )
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise

    @SQLiteStorageBase.handle_exceptions
    def get_agent_success_evaluation_results(
        self, limit: int = 100, agent_version: str | None = None
    ) -> list[AgentSuccessEvaluationResult]:
        sql = "SELECT * FROM agent_success_evaluation_result"
        params: list[Any] = []
        if agent_version is not None:
            sql += " WHERE agent_version = ?"
            params.append(agent_version)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._fetchall(sql, params)
        return [_row_to_eval_result(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_agent_success_evaluation_results_in_window(
        self,
        from_ts: int,
        to_ts: int,
        agent_version: str | None = None,
        limit: int | None = None,
    ) -> list[AgentSuccessEvaluationResult]:
        sql = """SELECT * FROM agent_success_evaluation_result
                 WHERE created_at >= ? AND created_at <= ?"""
        params: list[Any] = [_epoch_to_iso(from_ts), _epoch_to_iso(to_ts)]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._fetchall(sql, params)
        return [_row_to_eval_result(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_agent_success_evaluation_result_ids(
        self,
        user_id: str,
        session_id: str,
        evaluation_name: str,
        agent_version: str,
    ) -> list[int]:
        rows = self._fetchall(
            """SELECT result_id FROM agent_success_evaluation_result
               WHERE user_id = ?
                 AND session_id = ?
                 AND evaluation_name = ?
                 AND agent_version = ?
               ORDER BY created_at DESC""",
            (user_id, session_id, evaluation_name, agent_version),
        )
        return [int(r["result_id"]) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_all_agent_success_evaluation_results(self) -> None:
        self._execute("DELETE FROM agent_success_evaluation_result")

    @SQLiteStorageBase.handle_exceptions
    def delete_agent_success_evaluation_results_for_session(
        self,
        user_id: str,
        session_id: str,
        evaluation_name: str,
        agent_version: str,
    ) -> int:
        """Delete results scoped to (user_id, session_id, evaluation_name, agent_version).

        Args:
            user_id (str): User whose session results to clear.
            session_id (str): Session whose results to clear.
            evaluation_name (str): Which evaluator's results to clear.
            agent_version (str): Agent version scope.

        Returns:
            int: Number of rows deleted.
        """
        cur = self._execute(
            """DELETE FROM agent_success_evaluation_result
               WHERE user_id = ?
                 AND session_id = ?
                 AND evaluation_name = ?
                 AND agent_version = ?""",
            (user_id, session_id, evaluation_name, agent_version),
        )
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def delete_agent_success_evaluation_results_by_ids(
        self, result_ids: list[int]
    ) -> int:
        """Delete agent success eval result rows by primary key.

        Args:
            result_ids (list[int]): Primary-key result_ids to delete. An empty
                list is a no-op that returns 0.

        Returns:
            int: Number of rows actually deleted (ignores non-existent ids).
        """
        if not result_ids:
            return 0
        placeholders = ",".join(["?"] * len(result_ids))
        cur = self._execute(
            f"DELETE FROM agent_success_evaluation_result WHERE result_id IN ({placeholders})",
            list(result_ids),
        )
        return cur.rowcount
