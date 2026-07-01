"""Playbook CRUD + search methods for SQLite storage."""

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

from reflexio.models.api_schema.service_schemas import (
    AgentSuccessEvaluationResult,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
)

from ._base import (
    SQLiteStorageBase,
    _epoch_to_iso,
    _json_dumps,
    _json_loads,
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


def _row_to_playbook_optimization_candidate(
    row: sqlite3.Row,
) -> PlaybookOptimizationCandidate:
    return PlaybookOptimizationCandidate(
        candidate_id=row["candidate_id"],
        job_id=row["job_id"],
        candidate_index=row["candidate_index"],
        content=row["content"],
        parent_candidate_ids=_json_loads(row["parent_candidate_ids"]) or [],
        aggregate_score=row["aggregate_score"],
        is_winner=bool(row["is_winner"]),
        metadata_json=row["metadata_json"] or "{}",
        created_at=row["created_at"],
    )


def _row_to_playbook_optimization_evaluation(
    row: sqlite3.Row,
) -> PlaybookOptimizationEvaluation:
    return PlaybookOptimizationEvaluation(
        evaluation_id=row["evaluation_id"],
        job_id=row["job_id"],
        candidate_id=row["candidate_id"],
        target_kind=row["target_kind"],
        target_id=row["target_id"],
        scenario_user_playbook_id=row["scenario_user_playbook_id"],
        source_interaction_ids=_json_loads(row["source_interaction_ids"]) or [],
        score=row["score"],
        verdict=row["verdict"],
        likert=row["likert"],
        rationale=row["rationale"],
        asi_json=row["asi_json"],
        incumbent_rollout_json=row["incumbent_rollout_json"],
        candidate_rollout_json=row["candidate_rollout_json"],
        created_at=row["created_at"],
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
    # Playbook optimizer methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def create_playbook_optimization_job(
        self, job: PlaybookOptimizationJob
    ) -> PlaybookOptimizationJob:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO playbook_optimization_jobs
                   (target_kind, target_id, status, best_candidate_id,
                    successor_target_id, decision_reason, metadata_json,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.target_kind,
                    job.target_id,
                    job.status,
                    job.best_candidate_id,
                    job.successor_target_id,
                    job.decision_reason,
                    job.metadata_json,
                    job.created_at,
                    job.updated_at,
                ),
            )
            job.job_id = cur.lastrowid or 0
            self.conn.commit()
        return job

    @SQLiteStorageBase.handle_exceptions
    def update_playbook_optimization_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        best_candidate_id: int | None = None,
        successor_target_id: int | None = None,
        decision_reason: str | None = None,
        metadata_json: str | None = None,
    ) -> None:
        updates: list[str] = ["updated_at = strftime('%s','now')"]
        params: list[Any] = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if best_candidate_id is not None:
            updates.append("best_candidate_id = ?")
            params.append(best_candidate_id)
        if successor_target_id is not None:
            updates.append("successor_target_id = ?")
            params.append(successor_target_id)
        if decision_reason is not None:
            updates.append("decision_reason = ?")
            params.append(decision_reason)
        if metadata_json is not None:
            updates.append("metadata_json = ?")
            params.append(metadata_json)
        params.append(job_id)
        self._execute(
            f"UPDATE playbook_optimization_jobs SET {', '.join(updates)} WHERE job_id = ?",  # noqa: S608
            tuple(params),
        )

    @SQLiteStorageBase.handle_exceptions
    def insert_playbook_optimization_candidate(
        self, candidate: PlaybookOptimizationCandidate
    ) -> PlaybookOptimizationCandidate:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO playbook_optimization_candidates
                   (job_id, candidate_index, content, parent_candidate_ids,
                    aggregate_score, is_winner, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.job_id,
                    candidate.candidate_index,
                    candidate.content,
                    _json_dumps(candidate.parent_candidate_ids) or "[]",
                    candidate.aggregate_score,
                    1 if candidate.is_winner else 0,
                    candidate.metadata_json,
                    candidate.created_at,
                ),
            )
            candidate.candidate_id = cur.lastrowid or 0
            self.conn.commit()
        return candidate

    @SQLiteStorageBase.handle_exceptions
    def list_playbook_optimization_candidates(
        self, job_id: int
    ) -> list[PlaybookOptimizationCandidate]:
        rows = self._fetchall(
            "SELECT * FROM playbook_optimization_candidates WHERE job_id = ? ORDER BY candidate_id ASC",
            (job_id,),
        )
        return [_row_to_playbook_optimization_candidate(row) for row in rows]

    @SQLiteStorageBase.handle_exceptions
    def update_playbook_optimization_candidate(
        self,
        candidate_id: int,
        *,
        aggregate_score: float | None = None,
        is_winner: bool | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        if aggregate_score is not None:
            updates.append("aggregate_score = ?")
            params.append(aggregate_score)
        if is_winner is not None:
            updates.append("is_winner = ?")
            params.append(1 if is_winner else 0)
        if not updates:
            return
        params.append(candidate_id)
        self._execute(
            f"UPDATE playbook_optimization_candidates SET {', '.join(updates)} WHERE candidate_id = ?",  # noqa: S608
            tuple(params),
        )

    @SQLiteStorageBase.handle_exceptions
    def insert_playbook_optimization_evaluation(
        self, evaluation: PlaybookOptimizationEvaluation
    ) -> PlaybookOptimizationEvaluation:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO playbook_optimization_evaluations
                   (job_id, candidate_id, target_kind, target_id,
                    scenario_user_playbook_id, source_interaction_ids, score,
                    verdict, likert, rationale, asi_json, incumbent_rollout_json,
                    candidate_rollout_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evaluation.job_id,
                    evaluation.candidate_id,
                    evaluation.target_kind,
                    evaluation.target_id,
                    evaluation.scenario_user_playbook_id,
                    _json_dumps(evaluation.source_interaction_ids) or "[]",
                    evaluation.score,
                    evaluation.verdict,
                    evaluation.likert,
                    evaluation.rationale,
                    evaluation.asi_json,
                    evaluation.incumbent_rollout_json,
                    evaluation.candidate_rollout_json,
                    evaluation.created_at,
                ),
            )
            evaluation.evaluation_id = cur.lastrowid or 0
            self.conn.commit()
        return evaluation

    @SQLiteStorageBase.handle_exceptions
    def list_playbook_optimization_evaluations(
        self, job_id: int
    ) -> list[PlaybookOptimizationEvaluation]:
        rows = self._fetchall(
            "SELECT * FROM playbook_optimization_evaluations WHERE job_id = ? ORDER BY evaluation_id ASC",
            (job_id,),
        )
        return [_row_to_playbook_optimization_evaluation(row) for row in rows]

    @SQLiteStorageBase.handle_exceptions
    def insert_playbook_optimization_event(
        self, event: PlaybookOptimizationEvent
    ) -> PlaybookOptimizationEvent:
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO playbook_optimization_events
                   (job_id, event_type, payload_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (event.job_id, event.event_type, event.payload_json, event.created_at),
            )
            event.event_id = cur.lastrowid or 0
            self.conn.commit()
        return event

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
