"""Playbook optimization job store methods for SQLite storage."""

import sqlite3
from typing import Any

from reflexio.models.api_schema.service_schemas import (
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
)

from .._base import (
    SQLiteStorageBase,
    _json_dumps,
    _json_loads,
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


class OptimizationJobStoreMixin:
    """Mixin providing playbook optimization job/candidate/evaluation CRUD for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _execute: Any
    _fetchall: Any

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
