"""Playbook optimization job store methods for SQLite storage."""

import sqlite3
import time
from typing import Any

from reflexio.models.api_schema.domain.entities import canonicalize_artifact_json
from reflexio.models.api_schema.service_schemas import (
    OptimizationArtifactKind,
    OptimizationJobClaim,
    OptimizationJobStage,
    OptimizationTerminalOutcome,
    PlaybookOptimizationArtifact,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
)

_FAILURE_OUTCOMES = {"generation_failed", "replay_failed", "publication_failed"}
_STAGE_PREDECESSORS: dict[str, str] = {
    "candidate_generated": "evidence_frozen",
    "replay_running": "candidate_generated",
    "replay_evaluated": "replay_running",
    "publishing": "replay_evaluated",
    "applied": "publishing",
}


def _row_to_playbook_optimization_job(row: sqlite3.Row) -> PlaybookOptimizationJob:
    return PlaybookOptimizationJob(
        job_id=row["job_id"],
        optimizer_kind=row["optimizer_kind"],
        target_kind=row["target_kind"],
        target_id=row["target_id"],
        status=row["status"],
        best_candidate_id=row["best_candidate_id"],
        successor_target_id=row["successor_target_id"],
        decision_reason=row["decision_reason"],
        metadata_json=row["metadata_json"] or "{}",
        discovery_key=row["discovery_key"],
        attempt_key=row["attempt_key"],
        lease_owner=row["lease_owner"],
        lease_fence=row["lease_fence"],
        lease_expires_at=row["lease_expires_at"],
        stage=row["stage"],
        terminal_outcome=row["terminal_outcome"],
        expected_population_manifest_digest=row["expected_population_manifest_digest"],
        generation_selection_manifest_digest=row[
            "generation_selection_manifest_digest"
        ],
        replay_manifest_digest=row["replay_manifest_digest"],
        candidate_content_digest=row["candidate_content_digest"],
        search_projection_digest=row["search_projection_digest"],
        publication_scope_digest=row["publication_scope_digest"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_playbook_optimization_artifact(
    row: sqlite3.Row,
) -> PlaybookOptimizationArtifact:
    return PlaybookOptimizationArtifact(
        artifact_id=row["artifact_id"],
        job_id=row["job_id"],
        artifact_kind=row["artifact_kind"],
        content_json=row["content_json"],
        content_digest=row["content_digest"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _job_insert_values(job: PlaybookOptimizationJob) -> tuple[Any, ...]:
    return (
        job.optimizer_kind,
        job.target_kind,
        job.target_id,
        job.status,
        job.best_candidate_id,
        job.successor_target_id,
        job.decision_reason,
        job.metadata_json,
        job.discovery_key,
        job.attempt_key,
        job.lease_owner,
        job.lease_fence,
        job.lease_expires_at,
        job.stage,
        job.terminal_outcome,
        job.expected_population_manifest_digest,
        job.generation_selection_manifest_digest,
        job.replay_manifest_digest,
        job.candidate_content_digest,
        job.search_projection_digest,
        job.publication_scope_digest,
        job.created_at,
        job.updated_at,
    )


_JOB_INSERT_SQL = """INSERT INTO playbook_optimization_jobs
    (optimizer_kind, target_kind, target_id, status, best_candidate_id,
     successor_target_id, decision_reason, metadata_json, discovery_key,
     attempt_key, lease_owner, lease_fence, lease_expires_at, stage,
     terminal_outcome, expected_population_manifest_digest,
     generation_selection_manifest_digest, replay_manifest_digest,
     candidate_content_digest, search_projection_digest,
     publication_scope_digest, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

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
    _fetchone: Any
    _fetchall: Any
    _own_transaction: Any

    # ------------------------------------------------------------------
    # Playbook optimizer methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def create_playbook_optimization_job(
        self, job: PlaybookOptimizationJob
    ) -> PlaybookOptimizationJob:
        with self._lock:
            cur = self.conn.execute(_JOB_INSERT_SQL, _job_insert_values(job))
            job.job_id = cur.lastrowid or 0
            self.conn.commit()
        return job

    @SQLiteStorageBase.handle_exceptions
    def create_or_get_playbook_optimization_job(
        self, job: PlaybookOptimizationJob
    ) -> PlaybookOptimizationJob:
        if job.discovery_key is None and job.attempt_key is None:
            raise ValueError(
                "durable optimizer jobs require a discovery or attempt key"
            )
        with self._lock:
            owns_transaction = self._own_transaction()
            if owns_transaction:
                self.conn.execute("BEGIN IMMEDIATE")
            try:
                by_discovery = None
                if job.discovery_key is not None:
                    by_discovery = self.conn.execute(
                        """SELECT * FROM playbook_optimization_jobs
                           WHERE optimizer_kind = ? AND discovery_key = ?
                             AND status IN ('pending', 'running')""",
                        (job.optimizer_kind, job.discovery_key),
                    ).fetchone()
                by_attempt = None
                if job.attempt_key is not None:
                    by_attempt = self.conn.execute(
                        """SELECT * FROM playbook_optimization_jobs
                           WHERE optimizer_kind = ? AND attempt_key = ?
                             AND status IN ('pending', 'running')""",
                        (job.optimizer_kind, job.attempt_key),
                    ).fetchone()
                if (
                    by_discovery is not None
                    and by_attempt is not None
                    and by_discovery["job_id"] != by_attempt["job_id"]
                ):
                    raise ValueError("conflicting immutable optimizer job identity")
                existing = by_discovery or by_attempt
                if existing is not None:
                    if (
                        existing["target_kind"] != job.target_kind
                        or existing["target_id"] != job.target_id
                        or (
                            by_discovery is not None
                            and existing["attempt_key"] != job.attempt_key
                        )
                    ):
                        raise ValueError("conflicting immutable optimizer job identity")
                    result = _row_to_playbook_optimization_job(existing)
                else:
                    cur = self.conn.execute(_JOB_INSERT_SQL, _job_insert_values(job))
                    row = self.conn.execute(
                        "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?",
                        (cur.lastrowid,),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("optimizer job insert returned no row")
                    result = _row_to_playbook_optimization_job(row)
                if owns_transaction:
                    self.conn.commit()
                return result
            except Exception:
                if owns_transaction:
                    self.conn.rollback()
                raise

    @staticmethod
    def _lease_now(now: int | None) -> int:
        return int(time.time()) if now is None else now

    @SQLiteStorageBase.handle_exceptions
    def claim_playbook_optimization_job(
        self,
        job_id: int,
        owner: str,
        lease_seconds: int,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        claimed_at = self._lease_now(now)
        with self._lock:
            row = self.conn.execute(
                """UPDATE playbook_optimization_jobs
                   SET lease_owner = ?,
                       lease_fence = lease_fence + 1,
                       lease_expires_at = ?,
                       status = 'running',
                       updated_at = ?
                   WHERE job_id = ?
                     AND status IN ('pending', 'running')
                     AND lease_owner IS NULL
                   RETURNING job_id, lease_owner, lease_fence, lease_expires_at""",
                (owner, claimed_at + lease_seconds, claimed_at, job_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("optimizer job is not available to claim")
            if self._own_transaction():
                self.conn.commit()
        return OptimizationJobClaim(
            job_id=row["job_id"],
            owner=row["lease_owner"],
            fence=row["lease_fence"],
            expires_at=row["lease_expires_at"],
        )

    @SQLiteStorageBase.handle_exceptions
    def reclaim_playbook_optimization_job(
        self,
        job_id: int,
        owner: str,
        lease_seconds: int = 60,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        reclaimed_at = self._lease_now(now)
        with self._lock:
            row = self.conn.execute(
                """UPDATE playbook_optimization_jobs
                   SET lease_owner = ?,
                       lease_fence = lease_fence + 1,
                       lease_expires_at = ?,
                       status = 'running',
                       updated_at = ?
                   WHERE job_id = ?
                     AND status IN ('pending', 'running')
                     AND lease_owner IS NOT NULL
                     AND lease_expires_at IS NOT NULL
                     AND lease_expires_at <= ?
                   RETURNING job_id, lease_owner, lease_fence, lease_expires_at""",
                (
                    owner,
                    reclaimed_at + lease_seconds,
                    reclaimed_at,
                    job_id,
                    reclaimed_at,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("optimizer job lease is not expired")
            if self._own_transaction():
                self.conn.commit()
        return OptimizationJobClaim(
            job_id=row["job_id"],
            owner=row["lease_owner"],
            fence=row["lease_fence"],
            expires_at=row["lease_expires_at"],
        )

    @SQLiteStorageBase.handle_exceptions
    def renew_playbook_optimization_job_lease(
        self,
        job_id: int,
        owner: str,
        fence: int,
        lease_seconds: int,
        *,
        now: int | None = None,
    ) -> OptimizationJobClaim:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        renewed_at = self._lease_now(now)
        with self._lock:
            row = self.conn.execute(
                """UPDATE playbook_optimization_jobs
                   SET lease_expires_at = ?, updated_at = ?
                   WHERE job_id = ?
                     AND status IN ('pending', 'running')
                     AND lease_owner = ?
                     AND lease_fence = ?
                     AND lease_expires_at > ?
                   RETURNING job_id, lease_owner, lease_fence, lease_expires_at""",
                (
                    renewed_at + lease_seconds,
                    renewed_at,
                    job_id,
                    owner,
                    fence,
                    renewed_at,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("optimizer job lease is no longer current")
            if self._own_transaction():
                self.conn.commit()
        return OptimizationJobClaim(
            job_id=row["job_id"],
            owner=row["lease_owner"],
            fence=row["lease_fence"],
            expires_at=row["lease_expires_at"],
        )

    @SQLiteStorageBase.handle_exceptions
    def advance_playbook_optimization_stage(
        self,
        job_id: int,
        fence: int,
        stage: OptimizationJobStage,
        *,
        terminal_outcome: OptimizationTerminalOutcome | None = None,
        now: int | None = None,
    ) -> bool:
        advanced_at = self._lease_now(now)
        predecessor = _STAGE_PREDECESSORS.get(stage)
        terminal_status: str | None = None
        if stage == "applied":
            if terminal_outcome not in (None, "applied"):
                return False
            terminal_outcome = "applied"
            terminal_status = "completed"
        elif stage == "failed":
            if terminal_outcome not in _FAILURE_OUTCOMES:
                return False
            terminal_status = "failed"
        elif stage == "abstained":
            if terminal_outcome is None or terminal_outcome in {
                "applied",
                *_FAILURE_OUTCOMES,
            }:
                return False
            terminal_status = "skipped"
        elif predecessor is None or terminal_outcome is not None:
            return False
        with self._lock:
            if terminal_status is None:
                cur = self.conn.execute(
                    """UPDATE playbook_optimization_jobs
                       SET stage = ?, updated_at = ?
                       WHERE job_id = ?
                         AND status IN ('pending', 'running')
                         AND lease_fence = ?
                         AND lease_expires_at > ?
                         AND stage = ?""",
                    (stage, advanced_at, job_id, fence, advanced_at, predecessor),
                )
            else:
                cur = self.conn.execute(
                    """UPDATE playbook_optimization_jobs
                       SET stage = ?,
                           terminal_outcome = ?,
                           status = ?,
                           lease_owner = NULL,
                           lease_expires_at = NULL,
                           updated_at = ?
                       WHERE job_id = ?
                         AND status IN ('pending', 'running')
                         AND lease_fence = ?
                         AND lease_expires_at > ?
                         AND stage IN (
                             'evidence_frozen',
                             'candidate_generated',
                             'replay_running',
                             'replay_evaluated',
                             'publishing'
                         )""",
                    (
                        stage,
                        terminal_outcome,
                        terminal_status,
                        advanced_at,
                        job_id,
                        fence,
                        advanced_at,
                    ),
                )
            if self._own_transaction():
                self.conn.commit()
            return cur.rowcount == 1

    @SQLiteStorageBase.handle_exceptions
    def upsert_playbook_optimization_artifact(
        self,
        artifact: PlaybookOptimizationArtifact,
        fence: int,
        *,
        now: int | None = None,
    ) -> PlaybookOptimizationArtifact:
        artifact_content_json = canonicalize_artifact_json(artifact.content_json)
        written_at = self._lease_now(now)
        with self._lock:
            owns_transaction = self._own_transaction()
            if owns_transaction:
                self.conn.execute("BEGIN IMMEDIATE")
            try:
                lease = self.conn.execute(
                    """SELECT job_id FROM playbook_optimization_jobs
                       WHERE job_id = ?
                         AND status IN ('pending', 'running')
                         AND lease_fence = ?
                         AND lease_expires_at > ?""",
                    (artifact.job_id, fence, written_at),
                ).fetchone()
                if lease is None:
                    raise ValueError("optimizer job lease is no longer current")
                existing = self.conn.execute(
                    """SELECT * FROM playbook_optimization_artifacts
                       WHERE job_id = ? AND artifact_kind = ?""",
                    (artifact.job_id, artifact.artifact_kind),
                ).fetchone()
                if existing is not None:
                    if existing["content_digest"] != artifact.content_digest:
                        raise ValueError("optimizer artifact digest conflict")
                    if existing["content_json"] != artifact_content_json:
                        raise ValueError("optimizer artifact content conflict")
                    result = _row_to_playbook_optimization_artifact(existing)
                else:
                    cur = self.conn.execute(
                        """INSERT INTO playbook_optimization_artifacts
                           (job_id, artifact_kind, content_json, content_digest,
                            created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            artifact.job_id,
                            artifact.artifact_kind,
                            artifact_content_json,
                            artifact.content_digest,
                            artifact.created_at,
                            artifact.updated_at,
                        ),
                    )
                    row = self.conn.execute(
                        """SELECT * FROM playbook_optimization_artifacts
                           WHERE artifact_id = ?""",
                        (cur.lastrowid,),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("optimizer artifact insert returned no row")
                    result = _row_to_playbook_optimization_artifact(row)
                if owns_transaction:
                    self.conn.commit()
                return result
            except Exception:
                if owns_transaction:
                    self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def get_playbook_optimization_artifact(
        self,
        job_id: int,
        artifact_kind: OptimizationArtifactKind,
    ) -> PlaybookOptimizationArtifact | None:
        row = self._fetchone(
            """SELECT * FROM playbook_optimization_artifacts
               WHERE job_id = ? AND artifact_kind = ?""",
            (job_id, artifact_kind),
        )
        return None if row is None else _row_to_playbook_optimization_artifact(row)

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
