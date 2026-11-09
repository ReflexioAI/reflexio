"""Playbook optimization job store methods for SQLite storage."""

import json
import sqlite3
import time
from hashlib import sha256
from typing import Any

from pydantic import ValidationError

from reflexio.models.api_schema.domain.entities import canonicalize_artifact_json
from reflexio.models.api_schema.service_schemas import (
    OpenWorldQualificationClassCount,
    OpenWorldQualificationRecord,
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
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES,
    OPEN_WORLD_FAILED_TERMINAL_OUTCOMES,
)
from reflexio.server.services.playbook.publication import (
    PUBLICATION_PROJECTION_JSON_METADATA_KEY,
    PUBLICATION_PROOF_JSON_METADATA_KEY,
    PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY,
    canonical_json_bytes,
)
from reflexio.server.services.storage.error import (
    OpenWorldQualificationConflictError,
    OptimizationArtifactIntegrityError,
    OptimizationJobIdentityConflictError,
    OptimizationJobLeaseLiveError,
    StorageError,
)

_STAGE_PREDECESSORS_BY_OPTIMIZER: dict[str, dict[str, tuple[str, str]]] = {
    "offline_tuner_open_world": {
        "discovery_analyzed": ("evidence_frozen", "evidence_frozen"),
        "candidate_generated": ("discovery_analyzed", "discovery_analyzed"),
        "held_out_analyzed": ("candidate_generated", "candidate_generated"),
    },
}
_ACTIVE_STAGES_BY_OPTIMIZER = {
    "offline_tuner_open_world": (
        "evidence_frozen",
        "discovery_analyzed",
        "candidate_generated",
        "held_out_analyzed",
    ),
}
_TERMINAL_OUTCOMES_BY_OPTIMIZER = {
    "offline_tuner_open_world": {
        "failed": OPEN_WORLD_FAILED_TERMINAL_OUTCOMES,
        "abstained": OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES,
    },
}


#: Tables holding a job's rows without a cascading foreign key to it. Erasing
#: a job deletes these explicitly; every other child cascades from the job.
JOB_CHILD_TABLES_DELETED_EXPLICITLY: frozenset[str] = frozenset(
    {
        "playbook_optimization_candidates",
        "playbook_optimization_evaluations",
        "playbook_optimization_events",
    }
)

#: Tables that only reference a job by id and keep their row when it goes,
#: with the reason. The reference is cleared, never left dangling.
JOB_REFERENCING_TABLES_KEPT: dict[str, str] = {
    "open_world_sweep_progress": (
        "one row per playbook per day records that the tick ran; it holds an "
        "outcome label and no user content, and the queue reads it"
    ),
}

#: Every optimizer job whose stored content names the user, by id or by
#: governance subject. A job is evidence about each user it names, wherever the
#: name sits: the target playbook's owner; the subject vector the job froze
#: (every subject on the evidence it selected); a ``user_id`` anywhere in an
#: artifact (the frozen bundle names each session's user, and every memo,
#: candidate and decision is derived from those sessions); an analyst call or
#: attempt-context manifest entry; or a scenario playbook an evaluation scored.
_JOBS_NAMING_USER_SQL = """
SELECT jobs.job_id FROM playbook_optimization_jobs AS jobs
JOIN user_playbooks AS target
  ON jobs.target_kind = 'user_playbook'
 AND target.user_playbook_id = jobs.target_id
WHERE target.user_id = :user_id
UNION
SELECT jobs.job_id FROM playbook_optimization_jobs AS jobs,
     json_tree(jobs.metadata_json) AS node
WHERE node.atom = :subject_ref
UNION
SELECT artifacts.job_id FROM playbook_optimization_artifacts AS artifacts,
     json_tree(artifacts.content_json) AS node
WHERE (node.key = 'user_id' AND node.atom = :user_id)
   OR node.atom = :subject_ref
UNION
SELECT job_id FROM offline_tuner_open_world_invocations WHERE user_id = :user_id
UNION
SELECT context.job_id FROM offline_tuner_open_world_attempt_context AS context,
     json_tree(context.manifest_json) AS node
WHERE node.key = 'user_id' AND node.atom = :user_id
UNION
SELECT evaluations.job_id FROM playbook_optimization_evaluations AS evaluations
JOIN user_playbooks AS scenario
  ON scenario.user_playbook_id = evaluations.scenario_user_playbook_id
WHERE scenario.user_id = :user_id
"""


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
    try:
        return PlaybookOptimizationArtifact(
            artifact_id=row["artifact_id"],
            job_id=row["job_id"],
            artifact_kind=row["artifact_kind"],
            content_json=row["content_json"],
            content_digest=row["content_digest"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
    except (ValidationError, ValueError) as exc:
        raise OptimizationArtifactIntegrityError(
            "optimizer artifact row is malformed"
        ) from exc


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


def _gepa_publication_authority_from_metadata(metadata_json: str | None) -> dict:
    try:
        metadata = json.loads(metadata_json or "{}")
    except json.JSONDecodeError as exc:
        raise StorageError("GEPA durable publication authority is invalid") from exc
    authority = metadata.get("gepa_publication_authority")
    if not isinstance(authority, dict):
        raise StorageError("GEPA durable publication authority is missing")
    return authority


def _validate_gepa_publication_prepare_metadata(
    *,
    lease_seconds: int,
    candidate_content_digest: str,
    search_projection_digest: str,
    publication_proof_digest: str,
    projection_json: str,
    decision_proof_json: str,
    subject_epochs_json: str,
    metadata_json: str,
) -> dict[str, Any]:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    digests = (
        candidate_content_digest,
        search_projection_digest,
        publication_proof_digest,
    )
    if any(
        len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
        for digest in digests
    ):
        raise ValueError("GEPA publication digests must be lowercase SHA-256")
    try:
        metadata = json.loads(metadata_json)
        json.loads(subject_epochs_json)
    except json.JSONDecodeError as exc:
        raise ValueError("GEPA publication metadata must be valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("GEPA publication metadata must be an object")
    if sha256(projection_json.encode("utf-8")).hexdigest() != search_projection_digest:
        raise ValueError("GEPA projection digest does not match canonical JSON")
    if sha256(decision_proof_json.encode("utf-8")).hexdigest() != (
        publication_proof_digest
    ):
        raise ValueError("GEPA proof digest does not match canonical JSON")
    try:
        if canonical_json_bytes(json.loads(projection_json)).decode() != (
            projection_json
        ):
            raise ValueError("projection JSON is not canonical")
        if canonical_json_bytes(json.loads(decision_proof_json)).decode() != (
            decision_proof_json
        ):
            raise ValueError("proof JSON is not canonical")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("GEPA publication bytes must be canonical JSON") from exc
    return metadata


from .._base import (
    SQLiteStorageBase,
    _json_dumps,
    _json_loads,
)

_QUALIFICATION_SELECT_SQL = """SELECT * FROM offline_tuner_open_world_qualifications
   WHERE component_identity_digest = ? AND suite_digest = ?"""

_QUALIFICATION_INSERT_SQL = """INSERT INTO offline_tuner_open_world_qualifications
   (component_identity_digest, suite_digest, schema_version, result_digest,
    passed, class_counts_json, observation_digests_json, created_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""


def _qualification_insert_values(
    record: OpenWorldQualificationRecord,
) -> tuple[Any, ...]:
    return (
        record.component_identity_digest,
        record.suite_digest,
        record.schema_version,
        record.result_digest,
        int(record.passed),
        json.dumps([count.model_dump() for count in record.class_counts]),
        json.dumps(list(record.observation_digests)),
        record.created_at,
    )


def _row_to_open_world_qualification_record(
    row: sqlite3.Row,
) -> OpenWorldQualificationRecord:
    return OpenWorldQualificationRecord(
        schema_version=row["schema_version"],
        component_identity_digest=row["component_identity_digest"],
        suite_digest=row["suite_digest"],
        result_digest=row["result_digest"],
        class_counts=tuple(
            OpenWorldQualificationClassCount(**count)
            for count in json.loads(row["class_counts_json"])
        ),
        passed=bool(row["passed"]),
        observation_digests=tuple(json.loads(row["observation_digests_json"])),
        created_at=row["created_at"],
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
    commit_scope: Any
    get_user_playbook_publication_subject_epochs: Any

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
    def get_playbook_optimization_job(
        self, job_id: int
    ) -> PlaybookOptimizationJob | None:
        row = self._fetchone(
            "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?",
            (job_id,),
        )
        return None if row is None else _row_to_playbook_optimization_job(row)

    @SQLiteStorageBase.handle_exceptions
    def prepare_gepa_user_playbook_publication(
        self,
        *,
        job_id: int,
        owner: str,
        lease_seconds: int,
        winner_candidate_id: int,
        candidate_content_digest: str,
        search_projection_digest: str,
        publication_proof_digest: str,
        projection_json: str,
        decision_proof_json: str,
        subject_epochs_json: str,
        metadata_json: str,
    ) -> PlaybookOptimizationJob:
        metadata = _validate_gepa_publication_prepare_metadata(
            lease_seconds=lease_seconds,
            candidate_content_digest=candidate_content_digest,
            search_projection_digest=search_projection_digest,
            publication_proof_digest=publication_proof_digest,
            projection_json=projection_json,
            decision_proof_json=decision_proof_json,
            subject_epochs_json=subject_epochs_json,
            metadata_json=metadata_json,
        )
        subject_epochs = json.loads(subject_epochs_json)

        prepared_at = self._lease_now(None)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                job = self.conn.execute(
                    "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if job is None:
                    raise StorageError("GEPA publication job does not exist")
                if (
                    job["optimizer_kind"] != "gepa"
                    or job["target_kind"] != "user_playbook"
                ):
                    raise StorageError("GEPA publication job identity changed")
                if (
                    job["status"] != "running"
                    or job["stage"] is not None
                    or job["lease_owner"] is not None
                    or job["lease_expires_at"] is not None
                ):
                    raise StorageError("GEPA publication job is not prepareable")
                if not job["attempt_key"]:
                    raise StorageError("GEPA publication attempt identity is missing")
                durable_authority = _gepa_publication_authority_from_metadata(
                    job["metadata_json"]
                )
                incoming_authority = metadata.get("gepa_publication_authority")
                if incoming_authority != durable_authority:
                    raise StorageError("GEPA publication authority changed")
                candidate = self.conn.execute(
                    """SELECT * FROM playbook_optimization_candidates
                       WHERE candidate_id = ? AND job_id = ?""",
                    (winner_candidate_id, job_id),
                ).fetchone()
                if candidate is None or not candidate["is_winner"]:
                    raise StorageError("GEPA publication winner changed")
                if sha256(candidate["content"].encode("utf-8")).hexdigest() != (
                    candidate_content_digest
                ):
                    raise StorageError("GEPA publication winner content changed")
                expected_subject_epochs = (
                    self.get_user_playbook_publication_subject_epochs(job["target_id"])
                )
                if expected_subject_epochs != subject_epochs_json:
                    raise StorageError("GEPA publication subject vector changed")

                metadata["gepa_publication_authority"] = durable_authority
                metadata["publication_proof_digest"] = publication_proof_digest
                metadata[PUBLICATION_PROOF_JSON_METADATA_KEY] = decision_proof_json
                metadata[PUBLICATION_PROJECTION_JSON_METADATA_KEY] = projection_json
                metadata[PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY] = subject_epochs
                durable_metadata_json = json.dumps(
                    metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                updated = self.conn.execute(
                    """UPDATE playbook_optimization_jobs
                       SET best_candidate_id = ?, stage = 'publishing',
                           candidate_content_digest = ?, search_projection_digest = ?,
                           metadata_json = ?, decision_reason = 'publishing',
                           lease_owner = ?, lease_fence = lease_fence + 1,
                           lease_expires_at = ?,
                           updated_at = ?
                       WHERE job_id = ? AND optimizer_kind = 'gepa'
                         AND target_kind = 'user_playbook' AND status = 'running'
                         AND stage IS NULL AND lease_owner IS NULL
                         AND lease_expires_at IS NULL""",
                    (
                        winner_candidate_id,
                        candidate_content_digest,
                        search_projection_digest,
                        durable_metadata_json,
                        owner,
                        prepared_at + lease_seconds,
                        prepared_at,
                        job_id,
                    ),
                )
                if updated.rowcount != 1:
                    raise StorageError("GEPA publication preparation lost its fence")
                row = self.conn.execute(
                    "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if row is None:
                    raise StorageError("GEPA publication job disappeared")
                self.conn.commit()
                return _row_to_playbook_optimization_job(row)
            except Exception:
                self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def get_unconsumed_gepa_user_playbook_publishing_job(
        self, target_id: int
    ) -> PlaybookOptimizationJob | None:
        row = self._fetchone(
            """SELECT job.*
               FROM playbook_optimization_jobs AS job
               LEFT JOIN user_playbook_publication_results AS result
                 ON result.optimizer_kind = job.optimizer_kind
                AND result.job_id = job.job_id
               WHERE job.optimizer_kind = 'gepa'
                 AND job.target_kind = 'user_playbook'
                 AND job.target_id = ?
                 AND job.status = 'running'
                 AND job.stage = 'publishing'
                 AND result.job_id IS NULL
               ORDER BY job.job_id ASC
               LIMIT 1""",
            (target_id,),
        )
        return None if row is None else _row_to_playbook_optimization_job(row)

    @SQLiteStorageBase.handle_exceptions
    def reclaim_gepa_user_playbook_publishing_job(
        self,
        target_id: int,
        owner: str,
        lease_seconds: int,
        *,
        now: int | None = None,
    ) -> PlaybookOptimizationJob | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if not owner.strip():
            raise ValueError("owner must be non-empty")
        reclaimed_at = self._lease_now(now)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self.conn.execute(
                    """SELECT job.*
                       FROM playbook_optimization_jobs AS job
                       LEFT JOIN user_playbook_publication_results AS result
                         ON result.optimizer_kind = job.optimizer_kind
                        AND result.job_id = job.job_id
                       WHERE job.optimizer_kind = 'gepa'
                         AND job.target_kind = 'user_playbook'
                         AND job.target_id = ?
                         AND job.status = 'running'
                         AND job.stage = 'publishing'
                         AND result.job_id IS NULL
                       ORDER BY job.job_id ASC
                       LIMIT 1""",
                    (target_id,),
                ).fetchone()
                if existing is None:
                    self.conn.commit()
                    return None
                if (
                    existing["lease_expires_at"] is None
                    or existing["lease_expires_at"] > reclaimed_at
                ):
                    raise OptimizationJobLeaseLiveError(
                        "GEPA publication optimizer job lease is live"
                    )
                row = self.conn.execute(
                    """UPDATE playbook_optimization_jobs
                       SET lease_owner = ?,
                           lease_fence = lease_fence + 1,
                           lease_expires_at = ?,
                           updated_at = ?
                       WHERE job_id = ?
                         AND optimizer_kind = 'gepa'
                         AND target_kind = 'user_playbook'
                         AND status = 'running'
                         AND stage = 'publishing'
                         AND lease_expires_at <= ?
                       RETURNING *""",
                    (
                        owner,
                        reclaimed_at + lease_seconds,
                        reclaimed_at,
                        existing["job_id"],
                        reclaimed_at,
                    ),
                ).fetchone()
                if row is None:
                    raise OptimizationJobLeaseLiveError(
                        "GEPA publication optimizer job lease is live"
                    )
                self.conn.commit()
                return _row_to_playbook_optimization_job(row)
            except Exception:
                self.conn.rollback()
                raise

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
                    raise OptimizationJobIdentityConflictError(
                        "conflicting immutable optimizer job identity"
                    )
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
                        raise OptimizationJobIdentityConflictError(
                            "conflicting immutable optimizer job identity"
                        )
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
        with self._lock:
            advanced = self._advance_playbook_optimization_stage_locked(
                job_id,
                fence,
                stage,
                terminal_outcome=terminal_outcome,
                advanced_at=advanced_at,
            )
            if self._own_transaction():
                self.conn.commit()
            return advanced

    def _advance_playbook_optimization_stage_locked(
        self,
        job_id: int,
        fence: int,
        stage: str,
        *,
        terminal_outcome: str | None,
        advanced_at: int,
    ) -> bool:
        """Advance one job's stage under the caller's lock and transaction.

        A terminal stage takes only an outcome its optimizer's arm admits; any
        other request leaves the row untouched and returns ``False``.
        """
        row = self.conn.execute(
            "SELECT optimizer_kind FROM playbook_optimization_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return False
        optimizer_kind = row["optimizer_kind"]
        predecessors = _STAGE_PREDECESSORS_BY_OPTIMIZER.get(optimizer_kind, {}).get(
            stage
        )
        terminal_status: str | None = None
        # The 'applied' stage was reachable only for 'offline_tuner_replay',
        # whose 'publishing' -> 'applied' predecessor Phase 7 removed. With
        # no optimizer declaring an 'applied' predecessor the request now
        # falls through to the "unknown transition" arm below and is
        # refused -- exactly what the replay-only guard returned.
        if stage in ("failed", "abstained"):
            if terminal_outcome not in _TERMINAL_OUTCOMES_BY_OPTIMIZER.get(
                optimizer_kind, {}
            ).get(stage, frozenset()):
                return False
            terminal_status = "failed" if stage == "failed" else "skipped"
        elif predecessors is None or terminal_outcome is not None:
            return False
        if terminal_status is None:
            if predecessors is None:
                return False
            cur = self.conn.execute(
                """UPDATE playbook_optimization_jobs
                   SET stage = ?, updated_at = ?
                   WHERE job_id = ?
                     AND status IN ('pending', 'running')
                     AND optimizer_kind = ?
                     AND lease_fence = ?
                     AND lease_expires_at > ?
                     AND stage IN (?, ?)""",
                (
                    stage,
                    advanced_at,
                    job_id,
                    optimizer_kind,
                    fence,
                    advanced_at,
                    *predecessors,
                ),
            )
        else:
            active_stages = _ACTIVE_STAGES_BY_OPTIMIZER.get(optimizer_kind, ())
            placeholders = ", ".join("?" for _ in active_stages)
            cur = self.conn.execute(
                f"""UPDATE playbook_optimization_jobs
                   SET stage = ?,
                       terminal_outcome = ?,
                       status = ?,
                       lease_owner = NULL,
                       lease_expires_at = NULL,
                       updated_at = ?
                   WHERE job_id = ?
                     AND status IN ('pending', 'running')
                     AND optimizer_kind = ?
                     AND lease_fence = ?
                     AND lease_expires_at > ?
                     AND stage IN ({placeholders})""",  # noqa: S608
                (
                    stage,
                    terminal_outcome,
                    terminal_status,
                    advanced_at,
                    job_id,
                    optimizer_kind,
                    fence,
                    advanced_at,
                    *active_stages,
                ),
            )
        return cur.rowcount == 1

    def _erase_optimization_jobs_for_user_locked(
        self, user_id: str, subject_ref: str
    ) -> int:
        """Delete every optimizer job whose stored content names the user.

        Runs under the caller's lock and inside its transaction, before the
        user's playbooks are deleted or purged (ownership is read from them).
        The whole job goes rather than its user's parts: its bundle, memos,
        candidate and checkpoints are one derivation over all its sessions, so
        no remainder of it is evidence about anyone. Children with a cascading
        key go with the job; the rest are deleted here, and references that
        outlive it are cleared.

        Returns:
            int: Jobs deleted.
        """
        job_ids = [
            int(row[0])
            for row in self.conn.execute(
                _JOBS_NAMING_USER_SQL,
                {"user_id": user_id, "subject_ref": subject_ref},
            ).fetchall()
        ]
        for start in range(0, len(job_ids), 500):
            chunk = job_ids[start : start + 500]
            placeholders = ", ".join("?" * len(chunk))
            for table in sorted(JOB_CHILD_TABLES_DELETED_EXPLICITLY):
                self.conn.execute(
                    f"DELETE FROM {table} WHERE job_id IN ({placeholders})",  # noqa: S608
                    chunk,
                )
            for table in sorted(JOB_REFERENCING_TABLES_KEPT):
                self.conn.execute(
                    f"UPDATE {table} SET job_id = NULL"  # noqa: S608
                    f" WHERE job_id IN ({placeholders})",
                    chunk,
                )
            self.conn.execute(
                "DELETE FROM playbook_optimization_jobs"  # noqa: S608
                f" WHERE job_id IN ({placeholders})",
                chunk,
            )
        return len(job_ids)

    @SQLiteStorageBase.handle_exceptions
    def merge_playbook_optimization_job_metadata(
        self, job_id: int, patch: dict[str, Any]
    ) -> None:
        """Merge top-level keys into ``metadata_json`` without a lease predicate.

        Shallow: a key present in both takes ``patch``'s value. The read and
        the write run in one ``BEGIN IMMEDIATE`` transaction, so no concurrent
        writer can commit between them and lose its keys. No lease fence: the
        terminal-failure path writes its diagnostic precisely when a long
        attempt has outrun its lease. Only ``metadata_json`` and
        ``updated_at`` change, so no lifecycle column moves without its fence.

        Args:
            job_id (int): Optimizer job to annotate.
            patch (dict[str, Any]): Keys to merge.
        """
        if not patch:
            return
        with self._lock, self.commit_scope():
            row = self.conn.execute(
                "SELECT metadata_json FROM playbook_optimization_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return
            try:
                current = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError as exc:
                raise OptimizationArtifactIntegrityError(
                    "optimizer job metadata is not valid JSON"
                ) from exc
            if not isinstance(current, dict):
                raise OptimizationArtifactIntegrityError(
                    "optimizer job metadata is not an object"
                )
            merged = canonical_json_bytes({**current, **patch}).decode()
            self.conn.execute(
                "UPDATE playbook_optimization_jobs SET metadata_json = ?,"
                " updated_at = ? WHERE job_id = ?",
                (merged, int(time.time()), job_id),
            )

    @SQLiteStorageBase.handle_exceptions
    def upsert_playbook_optimization_artifact(
        self,
        artifact: PlaybookOptimizationArtifact,
        fence: int,
        *,
        now: int | None = None,
    ) -> PlaybookOptimizationArtifact:
        artifact_content_json = canonicalize_artifact_json(artifact.content_json)
        if (
            sha256(artifact_content_json.encode()).hexdigest()
            != artifact.content_digest
        ):
            raise OptimizationArtifactIntegrityError(
                "optimizer artifact digest does not match content"
            )
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
                        raise OptimizationArtifactIntegrityError(
                            "optimizer artifact digest conflict"
                        )
                    if existing["content_json"] != artifact_content_json:
                        raise OptimizationArtifactIntegrityError(
                            "optimizer artifact content conflict"
                        )
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
    def persist_open_world_qualification_record(
        self, record: OpenWorldQualificationRecord
    ) -> OpenWorldQualificationRecord:
        with self._lock:
            owns_transaction = self._own_transaction()
            if owns_transaction:
                self.conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self.conn.execute(
                    _QUALIFICATION_SELECT_SQL,
                    (record.component_identity_digest, record.suite_digest),
                ).fetchone()
                if existing is not None:
                    stored = _row_to_open_world_qualification_record(existing)
                    if stored.semantic_key() != record.semantic_key():
                        raise OpenWorldQualificationConflictError(
                            "open-world qualification record conflicts with the "
                            "record already cached for this identity and suite"
                        )
                else:
                    self.conn.execute(
                        _QUALIFICATION_INSERT_SQL,
                        _qualification_insert_values(record),
                    )
                    stored = record
                if owns_transaction:
                    self.conn.commit()
                return stored
            except Exception:
                if owns_transaction:
                    self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def load_open_world_qualification_record(
        self,
        *,
        component_identity_digest: str,
        suite_digest: str,
    ) -> OpenWorldQualificationRecord | None:
        row = self._fetchone(
            _QUALIFICATION_SELECT_SQL,
            (component_identity_digest, suite_digest),
        )
        return None if row is None else _row_to_open_world_qualification_record(row)

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
