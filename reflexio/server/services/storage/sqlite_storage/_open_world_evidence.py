"""Open-world evidence on SQLite: the bundle freeze and the window snapshot.

The four ``OpenWorldStoragePort`` members an attempt is frozen through:

* ``freeze_open_world_evidence_bundle`` / ``load_open_world_evidence_bundle``
  (``OpenWorldBundleStorage``). The bundle is a ``playbook_optimization_artifacts``
  row of kind ``open_world_evidence_bundle``.
* ``load_open_world_evidence_snapshot``, ``list_open_world_governance_epochs``
  and ``list_playbook_deployment_lifecycles`` (``OpenWorldFreezeReads``).

The semantics mirror the PostgreSQL implementation of the same port; the
shared contract (``reflexio.test_support.open_world_evidence_contract``) holds
both to the same answers. What SQLite answers differently, and why:

* **Governance.** SQLite keeps no governance-subject registry and no erasure
  epochs. Every subject is at epoch 0 and writable unless an erasure barrier
  (``subject_write_barriers``) holds it. Postgres also refuses a subject its
  registry does not list.
* **Outcome contracts.** SQLite stores no contract rows. An outcome's contract
  counts as registered when its stored digest is the one the session-outcome
  writer derives from the outcome's own ``source``.
* **Session-only exposures.** SQLite has no retrieval log, so an exposure with
  a session but no request never binds to a request; it stays
  ``session_only``. The SQLite publish path never writes one.
* **Deployment lifecycles.** Only the open-world provisional commit
  (``_open_world_publication``) writes a lifecycle.

The snapshot is one read transaction under WAL: rows a second connection
commits after the first read are invisible to the rest of it. Source rows are
read whole, in order, under a serialized-byte budget and an elapsed deadline;
exhausting either refuses the snapshot rather than returning a partial source.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from hashlib import sha256
from time import monotonic
from typing import Any

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
)
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceSnapshot,
    OpenWorldTargetIdentity,
)
from reflexio.server.services.offline_tuner.evidence.open_world_policy import (
    OpenWorldEvidenceReadLimits,
)
from reflexio.server.services.offline_tuner.evidence.open_world_snapshot_assembly import (
    ABSENT_TRAJECTORY,
    LoadedTrajectory,
    SessionKey,
    assemble_snapshot,
    assemble_trajectories,
    correlated_request_ids,
    digest_or_none,
    event_dependencies,
    governance_subject_refs_to_read,
    positive_int_or_none,
    prefix_pairs,
    resolve_exposure_correlations,
    session_keys,
    text_or_none,
    trajectory_cutovers,
)
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    OpenWorldEvidenceIntegrityError,
)
from reflexio.server.services.offline_tuner.exposure_receipts import (
    PUBLISHED_EXPOSURE_PREFIX,
)
from reflexio.server.services.offline_tuner.open_world_freeze_validation import (
    canonical_artifact_payload,
    freeze_validation_identity,
    validate_artifact_request,
    validate_freeze_job_request,
    validate_subject_epoch_shape,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_EVIDENCE_ARTIFACT_KIND,
    OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND,
    OpenWorldEvidenceGovernanceError,
    OpenWorldEvidenceLeaseConflictError,
    OpenWorldEvidenceStorageRecord,
    OpenWorldEvidenceTargetStaleError,
)
from reflexio.server.services.search_exposure import (
    user_playbook_full_version_fingerprint,
)
from reflexio.server.services.storage.error import (
    OptimizationArtifactIntegrityError,
    OptimizationJobIdentityConflictError,
    StorageError,
)

from ._base import (
    SQLiteStorageBase,
    _row_to_user_playbook,
    _through_prefix_predicate,
)
from ._open_world_publication import DeploymentLifecycleRow, lifecycle_row
from .playbook._optimization import (
    _JOB_INSERT_SQL,
    _job_insert_values,
    _row_to_playbook_optimization_artifact,
    _row_to_playbook_optimization_job,
)

logger = logging.getLogger(__name__)

_EXPOSURE_PAGE_SIZE = 500
#: Sessions, requests or pairs per ``IN`` / ``VALUES`` list.
_KEY_CHUNK_SIZE = 500
_SOURCE_FETCH_SIZE = 64
#: Serialized bytes of joined source rows one snapshot may consume, counting
#: every row read -- including rows of a session whose content is later dropped
#: as over its per-source cap. The same budget Postgres charges.
_SOURCE_READ_BYTES_LIMIT = 64 * 1024 * 1024
#: Elapsed seconds the source read may take before it refuses.
_SOURCE_READ_DEADLINE_SECONDS = 30.0
#: Retained canonical trajectory bytes across the whole snapshot.
_RETAINED_SOURCE_BYTES_LIMIT = 64 * 1024 * 1024
SOURCE_READ_BUDGET_EXCEEDED = "evidence_source_read_budget_exceeded"
SOURCE_READ_DEADLINE_EXCEEDED = "evidence_source_read_deadline_exceeded"

_ABANDONED_TERMINAL_OUTCOME = "abandoned"

#: Take a job's free lease under a new fence.
_CLAIM_SQL = """UPDATE playbook_optimization_jobs
    SET lease_owner = ?, lease_fence = lease_fence + 1,
        lease_expires_at = ?, status = 'running', updated_at = ?
    WHERE job_id = ? AND status IN ('pending', 'running')
      AND lease_owner IS NULL
    RETURNING job_id, lease_owner, lease_fence, lease_expires_at"""

#: Take a job's expired lease under a new fence.
_RECLAIM_SQL = """UPDATE playbook_optimization_jobs
    SET lease_owner = ?, lease_fence = lease_fence + 1,
        lease_expires_at = ?, status = 'running', updated_at = ?
    WHERE job_id = ? AND status IN ('pending', 'running')
      AND lease_owner IS NOT NULL
      AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?
    RETURNING job_id, lease_owner, lease_fence, lease_expires_at"""

_ACTIVE_JOB_SQL = """SELECT * FROM playbook_optimization_jobs
    WHERE optimizer_kind = ? AND {column} = ?
      AND status IN ('pending', 'running')"""

_EXPOSURE_PAGE_SQL = """
SELECT exposure_event_id, request_id, session_id, user_id,
       playbook_owner_user_id, user_playbook_id,
       served_semantic_digest, served_full_version_fingerprint,
       exposed_at, ingested_at,
       governance_subject_ref, playbook_owner_governance_subject_ref
FROM user_playbook_exposure_events
WHERE substr(exposure_event_id, 1, length(:prefix)) = :prefix
  AND COALESCE(exposed_at, ingested_at) >= :window_start
  AND COALESCE(exposed_at, ingested_at) < :frozen_at
  AND ingested_at < :frozen_at
  -- An unstamped row is charged to this playbook's bound, never a sibling's:
  -- it is the integrity signal that refuses an incomplete population.
  AND (user_playbook_id = :playbook_id OR user_playbook_id IS NULL)
  AND (COALESCE(exposed_at, ingested_at), ingested_at, exposure_event_id)
      > (:last_effective_at, :last_ingested_at, :last_event_id)
ORDER BY COALESCE(exposed_at, ingested_at), ingested_at, exposure_event_id
LIMIT :limit
"""

_TRAJECTORY_SQL = """
WITH requested(user_id, session_id, through_request_id) AS (VALUES {values})
SELECT requests.request_id,
       requests.user_id AS request_user_id,
       requests.created_at AS request_created_at,
       requests.source AS request_source,
       requests.agent_version AS request_agent_version,
       requests.session_id AS request_session_id,
       requests.evaluation_only AS request_evaluation_only,
       requests.retrieval_experiment_id AS request_retrieval_experiment_id,
       requests.retrieval_experiment_arm AS request_retrieval_experiment_arm,
       requests.governance_subject_ref AS request_governance_subject_ref,
       interactions.interaction_id,
       interactions.user_id AS interaction_user_id,
       interactions.request_id AS interaction_request_id,
       interactions.created_at AS interaction_created_at,
       interactions.content AS interaction_content,
       interactions.role AS interaction_role,
       interactions.token_count AS interaction_token_count,
       interactions.user_action AS interaction_user_action,
       interactions.user_action_description AS interaction_user_action_description,
       interactions.interacted_image_url AS interaction_interacted_image_url,
       interactions.image_encoding AS interaction_image_encoding,
       interactions.shadow_content AS interaction_shadow_content,
       interactions.expert_content AS interaction_expert_content,
       interactions.tools_used AS interaction_tools_used,
       interactions.citations AS interaction_citations,
       interactions.retrieved_learnings AS interaction_retrieved_learnings,
       interactions.governance_subject_ref AS interaction_governance_subject_ref
FROM requests
JOIN requested
  ON requests.user_id = requested.user_id
 AND requests.session_id = requested.session_id
LEFT JOIN interactions ON interactions.request_id = requests.request_id
WHERE requested.through_request_id IS NULL OR {prefix}
ORDER BY requests.user_id, requests.session_id,
         requests.created_at, requests.request_id,
         interactions.created_at, interactions.interaction_id
"""

#: The tables whose rows carry the governance subjects an evidence window can
#: touch, and the subject columns of each.
_SUBJECT_COLUMNS = (
    ("requests", "governance_subject_ref"),
    ("interactions", "governance_subject_ref"),
    ("user_playbooks", "governance_subject_ref"),
    ("session_outcomes", "governance_subject_ref"),
    ("user_playbook_exposure_events", "governance_subject_ref"),
    ("user_playbook_exposure_events", "playbook_owner_governance_subject_ref"),
)


def _chunks(values: Sequence[Any]) -> Iterator[Sequence[Any]]:
    for offset in range(0, len(values), _KEY_CHUNK_SIZE):
        yield values[offset : offset + _KEY_CHUNK_SIZE]


def _placeholders(count: int, width: int = 1) -> str:
    group = "(" + ", ".join("?" * width) + ")" if width > 1 else "?"
    return ", ".join([group] * count)


def _ordered_keys(keys: set[SessionKey]) -> list[SessionKey]:
    return sorted(keys, key=lambda key: (key[0].encode(), key[1].encode()))


def _job_from_row(row: sqlite3.Row) -> PlaybookOptimizationJob:
    """A stored job, with its metadata as canonical JSON like Postgres returns."""
    try:
        job = _row_to_playbook_optimization_job(row)
        metadata = canonical_json_bytes(json.loads(job.metadata_json or "{}"))
        return job.model_copy(update={"metadata_json": metadata.decode()})
    except (KeyError, TypeError, ValueError) as exc:
        raise OptimizationArtifactIntegrityError(
            "open-world optimizer job row is malformed"
        ) from exc


class SQLiteOpenWorldEvidenceMixin:
    """``OpenWorldBundleStorage`` and the freeze-time reads on SQLite."""

    conn: sqlite3.Connection
    org_id: str
    _lock: Any
    _scope_depth: int
    _own_transaction: Any
    commit_scope: Any
    _active_subject_barrier_locked: Any
    _outcome_contract_digest: Any
    _advance_playbook_optimization_stage_locked: Any

    # ------------------------------------------------------------------
    # OpenWorldBundleStorage
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def freeze_open_world_evidence_bundle(
        self,
        *,
        job: PlaybookOptimizationJob,
        artifact_content_json: str,
        artifact_content_digest: str,
        target_full_version_fingerprint: str,
        subject_epochs: tuple[tuple[str, int], ...],
        owner: str,
        lease_seconds: int,
        now: int,
    ) -> OpenWorldEvidenceStorageRecord:
        """Validate, lease and persist one exact bundle in one transaction.

        The job is created or found, leased and given its bundle artifact in
        one ``commit_scope``, joining the caller's when there is one, so a
        failure leaves none of them behind. A stored
        artifact is never rewritten: a replay with other bytes refuses. A
        replay while another owner's lease is live returns the stored record
        with ``claim=None``.

        Raises:
            OpenWorldEvidenceTargetStaleError: The target is not the current
                version ``target_full_version_fingerprint`` names.
            OpenWorldEvidenceGovernanceError: A subject is not at epoch 0 or is
                under erasure, or the job names another tenant.
            OptimizationJobIdentityConflictError: The target's active job has
                another immutable identity and a live lease.
            OpenWorldEvidenceLeaseConflictError: Another owner holds the live
                lease and no artifact is stored yet.
            OptimizationArtifactIntegrityError: The request does not agree with
                its own metadata, or the bytes differ from the stored bundle.
        """
        if job.optimizer_kind != OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND:
            raise ValueError("open-world freeze requires the open-world optimizer kind")
        if lease_seconds <= 0 or not owner.strip():
            raise ValueError("open-world freeze requires an owner and positive lease")
        validate_subject_epoch_shape(subject_epochs)
        metadata_fingerprint, metadata_epochs = freeze_validation_identity(
            job, org_id=self.org_id
        )
        if (
            metadata_fingerprint != target_full_version_fingerprint
            or metadata_epochs != subject_epochs
        ):
            raise OptimizationArtifactIntegrityError(
                "open-world freeze request does not match durable metadata"
            )
        _payload, canonical_content = canonical_artifact_payload(artifact_content_json)
        if sha256(canonical_content.encode()).hexdigest() != artifact_content_digest:
            raise OptimizationArtifactIntegrityError(
                "open-world artifact content/digest mismatch"
            )

        # A nested scope marks the caller's transaction rollback-only on
        # failure, so a caught freeze error cannot commit a job without its bundle.
        with self._lock, self.commit_scope():
            self._validate_evidence_target_locked(
                job.target_id, expected_fingerprint=target_full_version_fingerprint
            )
            self._validate_evidence_governance_locked(subject_epochs)
            persisted = self._retake_stranded_job_locked(
                self._create_or_get_evidence_job_locked(job),
                request=job,
                owner=owner,
                lease_seconds=lease_seconds,
                now=now,
            )
            validate_freeze_job_request(persisted, job)
            artifact = self._evidence_artifact_locked(persisted.job_id)
            if artifact is not None:
                validate_artifact_request(
                    artifact,
                    content_json=canonical_content,
                    content_digest=artifact_content_digest,
                )
            claim = self._claim_for_freeze_locked(
                persisted,
                artifact_exists=artifact is not None,
                owner=owner,
                lease_seconds=lease_seconds,
                now=now,
            )
            if artifact is None:
                if claim is None:
                    raise OpenWorldEvidenceLeaseConflictError(
                        "open-world artifact has no writable lease"
                    )
                artifact = self._insert_evidence_artifact_locked(
                    persisted.job_id,
                    content_json=canonical_content,
                    content_digest=artifact_content_digest,
                    now=now,
                )
            current = self._evidence_job_locked(persisted.job_id)
            if current is None:
                raise OptimizationArtifactIntegrityError(
                    "frozen optimizer job disappeared"
                )
        return OpenWorldEvidenceStorageRecord(
            job=current, artifact=artifact, claim=claim
        )

    @SQLiteStorageBase.handle_exceptions
    def load_open_world_evidence_bundle(
        self, job_id: int
    ) -> OpenWorldEvidenceStorageRecord:
        """Load the exact frozen bytes after revalidating target and governance.

        A terminal job loads too; the record never carries a claim.

        Raises:
            OpenWorldEvidenceTargetStaleError: The target moved since the freeze.
            OpenWorldEvidenceGovernanceError: A frozen subject is now under
                erasure.
            OptimizationArtifactIntegrityError: The job or its bundle is missing
                or malformed.
        """
        with self._lock:
            owns_transaction = self._own_transaction()
            if owns_transaction:
                self.conn.execute("BEGIN")
            try:
                job = self._evidence_job_locked(job_id)
                if job is None:
                    raise OptimizationArtifactIntegrityError(
                        "frozen optimizer job does not exist"
                    )
                fingerprint, subject_epochs = freeze_validation_identity(
                    job, org_id=self.org_id
                )
                self._validate_evidence_target_locked(
                    job.target_id, expected_fingerprint=fingerprint
                )
                self._validate_evidence_governance_locked(subject_epochs)
                artifact = self._evidence_artifact_locked(job_id)
                if artifact is None:
                    raise OptimizationArtifactIntegrityError(
                        "open-world evidence artifact missing"
                    )
            finally:
                if owns_transaction:
                    self.conn.rollback()
        return OpenWorldEvidenceStorageRecord(job=job, artifact=artifact)

    # ------------------------------------------------------------------
    # OpenWorldFreezeReads
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def list_open_world_governance_epochs(self) -> tuple[tuple[str, int], ...]:
        """``(subject_ref, 0)`` for every subject the stored evidence carries.

        SQLite keeps no erasure epochs, so every subject is at epoch 0. The
        vector covers every subject on a request, interaction, playbook,
        session outcome or exposure -- every row an evidence window can
        touch -- ordered by the subject's UTF-8 bytes. Subjects under an
        erasure barrier are listed too; the snapshot marks their events
        invalid.
        """
        union = " UNION ".join(
            f"SELECT {column} AS subject_ref FROM {table}"  # noqa: S608
            f" WHERE {column} IS NOT NULL"
            for table, column in _SUBJECT_COLUMNS
        )
        with self._lock:
            rows = self.conn.execute(union).fetchall()
        refs = {
            subject_ref
            for row in rows
            if (subject_ref := text_or_none(row["subject_ref"])) is not None
        }
        return tuple((ref, 0) for ref in sorted(refs, key=lambda ref: ref.encode()))

    def list_playbook_deployment_lifecycles(
        self, *, user_playbook_ids: Sequence[int]
    ) -> tuple[DeploymentLifecycleRow, ...]:
        """Every lifecycle naming one of these playbooks, at either end.

        Written by the open-world provisional commit. Ascending by
        ``(published_at, deployment_lifecycle_id)``, as PostgreSQL returns
        them; a playbook the tuner never published appears in none.
        """
        ids = [int(value) for value in user_playbook_ids]
        if not ids:
            return ()
        placeholders = ", ".join("?" * len(ids))
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM user_playbook_deployment_lifecycles"  # noqa: S608
                f" WHERE successor_id IN ({placeholders})"
                f" OR predecessor_id IN ({placeholders})"
                " ORDER BY published_at, deployment_lifecycle_id",
                (*ids, *ids),
            ).fetchall()
        return tuple(lifecycle_row(row) for row in rows)

    def load_open_world_evidence_snapshot(
        self,
        *,
        target: OpenWorldTargetIdentity,
        frozen_at: int,
        limits: OpenWorldEvidenceReadLimits,
    ) -> OpenWorldEvidenceSnapshot:
        """Read one complete half-open evidence window in one read transaction.

        Query, then pure step, then query: this method owns the transaction and
        every statement, and hands the rows between them to
        ``open_world_snapshot_assembly``, the functions the Postgres reader
        uses, which decide what the rows mean.

        Raises:
            OpenWorldEvidenceIntegrityError: ``target`` names another tenant.
            OpenWorldEvidenceTargetStaleError: The target is not the current
                version the target identity names.
            StorageError: The source read exhausted its byte budget or its
                deadline. No partial snapshot is returned.
            ValueError: ``frozen_at`` disagrees with the target, or the call is
                inside a ``commit_scope``.
        """
        if target.tenant_id != self.org_id:
            raise OpenWorldEvidenceIntegrityError(
                "target tenant does not match storage tenant"
            )
        if frozen_at != target.frozen_at:
            raise ValueError("frozen boundary does not match target identity")
        window_start = frozen_at - limits.historical_window_seconds
        with self._lock:
            if self._scope_depth > 0:
                raise ValueError(
                    "open-world evidence snapshot requires its own transaction"
                )
            # One read transaction: under WAL its snapshot is taken at the
            # first read, so a later commit on another connection is invisible
            # to every statement below.
            self.conn.execute("BEGIN")
            try:
                owner_subject_ref = self._validate_evidence_target_locked(
                    target.user_playbook_id,
                    expected_fingerprint=target.incumbent_full_version_fingerprint,
                )
                exposure_rows = self._evidence_exposure_rows_locked(
                    user_playbook_id=target.user_playbook_id,
                    window_start=window_start,
                    frozen_at=frozen_at,
                    maximum_population=limits.maximum_population,
                )
                if exposure_rows is None:
                    return OpenWorldEvidenceSnapshot(
                        tenant_id=target.tenant_id, population_limit_exceeded=True
                    )
                return self._build_evidence_snapshot_locked(
                    target=target,
                    exposure_rows=exposure_rows,
                    target_owner_subject_ref=owner_subject_ref,
                    max_trajectory_bytes=limits.max_trajectory_bytes,
                )
            finally:
                self.conn.rollback()

    # ------------------------------------------------------------------
    # Freeze helpers; every one runs under the lock, inside a transaction
    # ------------------------------------------------------------------

    def _validate_evidence_target_locked(
        self, user_playbook_id: int, *, expected_fingerprint: str
    ) -> str | None:
        """Refuse unless the target is current at ``expected_fingerprint``.

        Returns:
            str | None: The target owner's governance subject ref.
        """
        row = self.conn.execute(
            "SELECT * FROM user_playbooks"
            " WHERE user_playbook_id = ? AND status IS NULL",
            (user_playbook_id,),
        ).fetchone()
        if row is None:
            raise OpenWorldEvidenceTargetStaleError(
                "current target user playbook is missing or not current"
            )
        try:
            fingerprint = user_playbook_full_version_fingerprint(
                _row_to_user_playbook(row)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenWorldEvidenceTargetStaleError(
                "current target user playbook is invalid"
            ) from exc
        if fingerprint != expected_fingerprint:
            raise OpenWorldEvidenceTargetStaleError(
                "current target user playbook full version is stale"
            )
        return text_or_none(row["governance_subject_ref"])

    def _validate_evidence_governance_locked(
        self, subject_epochs: tuple[tuple[str, int], ...]
    ) -> None:
        """Refuse a subject not at SQLite's only epoch, or one under erasure."""
        if any(epoch != 0 for _subject, epoch in subject_epochs):
            raise OpenWorldEvidenceGovernanceError(
                "open-world evidence subject vector is not writable"
            )
        for subject_ref, _epoch in subject_epochs:
            if self._active_subject_barrier_locked(subject_ref) is not None:
                raise OpenWorldEvidenceGovernanceError(
                    "open-world evidence subject is under erasure"
                )

    def _evidence_job_locked(self, job_id: int) -> PlaybookOptimizationJob | None:
        row = self.conn.execute(
            "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return None if row is None else _job_from_row(row)

    def _create_or_get_evidence_job_locked(
        self, request: PlaybookOptimizationJob
    ) -> PlaybookOptimizationJob:
        """The active job for this identity or target, else a new one.

        The lookups and their order are the Postgres
        ``create_or_get_playbook_optimization_job``: by discovery key, by
        attempt key, then by target. The target fallback is what lets the
        identity check refuse a second attempt on a playbook that is already
        being tuned.
        """
        by_key: dict[str, sqlite3.Row | None] = {}
        for column in ("discovery_key", "attempt_key"):
            value = getattr(request, column)
            by_key[column] = (
                None
                if not value
                else self.conn.execute(
                    _ACTIVE_JOB_SQL.format(column=column),
                    (request.optimizer_kind, value),
                ).fetchone()
            )
        by_discovery, by_attempt = by_key["discovery_key"], by_key["attempt_key"]
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
                existing["target_kind"] != request.target_kind
                or existing["target_id"] != request.target_id
                or (
                    by_discovery is not None
                    and existing["attempt_key"] != request.attempt_key
                )
            ):
                raise OptimizationJobIdentityConflictError(
                    "conflicting immutable optimizer job identity"
                )
            return _job_from_row(existing)
        by_target = self.conn.execute(
            """SELECT * FROM playbook_optimization_jobs
               WHERE optimizer_kind = ? AND target_kind = ? AND target_id = ?
                 AND status IN ('pending', 'running')""",
            (request.optimizer_kind, request.target_kind, request.target_id),
        ).fetchone()
        if by_target is not None:
            return _job_from_row(by_target)
        try:
            metadata = canonical_json_bytes(json.loads(request.metadata_json or "{}"))
        except (TypeError, ValueError) as exc:
            raise OptimizationJobIdentityConflictError(
                "open-world optimizer job metadata is not canonicalizable"
            ) from exc
        stored = request.model_copy(
            update={
                "status": "pending",
                "metadata_json": metadata.decode(),
                "lease_owner": None,
                "lease_fence": 0,
                "lease_expires_at": None,
            }
        )
        cursor = self.conn.execute(_JOB_INSERT_SQL, _job_insert_values(stored))
        job = self._evidence_job_locked(int(cursor.lastrowid or 0))
        if job is None:
            raise OptimizationArtifactIntegrityError(
                "optimizer job create-or-get returned no row"
            )
        return job

    def _retake_stranded_job_locked(
        self,
        persisted: PlaybookOptimizationJob,
        *,
        request: PlaybookOptimizationJob,
        owner: str,
        lease_seconds: int,
        now: int,
    ) -> PlaybookOptimizationJob:
        """Terminalize an expired job of another identity, then create anew.

        A worker that dies mid-attempt leaves its job active. The next day's
        freeze carries new discovery and attempt keys, so the target lookup
        returns that stranded job and identity validation would refuse it
        forever. When the stranded lease can be taken -- it is free or has
        expired -- the job is terminalized as ``failed``/``abandoned`` and
        the freeze creates its own. A live lease is never taken: the lease
        predicates refuse it, and the identity conflict stands.

        Returns:
            PlaybookOptimizationJob: ``persisted`` when its identity agrees,
            otherwise the newly created job.

        Raises:
            OptimizationJobIdentityConflictError: The stranded job still holds
                a live lease, or could not be terminalized.
        """
        try:
            validate_freeze_job_request(persisted, request)
        except OptimizationJobIdentityConflictError:
            pass
        else:
            return persisted
        expires_at = now + lease_seconds
        if persisted.lease_owner is None:
            row = self.conn.execute(
                _CLAIM_SQL, (owner, expires_at, now, persisted.job_id)
            ).fetchone()
        else:
            row = self.conn.execute(
                _RECLAIM_SQL, (owner, expires_at, now, persisted.job_id, now)
            ).fetchone()
        if row is None:
            raise OptimizationJobIdentityConflictError(
                "active target has a different immutable open-world identity "
                "and its lease could not be retaken"
            )
        if not self._advance_playbook_optimization_stage_locked(
            persisted.job_id,
            int(row["lease_fence"]),
            "failed",
            terminal_outcome=_ABANDONED_TERMINAL_OUTCOME,
            advanced_at=now,
        ):
            raise OptimizationJobIdentityConflictError(
                "active target has a different immutable open-world identity "
                "and the abandoned attempt could not be terminalized"
            )
        logger.warning(
            "event=open_world_freeze_retook_stranded_attempt job_id=%s "
            "target_id=%s terminal_outcome=%s",
            persisted.job_id,
            persisted.target_id,
            _ABANDONED_TERMINAL_OUTCOME,
        )
        return self._create_or_get_evidence_job_locked(request)

    def _claim_for_freeze_locked(
        self,
        job: PlaybookOptimizationJob,
        *,
        artifact_exists: bool,
        owner: str,
        lease_seconds: int,
        now: int,
    ) -> OptimizationJobClaim | None:
        """Claim, reclaim or renew the freeze's lease, as Postgres decides it.

        ``None`` only when another owner's lease is live and the bundle is
        already stored: the replay reads the record without a lease.
        """
        expires_at = now + lease_seconds
        if job.lease_owner is None:
            statement = _CLAIM_SQL
            params: tuple[object, ...] = (owner, expires_at, now, job.job_id)
        elif job.lease_expires_at is not None and job.lease_expires_at <= now:
            statement = _RECLAIM_SQL
            params = (owner, expires_at, now, job.job_id, now)
        elif job.lease_owner == owner:
            statement = """UPDATE playbook_optimization_jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE job_id = ? AND status IN ('pending', 'running')
                  AND lease_owner = ? AND lease_fence = ? AND lease_expires_at > ?
                RETURNING job_id, lease_owner, lease_fence, lease_expires_at"""
            params = (expires_at, now, job.job_id, owner, job.lease_fence, now)
        elif artifact_exists:
            return None
        else:
            raise OpenWorldEvidenceLeaseConflictError(
                "active target has a live open-world evidence lease"
            )
        row = self.conn.execute(statement, params).fetchone()
        if row is None:
            raise OpenWorldEvidenceLeaseConflictError(
                "open-world evidence lease is unavailable"
            )
        return OptimizationJobClaim(
            job_id=int(row["job_id"]),
            owner=str(row["lease_owner"]),
            fence=int(row["lease_fence"]),
            expires_at=int(row["lease_expires_at"]),
        )

    def _evidence_artifact_locked(
        self, job_id: int
    ) -> PlaybookOptimizationArtifact | None:
        row = self.conn.execute(
            """SELECT * FROM playbook_optimization_artifacts
               WHERE job_id = ? AND artifact_kind = ?""",
            (job_id, OPEN_WORLD_EVIDENCE_ARTIFACT_KIND),
        ).fetchone()
        return None if row is None else _row_to_playbook_optimization_artifact(row)

    def _insert_evidence_artifact_locked(
        self, job_id: int, *, content_json: str, content_digest: str, now: int
    ) -> PlaybookOptimizationArtifact:
        """Write the bundle once. There is no update path: a stored one is final."""
        cursor = self.conn.execute(
            """INSERT INTO playbook_optimization_artifacts
               (job_id, artifact_kind, content_json, content_digest,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                OPEN_WORLD_EVIDENCE_ARTIFACT_KIND,
                content_json,
                content_digest,
                now,
                now,
            ),
        )
        row = self.conn.execute(
            "SELECT * FROM playbook_optimization_artifacts WHERE artifact_id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        if row is None:
            raise OptimizationArtifactIntegrityError(
                "open-world artifact insert returned no row"
            )
        return _row_to_playbook_optimization_artifact(row)

    # ------------------------------------------------------------------
    # Snapshot reads; every one runs inside the snapshot's read transaction
    # ------------------------------------------------------------------

    def _evidence_exposure_rows_locked(
        self,
        *,
        user_playbook_id: int,
        window_start: int,
        frozen_at: int,
        maximum_population: int,
    ) -> list[dict[str, Any]] | None:
        """This playbook's published exposures, or ``None`` above the bound."""
        rows: list[dict[str, Any]] = []
        params: dict[str, object] = {
            "prefix": PUBLISHED_EXPOSURE_PREFIX,
            "window_start": window_start,
            "frozen_at": frozen_at,
            "playbook_id": user_playbook_id,
            "last_effective_at": window_start - 1,
            "last_ingested_at": -1,
            "last_event_id": "",
        }
        while True:
            limit = min(_EXPOSURE_PAGE_SIZE, maximum_population - len(rows) + 1)
            page = [
                dict(row)
                for row in self.conn.execute(
                    _EXPOSURE_PAGE_SQL, params | {"limit": limit}
                ).fetchall()
            ]
            if len(rows) + len(page) > maximum_population:
                return None
            rows.extend(page)
            if len(page) < limit:
                return rows
            last = page[-1]
            params["last_effective_at"] = int(
                last["exposed_at"]
                if last["exposed_at"] is not None
                else last["ingested_at"]
            )
            params["last_ingested_at"] = int(last["ingested_at"])
            params["last_event_id"] = str(last["exposure_event_id"])

    def _build_evidence_snapshot_locked(
        self,
        *,
        target: OpenWorldTargetIdentity,
        exposure_rows: list[dict[str, Any]],
        target_owner_subject_ref: str | None,
        max_trajectory_bytes: int,
    ) -> OpenWorldEvidenceSnapshot:
        """Read each dependency of the exposure rows, then assemble the snapshot.

        SQLite has no retrieval log, so no session-only exposure binds.
        """
        bound_requests: dict[str, str] = {}
        resolved_rows = resolve_exposure_correlations(
            exposure_rows,
            bound_requests=bound_requests,
            retained_requests=self._retained_requests_locked(
                correlated_request_ids(exposure_rows, bound_requests)
            ),
        )
        playbooks = self._snapshot_playbooks_locked(resolved_rows)
        outcomes = self._snapshot_outcomes_locked(session_keys(resolved_rows))
        contracts = {
            digest
            for row in outcomes.values()
            if (digest := digest_or_none(row.get("outcome_contract_digest")))
            is not None
            and isinstance(row.get("source"), str)
            and digest == self._outcome_contract_digest(row["source"])
        }
        cutovers = trajectory_cutovers(resolved_rows, outcomes)
        trajectories = self._snapshot_trajectories_locked(
            cutovers, max_trajectory_bytes=max_trajectory_bytes
        )
        dependencies = event_dependencies(
            resolved_rows,
            outcomes=outcomes,
            trajectories=trajectories,
            target_owner_subject_ref=target_owner_subject_ref,
        )
        subject_refs = governance_subject_refs_to_read(target, dependencies)
        return assemble_snapshot(
            dependencies,
            target=target,
            playbooks=playbooks,
            contracts=contracts,
            cutovers=cutovers,
            in_prefix=self._prefix_membership_locked(
                prefix_pairs(resolved_rows, cutovers)
            ),
            governance_subjects=dict.fromkeys(subject_refs, (0, True)),
            blocked_subjects=self._blocked_subjects_locked(subject_refs),
        )

    def _retained_requests_locked(
        self, request_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        retained: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(request_ids):
            for row in self.conn.execute(
                "SELECT request_id, user_id, session_id FROM requests"  # noqa: S608
                f" WHERE request_id IN ({_placeholders(len(chunk))})",
                tuple(chunk),
            ):
                retained[str(row["request_id"])] = dict(row)
        return retained

    def _snapshot_playbooks_locked(
        self, rows: list[dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        playbook_ids = sorted(
            {
                playbook_id
                for row in rows
                if (playbook_id := positive_int_or_none(row.get("user_playbook_id")))
                is not None
            }
        )
        playbooks: dict[int, dict[str, Any]] = {}
        for chunk in _chunks(playbook_ids):
            for row in self.conn.execute(
                "SELECT user_playbook_id, user_id, content, trigger"  # noqa: S608
                " FROM user_playbooks"
                f" WHERE user_playbook_id IN ({_placeholders(len(chunk))})",
                tuple(chunk),
            ):
                playbooks[int(row["user_playbook_id"])] = dict(row)
        return playbooks

    def _snapshot_outcomes_locked(
        self, keys: set[SessionKey]
    ) -> dict[SessionKey, dict[str, Any]]:
        outcomes: dict[SessionKey, dict[str, Any]] = {}
        for chunk in _chunks(_ordered_keys(keys)):
            sql = (
                "WITH requested(user_id, session_id) AS"  # noqa: S608
                f" (VALUES {_placeholders(len(chunk), 2)})"
                " SELECT outcomes.outcome_id, outcomes.outcome_revision,"
                " outcomes.user_id, outcomes.session_id, outcomes.outcome,"
                " outcomes.outcome_contract_digest,"
                " outcomes.finalized_trajectory_digest,"
                " outcomes.governance_subject_ref, outcomes.is_inferred,"
                " outcomes.trajectory_through_request_id, outcomes.source"
                " FROM session_outcomes AS outcomes"
                " JOIN requested ON outcomes.user_id = requested.user_id"
                " AND outcomes.session_id = requested.session_id"
            )
            params = tuple(value for key in chunk for value in key)
            for row in self.conn.execute(sql, params):
                outcomes[(str(row["user_id"]), str(row["session_id"]))] = dict(row)
        return outcomes

    def _snapshot_trajectories_locked(
        self,
        cutovers: Mapping[SessionKey, str | None],
        *,
        max_trajectory_bytes: int,
    ) -> dict[SessionKey, LoadedTrajectory]:
        """Stream each session's whole source, through its cutover when it has one.

        No session is skipped for its length: every joined row is read and
        folded into ``assemble_trajectories``, whose per-source byte cap drops
        content mid-stream. What bounds the read is the serialized bytes of
        every row consumed and the elapsed time; exhausting either raises, and
        no partial source is returned as reconstructed.
        """
        loaded = dict.fromkeys(cutovers, ABSENT_TRAJECTORY)
        deadline = monotonic() + _SOURCE_READ_DEADLINE_SECONDS
        consumed_bytes = 0

        def bounded_rows(cursor: sqlite3.Cursor) -> Iterator[dict[str, Any]]:
            nonlocal consumed_bytes
            while rows := cursor.fetchmany(_SOURCE_FETCH_SIZE):
                for raw_row in rows:
                    if monotonic() >= deadline:
                        raise StorageError(SOURCE_READ_DEADLINE_EXCEEDED)
                    row = dict(raw_row)
                    consumed_bytes += len(
                        json.dumps(row, ensure_ascii=False, default=str).encode()
                    )
                    if consumed_bytes > _SOURCE_READ_BYTES_LIMIT:
                        raise StorageError(SOURCE_READ_BUDGET_EXCEEDED)
                    yield row

        for chunk in _chunks(_ordered_keys(set(cutovers))):
            sql = _TRAJECTORY_SQL.format(
                values=_placeholders(len(chunk), 3),
                prefix=_through_prefix_predicate(
                    "requests", cutover="requested.through_request_id"
                ),
            )
            params = tuple(
                value for key in chunk for value in (key[0], key[1], cutovers[key])
            )
            cursor = self.conn.execute(sql, params)
            try:
                for key, trajectory in assemble_trajectories(
                    bounded_rows(cursor), max_trajectory_bytes=max_trajectory_bytes
                ):
                    if monotonic() >= deadline:
                        raise StorageError(SOURCE_READ_DEADLINE_EXCEEDED)
                    loaded[key] = trajectory
                    if _retained_bytes(loaded.values()) > _RETAINED_SOURCE_BYTES_LIMIT:
                        raise StorageError(
                            "evidence snapshot exceeds the 64-MiB source read budget"
                        )
            finally:
                cursor.close()
        return loaded

    def _prefix_membership_locked(
        self, pairs: list[tuple[str, str]]
    ) -> set[tuple[str, str]]:
        """``(request, cutover)`` pairs whose request is in the cutover's prefix."""
        members: set[tuple[str, str]] = set()
        for chunk in _chunks(pairs):
            sql = (
                "WITH pair(request_id, through_request_id) AS"  # noqa: S608
                f" (VALUES {_placeholders(len(chunk), 2)})"
                " SELECT pair.request_id, pair.through_request_id FROM pair"
                " JOIN requests AS req ON req.request_id = pair.request_id"
                " WHERE "
                + _through_prefix_predicate("req", cutover="pair.through_request_id")
            )
            params = tuple(value for pair in chunk for value in pair)
            members.update(
                (str(row["request_id"]), str(row["through_request_id"]))
                for row in self.conn.execute(sql, params)
            )
        return members

    def _blocked_subjects_locked(self, subject_refs: set[str]) -> set[str]:
        blocked: set[str] = set()
        for chunk in _chunks(sorted(subject_refs)):
            blocked.update(
                str(row["subject_ref"])
                for row in self.conn.execute(
                    "SELECT subject_ref FROM subject_write_barriers"  # noqa: S608
                    " WHERE org_id = ? AND status IN ('erasing', 'erased')"
                    f" AND subject_ref IN ({_placeholders(len(chunk))})",
                    (self.org_id, *chunk),
                )
            )
        return blocked


def _retained_bytes(trajectories: Iterable[LoadedTrajectory]) -> int:
    return sum(
        trajectory.byte_count or 0
        for trajectory in trajectories
        if trajectory.trajectory is not None
    )


__all__ = [
    "SOURCE_READ_BUDGET_EXCEEDED",
    "SOURCE_READ_DEADLINE_EXCEEDED",
    "SQLiteOpenWorldEvidenceMixin",
]
