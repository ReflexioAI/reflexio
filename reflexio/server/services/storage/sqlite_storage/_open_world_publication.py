"""Open-world provisional publication on SQLite.

The ``OpenWorldProvisionalPublicationStore`` member of ``OpenWorldStoragePort``:
claim, stage, commit and clean up one provisional publication, load its
committed result, and recover a commit whose response was lost.

Every call runs in one ``BEGIN IMMEDIATE`` transaction (``commit_scope``) and
first rechecks what authorizes it: the job's live lease and fence, the
publication claim, the incumbent's exact full version, the frozen governance
subjects, the qualification authority and the durable attempt artifacts. The
commit re-runs the stage checks and then the decision-proof check before it
writes; the successor, the superseded incumbent, the deployment lifecycle,
the lineage and optimizer events, the result and the job's ``applied``
transition commit together or not at all. A commit that finds the incumbent
drifted cleans its own uncommitted residue, writes no publication, and raises
``OpenWorldPublicationIncumbentChangedError``; a commit replayed after it
committed returns the committed result.

The semantics mirror the PostgreSQL implementation of the same port, whose
checks run in the tenant functions ``claim_``/``stage_``/``commit_user_playbook_
provisional_publication``, ``cleanup_open_world_provisional_stale_attempt``,
``load_open_world_provisional_publication_retry``,
``assert_open_world_provisional_authority`` and
``assert_open_world_provisional_decision_proof``. This module restates them in
Python with the same messages; where PostgreSQL attaches an
``open_world_refusal`` code, SQLite raises the one class that carries it. The
shared contract (``reflexio.test_support.open_world_publication_contract``)
holds both to the same answers. What SQLite answers differently, and why:

* **Governance.** SQLite keeps no subject registry. The job's frozen subject
  vector (every subject at epoch 0) is the governance vector, and a subject is
  writable unless an erasure barrier holds it.
* **Projection columns.** PostgreSQL keeps the successor's lexical document and
  projection on ``user_playbooks``; SQLite writes the lexical document to
  ``user_playbooks_fts`` and the projection digest only to the lifecycle, so a
  lifecycle's ``predecessor_projection_digest`` is the projection digest of the
  lifecycle that published the incumbent (NULL for a hand-authored one).
* **No aggregation outbox.** The provisional lifecycle is written terminal
  (``confirmed``, ``published_unconditional``, ``not_required``), as on
  PostgreSQL, whose outbox trigger fires only for an ``applied`` result.
* **Projects.** A SQLite database is one project; rows carry no ``project_id``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    RULE_EVIDENCE_POLICY_ID,
)
from reflexio.server.services.offline_tuner.open_world_freeze_validation import (
    freeze_validation_identity,
)
from reflexio.server.services.offline_tuner.open_world_invocations import (
    OpenWorldRefusal,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND,
    OPEN_WORLD_CANDIDATE_ARTIFACT_KIND,
    OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND,
    OPEN_WORLD_EVIDENCE_ARTIFACT_KIND,
    OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND,
    OpenWorldEvidenceGovernanceError,
    OpenWorldPublicationIncumbentChangedError,
    OpenWorldPublicationRetryIdentityMismatchError,
)
from reflexio.server.services.playbook.publication import (
    LIFECYCLE_TERMINAL_REASONS,
    LIFECYCLE_TERMINAL_STATES,
    ProvisionalPublicationRequest,
    ProvisionalPublicationResult,
    PublicationClaim,
    canonical_json_bytes,
    incumbent_user_playbook_semantic_digest,
)
from reflexio.server.services.search_exposure import (
    user_playbook_full_version_fingerprint,
)
from reflexio.server.services.storage.error import StorageError

from . import _open_world_rule_evidence as rule
from ._base import (
    SQLiteStorageBase,
    _epoch_to_iso,
    _row_to_user_playbook,
)
from ._lineage import _append_event_stmt

_OPTIMIZER = OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND
_PUBLICATION_ACTOR = OPEN_WORLD_EVIDENCE_OPTIMIZER_KIND
_PROOF_SCHEMA_VERSION = "offline-tuner-rule-evidence-publication-proof-v1"
_FULL_VERSION_SCHEMA = "user-playbook-full-version-v1"
_SEMANTIC_SCHEMA = "user-playbook-incumbent-semantic-v1"
#: Fourteen days, the PostgreSQL literal both the result's and the lifecycle's
#: ``observation_deadline`` are ``published_at`` plus.
OBSERVATION_SECONDS = 1_209_600
#: The terminal tuple a provisional commit writes: nothing observes a SQLite
#: successor online, so it is published unconditionally, as on PostgreSQL.
_PUBLISHED_STATE = "confirmed"
_PUBLISHED_REASON = "published_unconditional"


def _hex64(column: str) -> str:
    return f"(length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*')"


def _literals(values: frozenset[str]) -> str:
    return ", ".join(f"'{value}'" for value in sorted(values))


#: ``user_playbook_deployment_lifecycles`` as PostgreSQL declares it, minus
#: ``project_id`` and the restoration fence columns (SQLite has no restoration
#: path). The state and reason vocabularies are the Python ones the
#: publication module derives, so the CHECKs cannot drift from them.
LIFECYCLE_DDL = f"""
CREATE TABLE IF NOT EXISTS user_playbook_deployment_lifecycles (
    deployment_lifecycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL UNIQUE
        REFERENCES playbook_optimization_jobs(job_id) ON DELETE CASCADE,
    state TEXT NOT NULL DEFAULT 'confirmed'
        CHECK (state IN ({_literals(LIFECYCLE_TERMINAL_STATES | {"provisional"})})
               AND state != 'provisional'),
    predecessor_id INTEGER NOT NULL CHECK (predecessor_id > 0),
    successor_id INTEGER NOT NULL CHECK (successor_id > 0),
    predecessor_full_fingerprint TEXT NOT NULL
        CHECK {_hex64("predecessor_full_fingerprint")},
    successor_full_fingerprint TEXT NOT NULL
        CHECK {_hex64("successor_full_fingerprint")},
    predecessor_projection_digest TEXT CHECK (
        predecessor_projection_digest IS NULL
        OR {_hex64("predecessor_projection_digest")}
    ),
    successor_projection_digest TEXT NOT NULL
        CHECK {_hex64("successor_projection_digest")},
    evidence_digest TEXT NOT NULL CHECK {_hex64("evidence_digest")},
    candidate_digest TEXT NOT NULL CHECK {_hex64("candidate_digest")},
    proof_digest TEXT NOT NULL CHECK {_hex64("proof_digest")},
    qualification_authority_epoch INTEGER NOT NULL
        CHECK (qualification_authority_epoch > 0),
    qualification_authority_digest TEXT NOT NULL
        CHECK {_hex64("qualification_authority_digest")},
    governance_epochs_json TEXT NOT NULL CHECK (trim(governance_epochs_json) != ''),
    published_at INTEGER NOT NULL CHECK (published_at >= 0),
    observation_deadline INTEGER NOT NULL,
    terminal_reason TEXT CHECK (
        terminal_reason IS NULL OR terminal_reason IN (
            {_literals(LIFECYCLE_TERMINAL_REASONS | {_PUBLISHED_REASON})}
        )
    ),
    terminal_at INTEGER,
    aggregation_obligation TEXT NOT NULL DEFAULT 'pending_confirmation'
        CHECK (aggregation_obligation IN
               ('pending_confirmation', 'enqueued', 'not_required')),
    CHECK (predecessor_id != successor_id),
    CHECK (observation_deadline = published_at + {OBSERVATION_SECONDS}),
    CHECK (
        (state = 'provisional' AND terminal_reason IS NULL AND terminal_at IS NULL)
        OR (state != 'provisional' AND trim(terminal_reason) != ''
            AND terminal_at >= published_at)
    )
);
CREATE INDEX IF NOT EXISTS idx_updl_predecessor
    ON user_playbook_deployment_lifecycles(predecessor_id);
CREATE INDEX IF NOT EXISTS idx_updl_successor
    ON user_playbook_deployment_lifecycles(successor_id);
"""


def migrate_open_world_publication(conn: sqlite3.Connection) -> None:
    """Create the deployment lifecycle table when absent. Idempotent.

    The publication claim, staging and result tables are base tables; their
    provisional columns are added by ``SQLiteStorageBase.migrate``.

    Args:
        conn (sqlite3.Connection): An open SQLite connection.
    """
    with conn:
        conn.executescript("BEGIN;\n" + LIFECYCLE_DDL + "\nCOMMIT;")


@dataclass(frozen=True)
class DeploymentLifecycleRow:
    """One stored lifecycle, as ``list_playbook_deployment_lifecycles`` returns it.

    Satisfies the port's ``PublishedLifecycle`` (``successor_id``,
    ``published_at``) and carries the rest of the identity a caller may read.
    """

    deployment_lifecycle_id: int
    job_id: int
    state: str
    predecessor_id: int
    successor_id: int
    published_at: int
    observation_deadline: int
    terminal_reason: str | None
    terminal_at: int | None
    successor_full_fingerprint: str


def lifecycle_row(row: sqlite3.Row) -> DeploymentLifecycleRow:
    return DeploymentLifecycleRow(
        deployment_lifecycle_id=int(row["deployment_lifecycle_id"]),
        job_id=int(row["job_id"]),
        state=str(row["state"]),
        predecessor_id=int(row["predecessor_id"]),
        successor_id=int(row["successor_id"]),
        published_at=int(row["published_at"]),
        observation_deadline=int(row["observation_deadline"]),
        terminal_reason=row["terminal_reason"],
        terminal_at=None if row["terminal_at"] is None else int(row["terminal_at"]),
        successor_full_fingerprint=str(row["successor_full_fingerprint"]),
    )


# ---------------------------------------------------------------------------
# The request as storage binds it
# ---------------------------------------------------------------------------


def _authority_payload(request: ProvisionalPublicationRequest) -> dict[str, Any]:
    authority = request.qualification_authority
    payload: dict[str, Any] = {
        "epoch": authority.epoch,
        "authority_digest": authority.authority_digest,
        "discovery_component_identity_digest": (
            authority.discovery_component_identity_digest
        ),
        "discovery_qualification_suite_digest": (
            authority.discovery_qualification_suite_digest
        ),
        "discovery_qualification_result_digest": (
            authority.discovery_qualification_result_digest
        ),
        "held_out_component_identity_digest": (
            authority.held_out_component_identity_digest
        ),
        "held_out_qualification_suite_digest": (
            authority.held_out_qualification_suite_digest
        ),
        "held_out_qualification_result_digest": (
            authority.held_out_qualification_result_digest
        ),
        "candidate_generator_identity_digest": (
            authority.candidate_generator_identity_digest
        ),
        "candidate_generator_authorization_digest": (
            authority.candidate_generator_authorization_digest
        ),
    }
    if authority.policy_id == RULE_EVIDENCE_POLICY_ID:
        payload["policy_id"] = authority.policy_id
    return payload


def staging_digest(request: ProvisionalPublicationRequest) -> str:
    """The digest of the request's immutable fields.

    Everything except the claim owner and the two fences, which a reclaim may
    move: the identity a stage replay and a committed result are matched on.
    """
    immutable = {
        "attempt_key": request.attempt_key,
        "candidate_digest": request.candidate_digest,
        "content_digest": request.projection.candidate_content_digest,
        "evidence_bundle_digest": request.evidence_bundle_digest,
        "incumbent_full_version_fingerprint": (
            request.incumbent_full_version_fingerprint
        ),
        "incumbent_snapshot_json": request.incumbent_snapshot_json,
        "incumbent_user_playbook_id": request.incumbent_user_playbook_id,
        "job_id": request.job_id,
        "optimizer_kind": request.optimizer_kind,
        "projection_digest": request.projection.digest,
        "projection_json": request.projection.canonical_json,
        "proof_digest": request.decision_proof.digest,
        "proof_json": request.decision_proof.canonical_json,
        "qualification_authority": _authority_payload(request),
        "revised_content": request.revised_content,
        "subject_epochs_json": request.subject_epochs_json,
    }
    return sha256(canonical_json_bytes(immutable)).hexdigest()


#: Staging columns that must equal the request (and the live incumbent) for a
#: staged row to be this request's, with the request attribute each holds.
def _staged_identity(
    request: ProvisionalPublicationRequest, digest: str
) -> dict[str, object]:
    authority = request.qualification_authority
    return {
        "optimizer_kind": _OPTIMIZER,
        "job_id": request.job_id,
        "attempt_key": request.attempt_key,
        "incumbent_user_playbook_id": request.incumbent_user_playbook_id,
        "revised_content": request.revised_content,
        "content_digest": request.projection.candidate_content_digest,
        "projection_json": request.projection.canonical_json,
        "projection_digest": request.projection.digest,
        "proof_json": request.decision_proof.canonical_json,
        "proof_digest": request.decision_proof.digest,
        "subject_epochs_json": request.subject_epochs_json,
        "request_id": _request_id(request.job_id),
        "staging_digest": digest,
        "incumbent_full_version_fingerprint": (
            request.incumbent_full_version_fingerprint
        ),
        "incumbent_snapshot_json": request.incumbent_snapshot_json,
        "evidence_bundle_digest": request.evidence_bundle_digest,
        "candidate_digest": request.candidate_digest,
        "qualification_authority_epoch": authority.epoch,
        "qualification_authority_digest": authority.authority_digest,
        "discovery_component_identity_digest": (
            authority.discovery_component_identity_digest
        ),
        "discovery_qualification_suite_digest": (
            authority.discovery_qualification_suite_digest
        ),
        "discovery_qualification_result_digest": (
            authority.discovery_qualification_result_digest
        ),
        "held_out_component_identity_digest": (
            authority.held_out_component_identity_digest
        ),
        "held_out_qualification_suite_digest": (
            authority.held_out_qualification_suite_digest
        ),
        "held_out_qualification_result_digest": (
            authority.held_out_qualification_result_digest
        ),
        "candidate_generator_identity_digest": (
            authority.candidate_generator_identity_digest
        ),
        "candidate_generator_authorization_digest": (
            authority.candidate_generator_authorization_digest
        ),
    }


def _request_id(job_id: int) -> str:
    return f"open_world_provisional:{job_id}"


def _semantic_digest(incumbent: sqlite3.Row) -> tuple[str, str]:
    content_digest = sha256(str(incumbent["content"]).encode("utf-8")).hexdigest()
    return content_digest, incumbent_user_playbook_semantic_digest(
        content_digest=content_digest, trigger=incumbent["trigger"]
    )


def _snapshot_of(row: sqlite3.Row) -> str:
    """The canonical full-version snapshot of a stored playbook row.

    The bytes ``open_world/publication`` builds from the same row: every
    persisted field but the embedding, with the governance ref and the row's
    own ``retired_at``.
    """
    playbook = _row_to_user_playbook(row)
    return canonical_json_bytes(
        {
            "schema_version": _FULL_VERSION_SCHEMA,
            "user_playbook": playbook.model_dump(mode="json", exclude={"embedding"})
            | {
                "governance_subject_ref": playbook.governance_subject_ref,
                "retired_at": row["retired_at"],
            },
        }
    ).decode("utf-8")


def _incumbent_matches(
    incumbent: sqlite3.Row,
    *,
    expected_id: int,
    expected_fingerprint: str,
    snapshot_json: str,
) -> bool:
    """``open_world_provisional_incumbent_matches``."""
    if (
        expected_id <= 0
        or not rule.is_hex64(expected_fingerprint)
        or not snapshot_json
        or sha256(snapshot_json.encode("utf-8")).hexdigest() != expected_fingerprint
    ):
        return False
    try:
        current = _snapshot_of(incumbent)
    except (KeyError, TypeError, ValueError):
        return False
    return incumbent["user_playbook_id"] == expected_id and current == snapshot_json


def _incumbent_is_current_snapshot(incumbent: sqlite3.Row, snapshot_json: str) -> bool:
    """The stage and commit drift check: the frozen, current, governed version."""
    try:
        current = _snapshot_of(incumbent)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        current == snapshot_json
        and incumbent["status"] is None
        and incumbent["merged_into"] is None
        and incumbent["superseded_by"] is None
        and incumbent["retired_at"] is None
        and not _blank(incumbent["governance_subject_ref"])
    )


def _blank(value: object) -> bool:
    return not isinstance(value, str) or value.strip(" ") == ""


def _result(row: sqlite3.Row) -> ProvisionalPublicationResult:
    return ProvisionalPublicationResult(
        job_id=int(row["job_id"]),
        outcome=row["outcome"],
        successor_user_playbook_id=row["successor_user_playbook_id"],
        deployment_lifecycle_id=row["deployment_lifecycle_id"],
        observation_deadline=row["observation_deadline"],
    )


def _payload_of(artifact: PlaybookOptimizationArtifact | None) -> Any:
    return None if artifact is None else json.loads(artifact.content_json)


class SQLiteOpenWorldPublicationMixin:
    """``OpenWorldProvisionalPublicationStore`` on SQLite."""

    conn: sqlite3.Connection
    org_id: str
    embedding_dimensions: int
    _lock: Any
    commit_scope: Any
    _has_sqlite_vec: bool
    _evidence_job_locked: Any
    _artifact_of_locked: Any
    _authorize_artifact_locked: Any
    _active_subject_barrier_locked: Any

    #: The publication clock, in epoch seconds. A test pins it.
    _publication_clock: Callable[[], float] = time.time

    def _publication_now(self) -> int:
        return int(self._publication_clock())

    # ------------------------------------------------------------------
    # The port members
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def cleanup_user_playbook_provisional_stale_attempt(
        self, *, job_id: int, owner: str, worker_fence: int
    ) -> bool:
        """Clean only this worker's uncommitted residue; false when it is not."""
        if job_id <= 0 or worker_fence <= 0 or _blank(owner):
            raise ValueError("provisional publication cleanup identity is invalid")
        with self._lock, self.commit_scope():
            return self._cleanup_locked(
                job_id=job_id,
                owner=owner,
                publication_fence=None,
                worker_fence=worker_fence,
            )

    @SQLiteStorageBase.handle_exceptions
    def claim_user_playbook_provisional_publication(
        self,
        *,
        job_id: int,
        owner: str,
        worker_fence: int,
        incumbent_user_playbook_id: int,
        incumbent_full_version_fingerprint: str,
        incumbent_snapshot_json: str,
    ) -> PublicationClaim:
        """Claim publication only while the frozen incumbent is still current.

        Raises:
            OpenWorldPublicationIncumbentChangedError: The incumbent is no longer
                the frozen full version; nothing is written.
        """
        if (
            job_id <= 0
            or worker_fence <= 0
            or incumbent_user_playbook_id <= 0
            or _blank(owner)
            or not rule.is_hex64(incumbent_full_version_fingerprint)
            or not incumbent_snapshot_json
            or sha256(incumbent_snapshot_json.encode("utf-8")).hexdigest()
            != incumbent_full_version_fingerprint
        ):
            raise ValueError("provisional publication claim identity is invalid")
        with self._lock, self.commit_scope():
            now = self._publication_now()
            result = self._result_row_locked(job_id)
            if result is not None:
                claim = self._claim_row_locked(job_id)
                if (
                    claim is None
                    or claim["owner"] != owner
                    or claim["worker_fence"] != worker_fence
                ):
                    raise StorageError(
                        "committed provisional publication claim changed"
                    )
                return PublicationClaim(
                    job_id=job_id, owner=owner, fence=int(claim["publication_fence"])
                )
            job = self._evidence_job_locked(job_id)
            if (
                job is None
                or job.optimizer_kind != _OPTIMIZER
                or job.target_kind != "user_playbook"
                or job.status not in ("pending", "running")
                or job.stage not in ("held_out_analyzed", "publishing")
                or job.lease_owner != owner
                or job.lease_fence != worker_fence
                or job.lease_expires_at is None
                or job.lease_expires_at <= now
            ):
                raise StorageError(
                    "provisional publication optimizer lease is not current"
                )
            claim = self._claim_row_locked(job_id)
            incumbent = self._playbook_row_locked(incumbent_user_playbook_id)
            if incumbent is None:
                raise StorageError("provisional publication incumbent is missing")
            if _blank(incumbent["governance_subject_ref"]):
                raise StorageError(
                    "provisional publication incumbent governance subject is missing"
                )
            if job.target_id != incumbent_user_playbook_id or not _incumbent_matches(
                incumbent,
                expected_id=incumbent_user_playbook_id,
                expected_fingerprint=incumbent_full_version_fingerprint,
                snapshot_json=incumbent_snapshot_json,
            ):
                raise OpenWorldPublicationIncumbentChangedError(
                    "provisional publication incumbent fingerprint changed"
                )
            self._lock_governance_locked(job, None)
            self._assert_authority_locked(job, None)
            candidate = _payload_of(
                self._artifact_of_locked(job_id, OPEN_WORLD_CANDIDATE_ARTIFACT_KIND)
            )
            if candidate is None or rule.text_at(
                candidate, "payload", "incumbent_full_version_fingerprint"
            ) != (incumbent_full_version_fingerprint):
                raise StorageError(
                    "provisional publication durable artifact binding changed"
                )
            if claim is not None:
                if claim["consumed"]:
                    raise StorageError("provisional publication claim was consumed")
                if claim["owner"] == owner and claim["worker_fence"] == worker_fence:
                    return PublicationClaim(
                        job_id=job_id,
                        owner=owner,
                        fence=int(claim["publication_fence"]),
                    )
                fence = int(claim["publication_fence"]) + 1
                self.conn.execute(
                    "UPDATE user_playbook_publication_claims"
                    " SET owner = ?, publication_fence = ?, worker_fence = ?,"
                    " updated_at = ? WHERE optimizer_kind = ? AND job_id = ?",
                    (owner, fence, worker_fence, now, _OPTIMIZER, job_id),
                )
            else:
                fence = 1
                self.conn.execute(
                    "INSERT INTO user_playbook_publication_claims"
                    " (optimizer_kind, job_id, owner, publication_fence,"
                    " worker_fence, consumed, updated_at)"
                    " VALUES (?, ?, ?, 1, ?, 0, ?)",
                    (_OPTIMIZER, job_id, owner, worker_fence, now),
                )
            moved = self.conn.execute(
                "UPDATE playbook_optimization_jobs SET stage = 'publishing',"
                " updated_at = ? WHERE job_id = ?"
                " AND stage IN ('held_out_analyzed', 'publishing')"
                " AND lease_owner = ? AND lease_fence = ? AND lease_expires_at > ?",
                (now, job_id, owner, worker_fence, now),
            )
            if moved.rowcount != 1:
                raise StorageError(
                    "provisional publication optimizer transition lost its fence"
                )
        return PublicationClaim(job_id=job_id, owner=owner, fence=fence)

    @SQLiteStorageBase.handle_exceptions
    def stage_user_playbook_provisional_publication(
        self, request: ProvisionalPublicationRequest
    ) -> bool:
        """Stage the request, or clean this worker's residue and return false.

        False means the incumbent is no longer the frozen version.
        """
        request.__post_init__()
        with self._lock, self.commit_scope():
            return self._stage_locked(request, now=self._publication_now())

    @SQLiteStorageBase.handle_exceptions
    def commit_user_playbook_provisional_publication(
        self, request: ProvisionalPublicationRequest
    ) -> ProvisionalPublicationResult:
        """Publish the staged successor provisionally, atomically, or not at all.

        Raises:
            OpenWorldPublicationIncumbentChangedError: The incumbent drifted.
                The worker's uncommitted residue was cleaned and no publication
                was written.
        """
        request.__post_init__()
        with self._lock:
            with self.commit_scope():
                result = self._commit_locked(request, now=self._publication_now())
            if result is None:
                raise OpenWorldPublicationIncumbentChangedError(
                    "provisional publication incumbent fingerprint changed"
                )
        return result

    @SQLiteStorageBase.handle_exceptions
    def load_user_playbook_provisional_publication_result(
        self, job_id: int
    ) -> ProvisionalPublicationResult | None:
        with self._lock:
            row = self._result_row_locked(job_id)
        return None if row is None else _result(row)

    @SQLiteStorageBase.handle_exceptions
    def load_open_world_provisional_publication_retry(
        self,
        *,
        job_id: int,
        claim: OptimizationJobClaim,
    ) -> ProvisionalPublicationResult | None:
        """The committed result, when it was committed under this claim.

        Raises:
            OpenWorldPublicationRetryIdentityMismatchError: A result exists but
                another claim, attempt or authority committed it.
        """
        if (
            type(claim) is not OptimizationJobClaim
            or claim.job_id != job_id
            or job_id <= 0
            or claim.fence <= 0
            or _blank(claim.owner)
        ):
            raise ValueError("provisional publication retry claim is invalid")
        with self._lock, self.commit_scope():
            result = self._result_row_locked(job_id)
            if result is None:
                return None
            job = self._evidence_job_locked(job_id)
            stored_claim = self._claim_row_locked(job_id)
            context = self._context_row_locked(job_id)
            self._lock_governance_locked(job, None)
            context_json = (
                None
                if context is None
                else rule.decode(
                    context["context_json"],
                    "committed provisional retry context snapshot changed",
                )
            )
            if (
                job is None
                or stored_claim is None
                or context is None
                or job.attempt_key != result["retry_attempt_key"]
                or job.successor_target_id != result["successor_user_playbook_id"]
                or stored_claim["owner"] != result["retry_claim_owner"]
                or stored_claim["worker_fence"] != result["retry_worker_fence"]
                or not stored_claim["consumed"]
                or result["retry_claim_owner"] != claim.owner
                or result["retry_worker_fence"] != claim.fence
                or context["context_digest"] != result["retry_attempt_context_digest"]
                or context["context_digest"]
                != rule.text_digest(context["context_json"])
                or rule.path(context_json, "qualification_authority", "epoch")
                != result["retry_qualification_authority_epoch"]
                or rule.text_at(
                    context_json, "qualification_authority", "authority_digest"
                )
                != result["retry_qualification_authority_digest"]
            ):
                raise OpenWorldPublicationRetryIdentityMismatchError(
                    "committed provisional retry identity changed"
                )
            return _result(result)

    # ------------------------------------------------------------------
    # Stage and commit
    # ------------------------------------------------------------------

    def _stage_locked(
        self, request: ProvisionalPublicationRequest, *, now: int
    ) -> bool:
        """``stage_user_playbook_provisional_publication``, in the caller's tx."""
        self._assert_request_shape(request)
        digest = staging_digest(request)
        job_id = request.job_id
        result = self._result_row_locked(job_id)
        if result is not None:
            if result["staging_digest"] != digest:
                raise StorageError("committed provisional publication identity changed")
            return True
        job = self._evidence_job_locked(job_id)
        claim = self._claim_row_locked(job_id)
        staging = self._staging_row_locked(job_id)
        incumbent = self._playbook_row_locked(request.incumbent_user_playbook_id)
        if incumbent is None:
            raise StorageError("provisional publication incumbent is missing")
        self._lock_governance_locked(job, request.subject_epochs_json)
        self._assert_authority_locked(job, request)
        owner = request.publication_claim.owner
        if (
            job is None
            or job.optimizer_kind != _OPTIMIZER
            or job.target_kind != "user_playbook"
            or job.target_id != incumbent["user_playbook_id"]
            or job.attempt_key != request.attempt_key
            or job.status not in ("pending", "running")
            or job.stage != "publishing"
            or job.lease_owner != owner
            or job.lease_fence != request.worker_fence
            or job.lease_expires_at is None
            or job.lease_expires_at <= now
        ):
            raise StorageError("provisional publication optimizer lease is not current")
        if (
            claim is None
            or claim["optimizer_kind"] != _OPTIMIZER
            or claim["owner"] != owner
            or claim["publication_fence"] != request.publication_claim.fence
            or claim["worker_fence"] != request.worker_fence
            or claim["consumed"]
        ):
            raise StorageError("provisional publication claim lost its fence")
        if not _incumbent_is_current_snapshot(
            incumbent, request.incumbent_snapshot_json
        ):
            self._cleanup_locked(
                job_id=job_id,
                owner=owner,
                publication_fence=request.publication_claim.fence,
                worker_fence=request.worker_fence,
            )
            return False
        content_digest, semantic_digest = _semantic_digest(incumbent)
        expected = _staged_identity(request, digest) | {
            "incumbent_content_digest": content_digest,
            "incumbent_trigger": incumbent["trigger"],
            "incumbent_semantic_digest": semantic_digest,
        }
        if staging is not None:
            if any(staging[column] != value for column, value in expected.items()):
                raise StorageError("staged provisional publication identity changed")
            if (
                staging["claim_owner"] != owner
                or staging["publication_fence"] != request.publication_claim.fence
                or staging["worker_fence"] != request.worker_fence
            ):
                self.conn.execute(
                    "UPDATE user_playbook_publication_staging"
                    " SET claim_owner = ?, publication_fence = ?, worker_fence = ?"
                    " WHERE optimizer_kind = ? AND job_id = ?",
                    (
                        owner,
                        request.publication_claim.fence,
                        request.worker_fence,
                        _OPTIMIZER,
                        job_id,
                    ),
                )
            return True
        # PostgreSQL rechecks the decision proof in a trigger on this insert.
        self._assert_decision_proof_locked(
            job_id,
            request.decision_proof.canonical_json,
            request.decision_proof.digest,
        )
        row = expected | {
            "claim_owner": owner,
            "publication_fence": request.publication_claim.fence,
            "worker_fence": request.worker_fence,
            "created_at": now,
        }
        columns = ", ".join(row)
        self.conn.execute(
            f"INSERT INTO user_playbook_publication_staging ({columns})"  # noqa: S608
            f" VALUES ({', '.join('?' * len(row))})",
            tuple(row.values()),
        )
        return True

    def _commit_locked(
        self, request: ProvisionalPublicationRequest, *, now: int
    ) -> ProvisionalPublicationResult | None:
        """``commit_user_playbook_provisional_publication``; ``None`` on drift.

        Re-runs the stage, then the decision-proof check, then every staged,
        claimed and leased binding, before any publication write.
        """
        job_id = request.job_id
        if not self._stage_locked(request, now=now):
            return None
        self._assert_decision_proof_locked(
            job_id,
            request.decision_proof.canonical_json,
            request.decision_proof.digest,
        )
        digest = staging_digest(request)
        result = self._result_row_locked(job_id)
        if result is not None:
            if result["staging_digest"] != digest:
                raise StorageError("committed provisional publication identity changed")
            return _result(result)
        job = self._evidence_job_locked(job_id)
        claim = self._claim_row_locked(job_id)
        staging = self._staging_row_locked(job_id)
        if job is None or staging is None:
            raise StorageError("provisional publication staged lease lost its fence")
        incumbent = self._playbook_row_locked(staging["incumbent_user_playbook_id"])
        self._lock_governance_locked(job, staging["subject_epochs_json"])
        context = self._assert_authority_locked(job, request)
        if incumbent is None:
            raise StorageError("provisional publication incumbent fingerprint changed")
        content_digest, semantic_digest = _semantic_digest(incumbent)
        expected = _staged_identity(request, digest) | {
            "incumbent_content_digest": content_digest,
            "incumbent_trigger": incumbent["trigger"],
            "incumbent_semantic_digest": semantic_digest,
            "claim_owner": request.publication_claim.owner,
            "publication_fence": request.publication_claim.fence,
            "worker_fence": request.worker_fence,
        }
        if (
            any(staging[column] != value for column, value in expected.items())
            or job.status not in ("pending", "running")
            or job.stage != "publishing"
            or job.lease_owner != staging["claim_owner"]
            or job.lease_fence != staging["worker_fence"]
            or job.lease_expires_at is None
            or job.lease_expires_at <= now
            or claim is None
            or claim["consumed"]
            or claim["owner"] != staging["claim_owner"]
            or claim["publication_fence"] != staging["publication_fence"]
            or claim["worker_fence"] != staging["worker_fence"]
        ):
            raise StorageError("provisional publication staged lease lost its fence")
        if not _incumbent_is_current_snapshot(
            incumbent, staging["incumbent_snapshot_json"]
        ):
            raise StorageError("provisional publication incumbent fingerprint changed")
        return self._publish_locked(
            job=job,
            staging=staging,
            incumbent=incumbent,
            context=context,
            now=now,
        )

    def _publish_locked(
        self,
        *,
        job: PlaybookOptimizationJob,
        staging: sqlite3.Row,
        incumbent: sqlite3.Row,
        context: sqlite3.Row,
        now: int,
    ) -> ProvisionalPublicationResult:
        """Every publication write, in the caller's one transaction."""
        job_id = job.job_id
        request_id = _request_id(job_id)
        projection = json.loads(staging["projection_json"])
        embedding = [float(value) for value in projection["embedding"]]
        inserted = self.conn.execute(
            "INSERT INTO user_playbooks"
            " (user_id, playbook_name, created_at, request_id, agent_version,"
            " content, trigger, rationale, blocking_issue, source_interaction_ids,"
            " status, source, embedding, expanded_terms, source_span, notes,"
            " reader_angle, tags, merged_into, superseded_by, retired_at,"
            " governance_subject_ref)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?,"
            " NULL, NULL, NULL, ?)",
            (
                incumbent["user_id"],
                incumbent["playbook_name"],
                _epoch_to_iso(now),
                incumbent["request_id"],
                incumbent["agent_version"],
                staging["revised_content"],
                incumbent["trigger"],
                incumbent["rationale"],
                incumbent["blocking_issue"],
                incumbent["source_interaction_ids"],
                incumbent["source"],
                json.dumps(embedding, separators=(",", ":")),
                " ".join(projection["expanded_terms"]),
                incumbent["source_span"],
                incumbent["notes"],
                incumbent["reader_angle"],
                incumbent["tags"],
                incumbent["governance_subject_ref"],
            ),
        )
        successor_id = inserted.lastrowid
        if successor_id is None:
            raise StorageError(
                "provisional publication successor insert returned no id"
            )
        self.conn.execute(
            "INSERT INTO user_playbooks_fts(rowid, search_text) VALUES (?, ?)",
            (successor_id, projection["lexical_document"]),
        )
        if self._has_sqlite_vec:
            self.conn.execute(
                "INSERT INTO user_playbooks_vec(rowid, embedding) VALUES (?, ?)",
                (successor_id, json.dumps(embedding)),
            )
        successor_row = self._playbook_row_locked(successor_id)
        successor_fingerprint = user_playbook_full_version_fingerprint(
            _row_to_user_playbook(successor_row)
        )
        superseded = self.conn.execute(
            "UPDATE user_playbooks SET status = 'superseded', superseded_by = ?,"
            " retired_at = ? WHERE user_playbook_id = ? AND status IS NULL",
            (successor_id, now, incumbent["user_playbook_id"]),
        )
        if superseded.rowcount != 1:
            raise StorageError(
                "provisional publication incumbent switch was not atomic"
            )
        predecessor_projection = self.conn.execute(
            "SELECT successor_projection_digest FROM user_playbook_deployment_lifecycles"
            " WHERE successor_id = ?"
            " ORDER BY published_at DESC, deployment_lifecycle_id DESC LIMIT 1",
            (incumbent["user_playbook_id"],),
        ).fetchone()
        deadline = now + OBSERVATION_SECONDS
        lifecycle = self.conn.execute(
            "INSERT INTO user_playbook_deployment_lifecycles"
            " (job_id, state, predecessor_id, successor_id,"
            " predecessor_full_fingerprint, successor_full_fingerprint,"
            " predecessor_projection_digest, successor_projection_digest,"
            " evidence_digest, candidate_digest, proof_digest,"
            " qualification_authority_epoch, qualification_authority_digest,"
            " governance_epochs_json, published_at, observation_deadline,"
            " terminal_reason, terminal_at, aggregation_obligation)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
            " 'not_required')",
            (
                job_id,
                _PUBLISHED_STATE,
                incumbent["user_playbook_id"],
                successor_id,
                staging["incumbent_full_version_fingerprint"],
                successor_fingerprint,
                None
                if predecessor_projection is None
                else predecessor_projection["successor_projection_digest"],
                staging["projection_digest"],
                staging["evidence_bundle_digest"],
                staging["candidate_digest"],
                staging["proof_digest"],
                staging["qualification_authority_epoch"],
                staging["qualification_authority_digest"],
                staging["subject_epochs_json"],
                now,
                deadline,
                _PUBLISHED_REASON,
                now,
            ),
        )
        lifecycle_id = lifecycle.lastrowid
        if lifecycle_id is None:
            raise StorageError(
                "provisional publication lifecycle insert returned no id"
            )
        _append_event_stmt(
            self.conn,
            org_id=self.org_id,
            entity_type="user_playbook",
            entity_id=str(incumbent["user_playbook_id"]),
            op="status_change",
            prov="wasInvalidatedBy",
            source_ids=[str(successor_id)],
            actor=_PUBLICATION_ACTOR,
            request_id=request_id,
            reason=request_id,
            created_at=now,
            from_status=None,
            to_status="superseded",
            status_namespace="user_playbook",
        )
        _append_event_stmt(
            self.conn,
            org_id=self.org_id,
            entity_type="user_playbook",
            entity_id=str(successor_id),
            op="revise",
            prov="wasRevisionOf",
            source_ids=[str(incumbent["user_playbook_id"])],
            actor=_PUBLICATION_ACTOR,
            request_id=request_id,
            reason=request_id,
            created_at=now,
        )
        self.conn.execute(
            "INSERT INTO playbook_optimization_events"
            " (job_id, event_type, payload_json, created_at)"
            " VALUES (?, 'provisional_publication', ?, ?)",
            (
                job_id,
                rule.canonical_text(
                    {
                        "outcome": "provisional",
                        "deployment_lifecycle_id": lifecycle_id,
                        "predecessor_user_playbook_id": incumbent["user_playbook_id"],
                        "successor_user_playbook_id": successor_id,
                        "evidence_digest": staging["evidence_bundle_digest"],
                        "candidate_digest": staging["candidate_digest"],
                        "proof_digest": staging["proof_digest"],
                        "projection_digest": staging["projection_digest"],
                    }
                ),
                now,
            ),
        )
        self._assert_retry_identity_locked(job, staging, context)
        self.conn.execute(
            "INSERT INTO user_playbook_publication_results"
            " (optimizer_kind, job_id, outcome, successor_user_playbook_id,"
            " staging_digest, created_at, deployment_lifecycle_id,"
            " observation_deadline, retry_attempt_key, retry_claim_owner,"
            " retry_worker_fence, retry_qualification_authority_epoch,"
            " retry_qualification_authority_digest, retry_attempt_context_digest)"
            " VALUES (?, ?, 'provisional', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _OPTIMIZER,
                job_id,
                successor_id,
                staging["staging_digest"],
                now,
                lifecycle_id,
                deadline,
                staging["attempt_key"],
                staging["claim_owner"],
                staging["worker_fence"],
                staging["qualification_authority_epoch"],
                staging["qualification_authority_digest"],
                context["context_digest"],
            ),
        )
        consumed = self.conn.execute(
            "UPDATE user_playbook_publication_claims SET consumed = 1, updated_at = ?"
            " WHERE optimizer_kind = ? AND job_id = ? AND owner = ?"
            " AND publication_fence = ? AND worker_fence = ? AND consumed = 0",
            (
                now,
                _OPTIMIZER,
                job_id,
                staging["claim_owner"],
                staging["publication_fence"],
                staging["worker_fence"],
            ),
        )
        if consumed.rowcount != 1:
            raise StorageError(
                "provisional publication claim consumption lost its fence"
            )
        applied = self.conn.execute(
            "UPDATE playbook_optimization_jobs SET stage = 'applied',"
            " terminal_outcome = 'applied', status = 'completed',"
            " successor_target_id = ?, candidate_content_digest = ?,"
            " search_projection_digest = ?, lease_owner = NULL,"
            " lease_expires_at = NULL, updated_at = ?"
            " WHERE job_id = ? AND status IN ('pending', 'running')"
            " AND stage = 'publishing' AND lease_owner = ? AND lease_fence = ?"
            " AND lease_expires_at > ?",
            (
                successor_id,
                staging["content_digest"],
                staging["projection_digest"],
                now,
                job_id,
                staging["claim_owner"],
                staging["worker_fence"],
                now,
            ),
        )
        if applied.rowcount != 1:
            raise StorageError("provisional publication job transition lost its fence")
        self.conn.execute(
            "DELETE FROM user_playbook_publication_staging"
            " WHERE optimizer_kind = ? AND job_id = ?",
            (_OPTIMIZER, job_id),
        )
        return ProvisionalPublicationResult(
            job_id=job_id,
            outcome="provisional",
            successor_user_playbook_id=int(successor_id),
            deployment_lifecycle_id=int(lifecycle_id),
            observation_deadline=deadline,
        )

    def _assert_retry_identity_locked(
        self,
        job: PlaybookOptimizationJob,
        staging: sqlite3.Row,
        context: sqlite3.Row,
    ) -> None:
        """``bind_open_world_provisional_retry_identity``: what a retry matches."""
        context_json = rule.decode(
            context["context_json"],
            "committed provisional retry context snapshot changed",
        )
        if job.attempt_key != staging["attempt_key"]:
            raise StorageError("committed provisional retry attempt identity changed")
        if context["context_digest"] != rule.text_digest(context["context_json"]):
            raise StorageError("committed provisional retry context snapshot changed")
        if (
            rule.path(context_json, "qualification_authority", "epoch")
            != staging["qualification_authority_epoch"]
            or rule.text_at(context_json, "qualification_authority", "authority_digest")
            != (staging["qualification_authority_digest"])
        ):
            raise StorageError("committed provisional retry authority context changed")

    def _cleanup_locked(
        self,
        *,
        job_id: int,
        owner: str,
        publication_fence: int | None,
        worker_fence: int,
    ) -> bool:
        """``cleanup_open_world_provisional_stale_attempt``."""
        if self._result_row_locked(job_id) is not None:
            return False
        job = self._evidence_job_locked(job_id)
        claim = self._claim_row_locked(job_id)
        staging = self._staging_row_locked(job_id)
        if claim is None:
            if staging is not None:
                return False
            return (
                job is not None
                and job.optimizer_kind == _OPTIMIZER
                and job.target_kind == "user_playbook"
                and job.status in ("pending", "running")
                and job.stage == "held_out_analyzed"
                and job.lease_owner == owner
                and job.lease_fence == worker_fence
            )
        if (
            claim["consumed"]
            or claim["owner"] != owner
            or claim["worker_fence"] != worker_fence
            or (
                publication_fence is not None
                and claim["publication_fence"] != publication_fence
            )
            or job is None
            or job.optimizer_kind != _OPTIMIZER
            or job.target_kind != "user_playbook"
            or job.status not in ("pending", "running")
            or job.stage != "publishing"
            or job.lease_owner != owner
            or job.lease_fence != worker_fence
        ):
            return False
        if staging is not None and (
            staging["claim_owner"] != owner
            or staging["publication_fence"] != claim["publication_fence"]
            or staging["worker_fence"] != worker_fence
        ):
            return False
        self.conn.execute(
            "DELETE FROM user_playbook_publication_staging"
            " WHERE optimizer_kind = ? AND job_id = ? AND claim_owner = ?"
            " AND publication_fence = ? AND worker_fence = ?",
            (_OPTIMIZER, job_id, owner, claim["publication_fence"], worker_fence),
        )
        deleted = self.conn.execute(
            "DELETE FROM user_playbook_publication_claims"
            " WHERE optimizer_kind = ? AND job_id = ? AND owner = ?"
            " AND publication_fence = ? AND worker_fence = ? AND consumed = 0",
            (_OPTIMIZER, job_id, owner, claim["publication_fence"], worker_fence),
        )
        if deleted.rowcount != 1:
            raise StorageError("provisional publication stale cleanup lost claim fence")
        moved = self.conn.execute(
            "UPDATE playbook_optimization_jobs SET stage = 'held_out_analyzed',"
            " updated_at = ? WHERE job_id = ? AND optimizer_kind = ?"
            " AND target_kind = 'user_playbook' AND status IN ('pending', 'running')"
            " AND stage = 'publishing' AND lease_owner = ? AND lease_fence = ?",
            (self._publication_now(), job_id, _OPTIMIZER, owner, worker_fence),
        )
        if moved.rowcount != 1:
            raise StorageError("provisional publication stale cleanup lost job fence")
        return True

    # ------------------------------------------------------------------
    # The checks every member shares
    # ------------------------------------------------------------------

    def _assert_request_shape(self, request: ProvisionalPublicationRequest) -> None:
        """``assert_open_world_provisional_request``, past the shared validator.

        ``ProvisionalPublicationRequest`` already validates digests, canonical
        bytes and the content bindings; what remains is the store's own shape.
        """
        projection = json.loads(request.projection.canonical_json)
        snapshot = json.loads(request.incumbent_snapshot_json)
        if (
            set(snapshot) != {"schema_version", "user_playbook"}
            or rule.path(snapshot, "user_playbook", "user_playbook_id")
            != request.incumbent_user_playbook_id
        ):
            rule.integrity("provisional publication incumbent snapshot is invalid")
        if len(
            request.projection.embedding
        ) != self.embedding_dimensions or not rule.trim(
            str(projection.get("lexical_document", ""))
        ):
            rule.integrity("provisional publication projection binding changed")
        if request.qualification_authority.policy_id != RULE_EVIDENCE_POLICY_ID:
            rule.integrity("unsupported publication method policy")

    def _lock_governance_locked(
        self, job: PlaybookOptimizationJob | None, subject_epochs_json: str | None
    ) -> None:
        """``lock_open_world_provisional_governance``.

        SQLite's governance vector is the one the job froze, every subject at
        epoch 0; a subject held by an erasure barrier is erased.
        """
        if job is None or job.optimizer_kind != _OPTIMIZER:
            raise StorageError("provisional publication governance vector is missing")
        _fingerprint, subject_epochs = freeze_validation_identity(
            job, org_id=self.org_id
        )
        if not subject_epochs:
            raise StorageError("provisional publication governance vector is missing")
        ordered = sorted(subject_epochs, key=lambda item: item[0].encode())
        expected = rule.canonical_text(
            {"subjects": [{"epoch": epoch, "ref": ref} for ref, epoch in ordered]}
        )
        if subject_epochs_json is not None and subject_epochs_json != expected:
            raise StorageError("provisional publication governance epoch changed")
        for subject_ref, epoch in ordered:
            if self._active_subject_barrier_locked(subject_ref) is not None:
                raise StorageError(
                    "provisional publication governance subject is erased"
                )
            if epoch != 0:
                raise StorageError("provisional publication governance epoch changed")

    def _assert_authority_locked(
        self,
        job: PlaybookOptimizationJob | None,
        request: ProvisionalPublicationRequest | None,
    ) -> sqlite3.Row:
        """``assert_open_world_provisional_authority``; returns the context."""
        authority = self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_qualification_authority"
            " WHERE singleton = 1"
        ).fetchone()
        if authority is None or authority["state"] != "active":
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "provisional publication qualification authority is unavailable",
            )
        if authority["method_authority_json"] is None:
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "publication requires rule-evidence method authority",
            )
        method = rule.decode(
            authority["method_authority_json"],
            "provisional publication qualification authority digest is invalid",
        )
        if authority["authority_digest"] != rule.json_digest(method) or (
            rule.text(rule.get(method, "policy_id")) != RULE_EVIDENCE_POLICY_ID
        ):
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "provisional publication qualification authority digest is invalid",
            )
        if request is not None and any(
            authority[column] != value
            for column, value in _authority_columns(request).items()
        ):
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "provisional publication qualification authority is stale",
            )
        context = None if job is None else self._context_row_locked(job.job_id)
        if context is None or any(
            context[column] is None
            for column in (
                "qualification_authority_epoch",
                "qualification_authority_digest",
                "discovery_component_identity_digest",
                "discovery_qualification_suite_digest",
                "discovery_qualification_result_digest",
                "candidate_generator_identity_digest",
                "candidate_generator_authorization_digest",
            )
        ):
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "provisional publication rejects legacy nullable authority context",
            )
        if _context_authority_drifted(context, authority):
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "provisional publication attempt authority is stale",
            )
        if job is None:  # a context row implies its job; for the type checker
            raise StorageError("provisional publication optimizer job is missing")
        evidence, candidate, decision = self._attempt_artifacts_locked(job.job_id)
        decision_payload = _payload_of(decision)
        if (
            evidence is None
            or candidate is None
            or decision is None
            or context["bundle_digest"] != evidence.content_digest
            or rule.text_at(
                json.loads(job.metadata_json or "{}"),
                "open_world_evidence",
                "bundle_digest",
            )
            != evidence.content_digest
            or rule.text_at(decision_payload, "payload", "bundle_digest")
            != evidence.content_digest
            or rule.text_at(decision_payload, "payload", "candidate_digest")
            != candidate.content_digest
            or rule.text_at(decision_payload, "payload", "decision", "decision")
            != "accept"
        ):
            rule.conflict("provisional publication durable decision is not accepted")
        candidate_payload = _payload_of(candidate)
        if request is not None and (
            evidence.content_digest != request.evidence_bundle_digest
            or candidate.content_digest != request.candidate_digest
            or rule.text_at(candidate_payload, "payload", "revised_content")
            != request.revised_content
            or rule.text_at(
                candidate_payload, "payload", "incumbent_full_version_fingerprint"
            )
            != request.incumbent_full_version_fingerprint
        ):
            rule.conflict("provisional publication durable artifact binding changed")
        self._authorize_artifact_locked(
            job.job_id, OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND, decision_payload
        )
        return context

    def _assert_decision_proof_locked(
        self, job_id: int, proof_json: str, proof_digest: str
    ) -> None:
        """``assert_open_world_provisional_decision_proof``.

        The proof must be exactly the one the durable attempt implies: its
        job, decision, bundle, outcome contract, candidate, population,
        incumbent version, attempt context, authority and governance subjects.
        """
        try:
            proof = json.loads(proof_json)
        except json.JSONDecodeError:
            rule.integrity("provisional publication proof is not valid JSON")
        if (
            rule.canonical_text(proof) != proof_json
            or rule.text_digest(proof_json) != proof_digest
        ):
            rule.integrity("provisional publication proof canonical binding changed")
        job = self._evidence_job_locked(job_id)
        authority = self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_qualification_authority"
            " WHERE singleton = 1"
        ).fetchone()
        context = self._context_row_locked(job_id)
        evidence, candidate, decision = self._attempt_artifacts_locked(job_id)
        if (
            job is None
            or job.optimizer_kind != _OPTIMIZER
            or context is None
            or authority is None
            or authority["state"] != "active"
            or evidence is None
            or candidate is None
            or decision is None
        ):
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "provisional publication proof durable authority is incomplete",
            )
        evidence_payload = _payload_of(evidence)
        candidate_payload = _payload_of(candidate)
        decision_payload = _payload_of(decision)
        if (
            rule.text_digest(context["context_json"]) != context["context_digest"]
            or context["bundle_digest"] != evidence.content_digest
            or context["outcome_contract_digest"]
            != rule.text_at(evidence_payload, "target", "success_definition_digest")
            or rule.text_at(candidate_payload, "payload", "bundle_digest")
            != evidence.content_digest
            or rule.text_at(decision_payload, "payload", "bundle_digest")
            != evidence.content_digest
            or rule.text_at(decision_payload, "payload", "candidate_digest")
            != candidate.content_digest
            or rule.text_at(decision_payload, "payload", "decision", "decision")
            != "accept"
        ):
            rule.conflict(
                "provisional publication proof durable decision is inconsistent"
            )
        if _context_authority_drifted(context, authority):
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "provisional publication proof qualification authority changed",
            )
        self._authorize_artifact_locked(
            job_id, OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND, decision_payload
        )
        memo = _payload_of(
            self._artifact_of_locked(job_id, OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND)
        )
        population = rule.text_at(memo, "payload", "diagnostic_population_digest")
        if (
            rule.text_at(decision_payload, "payload", "diagnostic_population_digest")
            != population
        ):
            rule.conflict("provisional publication diagnostic population changed")
        governance_subjects = self._writable_subjects_locked(job)
        expected = {
            "schema_version": _PROOF_SCHEMA_VERSION,
            "optimizer_kind": _OPTIMIZER,
            "decision": "apply",
            "job_id": job.job_id,
            "attempt_key": job.attempt_key,
            "attempt_decision_digest": decision.content_digest,
            "attempt_decision_json": rule.canonical_text(decision_payload),
            "evidence_bundle_digest": evidence.content_digest,
            "outcome_contract_digest": context["outcome_contract_digest"],
            "candidate_digest": candidate.content_digest,
            "candidate_content_digest": rule.text_at(
                candidate_payload, "payload", "content_diff", "candidate_content_digest"
            ),
            "diagnostic_population_digest": population,
            "policy_id": RULE_EVIDENCE_POLICY_ID,
            "incumbent_full_version_fingerprint": rule.text_at(
                candidate_payload, "payload", "incumbent_full_version_fingerprint"
            ),
            "attempt_context_digest": context["context_digest"],
            "qualification_authority": {
                "policy_id": RULE_EVIDENCE_POLICY_ID,
                "epoch": authority["epoch"],
                **{column: authority[column] for column in _AUTHORITY_TUPLE},
            },
            "governance_subjects": governance_subjects,
        }
        if proof != expected:
            rule.refuse(
                OpenWorldRefusal.AUTHORITY_STALE,
                "provisional publication proof does not match durable authority",
            )

    def _writable_subjects_locked(
        self, job: PlaybookOptimizationJob
    ) -> list[dict[str, object]]:
        """The job's governance subjects, refused unless every one is writable."""
        _fingerprint, subject_epochs = freeze_validation_identity(
            job, org_id=self.org_id
        )
        ordered = sorted(subject_epochs, key=lambda item: item[0].encode())
        if not ordered or any(
            epoch != 0 or self._active_subject_barrier_locked(ref) is not None
            for ref, epoch in ordered
        ):
            raise OpenWorldEvidenceGovernanceError(
                "provisional publication proof governance epoch changed"
            )
        return [{"ref": ref, "epoch": epoch} for ref, epoch in ordered]

    # ------------------------------------------------------------------
    # Row reads
    # ------------------------------------------------------------------

    def _attempt_artifacts_locked(
        self, job_id: int
    ) -> tuple[
        PlaybookOptimizationArtifact | None,
        PlaybookOptimizationArtifact | None,
        PlaybookOptimizationArtifact | None,
    ]:
        return (
            self._artifact_of_locked(job_id, OPEN_WORLD_EVIDENCE_ARTIFACT_KIND),
            self._artifact_of_locked(job_id, OPEN_WORLD_CANDIDATE_ARTIFACT_KIND),
            self._artifact_of_locked(job_id, OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND),
        )

    def _result_row_locked(self, job_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM user_playbook_publication_results"
            " WHERE optimizer_kind = ? AND job_id = ?",
            (_OPTIMIZER, job_id),
        ).fetchone()

    def _claim_row_locked(self, job_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM user_playbook_publication_claims"
            " WHERE optimizer_kind = ? AND job_id = ?",
            (_OPTIMIZER, job_id),
        ).fetchone()

    def _staging_row_locked(self, job_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM user_playbook_publication_staging"
            " WHERE optimizer_kind = ? AND job_id = ?",
            (_OPTIMIZER, job_id),
        ).fetchone()

    def _context_row_locked(self, job_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM offline_tuner_open_world_attempt_context WHERE job_id = ?",
            (job_id,),
        ).fetchone()

    def _playbook_row_locked(self, user_playbook_id: int) -> sqlite3.Row:
        return self.conn.execute(
            "SELECT * FROM user_playbooks WHERE user_playbook_id = ?",
            (user_playbook_id,),
        ).fetchone()


#: The qualification authority tuple, as columns of both the authority row and
#: the attempt context, in the order PostgreSQL compares them.
_AUTHORITY_TUPLE = (
    "epoch",
    "authority_digest",
    "discovery_component_identity_digest",
    "discovery_qualification_suite_digest",
    "discovery_qualification_result_digest",
    "held_out_component_identity_digest",
    "held_out_qualification_suite_digest",
    "held_out_qualification_result_digest",
    "candidate_generator_identity_digest",
    "candidate_generator_authorization_digest",
)


def _authority_columns(request: ProvisionalPublicationRequest) -> dict[str, object]:
    """The request's authority tuple, keyed by authority-row column."""
    payload = _authority_payload(request)
    return {column: payload[column] for column in _AUTHORITY_TUPLE}


def _context_authority_drifted(context: sqlite3.Row, authority: sqlite3.Row) -> bool:
    """True when the attempt context no longer names the current authority."""
    return any(
        context[
            "qualification_authority_epoch"
            if column == "epoch"
            else "qualification_authority_digest"
            if column == "authority_digest"
            else column
        ]
        != authority[column]
        for column in _AUTHORITY_TUPLE
    )


__all__ = [
    "LIFECYCLE_DDL",
    "OBSERVATION_SECONDS",
    "DeploymentLifecycleRow",
    "SQLiteOpenWorldPublicationMixin",
    "lifecycle_row",
    "migrate_open_world_publication",
    "staging_digest",
]
