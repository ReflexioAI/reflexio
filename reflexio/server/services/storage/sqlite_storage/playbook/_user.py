"""User playbook CRUD + search methods for SQLite storage."""

import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Any

from reflexio.models.api_schema.common import BlockingIssue
from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.api_schema.service_schemas import Status, UserPlaybook
from reflexio.models.config_schema import SearchMode, SearchOptions
from reflexio.server.services.embedding_text import resolve_retrieval_threshold
from reflexio.server.services.playbook.publication import (
    PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY,
    PublicationClaim,
    PublicationRequest,
    PublicationResult,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.lifecycle_filters import (
    validate_include_inactive,
)

from .._base import (
    _TOMBSTONE_STATUS_VALUES,
    SQLiteStorageBase,
    _build_status_sql,
    _effective_search_mode,
    _epoch_now,
    _epoch_to_iso,
    _json_dumps,
    _row_to_user_playbook,
    _sanitize_fts_query,
    _true_rrf_merge,
    _vector_rank_rows,
)
from .._lineage import _GC_ELIGIBLE_STATUSES, _append_event_stmt
from .._playbook import _build_tags_sql, _emit_hard_delete_playbook


def _emit_supersede_user_playbook(
    conn: sqlite3.Connection,
    *,
    org_id: str,
    entity_id: str,
    old_status: str | None,
    request_id: str,
) -> None:
    """Emit a single status_change->superseded lineage event for a user playbook."""
    _append_event_stmt(
        conn,
        org_id=org_id,
        entity_type="user_playbook",
        entity_id=entity_id,
        op="status_change",
        prov="wasInvalidatedBy",
        source_ids=[],
        actor="consolidator",
        request_id=request_id,
        reason=f"{old_status or 'None'}->superseded",
        from_status=old_status,
        to_status=Status.SUPERSEDED.value,
        status_namespace="lifecycle_status",
    )


def _publication_staging_payload(request: PublicationRequest) -> dict[str, object]:
    return {
        "attempt_key": request.attempt_key,
        "claim_owner": request.publication_claim.owner,
        "content_digest": request.projection.candidate_content_digest,
        "incumbent_user_playbook_id": request.incumbent_user_playbook_id,
        "job_id": request.job_id,
        "optimizer_kind": request.optimizer_kind,
        "projection_digest": request.projection.digest,
        "projection_json": request.projection.canonical_json,
        "proof_digest": request.decision_proof.digest,
        "proof_json": request.decision_proof.canonical_json,
        "publication_fence": request.publication_claim.fence,
        "request_id": request.request_id,
        "revised_content": request.revised_content,
        "subject_epochs_json": request.subject_epochs_json,
        "worker_fence": request.worker_fence,
    }


def _publication_staging_digest(request: PublicationRequest) -> str:
    payload = _publication_staging_payload(request)
    for mutable_field in ("claim_owner", "worker_fence", "publication_fence"):
        del payload[mutable_field]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


_STAGING_CONFLICT_FIELDS = (
    ("optimizer_kind", "optimizer kind"),
    ("job_id", "job id"),
    ("attempt_key", "attempt key"),
    ("incumbent_user_playbook_id", "incumbent"),
    ("revised_content", "revised content"),
    ("content_digest", "content digest"),
    ("projection_digest", "projection digest"),
    ("projection_json", "projection bytes"),
    ("proof_digest", "proof digest"),
    ("proof_json", "proof bytes"),
    ("subject_epochs_json", "subject epochs"),
    ("request_id", "request identity"),
)

_STAGING_BINDING_FIELDS = (
    ("claim_owner", "publication owner"),
    ("worker_fence", "worker fence"),
    ("publication_fence", "publication fence"),
)


def _assert_staging_matches(row: sqlite3.Row, request: PublicationRequest) -> None:
    expected = _publication_staging_payload(request)
    for field, label in _STAGING_CONFLICT_FIELDS:
        if row[field] != expected[field]:
            raise StorageError(f"staged publication conflicts on {label}")
    if row["staging_digest"] != _publication_staging_digest(request):
        raise StorageError("staged publication conflicts on staging digest")


def _assert_staging_binding_matches(
    row: sqlite3.Row, request: PublicationRequest
) -> None:
    expected = _publication_staging_payload(request)
    for field, label in _STAGING_BINDING_FIELDS:
        if row[field] != expected[field]:
            raise StorageError(f"staged publication conflicts on {label}")


def _assert_publication_lease_live(row: sqlite3.Row, *, now: int) -> None:
    lease_expires_at = row["lease_expires_at"]
    if lease_expires_at is None or lease_expires_at <= now:
        raise StorageError("publication optimizer job lease expired")


class UserPlaybookStoreMixin:
    """Mixin providing user playbook CRUD + search for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    org_id: str
    embedding_model_name: str
    _fetchone: Any
    _fetchall: Any
    _get_embedding: Any
    _should_expand_documents: Any
    _expand_document: Any
    _fts_upsert: Any
    _vec_upsert: Any
    _delete_playbook_search_rows: Any
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any
    _own_transaction: Any
    commit_scope: Any
    _has_sqlite_vec: bool

    def _publication_job_locked(
        self,
        request: PublicationRequest,
        *,
        now: int,
    ) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?",
            (request.job_id,),
        ).fetchone()
        if row is None:
            raise StorageError("publication optimizer job does not exist")
        if row["optimizer_kind"] != request.optimizer_kind:
            raise StorageError("publication optimizer kind changed")
        if row["target_kind"] != "user_playbook":
            raise StorageError("publication target is not a user playbook")
        if row["target_id"] != request.incumbent_user_playbook_id:
            raise StorageError("publication incumbent changed in optimizer job")
        if row["attempt_key"] != request.attempt_key:
            raise StorageError("publication attempt identity changed")
        if row["status"] not in {"pending", "running"}:
            raise StorageError("publication optimizer job is terminal")
        if row["stage"] != "publishing":
            raise StorageError("publication optimizer job is not at publishing stage")
        if row["lease_owner"] != request.publication_claim.owner:
            raise StorageError("publication worker owner changed")
        if row["lease_fence"] != request.worker_fence:
            raise StorageError("publication worker fence changed")
        _assert_publication_lease_live(row, now=now)
        if (
            row["candidate_content_digest"]
            != request.projection.candidate_content_digest
        ):
            raise StorageError("publication content digest changed")
        if row["search_projection_digest"] != request.projection.digest:
            raise StorageError("publication projection digest changed")
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError as exc:
            raise StorageError("publication optimizer metadata is invalid") from exc
        if metadata.get("publication_proof_digest") != request.decision_proof.digest:
            raise StorageError("publication proof digest changed")
        try:
            request_subject_epochs = json.loads(request.subject_epochs_json)
        except json.JSONDecodeError as exc:
            raise StorageError("publication subject epoch vector is invalid") from exc
        if (
            metadata.get(PUBLICATION_SUBJECT_EPOCHS_METADATA_KEY)
            != request_subject_epochs
        ):
            raise StorageError("publication subject epochs vector changed")
        return row

    def _publication_incumbent_and_subjects_locked(
        self,
        request: PublicationRequest,
    ) -> sqlite3.Row:
        incumbent = self.conn.execute(
            "SELECT * FROM user_playbooks WHERE user_playbook_id = ?",
            (request.incumbent_user_playbook_id,),
        ).fetchone()
        if incumbent is None:
            raise StorageError("publication incumbent does not exist")
        incumbent_subject_ref = self._subject_ref_from_user_playbook_row(incumbent)
        subjects = json.loads(request.subject_epochs_json)["subjects"]
        subject_refs = tuple(str(item["ref"]) for item in subjects)
        if incumbent_subject_ref not in subject_refs:
            raise StorageError(
                "publication incumbent governance subject is absent from frozen vector"
            )
        for subject_ref in subject_refs:
            self._assert_subject_writable_locked(subject_ref)
        return incumbent

    def _publication_claim_locked(
        self,
        request: PublicationRequest,
    ) -> sqlite3.Row:
        row = self.conn.execute(
            """SELECT * FROM user_playbook_publication_claims
               WHERE optimizer_kind = ? AND job_id = ?""",
            (request.optimizer_kind, request.job_id),
        ).fetchone()
        if row is None:
            raise StorageError("publication claim does not exist")
        if row["owner"] != request.publication_claim.owner:
            raise StorageError("publication claim owner changed")
        if row["publication_fence"] != request.publication_claim.fence:
            raise StorageError("publication fence changed")
        if row["worker_fence"] != request.worker_fence:
            raise StorageError("publication worker fence changed")
        if row["consumed"]:
            raise StorageError("publication claim was already consumed")
        return row

    @SQLiteStorageBase.handle_exceptions
    def claim_user_playbook_publication(
        self, *, job_id: int, owner: str, worker_fence: int
    ) -> PublicationClaim:
        if job_id <= 0 or worker_fence <= 0 or not owner.strip():
            raise ValueError("publication claim identity is invalid")
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                now = _epoch_now()
                job = self.conn.execute(
                    "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if job is None:
                    raise StorageError("publication optimizer job does not exist")
                if job["optimizer_kind"] not in {"gepa", "offline_tuner_replay"}:
                    raise StorageError("publication optimizer kind is not publishable")
                if job["target_kind"] != "user_playbook":
                    raise StorageError("publication target is not a user playbook")
                if job["status"] not in {"pending", "running"}:
                    raise StorageError("publication optimizer job is terminal")
                if job["stage"] != "publishing":
                    raise StorageError(
                        "publication optimizer job is not at publishing stage"
                    )
                if job["lease_owner"] != owner:
                    raise StorageError("publication worker owner changed")
                if job["lease_fence"] != worker_fence:
                    raise StorageError("publication worker fence changed")
                _assert_publication_lease_live(job, now=now)
                existing = self.conn.execute(
                    """SELECT * FROM user_playbook_publication_claims
                       WHERE optimizer_kind = ? AND job_id = ?""",
                    (job["optimizer_kind"], job_id),
                ).fetchone()
                if existing is not None:
                    if existing["consumed"]:
                        raise StorageError("publication claim was already consumed")
                    if (
                        existing["owner"] == owner
                        and existing["worker_fence"] == worker_fence
                    ):
                        fence = int(existing["publication_fence"])
                    else:
                        fence = int(existing["publication_fence"]) + 1
                        self.conn.execute(
                            """UPDATE user_playbook_publication_claims
                               SET owner = ?, publication_fence = ?, worker_fence = ?,
                                   updated_at = ?
                               WHERE optimizer_kind = ? AND job_id = ?""",
                            (
                                owner,
                                fence,
                                worker_fence,
                                now,
                                job["optimizer_kind"],
                                job_id,
                            ),
                        )
                else:
                    fence = 1
                    self.conn.execute(
                        """INSERT INTO user_playbook_publication_claims
                           (optimizer_kind, job_id, owner, publication_fence,
                            worker_fence, consumed, updated_at)
                           VALUES (?, ?, ?, ?, ?, 0, ?)""",
                        (
                            job["optimizer_kind"],
                            job_id,
                            owner,
                            fence,
                            worker_fence,
                            now,
                        ),
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return PublicationClaim(job_id=job_id, owner=owner, fence=fence)

    @SQLiteStorageBase.handle_exceptions
    def stage_user_playbook_publication(self, request: PublicationRequest) -> None:
        request.__post_init__()
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                now = _epoch_now()
                existing = self.conn.execute(
                    """SELECT * FROM user_playbook_publication_staging
                       WHERE optimizer_kind = ? AND job_id = ?""",
                    (request.optimizer_kind, request.job_id),
                ).fetchone()
                terminal = self.conn.execute(
                    """SELECT staging_digest FROM user_playbook_publication_results
                       WHERE optimizer_kind = ? AND job_id = ?""",
                    (request.optimizer_kind, request.job_id),
                ).fetchone()
                if terminal is not None:
                    if existing is None:
                        raise StorageError(
                            "committed publication lost its staging record"
                        )
                    _assert_staging_matches(existing, request)
                    if terminal["staging_digest"] != existing["staging_digest"]:
                        raise StorageError(
                            "committed publication staging digest changed"
                        )
                    self.conn.commit()
                    return
                if existing is not None:
                    _assert_staging_matches(existing, request)
                    self._publication_job_locked(request, now=now)
                    self._publication_claim_locked(request)
                    self._publication_incumbent_and_subjects_locked(request)
                    self.conn.execute(
                        """UPDATE user_playbook_publication_staging
                           SET claim_owner = ?, publication_fence = ?, worker_fence = ?
                           WHERE optimizer_kind = ? AND job_id = ?""",
                        (
                            request.publication_claim.owner,
                            request.publication_claim.fence,
                            request.worker_fence,
                            request.optimizer_kind,
                            request.job_id,
                        ),
                    )
                    self.conn.commit()
                    return
                self._publication_job_locked(request, now=now)
                self._publication_claim_locked(request)
                self._publication_incumbent_and_subjects_locked(request)
                self.conn.execute(
                    """INSERT INTO user_playbook_publication_staging
                       (optimizer_kind, job_id, attempt_key, claim_owner,
                        publication_fence, worker_fence, incumbent_user_playbook_id,
                        revised_content, content_digest, projection_json,
                        projection_digest, proof_json, proof_digest,
                        subject_epochs_json, request_id, staging_digest, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        request.optimizer_kind,
                        request.job_id,
                        request.attempt_key,
                        request.publication_claim.owner,
                        request.publication_claim.fence,
                        request.worker_fence,
                        request.incumbent_user_playbook_id,
                        request.revised_content,
                        request.projection.candidate_content_digest,
                        request.projection.canonical_json,
                        request.projection.digest,
                        request.decision_proof.canonical_json,
                        request.decision_proof.digest,
                        request.subject_epochs_json,
                        request.request_id,
                        _publication_staging_digest(request),
                        now,
                    ),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _finish_publication_locked(
        self,
        request: PublicationRequest,
        *,
        outcome: str,
        successor_id: int | None,
        staging_digest: str,
        now: int,
    ) -> None:
        self.conn.execute(
            """INSERT INTO user_playbook_publication_results
               (optimizer_kind, job_id, outcome, successor_user_playbook_id,
                staging_digest, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                request.optimizer_kind,
                request.job_id,
                outcome,
                successor_id,
                staging_digest,
                now,
            ),
        )
        consumed = self.conn.execute(
            """UPDATE user_playbook_publication_claims
               SET consumed = 1, updated_at = ?
               WHERE optimizer_kind = ? AND job_id = ? AND owner = ?
                 AND publication_fence = ? AND worker_fence = ? AND consumed = 0""",
            (
                now,
                request.optimizer_kind,
                request.job_id,
                request.publication_claim.owner,
                request.publication_claim.fence,
                request.worker_fence,
            ),
        )
        if consumed.rowcount != 1:
            raise StorageError("publication claim consumption lost its fence")
        job_stage = "applied" if outcome == "applied" else "abstained"
        job_status = "completed" if outcome == "applied" else "skipped"
        updated = self.conn.execute(
            """UPDATE playbook_optimization_jobs
               SET stage = ?, terminal_outcome = ?, status = ?,
                   successor_target_id = ?, lease_owner = NULL,
                   lease_expires_at = NULL, updated_at = ?
               WHERE job_id = ? AND optimizer_kind = ? AND attempt_key = ?
                 AND target_kind = 'user_playbook' AND target_id = ?
                 AND status IN ('pending', 'running') AND stage = 'publishing'
                 AND lease_owner = ? AND lease_fence = ?
                 AND lease_expires_at > ?""",
            (
                job_stage,
                outcome,
                job_status,
                successor_id,
                now,
                request.job_id,
                request.optimizer_kind,
                request.attempt_key,
                request.incumbent_user_playbook_id,
                request.publication_claim.owner,
                request.worker_fence,
                now,
            ),
        )
        if updated.rowcount != 1:
            raise StorageError("publication optimizer job transition lost its fence")

    @SQLiteStorageBase.handle_exceptions
    def commit_user_playbook_publication(
        self, request: PublicationRequest
    ) -> PublicationResult:
        request.__post_init__()
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                now = _epoch_now()
                terminal = self.conn.execute(
                    """SELECT * FROM user_playbook_publication_results
                       WHERE optimizer_kind = ? AND job_id = ?""",
                    (request.optimizer_kind, request.job_id),
                ).fetchone()
                staged = self.conn.execute(
                    """SELECT * FROM user_playbook_publication_staging
                       WHERE optimizer_kind = ? AND job_id = ?""",
                    (request.optimizer_kind, request.job_id),
                ).fetchone()
                if staged is None:
                    raise StorageError("publication successor is not staged")
                _assert_staging_matches(staged, request)
                if terminal is not None:
                    if terminal["staging_digest"] != staged["staging_digest"]:
                        raise StorageError(
                            "committed publication staging digest changed"
                        )
                    result = PublicationResult(
                        job_id=request.job_id,
                        outcome=terminal["outcome"],
                        successor_user_playbook_id=terminal[
                            "successor_user_playbook_id"
                        ],
                    )
                    self.conn.commit()
                    return result
                _assert_staging_binding_matches(staged, request)
                self._publication_job_locked(request, now=now)
                self._publication_claim_locked(request)
                incumbent = self._publication_incumbent_and_subjects_locked(request)
                subject_ref = self._subject_ref_from_user_playbook_row(incumbent)
                if incumbent["status"] is not None:
                    self._finish_publication_locked(
                        request,
                        outcome="incumbent_changed",
                        successor_id=None,
                        staging_digest=staged["staging_digest"],
                        now=now,
                    )
                    self.conn.execute(
                        """INSERT INTO playbook_optimization_events
                           (job_id, event_type, payload_json, created_at)
                           VALUES (?, 'publication_incumbent_changed', ?, ?)""",
                        (
                            request.job_id,
                            json.dumps(
                                {
                                    "outcome": "incumbent_changed",
                                    "request_id": request.request_id,
                                },
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            now,
                        ),
                    )
                    self.conn.commit()
                    return PublicationResult(
                        job_id=request.job_id,
                        outcome="incumbent_changed",
                        successor_user_playbook_id=None,
                    )
                embedding = [float(value) for value in request.projection.embedding]
                created_at = _epoch_to_iso(now)
                inserted = self.conn.execute(
                    """INSERT INTO user_playbooks
                       (user_id, playbook_name, created_at, request_id, agent_version,
                        content, trigger, rationale, blocking_issue,
                        source_interaction_ids, status, source, embedding,
                        expanded_terms, source_span, notes, reader_angle, tags,
                        merged_into, superseded_by, governance_subject_ref)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?,
                               NULL, NULL, ?)""",
                    (
                        incumbent["user_id"],
                        incumbent["playbook_name"],
                        created_at,
                        request.request_id,
                        incumbent["agent_version"],
                        request.revised_content,
                        request.projection.preserved_trigger,
                        incumbent["rationale"],
                        incumbent["blocking_issue"],
                        incumbent["source_interaction_ids"],
                        request.optimizer_kind,
                        json.dumps(embedding, separators=(",", ":")),
                        " ".join(request.projection.expanded_terms),
                        incumbent["source_span"],
                        incumbent["notes"],
                        incumbent["reader_angle"],
                        incumbent["tags"],
                        subject_ref,
                    ),
                )
                successor_id = inserted.lastrowid
                if successor_id is None:
                    raise StorageError("publication successor insert returned no id")
                self.conn.execute(
                    "INSERT INTO user_playbooks_fts(rowid, search_text) VALUES (?, ?)",
                    (successor_id, request.projection.lexical_document),
                )
                if self._has_sqlite_vec:
                    self.conn.execute(
                        "INSERT INTO user_playbooks_vec(rowid, embedding) VALUES (?, ?)",
                        (successor_id, json.dumps(embedding)),
                    )
                superseded = self.conn.execute(
                    """UPDATE user_playbooks
                       SET status = ?, superseded_by = ?, retired_at = ?
                       WHERE user_playbook_id = ? AND status IS NULL""",
                    (
                        Status.SUPERSEDED.value,
                        successor_id,
                        now,
                        request.incumbent_user_playbook_id,
                    ),
                )
                if superseded.rowcount != 1:
                    raise StorageError("publication incumbent changed during commit")
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="user_playbook",
                    entity_id=str(successor_id),
                    op="revise",
                    prov="wasRevisionOf",
                    source_ids=[str(request.incumbent_user_playbook_id)],
                    actor=request.optimizer_kind,
                    request_id=request.request_id,
                    reason="atomic optimizer publication",
                    created_at=now,
                )
                event_payload = json.dumps(
                    {
                        "outcome": "applied",
                        "proof_digest": request.decision_proof.digest,
                        "projection_digest": request.projection.digest,
                        "request_id": request.request_id,
                        "successor_user_playbook_id": successor_id,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                self.conn.execute(
                    """INSERT INTO playbook_optimization_events
                       (job_id, event_type, payload_json, created_at)
                       VALUES (?, 'publication_applied', ?, ?)""",
                    (request.job_id, event_payload, now),
                )
                self._finish_publication_locked(
                    request,
                    outcome="applied",
                    successor_id=successor_id,
                    staging_digest=staged["staging_digest"],
                    now=now,
                )
                self.conn.commit()
                return PublicationResult(
                    job_id=request.job_id,
                    outcome="applied",
                    successor_user_playbook_id=successor_id,
                )
            except Exception:
                self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def load_user_playbook_publication_result(
        self, job_id: int
    ) -> PublicationResult | None:
        row = self._fetchone(
            """SELECT job_id, outcome, successor_user_playbook_id
               FROM user_playbook_publication_results WHERE job_id = ?""",
            (job_id,),
        )
        if row is None:
            return None
        return PublicationResult(
            job_id=row["job_id"],
            outcome=row["outcome"],
            successor_user_playbook_id=row["successor_user_playbook_id"],
        )

    def _subject_ref_from_user_playbook_row(self, row: sqlite3.Row) -> str:
        subject_ref = row["governance_subject_ref"]
        if subject_ref:
            return str(subject_ref)
        user_id = row["user_id"]
        if user_id is None or str(user_id) == "":
            raise ValueError("User playbook subject identity is missing")
        return self._subject_ref_for_user_id(str(user_id))

    def _assert_user_playbook_writable_locked(
        self,
        user_playbook_id: int,
    ) -> sqlite3.Row | None:
        row = self.conn.execute(
            "SELECT user_id, governance_subject_ref FROM user_playbooks WHERE user_playbook_id = ?",
            (user_playbook_id,),
        ).fetchone()
        if row is None:
            return None
        self._assert_subject_writable_locked(
            self._subject_ref_from_user_playbook_row(row)
        )
        return row

    def precompute_user_playbook_embeddings(
        self, playbooks: list[UserPlaybook]
    ) -> None:
        """Populate ``.embedding`` / ``.expanded_terms`` in place; no DB write.

        Extracted verbatim from the former ``save_user_playbooks`` prelude
        (including the ``if embedding_text:`` guard) so the durable
        compute/persist split can embed outside the writer transaction and then
        persist with ``skip_embedding=True``.
        """
        for up in playbooks:
            embedding_text = up.trigger or up.content
            if embedding_text:
                if self._should_expand_documents():
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        emb_future = executor.submit(
                            self._get_embedding, embedding_text
                        )
                        exp_future = executor.submit(
                            self._expand_document, embedding_text
                        )
                        up.embedding = emb_future.result(timeout=15)
                        up.expanded_terms = exp_future.result(timeout=15)
                else:
                    up.embedding = self._get_embedding(embedding_text)

    @SQLiteStorageBase.handle_exceptions
    def save_user_playbooks(
        self,
        user_playbooks: list[UserPlaybook],
        *,
        skip_embedding: bool = False,
        lineage_contexts: list[LineageContext] | None = None,
    ) -> None:
        if lineage_contexts is not None and len(lineage_contexts) != len(
            user_playbooks
        ):
            raise ValueError("lineage_contexts must match user_playbooks length")
        if any(context.op_kind != "create" for context in lineage_contexts or []):
            raise ValueError(
                "user playbook create lineage context must use op_kind='create'"
            )
        contexts = lineage_contexts or [
            LineageContext(
                op_kind="create",
                actor=up.source or "",
                source_ids=[str(value) for value in up.source_interaction_ids],
                request_id=up.request_id,
            )
            for up in user_playbooks
        ]
        rows: list[tuple[UserPlaybook, LineageContext, str, str]] = []
        for up, lineage_context in zip(user_playbooks, contexts, strict=True):
            subject_ref = self._subject_ref_for_user_id(up.user_id)
            with self._lock:
                self._assert_subject_writable_locked(subject_ref)
            # Default (skip_embedding=False) recomputes unconditionally, exactly
            # as before — model_copy callers that change content while keeping
            # the old embedding depend on this. The durable persist path opts
            # out (embedding already set by precompute_user_playbook_embeddings).
            if not skip_embedding:
                self.precompute_user_playbook_embeddings([up])
            rows.append(
                (up, lineage_context, subject_ref, _epoch_to_iso(up.created_at))
            )

        with self.commit_scope():
            for up, lineage_context, subject_ref, created_at_iso in rows:
                with self._lock:
                    self._assert_subject_writable_locked(subject_ref)
                    cur = self.conn.execute(
                        """INSERT INTO user_playbooks
                           (user_id, playbook_name, created_at, request_id, agent_version,
                            content, trigger, rationale, blocking_issue,
                            source_interaction_ids,
                            status, source, embedding, expanded_terms,
                            source_span, notes, reader_angle, tags,
                            merged_into, superseded_by, governance_subject_ref)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            up.user_id,
                            up.playbook_name,
                            created_at_iso,
                            up.request_id,
                            up.agent_version,
                            up.content,
                            up.trigger,
                            up.rationale,
                            json.dumps(up.blocking_issue.model_dump())
                            if up.blocking_issue
                            else None,
                            _json_dumps(up.source_interaction_ids or None),
                            up.status.value if up.status else None,
                            up.source,
                            _json_dumps(up.embedding),
                            up.expanded_terms,
                            up.source_span,
                            up.notes,
                            up.reader_angle,
                            _json_dumps(up.tags),
                            up.merged_into,
                            up.superseded_by,
                            subject_ref,
                        ),
                    )
                    upid = cur.lastrowid or 0
                    up.user_playbook_id = upid
                    _append_event_stmt(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="user_playbook",
                        entity_id=str(upid),
                        op="create",
                        prov="wasGeneratedBy",
                        source_ids=lineage_context.source_ids,
                        actor=lineage_context.actor,
                        request_id=lineage_context.request_id
                        or up.request_id
                        or f"create_{upid}",
                        reason=lineage_context.reason,
                        model_name=lineage_context.model_name,
                        provider=lineage_context.provider,
                    )

        for up, _lineage_context, _subject_ref, _created_at_iso in rows:
            upid = up.user_playbook_id
            fts_parts = [up.trigger or "", up.content or ""]
            if up.expanded_terms:
                fts_parts.append(up.expanded_terms)
            self._fts_upsert(
                "user_playbooks_fts",
                upid,
                search_text=" ".join(p for p in fts_parts if p) or "",
            )
            if up.embedding:
                self._vec_upsert("user_playbooks_vec", upid, up.embedding)

    @SQLiteStorageBase.handle_exceptions
    def get_user_playbooks(
        self,
        limit: int = 100,
        user_id: str | None = None,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        include_embedding: bool = False,
        tags: list[str] | None = None,
        offset: int = 0,
        user_playbook_id: int | None = None,
        request_id: str | None = None,
        query: str | None = None,
    ) -> list[UserPlaybook]:
        sql = "SELECT * FROM user_playbooks WHERE 1=1"
        params: list[Any] = []

        if user_playbook_id is not None:
            sql += " AND user_playbook_id = ?"
            params.append(user_playbook_id)
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        if request_id is not None:
            sql += " AND request_id = ?"
            params.append(request_id)
        if query:
            like = f"%{query.lower()}%"
            sql += (
                " AND (LOWER(content) LIKE ? OR LOWER(trigger) LIKE ? "
                "OR LOWER(rationale) LIKE ? OR LOWER(request_id) LIKE ? "
                "OR LOWER(playbook_name) LIKE ? OR LOWER(user_id) LIKE ?)"
            )
            params.extend([like, like, like, like, like, like])
        if playbook_name:
            sql += " AND playbook_name = ?"
            params.append(playbook_name)
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        if start_time is not None:
            sql += " AND created_at >= ?"
            params.append(_epoch_to_iso(start_time))
        if end_time is not None:
            sql += " AND created_at <= ?"
            params.append(_epoch_to_iso(end_time))
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            sql += f" AND {frag}"
            params.extend(sparams)
        else:
            # Default: exclude tombstone statuses (MERGED/SUPERSEDED)
            _ph = ",".join("?" * len(_TOMBSTONE_STATUS_VALUES))
            sql += f" AND (status IS NULL OR status NOT IN ({_ph}))"
            params.extend(_TOMBSTONE_STATUS_VALUES)
        tag_frag, tag_params = _build_tags_sql("user_playbooks", tags)
        if tag_frag:
            sql += f" AND {tag_frag}"
            params.extend(tag_params)

        sql += " ORDER BY created_at DESC, user_playbook_id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._fetchall(sql, params)
        return [
            _row_to_user_playbook(r, include_embedding=include_embedding) for r in rows
        ]

    @SQLiteStorageBase.handle_exceptions
    def count_user_playbooks(
        self,
        user_id: str | None = None,
        playbook_name: str | None = None,
        min_user_playbook_id: int | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) as cnt FROM user_playbooks WHERE 1=1"
        params: list[Any] = []

        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        if playbook_name:
            sql += " AND playbook_name = ?"
            params.append(playbook_name)
        if min_user_playbook_id is not None:
            sql += " AND user_playbook_id > ?"
            params.append(min_user_playbook_id)
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            sql += f" AND {frag}"
            params.extend(sparams)
        else:
            # Default: exclude tombstone statuses (MERGED/SUPERSEDED)
            _ph = ",".join("?" * len(_TOMBSTONE_STATUS_VALUES))
            sql += f" AND (status IS NULL OR status NOT IN ({_ph}))"
            params.extend(_TOMBSTONE_STATUS_VALUES)

        row = self._fetchone(sql, params)
        return row["cnt"] if row else 0

    @SQLiteStorageBase.handle_exceptions
    def count_user_playbooks_by_session(self, session_id: str) -> int:
        _ph = ",".join("?" * len(_TOMBSTONE_STATUS_VALUES))
        row = self._fetchone(
            f"""SELECT COUNT(*) as cnt FROM user_playbooks up
               JOIN requests r ON up.request_id = r.request_id
               WHERE r.session_id = ?
                 AND (up.status IS NULL OR up.status NOT IN ({_ph}))""",  # noqa: S608
            (session_id, *_TOMBSTONE_STATUS_VALUES),
        )
        return row["cnt"] if row else 0

    @SQLiteStorageBase.handle_exceptions
    def delete_all_user_playbooks(self) -> None:
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            ids = [
                r["user_playbook_id"]
                for r in self.conn.execute(
                    "SELECT user_playbook_id FROM user_playbooks"
                ).fetchall()
            ]
            self.conn.execute("DELETE FROM user_playbooks")
            for upid in ids:
                _emit_hard_delete_playbook(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="user_playbook",
                    entity_id=str(upid),
                    request_id=batch_request_id,
                )
            self.conn.commit()
        self._delete_playbook_search_rows("user", ids)

    @SQLiteStorageBase.handle_exceptions
    def delete_user_playbook(self, user_playbook_id: int) -> None:
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM user_playbooks WHERE user_playbook_id = ?",
                (user_playbook_id,),
            )
            if cur.rowcount > 0:
                _emit_hard_delete_playbook(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="user_playbook",
                    entity_id=str(user_playbook_id),
                    request_id=uuid.uuid4().hex,
                )
            self._delete_playbook_search_rows("user", [user_playbook_id], commit=False)
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_all_user_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        sql = "SELECT user_playbook_id FROM user_playbooks WHERE playbook_name = ?"
        params: list[Any] = [playbook_name]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            ids = [
                r["user_playbook_id"] for r in self.conn.execute(sql, params).fetchall()
            ]
            if not ids:
                return
            ph = ",".join("?" for _ in ids)
            self.conn.execute(
                f"DELETE FROM user_playbooks WHERE user_playbook_id IN ({ph})", ids
            )
            for upid in ids:
                _emit_hard_delete_playbook(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="user_playbook",
                    entity_id=str(upid),
                    request_id=batch_request_id,
                )
            self.conn.commit()
        self._delete_playbook_search_rows("user", ids)

    @SQLiteStorageBase.handle_exceptions
    def delete_user_playbooks_by_ids(
        self, user_playbook_ids: list[int], *, emit_hard_delete: bool = True
    ) -> int:
        if not user_playbook_ids:
            return 0
        ph = ",".join("?" for _ in user_playbook_ids)
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            existing = [
                r["user_playbook_id"]
                for r in self.conn.execute(
                    f"SELECT user_playbook_id FROM user_playbooks WHERE user_playbook_id IN ({ph})",
                    user_playbook_ids,
                ).fetchall()
            ]
            cur = self.conn.execute(
                f"DELETE FROM user_playbooks WHERE user_playbook_id IN ({ph})",
                user_playbook_ids,
            )
            if emit_hard_delete:
                for upid in existing:
                    _emit_hard_delete_playbook(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="user_playbook",
                        entity_id=str(upid),
                        request_id=batch_request_id,
                        actor="system",
                    )
            self.conn.commit()
        self._delete_playbook_search_rows("user", user_playbook_ids)
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def update_all_user_playbooks_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> int:
        new_val = new_status.value if new_status else None
        now_ts = _epoch_now()
        old_val_str = old_status.value if old_status else "None"
        new_val_str = new_status.value if new_status else "None"
        reason = f"{old_val_str}->{new_val_str}"

        if old_status is None or (
            hasattr(old_status, "value") and old_status.value is None
        ):
            where = "status IS NULL"
            select_params: list[Any] = []
        else:
            where = "status = ?"
            select_params = [old_status.value]

        extra_params: list[Any] = []
        if agent_version is not None:
            where += " AND agent_version = ?"
            extra_params.append(agent_version)
        if playbook_name is not None:
            where += " AND playbook_name = ?"
            extra_params.append(playbook_name)

        # Set retired_at = now when transitioning to a GC-eligible status; clear to NULL otherwise.
        retired_at_val = now_ts if new_val in _GC_ELIGIBLE_STATUSES else None

        batch_request_id = uuid.uuid4().hex
        with self._lock:
            affected = list(
                self.conn.execute(
                    f"SELECT user_playbook_id, user_id, governance_subject_ref FROM user_playbooks WHERE {where}",
                    select_params + extra_params,
                ).fetchall()
            )
            for row in affected:
                self._assert_subject_writable_locked(
                    self._subject_ref_from_user_playbook_row(row)
                )
            cur = self.conn.execute(
                f"UPDATE user_playbooks SET status = ?, retired_at = ? WHERE {where}",
                [new_val, retired_at_val] + select_params + extra_params,
            )
            from_val = old_status.value if old_status else None
            to_val = new_status.value if new_status else None
            for row in affected:
                upid = row["user_playbook_id"]
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="user_playbook",
                    entity_id=str(upid),
                    op="status_change",
                    prov="wasInvalidatedBy",
                    source_ids=[],
                    actor="api",
                    request_id=batch_request_id,
                    reason=reason,
                    from_status=from_val,
                    to_status=to_val,
                    status_namespace="lifecycle_status",
                )
            self.conn.commit()
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def delete_all_user_playbooks_by_status(
        self,
        status: Status,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> int:
        # Bulk delete-by-status emits no hard_delete lineage events (parity with
        # the Supabase backend, which routes this through _hard_delete_and_log with
        # emit_hard_delete=False). Accepts any status: the upgrade flow legitimately
        # deletes old ARCHIVED playbooks via _delete_items_by_status(Status.ARCHIVED).
        where = "status = ?"
        params: list[Any] = [status.value]
        if agent_version is not None:
            where += " AND agent_version = ?"
            params.append(agent_version)
        if playbook_name is not None:
            where += " AND playbook_name = ?"
            params.append(playbook_name)

        with self._lock:
            ids = [
                r["user_playbook_id"]
                for r in self.conn.execute(
                    f"SELECT user_playbook_id FROM user_playbooks WHERE {where}",
                    params,  # noqa: S608
                ).fetchall()
            ]
            if not ids:
                return 0
            ph = ",".join("?" for _ in ids)
            cur = self.conn.execute(
                f"DELETE FROM user_playbooks WHERE user_playbook_id IN ({ph})",  # noqa: S608
                ids,
            )
            self._delete_playbook_search_rows("user", ids, commit=False)
            self.conn.commit()
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def get_user_playbooks_by_ids(
        self,
        user_id: str,
        user_playbook_ids: list[int],
        status_filter: list[Status | None] | None = None,
        *,
        include_inactive: bool = False,
    ) -> list[UserPlaybook]:
        validate_include_inactive(
            include_inactive=include_inactive, status_filter=status_filter
        )
        if not user_playbook_ids:
            return []
        ph = ",".join("?" for _ in user_playbook_ids)
        if include_inactive:
            rows = self._fetchall(
                "SELECT * FROM user_playbooks "
                f"WHERE user_id = ? AND user_playbook_id IN ({ph})",
                (user_id, *user_playbook_ids),
            )
            return [_row_to_user_playbook(r) for r in rows]
        if status_filter is None:
            status_filter = [None]
        frag, sparams = _build_status_sql(status_filter)
        sql = (
            f"SELECT * FROM user_playbooks "
            f"WHERE user_id = ? AND user_playbook_id IN ({ph}) AND {frag}"
        )
        params: list[Any] = [user_id, *user_playbook_ids, *sparams]
        return [_row_to_user_playbook(r) for r in self._fetchall(sql, params)]

    @SQLiteStorageBase.handle_exceptions
    def archive_user_playbook_by_id(self, user_id: str, user_playbook_id: int) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id, governance_subject_ref FROM user_playbooks WHERE user_playbook_id = ? AND user_id = ?",
                (user_playbook_id, user_id),
            ).fetchone()
            if row is None:
                return False
            self._assert_subject_writable_locked(
                self._subject_ref_from_user_playbook_row(row)
            )
            cur = self.conn.execute(
                "UPDATE user_playbooks SET status = ?, retired_at = ? "
                "WHERE user_playbook_id = ? AND user_id = ? AND status IS NULL",
                (Status.ARCHIVED.value, _epoch_now(), user_playbook_id, user_id),
            )
            self.conn.commit()
        return cur.rowcount > 0

    @SQLiteStorageBase.handle_exceptions
    def has_user_playbooks_with_status(
        self,
        status: Status | None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> bool:
        sql = "SELECT 1 FROM user_playbooks WHERE "
        params: list[Any] = []

        if status is None or (hasattr(status, "value") and status.value is None):
            sql += "status IS NULL"
        else:
            sql += "status = ?"
            params.append(status.value)

        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        if playbook_name is not None:
            sql += " AND playbook_name = ?"
            params.append(playbook_name)

        sql += " LIMIT 1"
        row = self._fetchone(sql, params)
        return row is not None

    @SQLiteStorageBase.handle_exceptions
    def search_user_playbooks(  # noqa: C901
        self,
        request: SearchUserPlaybookRequest,
        options: SearchOptions | None = None,
    ) -> list[UserPlaybook]:
        query = request.query
        user_id = request.user_id
        agent_version = request.agent_version
        playbook_name = request.playbook_name
        start_time = int(request.start_time.timestamp()) if request.start_time else None
        end_time = int(request.end_time.timestamp()) if request.end_time else None
        status_filter = request.status_filter
        match_count = request.top_k or 10
        query_embedding = options.query_embedding if options else None
        mode = _effective_search_mode(
            request.search_mode, query_embedding, request.query
        )
        threshold = resolve_retrieval_threshold(
            request.threshold,
            model_name=self.embedding_model_name,
        )
        rrf_k = options.rrf_k if options else 60
        vector_weight = options.vector_weight if options else 1.0
        fts_weight = options.fts_weight if options else 1.0

        conditions: list[str] = []
        params: list[Any] = []

        if user_id:
            conditions.append("up.user_id = ?")
            params.append(user_id)
        if agent_version:
            conditions.append("up.agent_version = ?")
            params.append(agent_version)
        if playbook_name:
            conditions.append("up.playbook_name = ?")
            params.append(playbook_name)
        if start_time:
            conditions.append("up.created_at >= ?")
            params.append(_epoch_to_iso(start_time))
        if end_time:
            conditions.append("up.created_at <= ?")
            params.append(_epoch_to_iso(end_time))
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            conditions.append(frag)
            params.extend(sparams)
        else:
            # Default: exclude tombstone statuses (MERGED/SUPERSEDED)
            _ph = ",".join("?" * len(_TOMBSTONE_STATUS_VALUES))
            conditions.append(f"(up.status IS NULL OR up.status NOT IN ({_ph}))")
            params.extend(_TOMBSTONE_STATUS_VALUES)
        tag_frag, tag_params = _build_tags_sql("up", request.tags)
        if tag_frag:
            conditions.append(tag_frag)
            params.extend(tag_params)

        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""
        overfetch = match_count * 5 if mode != SearchMode.FTS else match_count

        # Pure vector search: fetch all candidates, rank by cosine similarity
        if mode == SearchMode.VECTOR and query_embedding:
            base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"""SELECT * FROM user_playbooks up
                      {base_where}
                      ORDER BY up.created_at DESC"""
            rows = self._fetchall(sql, params)
            rows = _vector_rank_rows(
                rows,
                query_embedding,
                match_count,
                threshold=threshold,
            )
            return [_row_to_user_playbook(r) for r in rows]

        if query:
            fts_query = _sanitize_fts_query(query)
            sql = f"""SELECT up.* FROM user_playbooks up
                      JOIN user_playbooks_fts f ON up.user_playbook_id = f.rowid
                      WHERE user_playbooks_fts MATCH ?{where_extra}
                      ORDER BY bm25(user_playbooks_fts, 1.0)
                      LIMIT ?"""
            fts_rows = self._fetchall(sql, [fts_query, *params, overfetch])

            if mode == SearchMode.HYBRID and query_embedding:
                base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
                vec_limit = match_count * 10
                vec_sql = f"""SELECT * FROM user_playbooks up
                              {base_where}
                              ORDER BY up.created_at DESC
                              LIMIT ?"""
                vec_candidates = self._fetchall(vec_sql, [*params, vec_limit])
                vec_rows = _vector_rank_rows(
                    vec_candidates,
                    query_embedding,
                    overfetch,
                    threshold=threshold,
                )
                rows = _true_rrf_merge(
                    fts_rows,
                    vec_rows,
                    "user_playbook_id",
                    match_count,
                    rrf_k,
                    vector_weight,
                    fts_weight,
                )
                return [_row_to_user_playbook(r) for r in rows]
            return [_row_to_user_playbook(r) for r in fts_rows[:match_count]]

        # HYBRID without query text: rank by embedding only
        if query_embedding:
            base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"""SELECT * FROM user_playbooks up
                      {base_where}
                      ORDER BY up.created_at DESC"""
            rows = self._fetchall(sql, params)
            rows = _vector_rank_rows(
                rows,
                query_embedding,
                match_count,
                threshold=threshold,
            )
            return [_row_to_user_playbook(r) for r in rows]

        # No query text, no embedding -- recency fallback
        base_where = "WHERE " + " AND ".join(conditions) if conditions else "WHERE 1=1"
        sql = f"""SELECT * FROM user_playbooks up
                  {base_where}
                  ORDER BY up.created_at DESC LIMIT ?"""
        params.append(match_count)
        rows = self._fetchall(sql, params)
        return [_row_to_user_playbook(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_user_playbook_by_id(
        self, user_playbook_id: int, *, include_tombstones: bool = False
    ) -> UserPlaybook | None:
        sql = "SELECT * FROM user_playbooks WHERE user_playbook_id = ?"
        if not include_tombstones:
            _ph = ",".join("?" * len(_TOMBSTONE_STATUS_VALUES))
            sql += f" AND (status IS NULL OR status NOT IN ({_ph}))"
            row = self._fetchone(sql, (user_playbook_id, *_TOMBSTONE_STATUS_VALUES))
        else:
            row = self._fetchone(sql, (user_playbook_id,))
        return _row_to_user_playbook(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def get_user_playbooks_by_ids_any_user(
        self,
        user_playbook_ids: list[int],
        status_filter: list[Status | None] | None = None,
    ) -> list[UserPlaybook]:
        if not user_playbook_ids:
            return []
        ph = ",".join("?" for _ in user_playbook_ids)
        sql = f"SELECT * FROM user_playbooks WHERE user_playbook_id IN ({ph})"  # noqa: S608
        params: list[Any] = list(user_playbook_ids)
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            sql += f" AND {frag}"
            params.extend(sparams)
        rows = self._fetchall(sql, params)
        by_id = {
            _row_to_user_playbook(row).user_playbook_id: _row_to_user_playbook(row)
            for row in rows
        }
        return [by_id[upid] for upid in user_playbook_ids if upid in by_id]

    @SQLiteStorageBase.handle_exceptions
    def update_user_playbook(
        self,
        user_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        trigger: str | None = None,
        rationale: str | None = None,
        blocking_issue: BlockingIssue | None = None,
        tags: list[str] | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        if playbook_name is not None:
            updates.append("playbook_name = ?")
            params.append(playbook_name)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if trigger is not None:
            updates.append("trigger = ?")
            params.append(trigger)
        if rationale is not None:
            updates.append("rationale = ?")
            params.append(rationale)
        if blocking_issue is not None:
            updates.append("blocking_issue = ?")
            params.append(json.dumps(blocking_issue.model_dump()))
        if tags is not None:
            updates.append("tags = ?")
            params.append(_json_dumps(tags))
        if updates:
            params.append(user_playbook_id)
            semantic_change = any(
                value is not None for value in (content, trigger, rationale)
            )
            op = "revise" if semantic_change else "status_change"
            prov = "wasRevisionOf" if op == "revise" else "wasInvalidatedBy"
            with self._lock:
                if self._assert_user_playbook_writable_locked(user_playbook_id) is None:
                    return
                cur = self.conn.execute(
                    f"UPDATE user_playbooks SET {', '.join(updates)} WHERE user_playbook_id = ?",
                    tuple(params),
                )
                if cur.rowcount > 0:
                    _append_event_stmt(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="user_playbook",
                        entity_id=str(user_playbook_id),
                        op=op,
                        prov=prov,
                        source_ids=[],
                        actor="api",
                        request_id=uuid.uuid4().hex,
                        reason="in-place update",
                        from_status=None,
                        to_status=None,
                        status_namespace=None,
                    )
                if self._own_transaction():
                    self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def supersede_user_playbooks_by_ids(
        self, user_playbook_ids: list[int], request_id: str
    ) -> int:
        """Soft-delete user playbooks by setting status to SUPERSEDED.

        Preserves the row content for strict point-in-time attribution reads.
        Eligible rows are any non-tombstoned status (CURRENT / PENDING /
        ARCHIVED). Atomic: all updates and lineage events commit together.
        """
        if not user_playbook_ids:
            return 0
        if not request_id:
            raise ValueError("request_id must be non-empty for supersede")
        now_ts = _epoch_now()
        updated = 0
        with self._lock:
            for upid in user_playbook_ids:
                row = self.conn.execute(
                    "SELECT status, user_id, governance_subject_ref FROM user_playbooks WHERE user_playbook_id = ?",
                    (upid,),
                ).fetchone()
                if row is None:
                    continue
                self._assert_subject_writable_locked(
                    self._subject_ref_from_user_playbook_row(row)
                )
                old_status = row["status"]
                _ph = ",".join("?" * len(_TOMBSTONE_STATUS_VALUES))
                cur = self.conn.execute(
                    "UPDATE user_playbooks SET status = ?, retired_at = ?"
                    " WHERE user_playbook_id = ?"
                    f" AND (status IS NULL OR status NOT IN ({_ph}))",
                    (
                        Status.SUPERSEDED.value,
                        now_ts,
                        upid,
                        *_TOMBSTONE_STATUS_VALUES,
                    ),
                )
                if cur.rowcount > 0:
                    _emit_supersede_user_playbook(
                        self.conn,
                        org_id=self.org_id,
                        entity_id=str(upid),
                        old_status=old_status,
                        request_id=request_id,
                    )
                    updated += 1
            if self._own_transaction():
                self.conn.commit()
        return updated
