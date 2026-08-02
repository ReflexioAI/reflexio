"""SQLite purge-operation store methods.

Extracted verbatim from ``_governance.py`` (the PurgeOperationStore bucket): the
seven public methods (``begin_purge_operation``, ``record_purge_target``,
``list_purge_targets``, ``purge_targets_prepared``,
``prepare_governance_erase_targets``, ``fail_purge_operation``,
``get_purge_operation``) plus the Purge-owned private
``_record_purge_target_locked`` (called cross-bucket by the residual rebuild-hide
/ governance-erase-execution methods, which reach it via MRO co-composition).

The residual ``SQLiteGovernanceMixin`` stays composed alongside this mixin and
permanently holds the shared infra (``conn``, ``_lock``, ``org_id``,
``_deps()``), the cross-bucket residents it calls
(``_owned_user_playbook_ids_locked``, ``_planned_governance_delete_counts``),
and the module-level helpers (``_json_dumps``, ``_row_to_purge_operation``,
``_row_to_purge_target``), which are imported here rather than duplicated.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from reflexio.models.api_schema.domain.governance import (
    PurgeOperation,
    PurgeOperationTarget,
)
from reflexio.server.services.storage.governance_claims import (
    PurgeExecutionClaim,
    validate_purge_execution_claim,
)
from reflexio.server.services.storage.governance_validation import (
    _ALLOWED_PURGE_OPERATION_TYPES,
    _ALLOWED_PURGE_SCOPE_TYPES,
    _ALLOWED_PURGE_TARGET_DETAIL_KEYS,
    _ALLOWED_PURGE_TARGET_NAMES,
    _ALLOWED_PURGE_TARGET_PHASES,
    _ALLOWED_PURGE_TARGET_STATUSES,
    _PREPARE_PHASE,
    _SNAPSHOT_TARGET_NAME,
    _epoch_now,
    _validate_governance_deleted_count,
    _validate_governance_detail,
    _validate_governance_enum,
    _validate_governance_error_code,
    _validate_governance_error_detail,
    _validate_governance_idempotency_key,
    _validate_governance_prefixed_ref,
    _validate_governance_purge_id,
    _validate_governance_target_ref,
)

from .._governance import _json_dumps, _row_to_purge_operation, _row_to_purge_target

if TYPE_CHECKING:
    from .._governance import _SQLiteGovernanceDeps


class PurgeOperationStoreMixin:
    """SQLite purge-operation store primitives."""

    # Type hints for instance attributes/methods provided via MRO by the
    # co-composed residual SQLiteGovernanceMixin / SQLiteStorageBase.
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str
    _deps: Callable[[], _SQLiteGovernanceDeps]
    _owned_user_playbook_ids_locked: Callable[[str], set[int]]
    _planned_governance_delete_counts: Callable[[str, set[int]], dict[str, int]]
    _subject_ref_for_user_id: Callable[[str], str]

    @staticmethod
    def _authoritative_user_digest(purge_id: str, user_id: str) -> str:
        return hashlib.sha256(f"{purge_id}\0{user_id}".encode()).hexdigest()

    def _assert_authoritative_user_identity_locked(
        self, purge_id: str, user_id: str
    ) -> str:
        row = self.conn.execute(
            """SELECT operation_type, scope_type, subject_ref,
                      authoritative_user_digest
               FROM purge_operations
               WHERE org_id = ? AND purge_id = ?""",
            (self.org_id, purge_id),
        ).fetchone()
        expected_digest = self._authoritative_user_digest(purge_id, user_id)
        if (
            row is None
            or row["operation_type"] != "user_erasure"
            or row["scope_type"] != "user"
            or row["subject_ref"] != self._subject_ref_for_user_id(user_id)
            or row["authoritative_user_digest"] != expected_digest
        ):
            raise ValueError("Purge authoritative user identity does not match")
        return expected_digest

    def _record_purge_target_locked(
        self,
        *,
        purge_id: str,
        target_name: str,
        target_ref: str,
        phase: str,
        status: Literal["pending", "running", "failed", "complete"],
        detail: dict[str, object] | None,
        deleted_count: int,
        error_detail: str | None,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        _validate_governance_enum(
            "target_name",
            target_name,
            allowed=_ALLOWED_PURGE_TARGET_NAMES,
        )
        _validate_governance_enum(
            "phase",
            phase,
            allowed=_ALLOWED_PURGE_TARGET_PHASES,
        )
        _validate_governance_enum(
            "status",
            status,
            allowed=_ALLOWED_PURGE_TARGET_STATUSES,
        )
        detail = _validate_governance_detail(
            "detail",
            detail,
            allowed_keys=_ALLOWED_PURGE_TARGET_DETAIL_KEYS,
        )
        error_detail = _validate_governance_error_detail(error_detail)
        target_ref = _validate_governance_target_ref(
            target_name=target_name,
            phase=phase,
            target_ref=target_ref,
        )
        deleted_count = _validate_governance_deleted_count(deleted_count)
        now = _epoch_now()
        existing = self.conn.execute(
            """SELECT started_at, completed_at
               FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = ? AND phase = ?""",
            (self.org_id, purge_id, target_name, target_ref, phase),
        ).fetchone()
        started_at = existing["started_at"] if existing else None
        completed_at = existing["completed_at"] if existing else None
        if started_at is None and status in {"running", "failed", "complete"}:
            started_at = now
        if status in {"failed", "complete"}:
            completed_at = now
        self.conn.execute(
            """INSERT INTO purge_operation_targets (
                   org_id, purge_id, target_name, target_ref, phase, status, detail,
                   deleted_count, error_detail, started_at, completed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(org_id, purge_id, target_name, target_ref, phase) DO UPDATE SET
                   status = excluded.status,
                   detail = COALESCE(excluded.detail, purge_operation_targets.detail),
                   deleted_count = excluded.deleted_count,
                   error_detail = excluded.error_detail,
                   started_at = COALESCE(purge_operation_targets.started_at, excluded.started_at),
                   completed_at = excluded.completed_at""",
            (
                self.org_id,
                purge_id,
                target_name,
                target_ref,
                phase,
                status,
                _json_dumps(detail),
                deleted_count,
                error_detail,
                started_at,
                completed_at,
            ),
        )
        self.conn.execute(
            """UPDATE purge_operations
               SET status = CASE
                   WHEN status IN ('complete', 'failed') THEN status
                   WHEN ? IN ('running', 'complete') THEN 'running'
                   ELSE status
               END,
                   updated_at = ?
               WHERE purge_id = ? AND org_id = ?""",
            (status, now, purge_id, self.org_id),
        )

    def begin_purge_operation(
        self,
        purge_id: str,
        idempotency_key: str,
        operation_type: Literal["user_erasure", "org_purge"],
        scope_type: Literal["user", "org"],
        subject_ref: str | None,
        request_ref: str,
        authoritative_user_id: str | None = None,
    ) -> PurgeOperation:
        _validate_governance_enum(
            "operation_type",
            operation_type,
            allowed=_ALLOWED_PURGE_OPERATION_TYPES,
        )
        _validate_governance_enum(
            "scope_type",
            scope_type,
            allowed=_ALLOWED_PURGE_SCOPE_TYPES,
        )
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        _validate_governance_prefixed_ref(
            "request_ref", request_ref, prefix="reqref_v1_"
        )
        validated_purge_id = _validate_governance_purge_id("purge_id", purge_id)
        validated_idempotency_key = cast(
            str,
            _validate_governance_idempotency_key("idempotency_key", idempotency_key),
        )
        if operation_type == "user_erasure" and scope_type == "user":
            if not authoritative_user_id:
                raise ValueError("authoritative user identity is required")
            if subject_ref != self._subject_ref_for_user_id(authoritative_user_id):
                raise ValueError("authoritative user identity must match subject_ref")
        elif authoritative_user_id:
            raise ValueError(
                "authoritative user identity is only valid for user erasure"
            )
        authoritative_user_digest = (
            self._authoritative_user_digest(validated_purge_id, authoritative_user_id)
            if authoritative_user_id
            else None
        )
        now = _epoch_now()
        with self._lock:
            existing = self.conn.execute(
                """SELECT * FROM purge_operations
                   WHERE org_id = ? AND idempotency_key = ?""",
                (self.org_id, validated_idempotency_key),
            ).fetchone()
            if existing is not None:
                existing_operation = _row_to_purge_operation(existing)
                expected_identity = {
                    "purge_id": validated_purge_id,
                    "operation_type": operation_type,
                    "scope_type": scope_type,
                    "subject_ref": subject_ref,
                    "request_ref": request_ref,
                }
                for field_name, expected_value in expected_identity.items():
                    if getattr(existing_operation, field_name) != expected_value:
                        raise ValueError(
                            "Existing purge operation for idempotency_key has "
                            f"mismatched {field_name}"
                        )
                if existing["authoritative_user_digest"] != authoritative_user_digest:
                    raise ValueError(
                        "Existing purge operation has mismatched authoritative user identity"
                    )
                return _row_to_purge_operation(existing)
            self.conn.execute(
                """INSERT INTO purge_operations (
                       purge_id, org_id, operation_type, scope_type, subject_ref,
                       request_ref, idempotency_key, authoritative_user_digest,
                       status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    validated_purge_id,
                    self.org_id,
                    operation_type,
                    scope_type,
                    subject_ref,
                    request_ref,
                    validated_idempotency_key,
                    authoritative_user_digest,
                    now,
                    now,
                ),
            )
            self.conn.commit()
        return self.get_purge_operation(validated_purge_id)

    def claim_purge_operation_execution(
        self,
        purge_id: str,
        *,
        lease_owner: str,
        lease_ttl_seconds: int,
    ) -> PurgeExecutionClaim | None:
        validated_purge_id = _validate_governance_purge_id("purge_id", purge_id)
        if not lease_owner.strip():
            raise ValueError("lease_owner is required")
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        now = _epoch_now()
        expires_at = now + lease_ttl_seconds
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                cursor = self.conn.execute(
                    """UPDATE purge_operations
                       SET status = 'running', error_code = NULL, error_detail = NULL,
                           completed_at = NULL, updated_at = ?,
                           execution_claim_owner = ?,
                           execution_claim_fence = execution_claim_fence + 1,
                           execution_claim_expires_at = ?
                       WHERE purge_id = ? AND org_id = ?
                         AND (
                            status IN ('pending', 'failed')
                            OR (
                                status = 'running'
                                AND (
                                    execution_claim_expires_at IS NULL
                                    OR execution_claim_expires_at <= ?
                                )
                            )
                         )
                       RETURNING execution_claim_owner,
                                 execution_claim_fence,
                                 execution_claim_expires_at""",
                    (
                        now,
                        lease_owner,
                        expires_at,
                        validated_purge_id,
                        self.org_id,
                        now,
                    ),
                )
                row = cursor.fetchone()
                self.conn.commit()
                if row is None:
                    return None
                return PurgeExecutionClaim(
                    purge_id=validated_purge_id,
                    owner=str(row["execution_claim_owner"]),
                    fence=int(row["execution_claim_fence"]),
                    expires_at=int(row["execution_claim_expires_at"]),
                )
            except Exception:
                self.conn.rollback()
                raise

    def assert_purge_operation_execution_claim(
        self, purge_id: str, execution_claim: PurgeExecutionClaim
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        claim = validate_purge_execution_claim(purge_id, execution_claim)
        now = _epoch_now()
        row = self._deps()._fetchone(
            """SELECT status, execution_claim_owner, execution_claim_fence,
                      execution_claim_expires_at
               FROM purge_operations
               WHERE purge_id = ? AND org_id = ?""",
            (purge_id, self.org_id),
        )
        if row is None:
            raise ValueError(f"Purge operation {purge_id!r} not found")
        if (
            row["status"] != "running"
            or row["execution_claim_owner"] != claim.owner
            or int(row["execution_claim_fence"]) != claim.fence
            or row["execution_claim_expires_at"] is None
            or int(row["execution_claim_expires_at"]) <= now
        ):
            raise ValueError("purge execution claim is no longer active")

    def _assert_purge_operation_execution_claim_locked(
        self,
        purge_id: str,
        execution_claim: PurgeExecutionClaim,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        claim = validate_purge_execution_claim(purge_id, execution_claim)
        now = _epoch_now()
        row = self.conn.execute(
            """SELECT status, execution_claim_owner, execution_claim_fence,
                      execution_claim_expires_at
               FROM purge_operations
               WHERE purge_id = ? AND org_id = ?""",
            (purge_id, self.org_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"Purge operation {purge_id!r} not found")
        if (
            row["status"] != "running"
            or row["execution_claim_owner"] != claim.owner
            or int(row["execution_claim_fence"]) != claim.fence
            or row["execution_claim_expires_at"] is None
            or int(row["execution_claim_expires_at"]) <= now
        ):
            raise ValueError("purge execution claim is no longer active")

    def renew_purge_operation_execution_claim(
        self,
        purge_id: str,
        execution_claim: PurgeExecutionClaim,
        *,
        lease_ttl_seconds: int,
    ) -> PurgeExecutionClaim:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        claim = validate_purge_execution_claim(purge_id, execution_claim)
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        now = _epoch_now()
        expires_at = now + lease_ttl_seconds
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                cursor = self.conn.execute(
                    """UPDATE purge_operations
                       SET execution_claim_expires_at = ?, updated_at = ?
                       WHERE purge_id = ? AND org_id = ?
                         AND status = 'running'
                         AND execution_claim_owner = ?
                         AND execution_claim_fence = ?
                         AND execution_claim_expires_at IS NOT NULL
                         AND execution_claim_expires_at > ?
                       RETURNING execution_claim_owner,
                                 execution_claim_fence,
                                 execution_claim_expires_at""",
                    (
                        expires_at,
                        now,
                        purge_id,
                        self.org_id,
                        claim.owner,
                        claim.fence,
                        now,
                    ),
                )
                row = cursor.fetchone()
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        if row is None:
            raise ValueError("purge execution claim is no longer active")
        return PurgeExecutionClaim(
            purge_id=purge_id,
            owner=str(row["execution_claim_owner"]),
            fence=int(row["execution_claim_fence"]),
            expires_at=int(row["execution_claim_expires_at"]),
        )

    def record_purge_target(
        self,
        purge_id: str,
        target_name: str,
        phase: str,
        status: Literal["pending", "running", "failed", "complete"],
        *,
        execution_claim: PurgeExecutionClaim,
        target_ref: str = "",
        detail: dict[str, object] | None = None,
        deleted_count: int = 0,
        error_detail: str | None = None,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        _validate_governance_enum(
            "target_name",
            target_name,
            allowed=_ALLOWED_PURGE_TARGET_NAMES,
        )
        _validate_governance_enum(
            "phase",
            phase,
            allowed=_ALLOWED_PURGE_TARGET_PHASES,
        )
        _validate_governance_enum(
            "status",
            status,
            allowed=_ALLOWED_PURGE_TARGET_STATUSES,
        )
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_purge_operation_execution_claim_locked(
                    purge_id, execution_claim
                )
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name=target_name,
                    target_ref=target_ref,
                    phase=phase,
                    status=status,
                    detail=detail,
                    deleted_count=deleted_count,
                    error_detail=error_detail,
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def list_purge_targets(
        self, purge_id: str, phase: str | None = None
    ) -> list[PurgeOperationTarget]:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        deps = self._deps()
        sql = "SELECT * FROM purge_operation_targets WHERE org_id = ? AND purge_id = ?"
        params: list[Any] = [self.org_id, purge_id]
        if phase is not None:
            sql += " AND phase = ?"
            params.append(phase)
        sql += " ORDER BY phase ASC, target_name ASC, target_ref ASC"
        rows = deps._fetchall(sql, params)
        return [_row_to_purge_target(row) for row in rows]

    def purge_targets_prepared(self, purge_id: str) -> bool:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        row = self._deps()._fetchone(
            """SELECT 1 FROM purge_operation_targets
               WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                 AND phase = ? AND status = 'complete'""",
            (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
        )
        return row is not None

    def prepare_governance_erase_targets(
        self,
        purge_id: str,
        user_id: str,
        *,
        execution_claim: PurgeExecutionClaim,
        owned_user_playbook_ids: set[int] | None = None,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_purge_operation_execution_claim_locked(
                    purge_id, execution_claim
                )
                authoritative_user_digest = (
                    self._assert_authoritative_user_identity_locked(purge_id, user_id)
                )
                prepared = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                         AND phase = ? AND status = 'complete'""",
                    (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
                ).fetchone()
                if prepared is not None:
                    self.conn.commit()
                    return
                owned_user_playbook_ids = (
                    set(owned_user_playbook_ids)
                    if owned_user_playbook_ids is not None
                    else self._owned_user_playbook_ids_locked(user_id)
                )
                targets = self._planned_governance_delete_counts(
                    user_id,
                    owned_user_playbook_ids,
                )
                for target_name, count in targets.items():
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name=target_name,
                        target_ref="all",
                        phase="delete",
                        status="pending",
                        detail={"count": count},
                        deleted_count=0,
                        error_detail=None,
                    )
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name=_SNAPSHOT_TARGET_NAME,
                    target_ref="all",
                    phase=_PREPARE_PHASE,
                    status="complete",
                    detail={
                        "authoritative_user_digest": authoritative_user_digest,
                        "owned_user_playbook_ids": sorted(owned_user_playbook_ids),
                    },
                    deleted_count=0,
                    error_detail=None,
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def fail_purge_operation(
        self,
        purge_id: str,
        error_code: str,
        error_detail: str,
        *,
        execution_claim: PurgeExecutionClaim,
    ) -> PurgeOperation:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        validated_error_code = _validate_governance_error_code(error_code)
        validated_error_detail = _validate_governance_error_detail(error_detail)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_purge_operation_execution_claim_locked(
                    purge_id, execution_claim
                )
                cur = self.conn.execute(
                    """UPDATE purge_operations
                       SET status = 'failed', error_code = ?, error_detail = ?,
                       updated_at = ?, completed_at = ?,
                       execution_claim_owner = NULL,
                       execution_claim_expires_at = NULL
                       WHERE purge_id = ? AND org_id = ? AND status != 'complete'""",
                    (
                        validated_error_code,
                        validated_error_detail,
                        now,
                        now,
                        purge_id,
                        self.org_id,
                    ),
                )
                if cur.rowcount == 0:
                    existing = self.conn.execute(
                        "SELECT status FROM purge_operations WHERE purge_id = ? AND org_id = ?",
                        (purge_id, self.org_id),
                    ).fetchone()
                    if existing is not None and str(existing["status"]) == "complete":
                        raise ValueError("Purge operation is already complete")
                    raise ValueError(f"Purge operation {purge_id!r} not found")
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return self.get_purge_operation(purge_id)

    def get_purge_operation(self, purge_id: str) -> PurgeOperation:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        row = self._deps()._fetchone(
            "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
            (purge_id, self.org_id),
        )
        if row is None:
            raise ValueError(f"Purge operation {purge_id!r} not found")
        return _row_to_purge_operation(row)
