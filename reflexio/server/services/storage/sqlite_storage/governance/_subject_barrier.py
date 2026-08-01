"""SQLite subject-erasure-barrier store methods.

Extracted verbatim from ``_governance.py`` (the SubjectBarrier bucket): the five
public methods (``begin_subject_erasure_barrier``, ``assert_subject_writable``,
``complete_subject_erasure_barrier_after_empty_check``,
``fail_subject_erasure_barrier``, ``get_subject_write_barrier``) plus the seven
SubjectBarrier-owned privates (``_barrier_from_purge``,
``_active_subject_barrier_locked``, ``_assert_subject_writable_locked``,
``_subject_ref_for_user_id``, ``_legacy_request_ids_for_subject_locked``,
``_legacy_user_id_rows_remain_locked``, ``_same_subject_rows_remain_locked``).

``complete_subject_erasure_barrier_after_empty_check`` is one of the two methods
authorized to write a successful-ERASE audit row — the ERASE-audit-idempotency
block and the ``rowcount != 1``-guarded barrier flip are preserved byte-for-byte.

The residual ``SQLiteGovernanceMixin`` stays composed alongside this mixin and
permanently holds the shared infra (``conn``, ``_lock``, ``org_id``) and the
module-level helpers (``_row_to_audit_event``, ``_row_to_purge_operation``,
``_row_to_subject_write_barrier``), which are imported here rather than
duplicated. ``_append_audit_event_with_cursor`` (AuditEventStore) and
``get_purge_operation`` (PurgeOperationStore) resolve through the composed MRO.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable

from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
    SubjectBarrierStatus,
    SubjectWriteBarrier,
)
from reflexio.server.services.governance.config import (
    get_governance_ref_secret,
    governance_subject_ref,
)
from reflexio.server.services.storage.error import SubjectWriteBarrierError
from reflexio.server.services.storage.governance_claims import PurgeExecutionClaim
from reflexio.server.services.storage.governance_validation import (
    _CANONICAL_DELETE_TARGET_NAMES,
    _PREPARE_PHASE,
    _SNAPSHOT_TARGET_NAME,
    _canonicalize_audit_event_for_persistence,
    _epoch_now,
    _is_successful_erase_event,
    _successful_erase_identity,
    _validate_governance_error_code,
    _validate_governance_error_detail,
    _validate_governance_prefixed_ref,
    _validate_governance_purge_id,
)

from .._governance import (
    _row_to_audit_event,
    _row_to_purge_operation,
    _row_to_subject_write_barrier,
)


class SubjectBarrierMixin:
    """SQLite subject-erasure-barrier store primitives."""

    # Type hints for instance attributes/methods provided via MRO by the
    # co-composed residual SQLiteGovernanceMixin / SQLiteStorageBase.
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str

    # Provided via MRO by the co-composed AuditEventStoreMixin / the residual
    # PurgeOperationStoreMixin; reached here by the cross-bucket barrier
    # completion method.
    _append_audit_event_with_cursor: Callable[
        [sqlite3.Connection | sqlite3.Cursor, AuditEvent], bool
    ]
    get_purge_operation: Callable[[str], PurgeOperation]
    _assert_purge_operation_execution_claim_locked: Callable[
        [str, PurgeExecutionClaim | None], None
    ]

    def _barrier_from_purge(
        self,
        purge_operation: PurgeOperation,
        *,
        subject_ref: str,
    ) -> SubjectWriteBarrier:
        if purge_operation.subject_ref != subject_ref:
            raise ValueError(
                "Purge operation subject_ref must match the barrier subject_ref"
            )
        status_by_purge_status: dict[str, SubjectBarrierStatus] = {
            "pending": "erasing",
            "running": "erasing",
            "complete": "erased",
            "failed": "failed",
        }
        return SubjectWriteBarrier(
            org_id=purge_operation.org_id,
            subject_ref=subject_ref,
            purge_id=purge_operation.purge_id,
            status=status_by_purge_status[purge_operation.status],
            error_code=purge_operation.error_code,
            error_detail=purge_operation.error_detail,
            created_at=purge_operation.created_at,
            updated_at=purge_operation.updated_at,
        )

    def _active_subject_barrier_locked(self, subject_ref: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT * FROM subject_write_barriers
               WHERE org_id = ? AND subject_ref = ? AND status IN ('erasing', 'erased')""",
            (self.org_id, subject_ref),
        ).fetchone()

    def _assert_subject_writable_locked(self, subject_ref: str) -> None:
        row = self._active_subject_barrier_locked(subject_ref)
        if row is not None:
            raise SubjectWriteBarrierError(
                f"subject {subject_ref} is blocked by erasure barrier {row['purge_id']}"
            )

    def _subject_ref_for_user_id(self, user_id: str) -> str:
        return governance_subject_ref(
            self.org_id,
            user_id,
            get_governance_ref_secret(),
        )

    def _legacy_request_ids_for_subject_locked(self, subject_ref: str) -> set[str]:
        request_ids: set[str] = set()
        for row in self.conn.execute(
            """SELECT request_id, user_id
               FROM requests
               WHERE governance_subject_ref IS NULL"""
        ):
            user_id = str(row["user_id"])
            if self._subject_ref_for_user_id(user_id) != subject_ref:
                continue
            request_ids.add(str(row["request_id"]))
        return request_ids

    def _legacy_user_id_rows_remain_locked(
        self,
        *,
        table: str,
        subject_ref: str,
        request_ids: set[str] | None = None,
        request_id_column: str | None = None,
    ) -> bool:
        sql = f"SELECT user_id{', ' + request_id_column if request_id_column else ''} FROM {table} WHERE governance_subject_ref IS NULL"  # noqa: S608
        for row in self.conn.execute(sql):
            user_id = str(row["user_id"])
            if self._subject_ref_for_user_id(user_id) == subject_ref:
                return True
            if (
                request_ids
                and request_id_column is not None
                and str(row[request_id_column]) in request_ids
            ):
                return True
        return False

    def _same_subject_rows_remain_locked(self, subject_ref: str) -> bool:
        legacy_request_ids = self._legacy_request_ids_for_subject_locked(subject_ref)
        for table in (
            "requests",
            "interactions",
            "profiles",
            "user_playbooks",
            "agent_success_evaluation_result",
            "retrieved_learning_evaluation",
            "session_outcomes",
        ):
            row = self.conn.execute(
                f"""SELECT 1 FROM {table}
                    WHERE governance_subject_ref = ?
                    LIMIT 1""",
                (subject_ref,),
            ).fetchone()
            if row is not None:
                return True
        if legacy_request_ids:
            return True
        if self._legacy_user_id_rows_remain_locked(
            table="interactions",
            subject_ref=subject_ref,
            request_ids=legacy_request_ids,
            request_id_column="request_id",
        ):
            return True
        if self._legacy_user_id_rows_remain_locked(
            table="profiles",
            subject_ref=subject_ref,
            request_ids=legacy_request_ids,
            request_id_column="generated_from_request_id",
        ):
            return True
        if self._legacy_user_id_rows_remain_locked(
            table="user_playbooks",
            subject_ref=subject_ref,
            request_ids=legacy_request_ids,
            request_id_column="request_id",
        ):
            return True
        if self._legacy_user_id_rows_remain_locked(
            table="agent_success_evaluation_result",
            subject_ref=subject_ref,
        ):
            return True
        return self._legacy_user_id_rows_remain_locked(
            table="retrieved_learning_evaluation",
            subject_ref=subject_ref,
        )

    def begin_subject_erasure_barrier(
        self,
        subject_ref: str,
        purge_id: str,
        *,
        execution_claim: PurgeExecutionClaim,
    ) -> SubjectWriteBarrier:
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        validated_purge_id = _validate_governance_purge_id("purge_id", purge_id)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_purge_operation_execution_claim_locked(
                    validated_purge_id, execution_claim
                )
                purge_row = self.conn.execute(
                    """SELECT * FROM purge_operations
                       WHERE purge_id = ? AND org_id = ?""",
                    (validated_purge_id, self.org_id),
                ).fetchone()
                if purge_row is None:
                    raise ValueError(
                        f"Purge operation {validated_purge_id!r} not found"
                    )
                purge_operation = _row_to_purge_operation(purge_row)
                if purge_operation.subject_ref != subject_ref:
                    raise ValueError(
                        "Purge operation subject_ref must match the barrier subject_ref"
                    )
                existing_barrier = self.conn.execute(
                    """SELECT * FROM subject_write_barriers
                       WHERE org_id = ? AND subject_ref = ?""",
                    (self.org_id, subject_ref),
                ).fetchone()
                if (
                    existing_barrier is not None
                    and str(existing_barrier["purge_id"]) != validated_purge_id
                ):
                    raise ValueError(
                        "Existing barrier purge_id must match the requested purge_id"
                    )
                if (
                    existing_barrier is not None
                    and str(existing_barrier["status"]) == "erased"
                ):
                    row = existing_barrier
                    self.conn.commit()
                    return _row_to_subject_write_barrier(row)
                self.conn.execute(
                    """INSERT INTO subject_write_barriers
                       (org_id, subject_ref, purge_id, status, created_at, updated_at)
                       VALUES (?, ?, ?, 'erasing', ?, ?)
                       ON CONFLICT(org_id, subject_ref) DO UPDATE SET
                         purge_id = excluded.purge_id,
                         status = 'erasing',
                         error_code = NULL,
                         error_detail = NULL,
                         updated_at = excluded.updated_at""",
                    (self.org_id, subject_ref, validated_purge_id, now, now),
                )
                row = self.conn.execute(
                    """SELECT * FROM subject_write_barriers
                       WHERE org_id = ? AND subject_ref = ?""",
                    (self.org_id, subject_ref),
                ).fetchone()
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        if row is None:
            raise ValueError("subject erasure barrier insert failed")
        return _row_to_subject_write_barrier(row)

    def assert_subject_writable(self, subject_ref: str) -> None:
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_subject_writable_locked(subject_ref)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def complete_subject_erasure_barrier_after_empty_check(
        self,
        purge_id: str,
        audit_event: AuditEvent,
        *,
        execution_claim: PurgeExecutionClaim,
    ) -> PurgeOperation:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        if audit_event.org_id != self.org_id:
            raise ValueError("Audit event org_id must match storage org_id")
        if audit_event.idempotency_key != purge_id:
            raise ValueError("Audit event idempotency key must match purge_id")
        if not _is_successful_erase_event(audit_event, purge_id=purge_id):
            raise ValueError(
                "Completion requires a successful ERASE audit event for this purge"
            )
        audit_event = _canonicalize_audit_event_for_persistence(audit_event)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_purge_operation_execution_claim_locked(
                    purge_id, execution_claim
                )
                row = self.conn.execute(
                    "SELECT * FROM purge_operations WHERE purge_id = ? AND org_id = ?",
                    (purge_id, self.org_id),
                ).fetchone()
                if row is None:
                    raise ValueError(f"Purge operation {purge_id!r} not found")
                purge_operation = _row_to_purge_operation(row)
                if purge_operation.subject_ref != audit_event.subject_ref:
                    raise ValueError(
                        "Audit event subject_ref must match purge operation subject_ref"
                    )
                if purge_operation.request_ref != audit_event.request_ref:
                    raise ValueError(
                        "Audit event request_ref must match purge operation request_ref"
                    )
                snapshot = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = ? AND target_ref = 'all'
                         AND phase = ? AND status = 'complete'""",
                    (self.org_id, purge_id, _SNAPSHOT_TARGET_NAME, _PREPARE_PHASE),
                ).fetchone()
                if snapshot is None:
                    raise ValueError(
                        "Cannot complete purge without target snapshot marker"
                    )
                if self._same_subject_rows_remain_locked(audit_event.subject_ref or ""):
                    raise ValueError("same-subject rows remain")
                delete_rows = self.conn.execute(
                    """SELECT target_name, status FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND phase = 'delete'
                         AND target_ref = 'all'""",
                    (self.org_id, purge_id),
                ).fetchall()
                delete_statuses = {
                    str(target_row["target_name"]): str(target_row["status"])
                    for target_row in delete_rows
                }
                missing_delete_targets = [
                    target_name
                    for target_name in _CANONICAL_DELETE_TARGET_NAMES
                    if delete_statuses.get(target_name) != "complete"
                ]
                if missing_delete_targets:
                    raise ValueError(
                        "Cannot complete purge without complete delete target matrix: "
                        + ", ".join(missing_delete_targets)
                    )
                incomplete = self.conn.execute(
                    """SELECT 1 FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND status != 'complete'
                       LIMIT 1""",
                    (self.org_id, purge_id),
                ).fetchone()
                if incomplete is not None:
                    raise ValueError("Cannot complete purge with incomplete targets")
                existing_audit_row = self.conn.execute(
                    """SELECT * FROM audit_events
                       WHERE org_id = ? AND idempotency_key = ?""",
                    (self.org_id, purge_id),
                ).fetchone()
                if existing_audit_row is not None:
                    existing_event = _row_to_audit_event(existing_audit_row)
                    if not _is_successful_erase_event(
                        existing_event, purge_id=purge_id
                    ):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                    if _successful_erase_identity(
                        existing_event
                    ) != _successful_erase_identity(audit_event):
                        raise ValueError(
                            "Existing audit row for purge_id must be the matching "
                            "successful ERASE row"
                        )
                else:
                    self._append_audit_event_with_cursor(self.conn, audit_event)
                    existing_audit_row = self.conn.execute(
                        """SELECT * FROM audit_events
                           WHERE org_id = ? AND idempotency_key = ?""",
                        (self.org_id, purge_id),
                    ).fetchone()
                if existing_audit_row is None:
                    raise ValueError(
                        "Completion requires exactly one successful ERASE audit row "
                        "for the purge_id"
                    )
                existing_event = _row_to_audit_event(existing_audit_row)
                if not _is_successful_erase_event(existing_event, purge_id=purge_id):
                    raise ValueError(
                        "Completion requires exactly one matching successful ERASE "
                        "audit row for the purge_id"
                    )
                if _successful_erase_identity(
                    existing_event
                ) != _successful_erase_identity(audit_event):
                    raise ValueError(
                        "Completion requires exactly one matching successful ERASE "
                        "audit row for the purge_id"
                    )
                barrier_update = self.conn.execute(
                    """UPDATE subject_write_barriers
                       SET status = 'erased', error_code = NULL, error_detail = NULL, updated_at = ?
                       WHERE org_id = ? AND subject_ref = ? AND purge_id = ? AND status = 'erasing'""",
                    (now, self.org_id, audit_event.subject_ref, purge_id),
                )
                if barrier_update.rowcount != 1:
                    raise ValueError("subject erasure barrier is missing")
                self.conn.execute(
                    """UPDATE purge_operations
                       SET status = 'complete',
                           error_code = NULL,
                           error_detail = NULL,
                           updated_at = ?,
                           completed_at = ?,
                           execution_claim_owner = NULL,
                           execution_claim_expires_at = NULL
                       WHERE purge_id = ? AND org_id = ?""",
                    (now, now, purge_id, self.org_id),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return self.get_purge_operation(purge_id)

    def fail_subject_erasure_barrier(
        self,
        subject_ref: str,
        purge_id: str,
        error_code: str,
        error_detail: str,
        *,
        execution_claim: PurgeExecutionClaim,
    ) -> SubjectWriteBarrier:
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        validated_purge_id = _validate_governance_purge_id("purge_id", purge_id)
        validated_error_code = _validate_governance_error_code(error_code)
        validated_error_detail = _validate_governance_error_detail(error_detail)
        now = _epoch_now()
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_purge_operation_execution_claim_locked(
                    validated_purge_id, execution_claim
                )
                update_cursor = self.conn.execute(
                    """UPDATE subject_write_barriers
                       SET status = 'failed',
                           error_code = ?,
                           error_detail = ?,
                           updated_at = ?
                       WHERE org_id = ? AND subject_ref = ? AND purge_id = ?
                         AND status = 'erasing'""",
                    (
                        validated_error_code,
                        validated_error_detail,
                        now,
                        self.org_id,
                        subject_ref,
                        validated_purge_id,
                    ),
                )
                if update_cursor.rowcount != 1:
                    raise ValueError(
                        "subject erasure barrier failure requires a matching barrier"
                    )
                purge_row = self.conn.execute(
                    "SELECT 1 FROM purge_operations WHERE purge_id = ? AND org_id = ?",
                    (validated_purge_id, self.org_id),
                ).fetchone()
                if purge_row is not None:
                    self.conn.execute(
                        """UPDATE purge_operations
                           SET status = 'failed', error_code = ?, error_detail = ?,
                               updated_at = ?, completed_at = ?,
                               execution_claim_owner = NULL,
                               execution_claim_expires_at = NULL
                           WHERE purge_id = ? AND org_id = ?""",
                        (
                            validated_error_code,
                            validated_error_detail,
                            now,
                            now,
                            validated_purge_id,
                            self.org_id,
                        ),
                    )
                row = self.conn.execute(
                    """SELECT * FROM subject_write_barriers
                       WHERE org_id = ? AND subject_ref = ? AND purge_id = ?""",
                    (self.org_id, subject_ref, validated_purge_id),
                ).fetchone()
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        if row is None:
            raise ValueError("subject erasure barrier update failed")
        return _row_to_subject_write_barrier(row)

    def get_subject_write_barrier(self, subject_ref: str) -> SubjectWriteBarrier | None:
        _validate_governance_prefixed_ref(
            "subject_ref", subject_ref, prefix="subref_v1_"
        )
        row = self.conn.execute(
            """SELECT * FROM subject_write_barriers
               WHERE org_id = ? AND subject_ref = ?""",
            (self.org_id, subject_ref),
        ).fetchone()
        if row is None:
            return None
        return _row_to_subject_write_barrier(row)
