"""SQLite governance-erase-execution store methods.

Extracted verbatim from ``_governance.py`` (the GovernanceEraseExecution bucket):
the two public methods (``apply_governance_user_data_delete``,
``complete_purge_operation_with_audit``) plus the two EraseExecution-owned
privates (``_purge_governance_entity_content_locked``,
``_clear_user_data_for_governance_locked``).

``complete_purge_operation_with_audit`` is one of the two methods authorized to
write a successful-ERASE audit row — the ERASE-audit-idempotency block, the
re-read+identity-verify, and the ``status='complete'`` flip are preserved
byte-for-byte, committing together with the audit insert.
``_purge_governance_entity_content_locked`` emits the ``wasPurged`` lineage event
only on an actual content purge (``rowcount <= 0 -> return False`` before
``_append_event_stmt``); the ``_PURGE_SQL`` / ``_append_event_stmt`` symbols are
function-imported LOCAL-in-method (from ``.._lineage``) to avoid an import cycle.

The residual ``SQLiteGovernanceMixin`` stays composed alongside this mixin and
permanently holds the shared infra (``conn``, ``_lock``, ``org_id``,
``_deps()``), the delete-target / hide-for-rebuild / prepared-snapshot
validators it reaches cross-bucket
(``_validate_prepared_delete_target_matrix_locked``,
``_validate_hide_for_rebuild_targets_locked``,
``_prepared_owned_user_playbook_ids_locked``), and the module-level helpers
(``_row_to_audit_event``, ``_row_to_purge_operation``), which are imported here
rather than duplicated. ``_append_audit_event_with_cursor`` (AuditEventStore),
``get_purge_operation`` / ``_record_purge_target_locked`` (PurgeOperationStore)
resolve through the composed MRO.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
)
from reflexio.server.services.storage.governance_validation import (
    _CANONICAL_DELETE_TARGET_NAMES,
    _PREPARE_PHASE,
    _SNAPSHOT_TARGET_NAME,
    _canonicalize_audit_event_for_persistence,
    _epoch_now,
    _is_successful_erase_event,
    _successful_erase_identity,
    _validate_governance_purge_id,
)

from .._governance import _row_to_audit_event, _row_to_purge_operation

if TYPE_CHECKING:
    from .._governance import _SQLiteGovernanceDeps


class GovernanceEraseExecutionMixin:
    """SQLite governance-erase-execution store primitives."""

    # Type hints for instance attributes/methods provided via MRO by the
    # co-composed residual SQLiteGovernanceMixin / SQLiteStorageBase.
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str
    _deps: Callable[[], _SQLiteGovernanceDeps]

    # Cross-bucket residents reached via MRO: the delete-target / hide /
    # prepared-snapshot validators live on the residual SQLiteGovernanceMixin.
    _validate_prepared_delete_target_matrix_locked: Callable[[str], None]
    _validate_hide_for_rebuild_targets_locked: Callable[[str], None]
    _prepared_owned_user_playbook_ids_locked: Callable[[str], set[int]]

    # Provided via MRO by the co-composed AuditEventStoreMixin (audit bucket)
    # and PurgeOperationStoreMixin (purge bucket); reached here by the
    # cross-bucket erase-execution / purge completion methods.
    _append_audit_event_with_cursor: Callable[
        [sqlite3.Connection | sqlite3.Cursor, AuditEvent], bool
    ]
    get_purge_operation: Callable[[str], PurgeOperation]
    _record_purge_target_locked: Callable[..., None]

    def _purge_governance_entity_content_locked(
        self,
        *,
        entity_type: Literal["profile", "user_playbook"],
        entity_id: str,
        rowid: int,
    ) -> bool:
        from .._lineage import _PURGE_SQL, _append_event_stmt

        sql = _PURGE_SQL[entity_type]
        cur = self.conn.execute(sql, (entity_id,))
        if cur.rowcount <= 0:
            return False
        _append_event_stmt(
            self.conn,
            org_id=self.org_id,
            entity_type=entity_type,
            entity_id=entity_id,
            op="purge",
            prov="wasPurged",
            source_ids=[],
            actor="erasure",
            request_id=f"purge_{entity_id}",
            reason="content_purge",
        )
        if entity_type == "profile":
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?",
                (entity_id,),
            )
            if self._deps()._has_sqlite_vec:
                self.conn.execute(
                    "DELETE FROM profiles_vec WHERE rowid = ?",
                    (rowid,),
                )
        else:
            self.conn.execute(
                "DELETE FROM user_playbooks_fts WHERE rowid = ?",
                (rowid,),
            )
            if self._deps()._has_sqlite_vec:
                self.conn.execute(
                    "DELETE FROM user_playbooks_vec WHERE rowid = ?",
                    (rowid,),
                )
        return True

    def _clear_user_data_for_governance_locked(
        self,
        user_id: str,
        *,
        expected_user_playbook_ids: set[int] | None = None,
    ) -> dict[str, int]:
        deps = self._deps()
        interaction_ids = [
            int(row["interaction_id"])
            for row in self.conn.execute(
                "SELECT interaction_id FROM interactions WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        raw_upb_ids = [
            int(row["user_playbook_id"])
            for row in self.conn.execute(
                "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        if (
            expected_user_playbook_ids is not None
            and set(raw_upb_ids) != expected_user_playbook_ids
        ):
            raise ValueError(
                "Current user playbooks no longer match prepared purge snapshot"
            )
        request_ids = [
            str(row["request_id"])
            for row in self.conn.execute(
                "SELECT request_id FROM requests WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ]
        profile_rows = self.conn.execute(
            "SELECT rowid, profile_id FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        profile_rowid_by_id = {
            str(row["profile_id"]): int(row["rowid"]) for row in profile_rows
        }
        all_profile_ids = list(profile_rowid_by_id)

        purge_profile_ids, delete_profile_ids = deps._partition_purge_vs_delete(
            "profile",
            all_profile_ids,
        )
        purge_upb_str_ids, delete_upb_str_ids = deps._partition_purge_vs_delete(
            "user_playbook",
            [str(user_playbook_id) for user_playbook_id in raw_upb_ids],
        )
        purge_upb_ids = [int(entity_id) for entity_id in purge_upb_str_ids]
        delete_upb_ids = [int(entity_id) for entity_id in delete_upb_str_ids]
        erased_entity_ids = [
            *request_ids,
            *[str(interaction_id) for interaction_id in interaction_ids],
            *all_profile_ids,
            *[str(user_playbook_id) for user_playbook_id in raw_upb_ids],
        ]
        if erased_entity_ids:
            erased_entity_id_set = set(erased_entity_ids)
            lineage_source_event_ids: list[int] = []
            for row in self.conn.execute(
                "SELECT event_id, source_ids FROM lineage_event WHERE org_id = ?",
                (self.org_id,),
            ).fetchall():
                try:
                    source_ids = json.loads(str(row["source_ids"] or "[]"))
                except json.JSONDecodeError:
                    source_ids = []
                if any(
                    str(source_id) in erased_entity_id_set for source_id in source_ids
                ):
                    lineage_source_event_ids.append(int(row["event_id"]))
            deps._delete_in_chunks(
                "lineage_event", "event_id", lineage_source_event_ids
            )
            placeholders = ",".join("?" for _ in erased_entity_ids)
            self.conn.execute(
                f"""DELETE FROM lineage_event
                    WHERE org_id = ?
                      AND (
                        request_id IN ({placeholders})
                        OR entity_id IN ({placeholders})
                      )""",  # noqa: S608
                [self.org_id, *erased_entity_ids, *erased_entity_ids],
            )
        delete_profile_rowids = [
            profile_rowid_by_id[profile_id]
            for profile_id in delete_profile_ids
            if profile_id in profile_rowid_by_id
        ]

        deps._delete_in_chunks("interactions_fts", "rowid", interaction_ids)
        deps._delete_in_chunks("user_playbooks_fts", "rowid", delete_upb_ids)
        deps._delete_in_chunks("profiles_fts", "profile_id", delete_profile_ids)
        if deps._has_sqlite_vec:
            deps._delete_in_chunks("interactions_vec", "rowid", interaction_ids)
            deps._delete_in_chunks("user_playbooks_vec", "rowid", delete_upb_ids)
            deps._delete_in_chunks("profiles_vec", "rowid", delete_profile_rowids)

        interactions_cur = self.conn.execute(
            "DELETE FROM interactions WHERE user_id = ?",
            (user_id,),
        )
        eval_results_cur = self.conn.execute(
            """DELETE FROM agent_success_evaluation_result
               WHERE user_id = ?""",
            (user_id,),
        )
        requests_cur = self.conn.execute(
            "DELETE FROM requests WHERE user_id = ?",
            (user_id,),
        )
        if raw_upb_ids:
            deps._delete_source_windows_for_user_playbook_ids(raw_upb_ids)
        if delete_upb_ids:
            deps._delete_in_chunks("user_playbooks", "user_playbook_id", delete_upb_ids)
        if delete_profile_ids:
            deps._delete_in_chunks("profiles", "profile_id", delete_profile_ids)

        purged_profiles = 0
        for profile_id in purge_profile_ids:
            rowid = profile_rowid_by_id.get(profile_id)
            if rowid is None:
                continue
            purged_profiles += int(
                self._purge_governance_entity_content_locked(
                    entity_type="profile",
                    entity_id=profile_id,
                    rowid=rowid,
                )
            )

        purged_user_playbooks = 0
        for user_playbook_id in purge_upb_ids:
            purged_user_playbooks += int(
                self._purge_governance_entity_content_locked(
                    entity_type="user_playbook",
                    entity_id=str(user_playbook_id),
                    rowid=user_playbook_id,
                )
            )

        return {
            "interactions": interactions_cur.rowcount,
            "user_playbooks": len(delete_upb_ids),
            "profiles": len(delete_profile_ids),
            "requests": requests_cur.rowcount,
            "agent_success_evaluation_results": eval_results_cur.rowcount,
            "purged_profiles": purged_profiles,
            "purged_user_playbooks": purged_user_playbooks,
        }

    def apply_governance_user_data_delete(
        self, purge_id: str, user_id: str
    ) -> dict[str, int]:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        name_map = {
            "interactions": "interaction",
            "user_playbooks": "user_playbook",
            "profiles": "profile",
            "requests": "request",
            "agent_success_evaluation_results": "agent_success_evaluation_result",
            "purged_profiles": "profile_purge",
            "purged_user_playbooks": "user_playbook_purge",
        }
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                self._validate_prepared_delete_target_matrix_locked(purge_id)
                self._validate_hide_for_rebuild_targets_locked(purge_id)
                expected_user_playbook_ids = (
                    self._prepared_owned_user_playbook_ids_locked(purge_id)
                )
                counts = self._clear_user_data_for_governance_locked(
                    user_id,
                    expected_user_playbook_ids=expected_user_playbook_ids,
                )
                for key, value in counts.items():
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name=name_map.get(key, key),
                        target_ref="all",
                        phase="delete",
                        status="complete",
                        detail={"count": int(value)},
                        deleted_count=int(value),
                        error_detail=None,
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return counts

    def complete_purge_operation_with_audit(
        self, purge_id: str, audit_event: AuditEvent
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
                barrier_row = self.conn.execute(
                    """SELECT status FROM subject_write_barriers
                       WHERE org_id = ? AND subject_ref = ? AND purge_id = ?""",
                    (self.org_id, audit_event.subject_ref, purge_id),
                ).fetchone()
                if barrier_row is None or barrier_row["status"] != "erasing":
                    raise ValueError("subject erasure barrier is missing")
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
                delete_rows = self.conn.execute(
                    """SELECT target_name, status FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND phase = 'delete'
                         AND target_ref = 'all'""",
                    (self.org_id, purge_id),
                ).fetchall()
                delete_statuses = {
                    str(row["target_name"]): str(row["status"]) for row in delete_rows
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
                self.conn.execute(
                    """UPDATE purge_operations
                       SET status = 'complete',
                           error_code = NULL,
                           error_detail = NULL,
                           updated_at = ?,
                           completed_at = ?
                       WHERE purge_id = ? AND org_id = ?""",
                    (now, now, purge_id, self.org_id),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return self.get_purge_operation(purge_id)
