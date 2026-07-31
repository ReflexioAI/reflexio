"""SQLite session outcome storage."""

import json
import sqlite3
from typing import Any
from uuid import uuid4

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    SessionOutcomeFailureReason,
    SessionOutcomeRecord,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.error import SubjectWriteBarrierError
from reflexio.server.services.storage.session_outcome_identity import (
    outcome_contract_digest,
    trajectory_digest,
)
from reflexio.server.services.storage.storage_base._session_outcomes import (
    SessionOutcomeContext,
    SessionOutcomeWriteResult,
)

from ._base import SQLiteStorageBase, _canonical_session_snapshot, _iso_to_epoch

_OUTCOME_SCHEMA_VERSION = 1
_OUTCOME_FINALIZATION_RULE = "first_write"
_OUTCOME_ALLOWED_VALUES = ("success", "failure", "unknown")


class SessionOutcomeStoreMixin:
    conn: sqlite3.Connection
    _lock: Any
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any

    @SQLiteStorageBase.handle_exceptions
    def get_session_outcome_context(self, session_id: str) -> SessionOutcomeContext:
        with self._lock:
            existing = self.conn.execute(
                "SELECT user_id, source FROM session_outcomes WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                return SessionOutcomeContext(
                    user_id=str(existing["user_id"]),
                    source=str(existing["source"]),
                    existing=True,
                )
            rows = self.conn.execute(
                """SELECT user_id, source, created_at, request_id
                   FROM requests WHERE session_id = ?
                   ORDER BY created_at ASC, request_id ASC""",
                (session_id,),
            ).fetchall()
            if not rows:
                return SessionOutcomeContext()
            first = rows[0]
            return SessionOutcomeContext(
                user_id=str(first["user_id"]),
                source=str(first["source"]),
                first_request_at=_iso_to_epoch(first["created_at"]),
                user_contract_violation=len({str(row["user_id"]) for row in rows}) > 1,
                source_contract_violation=len({str(row["source"]) for row in rows}) > 1,
            )

    @SQLiteStorageBase.handle_exceptions
    def record_session_outcome(
        self,
        request: SetSessionOutcomeRequest,
        *,
        created_at: int,
        expected_context: SessionOutcomeContext,
    ) -> SessionOutcomeWriteResult:
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                existing = self.conn.execute(
                    "SELECT * FROM session_outcomes WHERE session_id = ?",
                    (request.session_id,),
                ).fetchone()
                if existing is not None:
                    first = self.conn.execute(
                        """SELECT source FROM requests WHERE session_id = ?
                           ORDER BY created_at ASC, request_id ASC LIMIT 1""",
                        (request.session_id,),
                    ).fetchone()
                    source = (
                        str(first["source"])
                        if first is not None
                        else str(existing["source"])
                    )
                    contract_digest = self._outcome_contract_digest(source)
                    snapshot_digest = (
                        trajectory_digest(
                            _canonical_session_snapshot(self.conn, request.session_id)
                        )
                        if first is not None
                        else str(existing["finalized_trajectory_digest"])
                    )
                    exact_retry = (
                        existing["outcome"] == str(request.outcome)
                        and int(existing["occurred_at"]) == request.occurred_at
                        and existing["label"] == request.label
                        and existing["value"] == request.value
                        and existing["metadata"] == self._metadata_json(request)
                        and existing["outcome_contract_digest"] == contract_digest
                        and existing["finalized_trajectory_digest"] == snapshot_digest
                    )
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        user_id=str(existing["user_id"]),
                        source=str(existing["source"]),
                        reason=(
                            None
                            if exact_retry
                            else SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
                        ),
                        outcome_id=str(existing["outcome_id"]),
                        outcome_revision=int(existing["outcome_revision"]),
                        outcome_contract_digest=str(
                            existing["outcome_contract_digest"]
                        ),
                        finalized_trajectory_digest=str(
                            existing["finalized_trajectory_digest"]
                        ),
                    )
                first = self.conn.execute(
                    """SELECT user_id, source, created_at, request_id
                       FROM requests WHERE session_id = ?
                       ORDER BY created_at ASC, request_id ASC LIMIT 1""",
                    (request.session_id,),
                ).fetchone()
                if first is None:
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        reason=SessionOutcomeFailureReason.UNKNOWN_SESSION,
                    )
                user_id = str(first["user_id"])
                source = str(first["source"])
                first_request_at = _iso_to_epoch(first["created_at"])
                if (
                    expected_context.user_id != user_id
                    or expected_context.source != source
                    or expected_context.first_request_at != first_request_at
                ):
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        user_id=user_id,
                        source=source,
                        context_changed=True,
                    )
                if request.occurred_at < first_request_at:
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        user_id=user_id,
                        source=source,
                        reason=SessionOutcomeFailureReason.OCCURRED_BEFORE_SESSION,
                    )
                subject_ref = self._subject_ref_for_user_id(user_id)
                try:
                    self._assert_subject_writable_locked(subject_ref)
                except SubjectWriteBarrierError:
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        user_id=user_id,
                        reason=SessionOutcomeFailureReason.SUBJECT_NOT_WRITABLE,
                    )
                contract_digest = self._outcome_contract_digest(source)
                snapshot_digest = trajectory_digest(
                    _canonical_session_snapshot(self.conn, request.session_id)
                )
                outcome_id = uuid4().hex
                self.conn.execute(
                    """INSERT INTO session_outcomes
                       (outcome_id, outcome_revision, user_id, session_id, outcome,
                        occurred_at, source, label, value, metadata,
                        outcome_contract_digest, finalized_trajectory_digest,
                        governance_subject_ref, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        outcome_id,
                        1,
                        user_id,
                        request.session_id,
                        str(request.outcome),
                        request.occurred_at,
                        source,
                        request.label,
                        request.value,
                        self._metadata_json(request),
                        contract_digest,
                        snapshot_digest,
                        subject_ref,
                        created_at,
                    ),
                )
                self.conn.commit()
                return SessionOutcomeWriteResult(
                    recorded=True,
                    user_id=user_id,
                    source=source,
                    outcome_id=outcome_id,
                    outcome_revision=1,
                    outcome_contract_digest=contract_digest,
                    finalized_trajectory_digest=snapshot_digest,
                )
            except Exception:
                self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def get_session_outcomes(
        self, request: GetSessionOutcomesRequest
    ) -> list[SessionOutcomeRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if request.session_ids:
            placeholders = ",".join("?" for _ in request.session_ids)
            clauses.append(f"session_id IN ({placeholders})")
            params.extend(request.session_ids)
        for column, value in (
            ("user_id", request.user_id),
            ("source", request.source),
            ("outcome", request.outcome.value if request.outcome else None),
            ("label", request.label),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if request.start_time is not None:
            clauses.append("occurred_at >= ?")
            params.append(request.start_time)
        if request.end_time is not None:
            clauses.append("occurred_at <= ?")
            params.append(request.end_time)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"""SELECT outcome_id, outcome_revision, user_id, session_id, outcome,
                       occurred_at, source, label, value, metadata,
                       outcome_contract_digest, finalized_trajectory_digest, created_at
                FROM session_outcomes{where}
                 ORDER BY occurred_at DESC, user_id ASC, session_id ASC LIMIT ? OFFSET ?""",
            [*params, request.top_k, request.offset],
        ).fetchall()
        return [
            SessionOutcomeRecord(
                outcome_id=row["outcome_id"],
                outcome_revision=row["outcome_revision"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                outcome=row["outcome"],
                occurred_at=row["occurred_at"],
                source=row["source"],
                label=row["label"],
                value=row["value"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                outcome_contract_digest=row["outcome_contract_digest"],
                finalized_trajectory_digest=row["finalized_trajectory_digest"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @SQLiteStorageBase.handle_exceptions
    def clear_session_outcomes_for_user(self, user_id: str) -> dict[str, int]:
        subject_ref = self._subject_ref_for_user_id(user_id)
        with self._lock:
            outcome_cursor = self.conn.execute(
                "DELETE FROM session_outcomes WHERE governance_subject_ref = ?",
                (subject_ref,),
            )
            self.conn.commit()
        return {
            "session_outcomes": int(outcome_cursor.rowcount or 0),
        }

    @staticmethod
    def _metadata_json(request: SetSessionOutcomeRequest) -> str | None:
        if request.metadata is None:
            return None
        return json.dumps(request.metadata, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _outcome_contract_digest(source: str) -> str:
        return outcome_contract_digest(
            source=source,
            schema_version=_OUTCOME_SCHEMA_VERSION,
            allowed_values=_OUTCOME_ALLOWED_VALUES,
            finalization_rule=_OUTCOME_FINALIZATION_RULE,
        )
