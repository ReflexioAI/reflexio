"""SQLite session outcome storage."""

import json
import sqlite3
from typing import Any

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    SessionOutcomeFailureReason,
    SessionOutcomeRecord,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.error import SubjectWriteBarrierError
from reflexio.server.services.storage.storage_base._session_outcomes import (
    SessionOutcomeContext,
    SessionOutcomeWriteResult,
)

from ._base import SQLiteStorageBase, _iso_to_epoch


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
                    "SELECT user_id, source FROM session_outcomes WHERE session_id = ?",
                    (request.session_id,),
                ).fetchone()
                if existing is not None:
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        user_id=str(existing["user_id"]),
                        source=str(existing["source"]),
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
                self.conn.execute(
                    """INSERT INTO session_outcomes
                       (user_id, session_id, outcome, occurred_at, source, label, value,
                        metadata, governance_subject_ref, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        request.session_id,
                        request.outcome.value,
                        request.occurred_at,
                        source,
                        request.label,
                        request.value,
                        json.dumps(
                            request.metadata, sort_keys=True, separators=(",", ":")
                        )
                        if request.metadata is not None
                        else None,
                        subject_ref,
                        created_at,
                    ),
                )
                self.conn.commit()
                return SessionOutcomeWriteResult(
                    recorded=True, user_id=user_id, source=source
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
            f"""SELECT user_id, session_id, outcome, occurred_at, source, label, value,
                       metadata, created_at FROM session_outcomes{where}
                 ORDER BY occurred_at DESC, user_id ASC, session_id ASC LIMIT ? OFFSET ?""",
            [*params, request.top_k, request.offset],
        ).fetchall()
        return [
            SessionOutcomeRecord(
                user_id=row["user_id"],
                session_id=row["session_id"],
                outcome=row["outcome"],
                occurred_at=row["occurred_at"],
                source=row["source"],
                label=row["label"],
                value=row["value"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
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
