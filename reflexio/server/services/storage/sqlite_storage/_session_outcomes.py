"""SQLite session outcome storage."""

import json
import sqlite3
from typing import Any, cast
from uuid import uuid4

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    SessionOutcomeFailureReason,
    SessionOutcomeRecord,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.error import SubjectWriteBarrierError
from reflexio.server.services.storage.session_outcome_identity import (
    OUTCOME_ALLOWED_VALUES,
    OUTCOME_FINALIZATION_RULE,
    OUTCOME_SCHEMA_VERSION,
    outcome_contract_digest,
)
from reflexio.server.services.storage.storage_base._session_outcomes import (
    SessionOutcomeContext,
    SessionOutcomeWriteResult,
)

from ._base import (
    SQLiteStorageBase,
    _canonical_session_trajectory_snapshot,
    _iso_to_epoch,
)


def _canonical_metadata_json(metadata: object) -> str | None:
    if metadata is None:
        return None
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _superseded_outcome_json(existing: Any, *, displaced_at: int) -> str:
    """Archive a displaced inferred outcome, whole, as canonical JSON.

    Kept on the surviving row rather than in a sibling table, and the reason is
    erasure rather than size: a sibling table would be a new subject-bearing
    surface that the erasure executor and the subject barrier would each have
    to learn about, and forgetting either leaves a data subject's outcome
    behind after a completed RTBF. Here it is removed by the DELETE that
    already removes the outcome.

    `metadata` is re-parsed rather than embedded as a string so the archive
    nests as JSON, matching what the Postgres side's `jsonb_build_object`
    produces -- otherwise the same field reads as an object on one backend and
    a quoted blob on the other.
    """
    stored_metadata = existing["metadata"]
    try:
        metadata = json.loads(stored_metadata) if stored_metadata else None
    except (TypeError, ValueError):
        metadata = None
    return json.dumps(
        {
            "outcome_id": existing["outcome_id"],
            "outcome_revision": existing["outcome_revision"],
            "outcome": existing["outcome"],
            "occurred_at": existing["occurred_at"],
            "source": existing["source"],
            "label": existing["label"],
            "value": existing["value"],
            "metadata": metadata,
            "outcome_contract_digest": existing["outcome_contract_digest"],
            "finalized_trajectory_digest": existing["finalized_trajectory_digest"],
            "created_at": existing["created_at"],
            "displaced_at": displaced_at,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _metadata_matches(*, stored_metadata: str | None, request_metadata: object) -> bool:
    if stored_metadata is None:
        stored_value = None
    else:
        try:
            stored_value = json.loads(stored_metadata)
        except (RecursionError, TypeError, ValueError):
            return False
    try:
        return _canonical_metadata_json(stored_value) == _canonical_metadata_json(
            request_metadata
        )
    except (RecursionError, TypeError, ValueError):
        return False


class SessionOutcomeStoreMixin:
    conn: sqlite3.Connection
    _lock: Any
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any

    @SQLiteStorageBase.handle_exceptions
    def get_session_outcome_context(self, session_id: str) -> SessionOutcomeContext:
        with self._lock:
            existing = self.conn.execute(
                """SELECT user_id, source, is_inferred
                   FROM session_outcomes WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
            existing_is_inferred = existing is not None and bool(
                existing["is_inferred"]
            )
            if existing is not None and not existing_is_inferred:
                # Settled by a real report: the only legal write left is a
                # byte-exact retry, so the first-request context below would
                # have nothing to validate.
                return SessionOutcomeContext(
                    user_id=str(existing["user_id"]),
                    source=str(existing["source"]),
                    existing=True,
                )
            # An INFERRED row is displaceable, so the write that replaces it is
            # a new outcome and needs the same first-request context a first
            # write gets. Returning early here -- which is what this method did
            # before displacement existed -- would leave `first_request_at`
            # None and make every displacing write answer UNKNOWN_SESSION.
            first = self.conn.execute(
                """SELECT user_id, source, created_at, request_id
                   FROM requests WHERE session_id = ?
                   ORDER BY created_at ASC, request_id ASC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if first is None:
                return SessionOutcomeContext(
                    existing=existing is not None,
                    existing_is_inferred=existing_is_inferred,
                )
            counts = self.conn.execute(
                """SELECT COUNT(DISTINCT user_id) AS user_count,
                          COUNT(DISTINCT COALESCE(source, '')) AS source_count
                   FROM requests WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
            return SessionOutcomeContext(
                user_id=str(first["user_id"]),
                source=str(first["source"] or ""),
                first_request_at=_iso_to_epoch(first["created_at"]),
                user_contract_violation=int(counts["user_count"]) > 1,
                source_contract_violation=int(counts["source_count"]) > 1,
                existing=existing is not None,
                existing_is_inferred=existing_is_inferred,
            )

    @SQLiteStorageBase.handle_exceptions
    def record_session_outcome(
        self,
        request: SetSessionOutcomeRequest,
        *,
        created_at: int,
        expected_context: SessionOutcomeContext,
        is_inferred: bool = False,
    ) -> SessionOutcomeWriteResult:
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                existing = self.conn.execute(
                    "SELECT * FROM session_outcomes WHERE session_id = ?",
                    (request.session_id,),
                ).fetchone()
                # A customer's own report replaces an outcome the tuner merely
                # INFERRED. Falling THROUGH to the ordinary write path rather
                # than branching here is deliberate: that path already runs the
                # subject barrier, the trajectory snapshot, the expected-context
                # re-check and the occurred-before-session bound, and a
                # displacing write is a real new outcome that earns every one of
                # them. Only the final statement differs.
                displacing = (
                    existing is not None
                    and bool(existing["is_inferred"])
                    and not is_inferred
                )
                if existing is not None and not displacing:
                    early_first = self.conn.execute(
                        """SELECT user_id, source, governance_subject_ref
                           FROM requests WHERE session_id = ?
                           ORDER BY created_at ASC, request_id ASC LIMIT 1""",
                        (request.session_id,),
                    ).fetchone()
                    snapshot = (
                        _canonical_session_trajectory_snapshot(
                            self.conn, request.session_id
                        )
                        if early_first is not None
                        else None
                    )
                    first = snapshot.first_request if snapshot is not None else None
                    source = (
                        str(first["source"] or "")
                        if first is not None
                        else str(existing["source"])
                    )
                    subject_ref = (
                        str(
                            first["governance_subject_ref"]
                            or self._subject_ref_for_user_id(str(first["user_id"]))
                        )
                        if first is not None
                        else str(existing["governance_subject_ref"])
                    )
                    contract_digest = self._outcome_contract_digest(source)
                    current_snapshot_digest = (
                        snapshot.digest if snapshot is not None else None
                    )
                    stored_contract_digest = existing["outcome_contract_digest"]
                    stored_snapshot_digest = existing["finalized_trajectory_digest"]
                    server_context_matches = first is None or (
                        str(existing["user_id"]) == str(first["user_id"])
                        and str(existing["source"]) == str(first["source"] or "")
                        and str(existing["governance_subject_ref"]) == subject_ref
                    )
                    exact_retry = (
                        existing["outcome"] == str(request.outcome)
                        and int(existing["occurred_at"]) == request.occurred_at
                        and existing["label"] == request.label
                        and existing["value"] == request.value
                        and _metadata_matches(
                            stored_metadata=existing["metadata"],
                            request_metadata=request.metadata,
                        )
                        and server_context_matches
                        and (
                            stored_contract_digest is None
                            or stored_contract_digest == contract_digest
                        )
                        and (
                            stored_snapshot_digest is None
                            or current_snapshot_digest is None
                            or stored_snapshot_digest == current_snapshot_digest
                        )
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
                        outcome_id=(
                            str(existing["outcome_id"])
                            if existing["outcome_id"] is not None
                            else None
                        ),
                        outcome_revision=(
                            int(existing["outcome_revision"])
                            if existing["outcome_revision"] is not None
                            else None
                        ),
                        outcome_contract_digest=(
                            str(stored_contract_digest)
                            if stored_contract_digest is not None
                            else None
                        ),
                        finalized_trajectory_digest=(
                            str(stored_snapshot_digest)
                            if stored_snapshot_digest is not None
                            else None
                        ),
                    )
                barrier_first = self.conn.execute(
                    """SELECT user_id, source, created_at, request_id,
                              governance_subject_ref
                       FROM requests WHERE session_id = ?
                       ORDER BY created_at ASC, request_id ASC LIMIT 1""",
                    (request.session_id,),
                ).fetchone()
                if barrier_first is None:
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        reason=SessionOutcomeFailureReason.UNKNOWN_SESSION,
                    )
                barrier_user_id = str(barrier_first["user_id"])
                subject_ref = str(
                    barrier_first["governance_subject_ref"]
                    or self._subject_ref_for_user_id(barrier_user_id)
                )
                try:
                    self._assert_subject_writable_locked(subject_ref)
                except SubjectWriteBarrierError:
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        user_id=barrier_user_id,
                        reason=SessionOutcomeFailureReason.SUBJECT_NOT_WRITABLE,
                    )
                snapshot = _canonical_session_trajectory_snapshot(
                    self.conn, request.session_id
                )
                if snapshot.request_count == 0 or snapshot.first_request is None:
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        reason=SessionOutcomeFailureReason.UNKNOWN_SESSION,
                    )
                first = snapshot.first_request
                user_id = str(first["user_id"])
                source = str(first["source"] or "")
                first_request_at = _iso_to_epoch(cast(str, first["created_at"]))
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
                subject_ref = str(
                    first["governance_subject_ref"]
                    or self._subject_ref_for_user_id(user_id)
                )
                # DISPLACEMENT MAY NOT CROSS A GOVERNANCE SUBJECT.
                #
                # The row is found by `session_id` alone, but it is ERASED by
                # `user_id` (`clear_session_outcomes_for_user`) and gated by
                # `governance_subject_ref`. The session's earliest request can
                # change owners after the inferred write -- delete the original
                # first request while another user's request remains in the
                # same session and the "first request" is now somebody else's.
                # Displacing there would rewrite `user_id` and
                # `governance_subject_ref` to the NEW owner while
                # `superseded_outcome` still holds the ORIGINAL subject's
                # outcome, so erasing the original user would no longer reach
                # it: a completed RTBF that silently leaves the subject's
                # outcome behind under a stranger's key. The primary key is
                # `(user_id, session_id)`, so the same rewrite can also collide
                # with a row the new owner already has.
                #
                # Refused rather than re-owned. This is the tenancy answer, and
                # it is deliberately conservative: a rotated governance secret
                # can make two requests from the SAME user carry different
                # stored refs, and refusing there costs one rejected report
                # while accepting there would re-key a row to a subject whose
                # barrier was never checked against the archived content. The
                # settled-row retry check above already treats a differing
                # stored ref the same way.
                #
                # A NULL stored ref is a mismatch too, for the same reason:
                # there is nothing to prove the archived outcome belongs to the
                # subject the row is about to be filed under.
                if displacing and existing is not None:
                    stored_subject_ref = existing["governance_subject_ref"]
                    same_subject = (
                        str(existing["user_id"]) == user_id
                        and stored_subject_ref is not None
                        and str(stored_subject_ref) == subject_ref
                    )
                    if not same_subject:
                        stored_contract_digest = existing["outcome_contract_digest"]
                        stored_snapshot_digest = existing["finalized_trajectory_digest"]
                        self.conn.rollback()
                        return SessionOutcomeWriteResult(
                            recorded=False,
                            user_id=str(existing["user_id"]),
                            source=str(existing["source"]),
                            reason=SessionOutcomeFailureReason.CONFLICTING_FINALIZATION,
                            outcome_id=(
                                str(existing["outcome_id"])
                                if existing["outcome_id"] is not None
                                else None
                            ),
                            outcome_revision=(
                                int(existing["outcome_revision"])
                                if existing["outcome_revision"] is not None
                                else None
                            ),
                            outcome_contract_digest=(
                                str(stored_contract_digest)
                                if stored_contract_digest is not None
                                else None
                            ),
                            finalized_trajectory_digest=(
                                str(stored_snapshot_digest)
                                if stored_snapshot_digest is not None
                                else None
                            ),
                        )
                if request.occurred_at < first_request_at:
                    self.conn.rollback()
                    return SessionOutcomeWriteResult(
                        recorded=False,
                        user_id=user_id,
                        source=source,
                        reason=SessionOutcomeFailureReason.OCCURRED_BEFORE_SESSION,
                    )
                contract_digest = self._outcome_contract_digest(source)
                snapshot_digest = snapshot.digest
                outcome_id = uuid4().hex
                if displacing and existing is not None:
                    # The displaced row is archived WHOLE rather than
                    # summarised: the analysis this exists for is "what did the
                    # judge say and what did the customer say", and a summary
                    # drops exactly that half.
                    revision = int(existing["outcome_revision"] or 1) + 1
                    self.conn.execute(
                        """UPDATE session_outcomes
                              SET outcome_id = ?, outcome_revision = ?,
                                  user_id = ?, outcome = ?, occurred_at = ?,
                                  source = ?, label = ?, value = ?, metadata = ?,
                                  outcome_contract_digest = ?,
                                  finalized_trajectory_digest = ?,
                                  governance_subject_ref = ?, created_at = ?,
                                  is_inferred = 0, superseded_outcome = ?
                            WHERE session_id = ?""",
                        (
                            outcome_id,
                            revision,
                            user_id,
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
                            _superseded_outcome_json(existing, displaced_at=created_at),
                            request.session_id,
                        ),
                    )
                else:
                    revision = 1
                    self.conn.execute(
                        """INSERT INTO session_outcomes
                           (outcome_id, outcome_revision, user_id, session_id, outcome,
                            occurred_at, source, label, value, metadata,
                            outcome_contract_digest, finalized_trajectory_digest,
                            governance_subject_ref, created_at, is_inferred)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            outcome_id,
                            revision,
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
                            1 if is_inferred else 0,
                        ),
                    )
                self.conn.commit()
                return SessionOutcomeWriteResult(
                    recorded=True,
                    user_id=user_id,
                    source=source,
                    outcome_id=outcome_id,
                    outcome_revision=revision,
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
        with self._lock:
            outcome_cursor = self.conn.execute(
                "DELETE FROM session_outcomes WHERE user_id = ?",
                (user_id,),
            )
            self.conn.commit()
        return {
            "session_outcomes": int(outcome_cursor.rowcount or 0),
        }

    @staticmethod
    def _metadata_json(request: SetSessionOutcomeRequest) -> str | None:
        return _canonical_metadata_json(request.metadata)

    @staticmethod
    def _outcome_contract_digest(source: str) -> str:
        return outcome_contract_digest(
            source=source,
            schema_version=OUTCOME_SCHEMA_VERSION,
            allowed_values=OUTCOME_ALLOWED_VALUES,
            finalization_rule=OUTCOME_FINALIZATION_RULE,
        )
