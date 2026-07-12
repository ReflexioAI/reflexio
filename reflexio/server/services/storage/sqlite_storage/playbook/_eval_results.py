"""Agent success evaluation result store methods for SQLite storage."""

import sqlite3
from datetime import UTC, datetime
from typing import Any

from reflexio.models.api_schema.service_schemas import (
    AgentSuccessEvaluationResult,
    RetrievedLearningEvaluationResult,
)

from ...storage_base.retrieved_learning_state import (
    CANONICAL_RETRIEVED_KINDS,
    DEFAULT_TRANSCRIPT_CHAR_LIMIT,
    TERMINAL_RETRIEVED_STATUSES,
    BoundedRetrievedLearningSnapshot,
    RetrievedLearningCommitResult,
    SessionFingerprintBuilder,
    append_bounded_snapshot_interaction,
    build_retrieved_learning_state_key,
)
from .._base import (
    SQLiteStorageBase,
    _epoch_to_iso,
    _iso_to_epoch,
    _json_dumps,
    _json_loads,
    _row_to_eval_result,
)


class AgentEvaluationResultStoreMixin:
    """Mixin providing agent success evaluation result CRUD for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _execute: Any
    _fetchall: Any
    _fetchone: Any
    _get_embedding: Any
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any
    _current_timestamp: Any
    get_operation_state: Any

    # ------------------------------------------------------------------
    # Agent Success Evaluation methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def save_agent_success_evaluation_results(
        self, results: list[AgentSuccessEvaluationResult]
    ) -> None:
        for result in results:
            if not result.embedding:
                embedding_text = f"{result.failure_type} {result.failure_reason}"
                if embedding_text.strip():
                    result.embedding = self._get_embedding(embedding_text)
                else:
                    result.embedding = []

            created_at_iso = _epoch_to_iso(result.created_at)
            subject_ref = self._subject_ref_for_user_id(result.user_id)
            with self._lock:
                try:
                    self.conn.execute("BEGIN IMMEDIATE")
                    self._assert_subject_writable_locked(subject_ref)
                    self.conn.execute(
                        """INSERT INTO agent_success_evaluation_result
                           (user_id, session_id, agent_version, evaluation_name, is_success,
                            failure_type, failure_reason, regular_vs_shadow,
                            number_of_correction_per_session, user_turns_to_resolution,
                            is_escalated, embedding, created_at, governance_subject_ref)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            result.user_id,
                            result.session_id,
                            result.agent_version,
                            result.evaluation_name,
                            int(result.is_success),
                            result.failure_type,
                            result.failure_reason,
                            result.regular_vs_shadow.value
                            if result.regular_vs_shadow
                            else None,
                            result.number_of_correction_per_session,
                            result.user_turns_to_resolution,
                            int(result.is_escalated),
                            _json_dumps(result.embedding) if result.embedding else None,
                            created_at_iso,
                            subject_ref,
                        ),
                    )
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise

    @SQLiteStorageBase.handle_exceptions
    def get_agent_success_evaluation_results(
        self, limit: int = 100, agent_version: str | None = None
    ) -> list[AgentSuccessEvaluationResult]:
        sql = "SELECT * FROM agent_success_evaluation_result"
        params: list[Any] = []
        if agent_version is not None:
            sql += " WHERE agent_version = ?"
            params.append(agent_version)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._fetchall(sql, params)
        return [_row_to_eval_result(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_agent_success_evaluation_results_in_window(
        self,
        from_ts: int,
        to_ts: int,
        agent_version: str | None = None,
        limit: int | None = None,
    ) -> list[AgentSuccessEvaluationResult]:
        sql = """SELECT * FROM agent_success_evaluation_result
                 WHERE created_at >= ? AND created_at <= ?"""
        params: list[Any] = [_epoch_to_iso(from_ts), _epoch_to_iso(to_ts)]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._fetchall(sql, params)
        return [_row_to_eval_result(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_agent_success_evaluation_result_ids(
        self,
        user_id: str,
        session_id: str,
        evaluation_name: str,
        agent_version: str,
    ) -> list[int]:
        rows = self._fetchall(
            """SELECT result_id FROM agent_success_evaluation_result
               WHERE user_id = ?
                 AND session_id = ?
                 AND evaluation_name = ?
                 AND agent_version = ?
               ORDER BY created_at DESC""",
            (user_id, session_id, evaluation_name, agent_version),
        )
        return [int(r["result_id"]) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_all_agent_success_evaluation_results(self) -> None:
        self._execute("DELETE FROM agent_success_evaluation_result")

    @SQLiteStorageBase.handle_exceptions
    def delete_agent_success_evaluation_results_for_session(
        self,
        user_id: str,
        session_id: str,
        evaluation_name: str,
        agent_version: str,
    ) -> int:
        """Delete results scoped to (user_id, session_id, evaluation_name, agent_version).

        Args:
            user_id (str): User whose session results to clear.
            session_id (str): Session whose results to clear.
            evaluation_name (str): Which evaluator's results to clear.
            agent_version (str): Agent version scope.

        Returns:
            int: Number of rows deleted.
        """
        cur = self._execute(
            """DELETE FROM agent_success_evaluation_result
               WHERE user_id = ?
                 AND session_id = ?
                 AND evaluation_name = ?
                 AND agent_version = ?""",
            (user_id, session_id, evaluation_name, agent_version),
        )
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def delete_agent_success_evaluation_results_by_ids(
        self, result_ids: list[int]
    ) -> int:
        """Delete agent success eval result rows by primary key.

        Args:
            result_ids (list[int]): Primary-key result_ids to delete. An empty
                list is a no-op that returns 0.

        Returns:
            int: Number of rows actually deleted (ignores non-existent ids).
        """
        if not result_ids:
            return 0
        placeholders = ",".join(["?"] * len(result_ids))
        cur = self._execute(
            f"DELETE FROM agent_success_evaluation_result WHERE result_id IN ({placeholders})",
            list(result_ids),
        )
        return cur.rowcount

    # ------------------------------------------------------------------
    # Retrieved-learning evaluation methods
    # ------------------------------------------------------------------
    # Generation fencing + session-fingerprint CAS; see
    # storage_base/retrieved_learning_state.py for the concurrency contract.
    # All state reads/writes inside a transaction use raw self.conn.execute
    # (never self._execute or self-committing helpers) so replacement stays
    # atomic on the shared connection.

    def _rle_state_row(self, state_key: str) -> dict[str, Any]:
        cur = self.conn.execute(
            "SELECT operation_state FROM _operation_state WHERE service_name = ?",
            (state_key,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        return _json_loads(row["operation_state"]) or {}

    def _rle_upsert_state(self, state_key: str, state: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT INTO _operation_state (service_name, operation_state, updated_at)
               VALUES (?,?,?)
               ON CONFLICT(service_name) DO UPDATE SET
                 operation_state = excluded.operation_state,
                 updated_at = excluded.updated_at""",
            (state_key, _json_dumps(state), self._current_timestamp()),
        )

    def _rle_fingerprint_now(self, user_id: str, session_id: str) -> str:
        """Recompute the session fingerprint from live rows (in-transaction)."""
        cur = self.conn.execute(
            """SELECT i.interaction_id, i.role,
                      substr(i.content, 1, ?) AS content, i.retrieved_learnings
               FROM interactions i JOIN requests r ON i.request_id = r.request_id
               WHERE r.session_id = ? AND i.user_id = ?
               ORDER BY i.created_at ASC, i.interaction_id ASC""",
            (DEFAULT_TRANSCRIPT_CHAR_LIMIT, session_id, user_id),
        )
        builder = SessionFingerprintBuilder()
        for row in cur:
            builder.add(
                int(row["interaction_id"]),
                _parse_attachment_refs(row["retrieved_learnings"]),
                row["role"] or "User",
                row["content"] or "",
            )
        return builder.hexdigest()

    def _rle_attached_refs(self, user_id: str, session_id: str) -> set[tuple[str, str]]:
        """Canonical ``(kind, learning_id)`` refs attached to the live session."""
        cur = self.conn.execute(
            """SELECT i.retrieved_learnings
               FROM interactions i JOIN requests r ON i.request_id = r.request_id
               WHERE r.session_id = ? AND i.user_id = ?""",
            (session_id, user_id),
        )
        attached: set[tuple[str, str]] = set()
        for row in cur:
            attached.update(_parse_attachment_refs(row["retrieved_learnings"]))
        return attached

    def _rle_resolvable_refs(
        self, user_id: str, results: list[RetrievedLearningEvaluationResult]
    ) -> set[tuple[str, str]]:
        """Recheck original source rows for historical resolution (in-transaction)."""
        by_kind: dict[str, list[str]] = {}
        for r in results:
            by_kind.setdefault(r.kind, []).append(r.learning_id)
        resolvable: set[tuple[str, str]] = set()
        for chunk in _chunked(by_kind.get("profile", [])):
            ph = ",".join("?" for _ in chunk)
            cur = self.conn.execute(
                f"""SELECT profile_id FROM profiles
                    WHERE user_id = ? AND profile_id IN ({ph})""",
                (user_id, *chunk),
            )
            resolvable.update(("profile", str(row[0])) for row in cur.fetchall())
        user_playbook_ids = [
            i for i in (_as_int(x) for x in by_kind.get("user_playbook", [])) if i
        ]
        for chunk in _chunked(user_playbook_ids):
            ph = ",".join("?" for _ in chunk)
            cur = self.conn.execute(
                f"""SELECT user_playbook_id FROM user_playbooks
                    WHERE user_id = ? AND user_playbook_id IN ({ph})""",
                (user_id, *chunk),
            )
            resolvable.update(("user_playbook", str(row[0])) for row in cur.fetchall())
        agent_playbook_ids = [
            i for i in (_as_int(x) for x in by_kind.get("agent_playbook", [])) if i
        ]
        for chunk in _chunked(agent_playbook_ids):
            ph = ",".join("?" for _ in chunk)
            cur = self.conn.execute(
                f"""SELECT agent_playbook_id FROM agent_playbooks
                    WHERE agent_playbook_id IN ({ph})""",
                chunk,
            )
            resolvable.update(("agent_playbook", str(row[0])) for row in cur.fetchall())
        return resolvable

    @SQLiteStorageBase.handle_exceptions
    def begin_retrieved_learning_evaluation_run(
        self, user_id: str, session_id: str
    ) -> int:
        state_key = build_retrieved_learning_state_key(user_id, session_id)
        subject_ref = self._subject_ref_for_user_id(user_id)
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_subject_writable_locked(subject_ref)
                state = self._rle_state_row(state_key)
                generation = int(state.get("generation") or 0) + 1
                state.update(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "generation": generation,
                        "attempted_at": int(self._current_epoch()),
                    }
                )
                state.setdefault("status", "pending")
                state.setdefault("session_fingerprint", "")
                self._rle_upsert_state(state_key, state)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return generation

    @SQLiteStorageBase.handle_exceptions
    def load_bounded_retrieved_learning_snapshot(
        self,
        user_id: str,
        session_id: str,
        raw_ref_limit: int = 5_000,
        transcript_char_limit: int = DEFAULT_TRANSCRIPT_CHAR_LIMIT,
    ) -> BoundedRetrievedLearningSnapshot:
        snapshot = BoundedRetrievedLearningSnapshot()
        row = self._fetchone(
            """SELECT MIN(created_at) AS earliest FROM requests
               WHERE session_id = ? AND user_id = ?""",
            (session_id, user_id),
        )
        if row and row["earliest"]:
            snapshot.earliest_request_created_at = _iso_to_epoch(row["earliest"])
        row = self._fetchone(
            """SELECT agent_version FROM requests
               WHERE session_id = ? AND user_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (session_id, user_id),
        )
        if row:
            snapshot.agent_version = row["agent_version"] or ""
        cursor = self.conn.execute(
            # ``fp_content`` is truncated to the fixed fingerprint limit,
            # independent of the caller's ``transcript_char_limit`` snapshot
            # budget, so the commit-side recompute (which has no such budget)
            # produces an identical digest.
            """SELECT i.interaction_id, i.role,
                      substr(i.content, 1, ?) AS content,
                      substr(i.content, 1, ?) AS fp_content,
                      i.created_at, i.retrieved_learnings
               FROM interactions i JOIN requests r ON i.request_id = r.request_id
               WHERE r.session_id = ? AND i.user_id = ?
               ORDER BY i.created_at ASC, i.interaction_id ASC""",
            (transcript_char_limit, DEFAULT_TRANSCRIPT_CHAR_LIMIT, session_id, user_id),
        )
        builder = SessionFingerprintBuilder()
        transcript_chars_remaining = transcript_char_limit
        for r in cursor:
            refs = _parse_attachment_refs(r["retrieved_learnings"])
            snapshot.raw_attachment_count += len(refs)
            builder.add(
                int(r["interaction_id"]),
                refs,
                r["role"] or "User",
                r["fp_content"] or "",
            )
            if snapshot.raw_attachment_count > raw_ref_limit:
                snapshot.attachment_limit_exceeded = True
                snapshot.interactions.clear()
                transcript_chars_remaining = 0
                continue
            transcript_chars_remaining = append_bounded_snapshot_interaction(
                snapshot,
                interaction_id=int(r["interaction_id"]),
                role=r["role"] or "User",
                content=r["content"] or "",
                created_at=_iso_to_epoch(r["created_at"]),
                refs=refs,
                transcript_chars_remaining=transcript_chars_remaining,
            )
        snapshot.precomputed_fingerprint = builder.hexdigest()
        return snapshot

    @SQLiteStorageBase.handle_exceptions
    def get_matching_retrieved_learning_terminal_state(
        self, user_id: str, session_id: str, session_fingerprint: str
    ) -> dict[str, Any] | None:
        state_key = build_retrieved_learning_state_key(user_id, session_id)
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                state = self._rle_state_row(state_key)
                live_fingerprint = self._rle_fingerprint_now(user_id, session_id)
                matched = (
                    state.get("status") in TERMINAL_RETRIEVED_STATUSES
                    and state.get("session_fingerprint") == session_fingerprint
                    and live_fingerprint == session_fingerprint
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return state if matched else None

    @SQLiteStorageBase.handle_exceptions
    def replace_retrieved_learning_evaluation_results(
        self,
        user_id: str,
        session_id: str,
        generation: int,
        session_fingerprint: str,
        proposed_status: str,
        diagnostics: dict[str, Any],
        results: list[RetrievedLearningEvaluationResult],
    ) -> RetrievedLearningCommitResult:
        if proposed_status not in ("complete", "degraded"):
            raise ValueError(f"invalid proposed_status: {proposed_status}")
        state_key = build_retrieved_learning_state_key(user_id, session_id)
        subject_ref = self._subject_ref_for_user_id(user_id)
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_subject_writable_locked(subject_ref)
                state = self._rle_state_row(state_key)
                if int(state.get("generation") or 0) != generation:
                    self.conn.rollback()
                    return RetrievedLearningCommitResult(disposition="superseded")
                if self._rle_fingerprint_now(user_id, session_id) != (
                    session_fingerprint
                ):
                    self.conn.rollback()
                    return RetrievedLearningCommitResult(disposition="stale")
                # Persist only records that are still attached to the live
                # session AND whose original row still exists. A duplicate identity is left
                # to trip the UNIQUE index and roll back the commit — caller
                # bugs fail loud rather than silently dropping data.
                attached = self._rle_attached_refs(user_id, session_id)
                resolvable = self._rle_resolvable_refs(user_id, results)
                kept = [
                    r
                    for r in results
                    if (r.kind, r.learning_id) in attached
                    and (r.kind, r.learning_id) in resolvable
                ]
                final_status = proposed_status if kept else "not_applicable"
                self.conn.execute(
                    """DELETE FROM retrieved_learning_evaluation
                       WHERE user_id = ? AND session_id = ?""",
                    (user_id, session_id),
                )
                for r in kept:
                    self.conn.execute(
                        """INSERT INTO retrieved_learning_evaluation
                           (user_id, session_id, agent_version, kind,
                            learning_id, is_relevant, relevance_reason, impact,
                            impact_reason, created_at, governance_subject_ref)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            user_id,
                            session_id,
                            r.agent_version,
                            r.kind,
                            r.learning_id,
                            None if r.is_relevant is None else int(r.is_relevant),
                            r.relevance_reason,
                            r.impact,
                            r.impact_reason,
                            r.created_at,
                            subject_ref,
                        ),
                    )
                state.update(diagnostics)
                state.update(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "generation": generation,
                        "status": final_status,
                        "session_fingerprint": session_fingerprint,
                        "resolvable_count": len(resolvable),
                        "committed_count": len(kept),
                        "completed_at": int(self._current_epoch()),
                    }
                )
                self._rle_upsert_state(state_key, state)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return RetrievedLearningCommitResult(
            disposition="applied", status=final_status, committed_count=len(kept)
        )

    @SQLiteStorageBase.handle_exceptions
    def finish_retrieved_learning_evaluation_run(
        self,
        user_id: str,
        session_id: str,
        generation: int,
        status: str,
        diagnostics: dict[str, Any],
    ) -> None:
        if status not in ("failed", "pending"):
            raise ValueError(f"invalid finish status: {status}")
        state_key = build_retrieved_learning_state_key(user_id, session_id)
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                state = self._rle_state_row(state_key)
                if int(state.get("generation") or 0) != generation:
                    self.conn.rollback()
                    return
                state.update(diagnostics)
                state["status"] = status
                self._rle_upsert_state(state_key, state)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def get_retrieved_learning_evaluation_results(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[RetrievedLearningEvaluationResult]:
        sql = "SELECT * FROM retrieved_learning_evaluation"
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, result_id DESC LIMIT ?"
        params.append(limit)
        rows = self._fetchall(sql, params)
        return [_row_to_retrieved_learning_result(r) for r in rows]

    def _current_epoch(self) -> int:
        return int(datetime.now(UTC).timestamp())


def _parse_attachment_refs(raw: Any) -> list[tuple[str, str]]:
    """Parse a stored retrieved_learnings JSON blob into (kind, learning_id) tuples."""
    parsed = _json_loads(raw) if isinstance(raw, str) else raw
    if not parsed or not isinstance(parsed, list):
        return []
    refs: list[tuple[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        learning_id = str(item.get("learning_id") or "")
        if kind in CANONICAL_RETRIEVED_KINDS and learning_id:
            refs.append((kind, learning_id))
    return refs


def _row_to_retrieved_learning_result(
    row: sqlite3.Row,
) -> RetrievedLearningEvaluationResult:
    d = dict(row)
    return RetrievedLearningEvaluationResult(
        result_id=int(d["result_id"]),
        user_id=d["user_id"],
        session_id=d["session_id"],
        agent_version=d.get("agent_version") or "",
        kind=d["kind"],
        learning_id=d["learning_id"],
        is_relevant=None if d.get("is_relevant") is None else bool(d["is_relevant"]),
        relevance_reason=d.get("relevance_reason") or "",
        impact=d.get("impact"),
        impact_reason=d.get("impact_reason") or "",
        created_at=int(d["created_at"]),
    )


def _as_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _chunked(items: list, size: int = 500) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]
