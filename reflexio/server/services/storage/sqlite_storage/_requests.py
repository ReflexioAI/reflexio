"""Request CRUD methods for SQLite storage."""

import sqlite3
from typing import Any, Literal, cast

from reflexio.models.api_schema.internal_schema import (
    RequestInteractionDataModel,
    SessionDescriptor,
    SessionFirstRequest,
)
from reflexio.models.api_schema.service_schemas import (
    Request,
)
from reflexio.models.api_schema.validators import validate_session_outcome_source

from ._base import (
    SQLiteStorageBase,
    _epoch_to_iso,
    _iso_to_epoch,
    _row_to_interaction,
    _row_to_request,
)


class RequestMixin:
    """Mixin providing request CRUD operations."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _execute: Any
    _fetchone: Any
    _fetchall: Any
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any
    _own_transaction: Any

    # ------------------------------------------------------------------
    # Request methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def add_request(self, request: Request) -> None:
        source = validate_session_outcome_source(request.source)
        created_at_iso = _epoch_to_iso(request.created_at)
        subject_ref = self._subject_ref_for_user_id(request.user_id)
        with self._lock:
            own_txn = self._own_transaction()
            try:
                if own_txn:
                    self.conn.execute("BEGIN IMMEDIATE")
                self._assert_subject_writable_locked(subject_ref)
                self.conn.execute(
                    """INSERT OR REPLACE INTO requests
                       (request_id, user_id, created_at, source, agent_version, session_id,
                        evaluation_only, governance_subject_ref,
                        retrieval_experiment_id, retrieval_experiment_arm)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        request.request_id,
                        request.user_id,
                        created_at_iso,
                        source,
                        request.agent_version,
                        request.session_id,
                        1 if request.evaluation_only else 0,
                        subject_ref,
                        request.retrieval_experiment_id,
                        request.retrieval_experiment_arm,
                    ),
                )
                if own_txn:
                    self.conn.commit()
            except Exception:
                if own_txn:
                    self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def get_request(self, request_id: str) -> Request | None:
        row = self._fetchone(
            "SELECT * FROM requests WHERE request_id = ?", (request_id,)
        )
        return _row_to_request(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def delete_request(self, request_id: str) -> None:
        # Delete FTS entries for interactions of this request
        ids = [
            r["interaction_id"]
            for r in self._fetchall(
                "SELECT interaction_id FROM interactions WHERE request_id = ?",
                (request_id,),
            )
        ]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            with self._lock:
                self.conn.execute(
                    f"DELETE FROM interactions_fts WHERE rowid IN ({placeholders})", ids
                )
                self.conn.commit()
        self._execute("DELETE FROM interactions WHERE request_id = ?", (request_id,))
        self._execute("DELETE FROM requests WHERE request_id = ?", (request_id,))

    @SQLiteStorageBase.handle_exceptions
    def delete_session(self, session_id: str) -> int:
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                rows = self.conn.execute(
                    "SELECT request_id FROM requests WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
                request_ids = [str(row["request_id"]) for row in rows]
                if request_ids:
                    placeholders = ",".join("?" for _ in request_ids)
                    self.conn.execute(
                        f"""DELETE FROM interactions_fts WHERE rowid IN (
                            SELECT interaction_id FROM interactions
                            WHERE request_id IN ({placeholders})
                        )""",
                        request_ids,
                    )
                    self.conn.execute(
                        f"DELETE FROM interactions WHERE request_id IN ({placeholders})",
                        request_ids,
                    )
                    self.conn.execute(
                        "DELETE FROM requests WHERE session_id = ?", (session_id,)
                    )
                self.conn.commit()
                return len(request_ids)
            except Exception:
                self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def delete_all_requests(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM interactions_fts")
            self.conn.execute("DELETE FROM interactions")
            self.conn.execute("DELETE FROM requests")
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_requests_by_ids(self, request_ids: list[str]) -> int:
        if not request_ids:
            return 0
        # Delete FTS entries for interactions of these requests
        ph = ",".join("?" for _ in request_ids)
        interaction_ids = [
            r["interaction_id"]
            for r in self._fetchall(
                f"SELECT interaction_id FROM interactions WHERE request_id IN ({ph})",
                request_ids,
            )
        ]
        if interaction_ids:
            iph = ",".join("?" for _ in interaction_ids)
            with self._lock:
                self.conn.execute(
                    f"DELETE FROM interactions_fts WHERE rowid IN ({iph})",
                    interaction_ids,
                )
                self.conn.commit()
        self._execute(
            f"DELETE FROM interactions WHERE request_id IN ({ph})", request_ids
        )
        cur = self._execute(
            f"DELETE FROM requests WHERE request_id IN ({ph})", request_ids
        )
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def get_sessions(
        self,
        user_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        top_k: int | None = 30,
        offset: int = 0,
        source: str | None = None,
    ) -> dict[str, list[RequestInteractionDataModel]]:
        # Request-level filters shared by both the session-page query and the
        # request-fetch query. Pagination is applied at the SESSION level
        # (top_k/offset count sessions, not request rows) so a session with many
        # requests is never truncated to a subset of its rows.
        filter_sql = ""
        filter_params: list[Any] = []
        if user_id:
            filter_sql += " AND user_id = ?"
            filter_params.append(user_id)
        if request_id:
            filter_sql += " AND request_id = ?"
            filter_params.append(request_id)
        if session_id:
            filter_sql += " AND session_id = ?"
            filter_params.append(session_id)
        if source is not None:
            filter_sql += " AND source = ?"
            filter_params.append(source)
        if start_time is not None:
            filter_sql += " AND created_at >= ?"
            filter_params.append(_epoch_to_iso(start_time))
        if end_time is not None:
            filter_sql += " AND created_at <= ?"
            filter_params.append(_epoch_to_iso(end_time))

        # Step 1: select the page of session_ids, ordered by each session's most
        # recent matching request (latest-first), with session_id as a stable
        # tiebreak so pages don't overlap or skip.
        effective_limit = top_k or 100
        page_rows = self._fetchall(
            f"SELECT session_id, MAX(created_at) AS latest FROM requests "
            f"WHERE 1=1{filter_sql} "
            f"GROUP BY session_id ORDER BY latest DESC, session_id DESC "
            f"LIMIT ? OFFSET ?",
            [*filter_params, effective_limit, offset],
        )
        if not page_rows:
            return {}
        page_session_ids = [r["session_id"] for r in page_rows]

        # Step 2: fetch ALL matching requests for the sessions in this page.
        placeholders = ",".join("?" for _ in page_session_ids)
        req_rows = self._fetchall(
            f"SELECT * FROM requests WHERE 1=1{filter_sql} "
            f"AND session_id IN ({placeholders}) ORDER BY created_at DESC",
            [*filter_params, *page_session_ids],
        )
        if not req_rows:
            return {}

        # Preserve the latest-first session ordering from step 1.
        grouped: dict[str, list[RequestInteractionDataModel]] = {
            (sid or ""): [] for sid in page_session_ids
        }
        for rr in req_rows:
            req = _row_to_request(rr)
            group_name = req.session_id or ""
            int_rows = self._fetchall(
                "SELECT * FROM interactions WHERE request_id = ? ORDER BY created_at ASC",
                (req.request_id,),
            )
            interactions = [_row_to_interaction(ir) for ir in int_rows]
            grouped.setdefault(group_name, []).append(
                RequestInteractionDataModel(
                    session_id=group_name,
                    request=req,
                    interactions=interactions,
                )
            )
        return grouped

    @SQLiteStorageBase.handle_exceptions
    def get_rerun_user_ids(
        self,
        user_id: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        source: str | None = None,
        agent_version: str | None = None,
    ) -> list[str]:
        sql = "SELECT DISTINCT user_id FROM requests WHERE 1=1"
        params: list[Any] = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if start_time:
            sql += " AND created_at >= ?"
            params.append(_epoch_to_iso(start_time))
        if end_time:
            sql += " AND created_at <= ?"
            params.append(_epoch_to_iso(end_time))
        if source:
            sql += " AND source = ?"
            params.append(source)
        if agent_version:
            sql += " AND agent_version = ?"
            params.append(agent_version)

        rows = self._fetchall(sql, params)
        return sorted(r["user_id"] for r in rows)

    @SQLiteStorageBase.handle_exceptions
    def get_requests_by_session(self, user_id: str, session_id: str) -> list[Request]:
        rows = self._fetchall(
            "SELECT * FROM requests WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        )
        return [_row_to_request(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_retrieval_experiment_assignments(
        self, experiment_id: str
    ) -> dict[tuple[str, str], Literal["treatment", "holdout"]]:
        rows = self._fetchall(
            """SELECT user_id, session_id, retrieval_experiment_arm
               FROM (
                   SELECT user_id, session_id, retrieval_experiment_arm,
                          ROW_NUMBER() OVER (
                              PARTITION BY user_id, session_id
                              ORDER BY created_at ASC, request_id ASC
                          ) AS row_number
                   FROM requests
                   WHERE retrieval_experiment_id = ?
               ) ranked
               WHERE row_number = 1""",
            (experiment_id,),
        )
        return {
            (str(row["user_id"]), str(row["session_id"])): cast(
                Literal["treatment", "holdout"], row["retrieval_experiment_arm"]
            )
            for row in rows
        }

    @SQLiteStorageBase.handle_exceptions
    def get_retrieval_experiment_output_token_counts(
        self, experiment_id: str
    ) -> dict[tuple[str, str], int]:
        rows = self._fetchall(
            """SELECT r.user_id,
                      r.session_id,
                      COALESCE(SUM(
                          CASE
                              WHEN LOWER(TRIM(
                                  i.role,
                                  char(9) || char(10) || char(11) ||
                                  char(12) || char(13) || ' '
                              )) <> 'user'
                              THEN i.token_count
                              ELSE 0
                          END
                      ), 0) AS output_token_count,
                      SUM(
                          CASE
                              WHEN LOWER(TRIM(
                                  i.role,
                                  char(9) || char(10) || char(11) ||
                                  char(12) || char(13) || ' '
                              )) <> 'user'
                                   AND i.token_count IS NULL
                              THEN 1
                              ELSE 0
                          END
                      ) AS missing_token_count
               FROM requests r
               LEFT JOIN interactions i ON i.request_id = r.request_id
               WHERE r.retrieval_experiment_id = ?
               GROUP BY r.user_id, r.session_id""",
            (experiment_id,),
        )
        return {
            (str(row["user_id"]), str(row["session_id"])): int(
                row["output_token_count"]
            )
            for row in rows
            if int(row["missing_token_count"] or 0) == 0
        }

    @SQLiteStorageBase.handle_exceptions
    def get_session_ids_in_window(
        self, from_ts: int, to_ts: int
    ) -> list[SessionDescriptor]:
        from_iso = _epoch_to_iso(from_ts)
        to_iso = _epoch_to_iso(to_ts)
        rows = self._fetchall(
            """SELECT DISTINCT user_id, session_id, agent_version, source
               FROM requests
               WHERE session_id IS NOT NULL
                 AND created_at BETWEEN ? AND ?
               ORDER BY session_id, user_id, agent_version""",
            (from_iso, to_iso),
        )
        return [
            SessionDescriptor(
                user_id=r["user_id"],
                session_id=r["session_id"],
                agent_version=r["agent_version"],
                source=r["source"],
            )
            for r in rows
        ]

    @SQLiteStorageBase.handle_exceptions
    def get_first_requests_by_session_ids(
        self, session_ids: list[str]
    ) -> dict[str, SessionFirstRequest]:
        if not session_ids:
            return {}
        out: dict[str, SessionFirstRequest] = {}
        ids = sorted(set(session_ids))
        chunk_size = 500
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            ph = ",".join("?" for _ in chunk)
            rows = self._fetchall(
                f"""SELECT session_id, user_id, source, created_at
                    FROM (
                        SELECT session_id, user_id, source, created_at,
                               ROW_NUMBER() OVER (
                                   PARTITION BY session_id
                                   ORDER BY created_at ASC, request_id ASC
                               ) AS rn
                        FROM requests
                        WHERE session_id IN ({ph})
                    )
                    WHERE rn = 1""",  # noqa: S608
                chunk,
            )
            for row in rows:
                session_id = row["session_id"]
                out[session_id] = SessionFirstRequest(
                    session_id=session_id,
                    user_id=row["user_id"],
                    source=row["source"] or "",
                    created_at=_iso_to_epoch(row["created_at"]),
                )
        return out

    @SQLiteStorageBase.handle_exceptions
    def get_first_requests_by_user_session_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], SessionFirstRequest]:
        if not pairs:
            return {}
        out: dict[tuple[str, str], SessionFirstRequest] = {}
        pair_list = sorted(set(pairs))
        chunk_size = 300
        for i in range(0, len(pair_list), chunk_size):
            chunk = pair_list[i : i + chunk_size]
            values = ",".join("(?, ?)" for _ in chunk)
            params = [value for pair in chunk for value in pair]
            rows = self._fetchall(
                f"""SELECT session_id, user_id, source, created_at
                    FROM (
                        SELECT session_id, user_id, source, created_at,
                               ROW_NUMBER() OVER (
                                   PARTITION BY user_id, session_id
                                   ORDER BY created_at ASC, request_id ASC
                               ) AS rn
                        FROM requests
                        WHERE (user_id, session_id) IN ({values})
                    )
                    WHERE rn = 1""",  # noqa: S608
                params,
            )
            for row in rows:
                key = (row["user_id"], row["session_id"])
                out[key] = SessionFirstRequest(
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                    source=row["source"] or "",
                    created_at=_iso_to_epoch(row["created_at"]),
                )
        return out
