"""Request CRUD methods for SQLite storage."""

import sqlite3
from typing import Any

from reflexio.models.api_schema.internal_schema import (
    RequestInteractionDataModel,
    SessionDescriptor,
    SessionFirstRequest,
)
from reflexio.models.api_schema.service_schemas import (
    Request,
)

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

    # ------------------------------------------------------------------
    # Request methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def add_request(self, request: Request) -> None:
        created_at_iso = _epoch_to_iso(request.created_at)
        self._execute(
            """INSERT OR REPLACE INTO requests
               (request_id, user_id, created_at, source, agent_version, session_id, evaluation_only)
               VALUES (?,?,?,?,?,?,?)""",
            (
                request.request_id,
                request.user_id,
                created_at_iso,
                request.source,
                request.agent_version,
                request.session_id,
                1 if request.evaluation_only else 0,
            ),
        )

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
        rows = self._fetchall(
            "SELECT request_id FROM requests WHERE session_id = ?", (session_id,)
        )
        if not rows:
            return 0
        request_ids = [r["request_id"] for r in rows]
        for rid in request_ids:
            # Delete FTS for interactions
            iids = [
                r["interaction_id"]
                for r in self._fetchall(
                    "SELECT interaction_id FROM interactions WHERE request_id = ?",
                    (rid,),
                )
            ]
            if iids:
                ph = ",".join("?" for _ in iids)
                with self._lock:
                    self.conn.execute(
                        f"DELETE FROM interactions_fts WHERE rowid IN ({ph})", iids
                    )
                    self.conn.commit()
            self._execute("DELETE FROM interactions WHERE request_id = ?", (rid,))
        self._execute("DELETE FROM requests WHERE session_id = ?", (session_id,))
        return len(request_ids)

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
    ) -> dict[str, list[RequestInteractionDataModel]]:
        sql = "SELECT * FROM requests WHERE 1=1"
        params: list[Any] = []

        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if request_id:
            sql += " AND request_id = ?"
            params.append(request_id)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if start_time:
            sql += " AND created_at >= ?"
            params.append(_epoch_to_iso(start_time))
        if end_time:
            sql += " AND created_at <= ?"
            params.append(_epoch_to_iso(end_time))

        effective_limit = top_k or 100
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([effective_limit, offset])

        req_rows = self._fetchall(sql, params)
        if not req_rows:
            return {}

        grouped: dict[str, list[RequestInteractionDataModel]] = {}
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
