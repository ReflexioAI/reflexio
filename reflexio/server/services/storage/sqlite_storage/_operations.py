"""Operation state methods for SQLite storage."""

import sqlite3
import time
from typing import Any

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)

from ._base import (
    SQLiteStorageBase,
    _epoch_to_iso,
    _iso_to_epoch,
    _json_dumps,
    _json_loads,
    _row_to_interaction,
)


class OperationMixin:
    """Mixin providing operation state CRUD and locking."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _execute: Any
    _fetchone: Any
    _fetchall: Any
    _current_timestamp: Any
    get_operation_state: Any

    # ------------------------------------------------------------------
    # Operation State methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def create_operation_state(self, service_name: str, operation_state: dict) -> None:
        self._execute(
            "INSERT INTO _operation_state (service_name, operation_state, updated_at) VALUES (?,?,?)",
            (service_name, _json_dumps(operation_state), self._current_timestamp()),
        )

    @SQLiteStorageBase.handle_exceptions
    def upsert_operation_state(self, service_name: str, operation_state: dict) -> None:
        self._execute(
            """INSERT INTO _operation_state (service_name, operation_state, updated_at)
               VALUES (?,?,?)
               ON CONFLICT(service_name) DO UPDATE SET
                 operation_state = excluded.operation_state,
                 updated_at = excluded.updated_at""",
            (service_name, _json_dumps(operation_state), self._current_timestamp()),
        )

    @SQLiteStorageBase.handle_exceptions
    def get_operation_state(self, service_name: str) -> dict | None:
        row = self._fetchone(
            "SELECT * FROM _operation_state WHERE service_name = ?", (service_name,)
        )
        if not row:
            return None
        return {
            "service_name": row["service_name"],
            "operation_state": _json_loads(row["operation_state"]),
            "updated_at": row["updated_at"],
        }

    @SQLiteStorageBase.handle_exceptions
    def get_operation_state_with_new_request_interaction(
        self,
        service_name: str,
        user_id: str | None,
        sources: list[str] | None = None,
    ) -> tuple[dict, list[RequestInteractionDataModel]]:
        # Get operation state
        state_record = self.get_operation_state(service_name)
        operation_state: dict = {}
        if state_record and isinstance(state_record.get("operation_state"), dict):
            operation_state = state_record["operation_state"]

        last_processed_ids = operation_state.get("last_processed_interaction_ids") or []
        last_processed_timestamp = operation_state.get("last_processed_timestamp")

        # Build query for new interactions
        sql = """
            SELECT i.*, r.request_id as r_request_id, r.user_id as r_user_id,
                   r.created_at as r_created_at, r.source as r_source,
                   r.agent_version as r_agent_version, r.session_id as r_session_id
            FROM interactions i
            JOIN requests r ON i.request_id = r.request_id
            WHERE 1=1
        """
        params: list[Any] = []

        if user_id:
            sql += " AND i.user_id = ?"
            params.append(user_id)

        if last_processed_timestamp is not None:
            sql += " AND i.created_at >= ?"
            params.append(_epoch_to_iso(last_processed_timestamp))

        if last_processed_ids:
            ph = ",".join("?" for _ in last_processed_ids)
            sql += f" AND i.interaction_id NOT IN ({ph})"
            params.extend(int(x) for x in last_processed_ids)

        if sources:
            ph = ",".join("?" for _ in sources)
            sql += f" AND r.source IN ({ph})"
            params.extend(sources)

        sql += " ORDER BY i.created_at ASC"
        rows = self._fetchall(sql, params)

        # Group by request
        requests_map: dict[str, Request] = {}
        interactions_by_request: dict[str, list[Interaction]] = {}

        for row in rows:
            d = dict(row)
            req_id = d["request_id"]
            if req_id not in requests_map:
                requests_map[req_id] = Request(
                    request_id=d["r_request_id"],
                    user_id=d["r_user_id"],
                    created_at=_iso_to_epoch(d["r_created_at"]),
                    source=d.get("r_source") or "",
                    agent_version=d.get("r_agent_version") or "",
                    session_id=d.get("r_session_id"),
                )
                interactions_by_request[req_id] = []
            interactions_by_request[req_id].append(_row_to_interaction(row))

        sessions: list[RequestInteractionDataModel] = []
        for req_id, req in requests_map.items():
            interactions = sorted(
                interactions_by_request[req_id], key=lambda x: x.created_at or 0
            )
            group_name = req.session_id or req.request_id
            sessions.append(
                RequestInteractionDataModel(
                    session_id=group_name,
                    request=req,
                    interactions=interactions,
                )
            )

        sessions.sort(
            key=lambda g: (
                min(i.created_at or 0 for i in g.interactions) if g.interactions else 0
            )
        )
        return operation_state, sessions

    @SQLiteStorageBase.handle_exceptions
    def get_last_k_interactions_grouped(
        self,
        user_id: str | None,
        k: int,
        sources: list[str] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        agent_version: str | None = None,
    ) -> tuple[list[RequestInteractionDataModel], list[Interaction]]:
        sql = """
            SELECT i.*, r.request_id as r_request_id, r.user_id as r_user_id,
                   r.created_at as r_created_at, r.source as r_source,
                   r.agent_version as r_agent_version, r.session_id as r_session_id
            FROM interactions i
            JOIN requests r ON i.request_id = r.request_id
            WHERE 1=1
        """
        params: list[Any] = []

        if user_id:
            sql += " AND i.user_id = ?"
            params.append(user_id)
        if sources:
            ph = ",".join("?" for _ in sources)
            sql += f" AND r.source IN ({ph})"
            params.extend(sources)
        if start_time:
            sql += " AND i.created_at >= ?"
            params.append(_epoch_to_iso(start_time))
        if end_time:
            sql += " AND i.created_at <= ?"
            params.append(_epoch_to_iso(end_time))
        if agent_version:
            sql += " AND r.agent_version = ?"
            params.append(agent_version)

        sql += " ORDER BY i.created_at DESC LIMIT ?"
        params.append(k)

        rows = self._fetchall(sql, params)

        flat_interactions: list[Interaction] = []
        requests_map: dict[str, Request] = {}
        interactions_by_request: dict[str, list[Interaction]] = {}

        for row in rows:
            d = dict(row)
            req_id = d["request_id"]
            if req_id not in requests_map:
                requests_map[req_id] = Request(
                    request_id=d["r_request_id"],
                    user_id=d["r_user_id"],
                    created_at=_iso_to_epoch(d["r_created_at"]),
                    source=d.get("r_source") or "",
                    agent_version=d.get("r_agent_version") or "",
                    session_id=d.get("r_session_id"),
                )
                interactions_by_request[req_id] = []

            interaction = _row_to_interaction(row)
            flat_interactions.append(interaction)
            interactions_by_request[req_id].append(interaction)

        sessions: list[RequestInteractionDataModel] = []
        for req_id, req in requests_map.items():
            interactions = sorted(
                interactions_by_request[req_id], key=lambda x: x.created_at or 0
            )
            group_name = req.session_id or req.request_id
            sessions.append(
                RequestInteractionDataModel(
                    session_id=group_name,
                    request=req,
                    interactions=interactions,
                )
            )

        sessions.sort(
            key=lambda g: (
                min(i.created_at or 0 for i in g.interactions) if g.interactions else 0
            )
        )
        return sessions, flat_interactions

    @SQLiteStorageBase.handle_exceptions
    def update_operation_state(self, service_name: str, operation_state: dict) -> None:
        self._execute(
            "UPDATE _operation_state SET operation_state = ?, updated_at = ? WHERE service_name = ?",
            (_json_dumps(operation_state), self._current_timestamp(), service_name),
        )

    @SQLiteStorageBase.handle_exceptions
    def get_all_operation_states(self) -> list[dict]:
        rows = self._fetchall("SELECT * FROM _operation_state")
        return [
            {
                "service_name": r["service_name"],
                "operation_state": _json_loads(r["operation_state"]),
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    @SQLiteStorageBase.handle_exceptions
    def delete_operation_state(self, service_name: str) -> None:
        self._execute(
            "DELETE FROM _operation_state WHERE service_name = ?", (service_name,)
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_all_operation_states(self) -> None:
        self._execute("DELETE FROM _operation_state")

    @SQLiteStorageBase.handle_exceptions
    def try_acquire_in_progress_lock(
        self, state_key: str, request_id: str, stale_lock_seconds: int = 300
    ) -> dict:
        with self._lock:
            row = self._fetchone(
                "SELECT operation_state, updated_at FROM _operation_state WHERE service_name = ?",
                (state_key,),
            )

            now = time.time()

            if row is None:
                # No state exists — acquire lock
                state = {
                    "status": "in_progress",
                    "current_request_id": request_id,
                    "pending_request_id": None,
                }
                self._execute(
                    """INSERT INTO _operation_state (service_name, operation_state, updated_at)
                       VALUES (?, ?, ?)""",
                    (state_key, _json_dumps(state), self._current_timestamp()),
                )
                return {"acquired": True, "state": state}

            current_state = _json_loads(row["operation_state"]) or {}

            # Check if lock is stale
            updated_at_str = row["updated_at"]
            is_stale = False
            if updated_at_str:
                updated_epoch = _iso_to_epoch(updated_at_str)
                is_stale = (now - updated_epoch) > stale_lock_seconds

            if current_state.get("status") != "in_progress" or is_stale:
                # Acquire lock
                state = {
                    "status": "in_progress",
                    "current_request_id": request_id,
                    "pending_request_id": None,
                }
                self._execute(
                    "UPDATE _operation_state SET operation_state = ?, updated_at = ? WHERE service_name = ?",
                    (_json_dumps(state), self._current_timestamp(), state_key),
                )
                return {"acquired": True, "state": state}

            # Lock is active — update pending
            current_state["pending_request_id"] = request_id
            self._execute(
                "UPDATE _operation_state SET operation_state = ?, updated_at = ? WHERE service_name = ?",
                (_json_dumps(current_state), self._current_timestamp(), state_key),
            )
            return {"acquired": False, "state": current_state}
