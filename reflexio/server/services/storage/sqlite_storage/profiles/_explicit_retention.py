"""SQLite explicit-retention jobs under the existing writer/commit owner."""

from __future__ import annotations

import sqlite3
from typing import Any

from reflexio.server.services.profile.learning.explicit_retention import (
    KEYS,
    ProfileExplicitRetentionMixin,
)

from .._base import _json_dumps, _json_loads
from .._lineage import _append_event_stmt

# Persisted expression uses only SQLite built-ins, so old releases/connections
# can continue writing this file without registering any Python function.
DEADLINE_INDEX = "idx_profiles_explicit_retention_due_v1"
_SENTINEL = "(-9e999)"


def deadline_projection_sql() -> str:
    """Conservative exact-integer candidate projection, never erase authority.

    Duplicate/escaped keys and numeric overflow go to an uncertainty bucket:
    SQLite JSON key selection/equality differs from Python's decoder. Invalid
    input must not make an old writer's INSERT/UPDATE fail index evaluation.
    """
    column = "custom_features"
    present = " + ".join(
        f"(json_type({column}, '$.{key}') IS NOT NULL)" for key in KEYS
    )
    repeated = " OR ".join(
        f'''length({column}) - length(replace({column}, '"{key}"', '')) > {len(key) + 2}'''
        for key in (*KEYS, "basis", "logical_deadline", "boundary")
    )
    branches = []
    for key in KEYS:
        path = f"$.{key}"
        value = f"json_extract({column}, '{path}.logical_deadline')"
        boundary = f"json_extract({column}, '{path}.boundary')"
        due = f"({value} + ({boundary} = 'through'))"
        branches.append(
            f"WHEN json_type({column}, '{path}') IS NOT NULL THEN CASE "
            f"WHEN json_type({column}, '{path}') != 'object' THEN {_SENTINEL} "
            f"WHEN json_extract({column}, '{path}.basis') IS NOT 'explicit_retention' THEN NULL "
            f"WHEN json_type({column}, '{path}.logical_deadline') = 'integer' "
            f"AND typeof({value}) = 'integer' "
            f"AND {boundary} IN ('stop_at', 'through') "
            f"AND typeof({due}) = 'integer' THEN {due} ELSE {_SENTINEL} END "
        )
    return (
        f"(CASE WHEN {column} IS NULL OR {column} = '' THEN NULL "
        f"WHEN typeof({column}) != 'text' THEN {_SENTINEL} "
        f"WHEN NOT json_valid({column}) THEN {_SENTINEL} "
        f"WHEN json_type({column}) = 'null' THEN NULL "
        f"WHEN json_type({column}) != 'object' THEN {_SENTINEL} "
        f"WHEN instr({column}, '\\u') > 0 THEN {_SENTINEL} "
        f"WHEN ({repeated}) THEN {_SENTINEL} "
        f"WHEN ({present}) > 1 THEN {_SENTINEL} " + "".join(branches) + "ELSE NULL END)"
    )


def install_deadline_index(conn: sqlite3.Connection) -> None:
    """Install an additive old-writer-compatible index through schema ownership."""
    expression = deadline_projection_sql()
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {DEADLINE_INDEX} "  # noqa: S608 -- static identifier/expression
        f"ON profiles ({expression}, profile_id) WHERE {expression} IS NOT NULL"
    )


class ExplicitRetentionMixin(ProfileExplicitRetentionMixin):
    conn: sqlite3.Connection
    org_id: str
    _lock: Any
    _delete_in_chunks: Any
    _fts_delete_profile: Any
    _vec_delete: Any
    _has_sqlite_vec: bool

    def _profile_deadline_project_id(self) -> str:
        # OSS SQLite is scoped to its org/database and has no project column.
        return ""

    def _profile_deadline_rows(
        self,
        *,
        user_id: str | None,
        after_profile_id: str,
        limit: int,
        profile_id: str | None = None,
        lock: bool = False,
        metadata_only: bool = False,
        candidate_mode: str | None = None,
        now: int | None = None,
        earliest: bool = False,
    ) -> list[dict[str, Any]]:
        del lock  # commit_scope owns BEGIN IMMEDIATE and the shared writer lock.
        conditions = ["profile_id > ?"]
        params: list[Any] = [after_profile_id]
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if profile_id is not None:
            conditions.append("profile_id = ?")
            params.append(profile_id)
        elif candidate_mode is not None:
            expression = deadline_projection_sql()
            conditions.append(f"{expression} IS NOT NULL")
            if candidate_mode == "uncertain":
                conditions.append(f"{expression} = {_SENTINEL}")
            elif candidate_mode == "regular":
                conditions.append(f"{expression} > {_SENTINEL}")
            elif candidate_mode != "all":
                raise ValueError("Unknown deadline candidate mode")
            if now is not None:
                conditions.append(f"{expression} <= ?")
                # Wide Python clocks retain their meaning after authoritative
                # Python filtering, without overflowing SQLite parameter binding.
                params.append(max(-(2**63), min(2**63 - 1, now)))
        else:
            conditions.append("custom_features IS NOT NULL")
        params.append(limit)
        columns = (
            "profile_id, user_id, custom_features"
            if metadata_only
            else "rowid, profile_id, user_id, content, last_modified_timestamp, "
            "expiration_timestamp, source_span, custom_features"
        )
        query = (
            "SELECT "
            + columns
            + " FROM profiles WHERE "
            + " AND ".join(conditions)
            + (
                f" ORDER BY {deadline_projection_sql()}, profile_id LIMIT ?"
                if earliest
                else " ORDER BY profile_id LIMIT ?"
            )
        )
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [
            dict(dict(row), custom_features=_json_loads(row["custom_features"]))
            for row in rows
        ]

    def _profile_deadline_update_features(
        self, row: dict[str, Any], features: dict[str, Any]
    ) -> None:
        cursor = self.conn.execute(
            "UPDATE profiles SET custom_features = ? WHERE profile_id = ? AND user_id = ?",
            (_json_dumps(features), row["profile_id"], row["user_id"]),
        )
        if cursor.rowcount != 1:
            raise ValueError("Locked deadline ancestor update did not affect one row")

    def _profile_deadline_delete(self, row: dict[str, Any]) -> int:
        identity = row["profile_id"]
        self._delete_in_chunks("profiles_fts", "profile_id", [identity])
        # Also order native deferred deletes after any upserts staged by this
        # publication. Otherwise an overdue newly-published row could leave a
        # dangling search entry when the outer commit flushes its index queue.
        self._fts_delete_profile(identity)
        if self._has_sqlite_vec:
            self._delete_in_chunks("profiles_vec", "rowid", [row["rowid"]])
            self._vec_delete("profiles_vec", row["rowid"])
        cursor = self.conn.execute(
            "DELETE FROM profiles WHERE profile_id = ? AND user_id = ?",
            (identity, row["user_id"]),
        )
        if cursor.rowcount == 1:
            _append_event_stmt(
                self.conn,
                org_id=self.org_id,
                entity_type="profile",
                entity_id=identity,
                op="hard_delete",
                prov="wasInvalidatedBy",
                source_ids=[],
                actor="system",
                request_id="explicit-retention-" + identity,
                reason="explicit_retention_deadline",
            )
        return cursor.rowcount
