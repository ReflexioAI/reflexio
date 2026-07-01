"""SQLite FTS5 + sqlite-vec index maintenance helpers.

Peeled verbatim from ``_base.py`` (Tier-1 storage decomposition). These helpers
are **self-committing**: each calls ``self.conn.commit()`` internally so the
index sidecar update is durable. They are sqlite-only — there is no
supabase/postgres counterpart. Composed into ``SQLiteStorage`` ahead of
``SQLiteStorageBase``; the boot-time ``_migrate_vec_tables`` (residual in
``_base.py``) resolves ``self._vec_upsert`` through the composed MRO.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


class SQLiteFtsVecMixin:
    """FTS5 + sqlite-vec index-maintenance helpers for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _fetchall: Any
    _has_sqlite_vec: bool

    # FTS helpers
    def _fts_upsert(self, table: str, rowid: int, **text_fields: str | None) -> None:
        """Insert or update an FTS row.  Deletes old entry first to avoid duplicates."""
        with self._lock:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            cols = list(text_fields.keys())
            vals = [text_fields[c] or "" for c in cols]
            placeholders = ",".join("?" for _ in cols)
            col_str = ",".join(cols)
            self.conn.execute(
                f"INSERT INTO {table}(rowid, {col_str}) VALUES (?, {placeholders})",
                [rowid, *vals],
            )
            self.conn.commit()

    def _fts_delete(self, table: str, rowid: int) -> None:
        with self._lock:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            self.conn.commit()

    def _fts_upsert_profile(self, profile_id: str, content: str) -> None:
        """FTS for profiles uses profile_id TEXT as key column."""
        with self._lock:
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?", (profile_id,)
            )
            self.conn.execute(
                "INSERT INTO profiles_fts(profile_id, content) VALUES (?, ?)",
                (profile_id, content),
            )
            self.conn.commit()

    def _fts_delete_profile(self, profile_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?", (profile_id,)
            )
            self.conn.commit()

    # Vec helpers (sqlite-vec)
    def _vec_upsert(self, table: str, rowid: int, embedding: list[float]) -> None:
        """Insert or update a vec table row. No-op when sqlite-vec is unavailable."""
        if not self._has_sqlite_vec:
            return
        with self._lock:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            self.conn.execute(
                f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",
                (rowid, json.dumps(embedding)),
            )
            self.conn.commit()

    def _vec_delete(self, table: str, rowid: int) -> None:
        """Delete a vec table row. No-op when sqlite-vec is unavailable."""
        if not self._has_sqlite_vec:
            return
        with self._lock:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            self.conn.commit()

    def _vec_knn_search(
        self,
        vec_table: str,
        main_table: str,
        query_embedding: list[float],
        match_count: int,
        conditions: list[str] | None = None,
        params: list[Any] | None = None,
    ) -> list[sqlite3.Row]:
        """Run a native KNN search via sqlite-vec and join back to the main table.

        Over-fetches from the KNN index (5x ``match_count``) so that post-filter
        WHERE conditions (org, user, status, etc.) don't silently reduce the
        result set below the requested count.

        Args:
            vec_table: Name of the vec0 virtual table.
            main_table: Name of the main data table.
            query_embedding: Query embedding vector.
            match_count: Number of results to return.
            conditions: Optional WHERE conditions for the main table.
            params: Parameters for the conditions.

        Returns:
            Up to ``match_count`` rows from the main table, ordered by vector
            distance (ascending).
        """
        knn_overfetch = match_count * 5
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""SELECT m.* FROM {main_table} m
                  JOIN (
                      SELECT rowid, distance FROM {vec_table}
                      WHERE embedding MATCH ?
                      ORDER BY distance
                      LIMIT ?
                  ) v ON m.rowid = v.rowid
                  WHERE {where_clause}
                  ORDER BY v.distance
                  LIMIT ?"""
        all_params = [
            json.dumps(query_embedding),
            knn_overfetch,
            *(params or []),
            match_count,
        ]
        return self._fetchall(sql, all_params)
