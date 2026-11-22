"""SQLite FTS5 + sqlite-vec index maintenance helpers.

Peeled verbatim from ``_base.py`` (Tier-1 storage decomposition). These helpers
are **self-committing**: each calls ``self.conn.commit()`` internally so the
index sidecar update is durable. They are sqlite-only — there is no
supabase/postgres counterpart. Composed into ``SQLiteStorage`` ahead of
``SQLiteStorageBase``; the boot-time ``_migrate_vec_tables`` (residual in
``_base.py``) resolves ``self._vec_upsert`` through the composed MRO.

When a ``commit_scope`` is active (``_scope_depth > 0``), the public helpers
buffer the operation into ``_deferred_index_ops`` for post-commit flushing;
the ``_..._now`` variants always execute immediately and are called by
``_flush_index_op`` after the outer commit.
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
    _scope_depth: int
    _deferred_index_ops: list[tuple[str, Any]]

    # ------------------------------------------------------------------
    # Dispatcher — called by commit_scope after the outer commit
    # ------------------------------------------------------------------

    def _flush_index_op(self, kind: str, args: Any) -> None:
        """Dispatch a buffered index op after the outer commit."""
        if kind == "fts_upsert":
            table, rowid, text_fields = args
            self._fts_upsert_now(table, rowid, **text_fields)
        elif kind == "fts_delete":
            table, rowid = args
            self._fts_delete_now(table, rowid)
        elif kind == "fts_upsert_profile":
            profile_id, content = args
            self._fts_upsert_profile_now(profile_id, content)
        elif kind == "fts_delete_profile":
            (profile_id,) = args
            self._fts_delete_profile_now(profile_id)
        else:
            raise ValueError(f"unknown index op kind: {kind!r}")

    # ------------------------------------------------------------------
    # FTS helpers — public (defer when in scope) + _now (immediate)
    # ------------------------------------------------------------------

    def _fts_upsert(self, table: str, rowid: int, **text_fields: str | None) -> None:
        """Insert or update an FTS row.  Deletes old entry first to avoid duplicates."""
        if self._scope_depth > 0:
            self._deferred_index_ops.append(("fts_upsert", (table, rowid, text_fields)))
            return
        self._fts_upsert_now(table, rowid, **text_fields)

    def _fts_upsert_now(
        self, table: str, rowid: int, **text_fields: str | None
    ) -> None:
        with self._lock, self.conn:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))  # noqa: S608
            cols = list(text_fields.keys())
            vals = [text_fields[c] or "" for c in cols]
            placeholders = ",".join("?" for _ in cols)
            col_str = ",".join(cols)
            self.conn.execute(
                f"INSERT INTO {table}(rowid, {col_str}) VALUES (?, {placeholders})",  # noqa: S608
                [rowid, *vals],
            )
            self.conn.commit()

    def _fts_delete(self, table: str, rowid: int) -> None:
        if self._scope_depth > 0:
            self._deferred_index_ops.append(("fts_delete", (table, rowid)))
            return
        self._fts_delete_now(table, rowid)

    def _fts_delete_now(self, table: str, rowid: int) -> None:
        with self._lock, self.conn:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))  # noqa: S608
            self.conn.commit()

    def _fts_upsert_profile(self, profile_id: str, content: str) -> None:
        """FTS for profiles uses profile_id TEXT as key column."""
        if self._scope_depth > 0:
            self._deferred_index_ops.append(
                ("fts_upsert_profile", (profile_id, content))
            )
            return
        self._fts_upsert_profile_now(profile_id, content)

    def _fts_upsert_profile_now(self, profile_id: str, content: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?", (profile_id,)
            )
            self.conn.execute(
                "INSERT INTO profiles_fts(profile_id, content) VALUES (?, ?)",
                (profile_id, content),
            )
            self.conn.commit()

    def _fts_delete_profile(self, profile_id: str) -> None:
        if self._scope_depth > 0:
            self._deferred_index_ops.append(("fts_delete_profile", (profile_id,)))
            return
        self._fts_delete_profile_now(profile_id)

    def _fts_delete_profile_now(self, profile_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?", (profile_id,)
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Vec helpers (sqlite-vec) — public (defer when in scope) + _now (immediate)
    # ------------------------------------------------------------------

    def _vec_upsert(self, table: str, rowid: int, embedding: list[float]) -> None:
        """Insert or update a vec table row. No-op when sqlite-vec is unavailable."""
        if not self._has_sqlite_vec:
            return
        if self._scope_depth > 0:
            self._deferred_index_ops.append(("vec_upsert", (table, rowid, embedding)))
            return
        self._vec_upsert_now(table, rowid, embedding)

    def _vec_upsert_now(self, table: str, rowid: int, embedding: list[float]) -> None:
        if not self._has_sqlite_vec:
            return
        with self._lock, self.conn:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))  # noqa: S608
            self.conn.execute(
                f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)",  # noqa: S608
                (rowid, json.dumps(embedding)),
            )
            self.conn.commit()

    def _vec_delete(self, table: str, rowid: int) -> None:
        """Delete a vec table row. No-op when sqlite-vec is unavailable."""
        if not self._has_sqlite_vec:
            return
        if self._scope_depth > 0:
            self._deferred_index_ops.append(("vec_delete", (table, rowid)))
            return
        self._vec_delete_now(table, rowid)

    def _vec_delete_now(self, table: str, rowid: int) -> None:
        if not self._has_sqlite_vec:
            return
        with self._lock, self.conn:
            self.conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))  # noqa: S608
            self.conn.commit()
