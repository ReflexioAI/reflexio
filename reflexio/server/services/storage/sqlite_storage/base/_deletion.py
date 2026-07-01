"""SQLite retention-hook + chunked bulk-delete helpers.

Peeled verbatim from ``_base.py`` (Tier-1 storage decomposition). Composed into
``SQLiteStorage`` ahead of ``SQLiteStorageBase`` so the atomic
``_retention_perform_delete`` override wins over ``RetentionMixin``'s default.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from reflexio.server.services.storage.retention import RetentionTarget
from reflexio.server.services.storage.retention_mixin import (
    RETENTION_DELETE_CHUNK,
    chunked,
)

from .._base import SQLiteStorageBase


class SQLiteDeletionMixin:
    """Retention-hook impls + chunked bulk-delete helpers for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _fetchone: Any
    _fetchall: Any
    _has_sqlite_vec: bool

    # -- Retention hooks (see RetentionMixin) --

    @SQLiteStorageBase.handle_exceptions
    def _retention_table_exists(self, table_name: str) -> bool:
        row = self._fetchone(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        )
        return row is not None

    @SQLiteStorageBase.handle_exceptions
    def _retention_count_rows(self, target: RetentionTarget) -> int:
        row = self._fetchone(f"SELECT COUNT(*) as cnt FROM {target.table_name}")  # noqa: S608
        return int(row["cnt"]) if row else 0

    @SQLiteStorageBase.handle_exceptions
    def _retention_select_oldest_keys(
        self, target: RetentionTarget, count: int
    ) -> list[tuple[Any, ...]]:
        id_sql = ", ".join(target.id_columns)
        tiebreak_sql = id_sql
        rows = self._fetchall(
            f"SELECT {id_sql} FROM {target.table_name} "  # noqa: S608
            f"ORDER BY {target.order_column} ASC, {tiebreak_sql} ASC LIMIT ?",
            (count,),
        )
        return [tuple(row[col] for col in target.id_columns) for row in rows]

    @SQLiteStorageBase.handle_exceptions
    def _retention_perform_delete(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        # Wrap dependency + target deletes in a single critical section so
        # concurrent writers see either both or neither.
        with self._lock:
            try:
                self._retention_delete_dependencies(target, keys)
                self._retention_delete_target_rows(target, keys)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def _retention_delete_dependencies(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        ids = [key[0] for key in keys]
        target_name = target.name
        if target_name == "requests":
            self._delete_interactions_for_request_ids([str(v) for v in ids])
        elif target_name == "interactions":
            self._delete_interaction_search_rows([int(v) for v in ids])
        elif target_name == "profiles":
            self._delete_profile_search_rows([str(v) for v in ids])
        elif target_name == "user_playbooks":
            self._delete_source_windows_for_user_playbook_ids([int(v) for v in ids])
            self._delete_playbook_search_rows(
                "user", [int(v) for v in ids], commit=False
            )
        elif target_name == "agent_playbooks":
            self._delete_source_windows_for_agent_playbook_ids([int(v) for v in ids])
            self._delete_playbook_search_rows(
                "agent", [int(v) for v in ids], commit=False
            )
        elif target_name == "playbook_optimization_jobs":
            self._delete_optimizer_rows_for_job_ids([int(v) for v in ids])
        elif target_name == "playbook_optimization_candidates":
            self._delete_optimizer_evaluations_for_candidate_ids([int(v) for v in ids])

    def _retention_delete_target_rows(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        if len(target.id_columns) == 1:
            self._delete_in_chunks(
                target.table_name,
                target.id_columns[0],
                [key[0] for key in keys],
            )
            return
        # Composite-key delete: chunk by row to bound parameter count.
        params_per_key = len(target.id_columns)
        rows_per_chunk = max(1, RETENTION_DELETE_CHUNK // params_per_key)
        for chunk in chunked(keys, rows_per_chunk):
            where = " OR ".join(
                "("
                + " AND ".join(f"{column} = ?" for column in target.id_columns)
                + ")"
                for _ in chunk
            )
            params = [value for key in chunk for value in key]
            self.conn.execute(
                f"DELETE FROM {target.table_name} WHERE {where}",  # noqa: S608
                params,
            )

    # -- Chunked-delete primitives shared by the cascade helpers --

    def _delete_in_chunks(
        self, table_name: str, column_name: str, values: list[Any]
    ) -> None:
        """Chunked ``DELETE FROM table WHERE col IN (...)``.

        Chunking keeps parameter count under ``SQLITE_MAX_VARIABLE_NUMBER``
        on older sqlite builds (default 999) and avoids degenerate plans
        on very large IN lists.
        """
        if not values:
            return
        for chunk in chunked(values):
            placeholders = ",".join("?" for _ in chunk)
            self.conn.execute(
                f"DELETE FROM {table_name} WHERE {column_name} IN ({placeholders})",  # noqa: S608
                chunk,
            )

    def _select_in_chunks(self, sql_template: str, values: list[Any]) -> list[Any]:
        """Run ``sql_template`` (containing ``{placeholders}``) over chunks of
        ``values`` and aggregate the result rows."""
        results: list[Any] = []
        for chunk in chunked(values):
            placeholders = ",".join("?" for _ in chunk)
            stmt = sql_template.format(placeholders=placeholders)
            results.extend(self.conn.execute(stmt, chunk).fetchall())
        return results

    def _delete_interactions_for_request_ids(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        rows = self._select_in_chunks(
            "SELECT interaction_id FROM interactions WHERE request_id IN ({placeholders})",
            request_ids,
        )
        self._delete_interaction_search_rows(
            [int(row["interaction_id"]) for row in rows]
        )
        self._delete_in_chunks("interactions", "request_id", request_ids)

    def _delete_interaction_search_rows(self, interaction_ids: list[int]) -> None:
        """Remove fts + vec index rows for the given interaction IDs.

        Non-committing: participates in the caller's transaction.  Only called
        from inside the retention atomic block (_retention_perform_delete).
        """
        if not interaction_ids:
            return
        self._delete_in_chunks("interactions_fts", "rowid", interaction_ids)
        if self._has_sqlite_vec:
            self._delete_in_chunks("interactions_vec", "rowid", interaction_ids)

    def _delete_profile_search_rows(self, profile_ids: list[str]) -> None:
        """Remove fts + vec index rows for the given profile IDs.

        Non-committing: participates in the caller's transaction.  Only called
        from inside the retention atomic block (_retention_perform_delete).
        profiles_fts is keyed by profile_id (TEXT); profiles_vec by rowid (INT).
        """
        if not profile_ids:
            return
        self._delete_in_chunks("profiles_fts", "profile_id", profile_ids)
        if self._has_sqlite_vec:
            rows = self._select_in_chunks(
                "SELECT rowid FROM profiles WHERE profile_id IN ({placeholders})",
                profile_ids,
            )
            rowids = [row["rowid"] for row in rows]
            if rowids:
                self._delete_in_chunks("profiles_vec", "rowid", rowids)

    def _delete_playbook_search_rows(
        self, kind: str, ids: list[int], *, commit: bool = True
    ) -> None:
        """Remove fts + vec index rows for the given playbook IDs.

        Args:
            kind: ``"user"`` or ``"agent"``.
            ids: Playbook row IDs to remove from the search indexes.
            commit: When ``True`` (default) commits after the deletes so the
                after-commit callers in ``_playbook.py`` get a clean, durable
                cleanup.  Pass ``commit=False`` from inside the retention atomic
                block so the deletes participate in the single block-level commit
                (``_retention_perform_delete``).

        Note: callers may already hold ``self._lock`` when calling this (the
        ``commit=False`` retention/atomic-delete call sites do). The internal
        ``with self._lock:`` re-acquire is safe ONLY because ``self._lock`` is a
        reentrant ``threading.RLock``; a non-reentrant lock would deadlock here.
        """
        if not ids:
            return
        with self._lock:
            self._delete_in_chunks(f"{kind}_playbooks_fts", "rowid", ids)
            if self._has_sqlite_vec:
                self._delete_in_chunks(f"{kind}_playbooks_vec", "rowid", ids)
            if commit:
                self.conn.commit()

    def _delete_source_windows_for_agent_playbook_ids(
        self, agent_playbook_ids: list[int]
    ) -> None:
        self._delete_in_chunks(
            "agent_playbook_source_user_playbooks",
            "agent_playbook_id",
            agent_playbook_ids,
        )

    def _delete_source_windows_for_user_playbook_ids(
        self, user_playbook_ids: list[int]
    ) -> None:
        self._delete_in_chunks(
            "agent_playbook_source_user_playbooks",
            "user_playbook_id",
            user_playbook_ids,
        )

    def _delete_optimizer_rows_for_job_ids(self, job_ids: list[int]) -> None:
        if not job_ids:
            return
        for table in (
            "playbook_optimization_evaluations",
            "playbook_optimization_events",
            "playbook_optimization_candidates",
        ):
            self._delete_in_chunks(table, "job_id", job_ids)

    def _delete_optimizer_evaluations_for_candidate_ids(
        self, candidate_ids: list[int]
    ) -> None:
        self._delete_in_chunks(
            "playbook_optimization_evaluations",
            "candidate_id",
            candidate_ids,
        )
