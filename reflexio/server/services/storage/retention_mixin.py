"""Shared scaffolding for SQL-backed row-retention cleanup.

Concrete SQL backends (SQLite, Postgres, Supabase) mix in
``RetentionMixin`` and implement a small set of hooks. The public surface
``count_retention_target_rows`` / ``delete_oldest_retention_target_rows``
lives here so the dispatch — limit lookup, key selection, cascade,
delete — cannot drift across the three backends.

Disk storage has a fundamentally different (file-based) implementation
and does not use this mixin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any

from reflexio.server.services.storage.retention import (
    RETENTION_TARGETS_BY_NAME,
    RetentionTarget,
)

# Conservative chunk size for IN-list deletes. Picked to stay well under:
#   - SQLite's SQLITE_MAX_VARIABLE_NUMBER (999 on builds before 3.32; 32766 after).
#   - PostgREST URL length limits (gateway caps are commonly 8-16 KB).
RETENTION_DELETE_CHUNK = 500


def chunked(
    values: Sequence[Any], chunk_size: int = RETENTION_DELETE_CHUNK
) -> Iterator[list[Any]]:
    """Yield consecutive ``chunk_size`` slices of ``values`` as lists.

    Args:
        values (Sequence[Any]): Items to split.
        chunk_size (int): Maximum number of items per chunk.

    Yields:
        list[Any]: Successive non-empty chunks.
    """
    for start in range(0, len(values), chunk_size):
        yield list(values[start : start + chunk_size])


def get_retention_target(target_name: str) -> RetentionTarget:
    """Resolve a retention target by name.

    Args:
        target_name (str): Registered name (e.g. ``"interactions"``).

    Returns:
        RetentionTarget: The matching target.

    Raises:
        ValueError: If ``target_name`` is not registered.
    """
    try:
        return RETENTION_TARGETS_BY_NAME[target_name]
    except KeyError as exc:
        raise ValueError(f"Unknown retention target: {target_name}") from exc


class RetentionMixin(ABC):
    """Backend-agnostic orchestration of row-retention cleanup.

    SQL backends mix this in and implement the abstract hooks. The public
    methods are intentionally defined once here so that the count/select/
    cascade/delete ordering is identical across all three backends.
    """

    def count_retention_target_rows(self, target_name: str) -> int:
        """Return the current row count for a retention target.

        Args:
            target_name (str): Registered retention target name.

        Returns:
            int: Row count, or 0 if the underlying table is missing.
        """
        target = get_retention_target(target_name)
        if not self._retention_table_exists(target.table_name):
            return 0
        return self._retention_count_rows(target)

    def delete_oldest_retention_target_rows(
        self, target_name: str, count: int
    ) -> int:
        """Delete up to ``count`` oldest rows for a retention target.

        Calls the dependency-cascade hook before the target-row delete so
        backends with foreign-key constraints stay consistent.

        Args:
            target_name (str): Registered retention target name.
            count (int): Maximum number of rows to delete.

        Returns:
            int: Number of rows actually selected for deletion.
        """
        if count <= 0:
            return 0
        target = get_retention_target(target_name)
        if not self._retention_table_exists(target.table_name):
            return 0
        keys = self._retention_select_oldest_keys(target, count)
        if not keys:
            return 0
        self._retention_perform_delete(target, keys)
        return len(keys)

    def _retention_perform_delete(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        """Run dependency cleanup then target-row delete.

        Default implementation runs the hooks in sequence. Backends that
        need both steps inside one transaction (notably SQLite) should
        override this method.
        """
        self._retention_delete_dependencies(target, keys)
        self._retention_delete_target_rows(target, keys)

    # -- Backend hooks --

    @abstractmethod
    def _retention_table_exists(self, table_name: str) -> bool:
        """Return whether ``table_name`` exists in the backing store."""
        raise NotImplementedError

    @abstractmethod
    def _retention_count_rows(self, target: RetentionTarget) -> int:
        """Return the live row count for ``target``'s table."""
        raise NotImplementedError

    @abstractmethod
    def _retention_select_oldest_keys(
        self, target: RetentionTarget, count: int
    ) -> list[tuple[Any, ...]]:
        """Return up to ``count`` oldest key tuples for ``target``.

        Each key tuple contains values aligned with ``target.id_columns``.
        Ordering is by ``target.order_column`` ascending with the id
        columns as a stable tiebreaker.
        """
        raise NotImplementedError

    @abstractmethod
    def _retention_delete_dependencies(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        """Remove rows in tables that depend on ``target`` for ``keys``."""
        raise NotImplementedError

    @abstractmethod
    def _retention_delete_target_rows(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> None:
        """Remove the rows for ``keys`` from ``target``'s table."""
        raise NotImplementedError
