"""Shared scaffolding for SQL-backed row-retention cleanup.

Concrete SQL backends (SQLite, Postgres, Supabase) mix in
``RetentionMixin`` and implement a small set of hooks. The public surface
``count_retention_target_rows`` / ``delete_oldest_retention_target_rows``
lives here so the dispatch — limit lookup, key selection, cascade,
delete — cannot drift across the three backends.

``probe_retention_targets`` is the sweep's read side: every target's size in
one call, so a backend whose reads are remote round trips can answer all of
them together. Its default -- ``probe_retention_targets_individually`` -- is
the per-target estimate-then-count the sweep always did, so a backend that
does not override the hook behaves exactly as before.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
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


@dataclass(frozen=True, slots=True)
class RetentionProbe:
    """One target's size, as the retention sweep needs it.

    Attributes:
        rows (int): The exact row count when ``exact``; otherwise an upper-bound
            ESTIMATE that was below the caller's exact-count threshold.
        exact (bool): Whether ``rows`` is an exact count. The sweep warns and
            deletes only on an exact count -- an estimate can only settle "far
            below the cap, nothing to do".
        seconds (float): Wall time spent probing this target alone, where it
            can be attributed (the per-target path). A batched probe leaves it 0.
        error (Exception | None): What probing this target raised, if anything.
            Recorded rather than raised so one bad table does not cost the rest;
            the sweep re-raises it inside that target's own isolation.
    """

    rows: int = 0
    exact: bool = False
    seconds: float = 0.0
    error: Exception | None = None


def probe_retention_targets_individually(
    storage: Any, exact_count_from: Mapping[str, float]
) -> dict[str, RetentionProbe]:
    """Probe each target on its own: the estimate first, then an exact count.

    The sweep's original per-target logic, unchanged: a target whose estimate is
    below its ``exact_count_from`` threshold is settled by the estimate; any
    other -- including one with no estimate -- is counted exactly. Each target's
    failure is isolated into its own :class:`RetentionProbe`.

    Takes ``Any`` rather than ``RetentionMixin`` because the sweep also falls
    back to it for a storage that never mixed the hook in (test doubles, a
    backend without retention), exactly as it used to call these methods.

    Args:
        storage (Any): Provides ``count_retention_target_rows`` and, optionally,
            ``estimate_retention_target_rows``.
        exact_count_from (Mapping[str, float]): Target name -> the estimate at
            or above which the target must be counted exactly.

    Returns:
        dict[str, RetentionProbe]: One entry per requested target.
    """
    estimate_rows = getattr(storage, "estimate_retention_target_rows", None)
    probes: dict[str, RetentionProbe] = {}
    for target_name, threshold in exact_count_from.items():
        started = time.monotonic()
        try:
            estimate = estimate_rows(target_name) if estimate_rows is not None else None
            if estimate is not None and estimate < threshold:
                probe = RetentionProbe(estimate, exact=False)
            else:
                probe = RetentionProbe(
                    storage.count_retention_target_rows(target_name), exact=True
                )
        except Exception as exc:  # noqa: BLE001 -- isolated per target, re-raised by the sweep
            probe = RetentionProbe(error=exc)
        probes[target_name] = replace(probe, seconds=time.monotonic() - started)
    return probes


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

    def estimate_retention_target_rows(self, target_name: str) -> int | None:
        """Return a cheap upper-bound estimate of a target's rows, or ``None``.

        The sweep uses it to skip the exact count for tables far below their
        cap (an exact ``count(*)`` per table per org was most of a slow sweep).
        It must never UNDER-estimate the rows the exact count would see, or a
        table over its cap would go untrimmed; ``None`` means "no estimate",
        and the sweep counts exactly.

        Args:
            target_name (str): Registered retention target name.

        Returns:
            int | None: An estimate no lower than the exact count, or ``None``.
        """
        return self._retention_estimate_rows(get_retention_target(target_name))

    def probe_retention_targets(
        self, exact_count_from: Mapping[str, float]
    ) -> dict[str, RetentionProbe]:
        """Size every requested target, for the retention sweep, in one call.

        For each target, either an exact count or -- only when the estimate is
        below that target's threshold -- the estimate, marked ``exact=False``.
        A target whose estimate is at or above its threshold, or that has no
        estimate, MUST come back exact: the sweep warns and deletes on exact
        counts only.

        Args:
            exact_count_from (Mapping[str, float]): Target name -> the estimate
                at or above which the target must be counted exactly.

        Returns:
            dict[str, RetentionProbe]: One entry per requested target, with a
            failure recorded in that target's ``error`` rather than raised.
        """
        return self._retention_probe_targets(exact_count_from)

    def delete_oldest_retention_target_rows(self, target_name: str, count: int) -> int:
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
        older_than_epoch = (
            int(time.time()) - target.minimum_age_seconds
            if target.minimum_age_seconds > 0
            else None
        )
        keys = self._retention_select_keys(
            target,
            count,
            older_than_epoch=older_than_epoch,
        )
        protect = getattr(self, "filter_extraction_retention", None)
        if protect is not None:
            keys = protect(target_name, keys)
        if not keys:
            return 0
        self._retention_perform_delete(target, keys)
        return len(keys)

    def gc_retired_optimization_jobs(
        self,
        *,
        older_than_epoch: int,
        stale_before_epoch: int,
        limit: int = 1000,
    ) -> int:
        """Apply fixed optimization terminal and staging retention ownership.

        Enterprise SQL backends override the protected hook. OSS backends have
        no provider/publication staging tables and therefore return zero.
        """
        if older_than_epoch < 0 or stale_before_epoch < 0:
            raise ValueError("optimization retention cutoffs must be non-negative")
        if limit <= 0:
            raise ValueError("optimization retention limit must be positive")
        return self._retention_gc_retired_optimization_jobs(
            older_than_epoch=older_than_epoch,
            stale_before_epoch=stale_before_epoch,
            limit=limit,
        )

    def _retention_gc_retired_optimization_jobs(
        self,
        *,
        older_than_epoch: int,
        stale_before_epoch: int,
        limit: int,
    ) -> int:
        del older_than_epoch, stale_before_epoch, limit
        return 0

    def _retention_select_keys(
        self,
        target: RetentionTarget,
        count: int,
        *,
        older_than_epoch: int | None,
    ) -> list[tuple[Any, ...]]:
        """Select tombstones first, then oldest rows when a target opts in.

        The tombstone pass can never under-delete: the fallback select returns
        ``min(count, table_rows)`` keys and ``seen`` only ever holds tombstones
        already counted, so the union still reaches ``count`` whenever the
        table holds that many rows.
        """
        if not target.priority_statuses:
            return self._retention_select_oldest_keys(
                target,
                count,
                older_than_epoch=older_than_epoch,
            )

        keys = self._retention_select_oldest_keys(
            target,
            count,
            statuses=target.priority_statuses,
            older_than_epoch=older_than_epoch,
        )
        if len(keys) >= count:
            return keys

        seen = set(keys)
        for key in self._retention_select_oldest_keys(
            target,
            count,
            older_than_epoch=older_than_epoch,
        ):
            if key not in seen:
                keys.append(key)
                seen.add(key)
            if len(keys) == count:
                break
        return keys

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

    def _retention_estimate_rows(self, target: RetentionTarget) -> int | None:
        """Backend hook for :meth:`estimate_retention_target_rows`.

        Default: no estimate. A backend overriding it must return an upper
        bound -- e.g. PostgreSQL's table-wide ``pg_class.reltuples``, which
        spans every project and so never undercounts a project-scoped count --
        and ``None`` when it has none (a table never analyzed).
        """
        del target
        return None

    def _retention_probe_targets(
        self, exact_count_from: Mapping[str, float]
    ) -> dict[str, RetentionProbe]:
        """Backend hook for :meth:`probe_retention_targets`.

        Default: one target at a time, through the public estimate and count --
        the right shape for a backend whose reads are local. A backend whose
        reads are remote round trips may answer the whole set at once, but must
        return the same decisions the default would: identical exact counts,
        and an estimate only where the default would have skipped on it.
        """
        return probe_retention_targets_individually(self, exact_count_from)

    @abstractmethod
    def _retention_select_oldest_keys(
        self,
        target: RetentionTarget,
        count: int,
        statuses: tuple[str, ...] | None = None,
        older_than_epoch: int | None = None,
    ) -> list[tuple[Any, ...]]:
        """Return up to ``count`` oldest key tuples for ``target``.

        Each key tuple contains values aligned with ``target.id_columns``.
        Ordering is by ``target.order_column`` ascending with the id
        columns as a stable tiebreaker.

        Args:
            target (RetentionTarget): Target whose keys are being selected.
            count (int): Maximum number of key tuples to return.
            statuses (tuple[str, ...] | None): When not None, restrict the
                select to rows whose ``status`` is one of these values. An
                empty tuple matches nothing (never every row).
            older_than_epoch (int | None): When set, restrict the select to
                rows whose target ordering column is strictly older.
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
