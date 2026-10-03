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
from datetime import UTC, datetime
from typing import Any

from reflexio.server.services.storage.retention import (
    RETENTION_CASCADES,
    RETENTION_TARGETS_BY_NAME,
    RetentionArchiver,
    RetentionTarget,
)

# Conservative chunk size for IN-list deletes. Picked to stay well under:
#   - SQLite's SQLITE_MAX_VARIABLE_NUMBER (999 on builds before 3.32; 32766 after).
#   - PostgREST URL length limits (gateway caps are commonly 8-16 KB).
RETENTION_DELETE_CHUNK = 500

#: The age pass examines at most this many rows per eligible row it may act on,
#: so held-back rows (protected, or with dependents) are stepped past without
#: making the walk unbounded.
AGE_EXAMINE_FACTOR = 10


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


@dataclass(frozen=True, slots=True)
class AgeRetentionResult:
    """What one target's age pass did.

    Attributes:
        eligible (int): Rows found older than the cutoff and deletable -- not
            protected by an unfinished extraction, with no dependent rows left.
            In a dry run this is what WOULD have been deleted.
        deleted (int): Rows archived and then deleted. Always 0 in a dry run.
        backlog (bool): The pass stopped on its row budget or deadline with
            more aged rows possibly left, rather than by running out of them.
        blocked (str | None): Why the pass stopped before finishing, if it did:
            ``"archive_failed"`` (the archiver declined; nothing in that batch
            was deleted) or ``"fetch_mismatch"`` (the rows read for the archive
            did not match the keys selected, so the batch was not deleted).
    """

    eligible: int = 0
    deleted: int = 0
    backlog: bool = False
    blocked: str | None = None


def retention_cutoff_value(target: RetentionTarget, epoch_seconds: int) -> int | str:
    """Express an age cutoff in the type of ``target.order_column``.

    Epoch columns compare against the integer. Every other ordering column is a
    timestamp -- ``timestamptz``/``timestamp`` on Postgres, ISO-8601 TEXT on
    SQLite -- and gets a UTC ISO string: Postgres casts the untyped literal to
    the column's type, and SQLite compares it as text, which is chronological
    to the second because every writer stores UTC ISO-8601.

    Args:
        target (RetentionTarget): Target whose ordering column is compared.
        epoch_seconds (int): The cutoff, in Unix seconds.

    Returns:
        int | str: The cutoff in the column's comparison type.
    """
    if target.order_column_epoch:
        return epoch_seconds
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat()


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
        older_than = (
            retention_cutoff_value(
                target, int(time.time()) - target.minimum_age_seconds
            )
            if target.minimum_age_seconds > 0
            else None
        )
        keys = self._retention_select_keys(
            target,
            count,
            older_than_epoch=older_than,
        )
        protect = getattr(self, "filter_extraction_retention", None)
        if protect is not None:
            keys = protect(target_name, keys)
        if not keys:
            return 0
        self._retention_perform_delete(target, keys)
        return len(keys)

    def expire_retention_target_rows(
        self,
        target_name: str,
        *,
        older_than_epoch: int,
        budget: int,
        batch_size: int,
        archiver: RetentionArchiver | None,
        deadline: float,
    ) -> AgeRetentionResult:
        """Archive, then delete, a target's rows older than a cutoff.

        Walks the target oldest-first by keyset -- ``(order_column, id)`` -- so a
        row that stays protected (an unfinished extraction window, a user whose
        extraction floor never advances) is stepped past instead of being
        selected again at the head of every batch, which would stall the pass.

        Per batch: select aged keys, drop the ones extraction still needs, drop
        the ones with dependent rows left (a request whose interactions have not
        aged out yet), read the full rows, hand them to ``archiver``, and delete
        ONLY after it returns True. A row is never deleted unless the archiver
        accepted it and it is older than the cutoff.

        Args:
            target_name (str): An ``age_retained`` target.
            older_than_epoch (int): Rows strictly older than this, in Unix
                seconds, are eligible.
            budget (int): Most eligible rows to act on this call. Rows that are
                held back (protected, or with dependents) do not spend it, so a
                long protected prefix cannot starve the rows behind it; the walk
                is instead bounded by ``AGE_EXAMINE_FACTOR * budget`` rows
                examined, and by ``deadline``.
            batch_size (int): Rows per select/archive/delete round.
            archiver (RetentionArchiver | None): ``None`` makes this a dry run:
                it counts what it would delete and deletes nothing.
            deadline (float): ``time.monotonic()`` value after which no further
                batch is started.

        Returns:
            AgeRetentionResult: Counts, and why the pass stopped early if it did.

        Raises:
            ValueError: If the target is not age-retained or has a composite key.
        """
        target = get_retention_target(target_name)
        if not target.age_retained or len(target.id_columns) != 1:
            raise ValueError(f"{target_name} is not a single-key age-retained target")
        if not self._retention_table_exists(target.table_name):
            return AgeRetentionResult()
        cutoff = retention_cutoff_value(target, older_than_epoch)
        id_column = target.id_columns[0]
        eligible = deleted = examined = 0
        max_examined = budget * AGE_EXAMINE_FACTOR
        after: tuple[Any, Any] | None = None
        while eligible < budget and examined < max_examined:
            if time.monotonic() >= deadline:
                return AgeRetentionResult(eligible, deleted, backlog=True)
            batch = self._retention_select_aged_keys(
                target,
                min(batch_size, budget - eligible, max_examined - examined),
                cutoff,
                after,
            )
            if not batch:
                return AgeRetentionResult(eligible, deleted)
            examined += len(batch)
            after = batch[-1]
            keys = self._retention_deletable_aged_keys(
                target, [(key,) for _, key in batch]
            )
            if not keys:
                continue
            eligible += len(keys)
            if archiver is None:
                continue
            ids = [key[0] for key in keys]
            rows = self._retention_fetch_rows(target.table_name, id_column, ids)
            if len(rows) != len(keys):
                return AgeRetentionResult(
                    eligible - len(keys), deleted, blocked="fetch_mismatch"
                )
            if not archiver(target.table_name, rows):
                return AgeRetentionResult(
                    eligible - len(keys), deleted, blocked="archive_failed"
                )
            deleted += self._retention_delete_aged_keys(target, keys)
        return AgeRetentionResult(eligible, deleted, backlog=True)

    def _retention_delete_aged_keys(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> int:
        """Delete archived aged keys; never cascade into a dependent row.

        The dependency check in :meth:`_retention_deletable_aged_keys` ran
        before the archive upload, and publishers do not take the cleanup
        lease -- so an interaction may have been added to a held request since.
        A target with cascades is therefore deleted by the guarded hook, which
        re-checks "no dependents" in the same statement as the delete, instead
        of by the cap path's unconditional cascade. Rows it skips stay in place
        (already archived; a later sweep retries them).

        Returns:
            int: Rows actually deleted.
        """
        if RETENTION_CASCADES.get(target.name):
            return self._retention_delete_childless_rows(target, keys)
        self._retention_perform_delete(target, keys)
        return len(keys)

    def _retention_deletable_aged_keys(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> list[tuple[Any, ...]]:
        """Drop keys extraction still needs, and keys with dependents left.

        A key with a dependent row (``RETENTION_CASCADES``) is kept back rather
        than cascaded: the cascade would delete that dependent without it ever
        being archived. It becomes deletable once its dependents age out first.
        """
        protect = getattr(self, "filter_extraction_retention", None)
        if protect is not None:
            keys = protect(target.name, keys)
        for cascade in RETENTION_CASCADES.get(target.name, ()):
            if not keys:
                break
            held = {
                row[cascade.fk_column]
                for row in self._retention_fetch_rows(
                    cascade.table_name,
                    cascade.fk_column,
                    [key[0] for key in keys],
                    columns=(cascade.fk_column,),
                )
            }
            keys = [key for key in keys if key[0] not in held]
        return keys

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
        older_than_epoch: int | str | None,
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
        older_than_epoch: int | str | None = None,
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
            older_than_epoch (int | str | None): When set, restrict the select
                to rows whose target ordering column is strictly older. Already
                in the column's type -- see :func:`retention_cutoff_value`.
        """
        raise NotImplementedError

    def _retention_select_aged_keys(
        self,
        target: RetentionTarget,
        count: int,
        older_than: int | str,
        after: tuple[Any, Any] | None,
    ) -> list[tuple[Any, Any]]:
        """Return up to ``count`` ``(order_value, id)`` pairs older than a cutoff.

        Single-key targets only. Ordered by ``(order_column, id)`` ascending and
        strictly after ``after`` -- the last pair of the previous batch -- so the
        caller pages by key rather than by offset.

        Args:
            target (RetentionTarget): Target being aged.
            count (int): Maximum pairs to return.
            older_than (int | str): Cutoff in the column's type.
            after (tuple[Any, Any] | None): Exclusive keyset lower bound.
        """
        raise NotImplementedError

    def _retention_delete_childless_rows(
        self, target: RetentionTarget, keys: list[tuple[Any, ...]]
    ) -> int:
        """Delete ``keys`` from a single-key target, skipping any that has a row
        in one of its ``RETENTION_CASCADES`` tables, in ONE atomic statement or
        transaction -- the existence check and the delete must not be separable.

        Returns:
            int: Rows actually deleted.
        """
        raise NotImplementedError

    def _retention_fetch_rows(
        self,
        table_name: str,
        column: str,
        values: list[Any],
        columns: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Read the rows of ``table_name`` whose ``column`` is in ``values``.

        Args:
            table_name (str): Table to read.
            column (str): Column matched against ``values``.
            values (list[Any]): Values to match.
            columns (tuple[str, ...] | None): DISTINCT values of these columns;
                ``None`` returns the whole row for archiving, minus derived
                search columns (embeddings, full-text vectors) that can be
                regenerated. Distinct, so a parent-key existence check reads one
                row per parent rather than one per child.

        Returns:
            list[dict[str, Any]]: One dict per matching row; ``[]`` when the
            table does not exist.
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
