"""Shared lifecycle-status sets and filter validation for every storage backend.

The status sets below are the one spelling of "which lifecycle statuses does
this read or cleanup treat as gone". SQLite binds them as parameters and the
Supabase backend renders them into PostgREST filters, so a change here moves
both backends together. The Postgres GC functions cannot import Python; their
literal status lists are held equal to ``GC_ELIGIBLE_STATUSES`` by an
enterprise drift guard (``test_lifecycle_status_sql_drift.py``).

The sets differ on purpose, per read:

- default list, count and by-id reads of user playbooks hide only merged and
  superseded rows (``TOMBSTONE_STATUS_VALUES``); archived rows stay visible;
- default user-playbook SEARCH returns current rows only (``status IS NULL``),
  on every backend and with or without a query;
- the agent-playbook by-id read also hides ``archive_in_progress``;
- the profile by-id read also hides TTL-expired profiles;
- GC and row retention treat archived and expired rows as dead too.

``include_inactive=True`` returns every owned row regardless of lifecycle
status, expiry, or approval — it is the historical-resolution mode used by the
retrieved-learning evaluator. Combining it with an explicit status filter is
contradictory: the filter would be silently discarded and the caller would get
back rows it asked to exclude. Fail loud instead.
"""

from __future__ import annotations

from reflexio.models.api_schema.domain import PlaybookStatus, Status
from reflexio.server.services.storage.error import StorageError

TOMBSTONE_STATUS_VALUES: tuple[str, ...] = (
    Status.MERGED.value,
    Status.SUPERSEDED.value,
)
AGENT_PLAYBOOK_DEFAULT_EXCLUDED_STATUSES: tuple[str, ...] = (
    *TOMBSTONE_STATUS_VALUES,
    Status.ARCHIVE_IN_PROGRESS.value,
)
PROFILE_TOMBSTONE_STATUS_VALUES: tuple[str, ...] = (
    *TOMBSTONE_STATUS_VALUES,
    Status.EXPIRED.value,
)
# Rows with these statuses may be hard-deleted by TTL GC and are trimmed first
# by row retention. Also the merge guard: a source already carrying one of these
# is skipped (no re-tombstone, no clock reset).
GC_ELIGIBLE_STATUSES: frozenset[str] = frozenset(
    {
        Status.MERGED.value,
        Status.SUPERSEDED.value,
        Status.ARCHIVED.value,
        Status.EXPIRED.value,
    }
)
# The lineage ``reason`` recorded on a GC hard-delete, on every backend.
GC_HARD_DELETE_REASON = "retention_gc"


def validate_include_inactive(
    *,
    include_inactive: bool,
    status_filter: list[Status | None] | None = None,
    playbook_status_filter: list[PlaybookStatus] | None = None,
) -> None:
    """Reject ``include_inactive=True`` combined with an explicit status filter.

    Raises ``StorageError`` rather than ``ValueError`` so the behavior is uniform
    across backends: both ``handle_exceptions`` decorators re-raise ``StorageError``
    untouched, whereas any other exception type is swallowed, wrapped, and (on
    Supabase) reported externally — turning a caller bug into noise.

    Args:
        include_inactive (bool): Whether the caller asked for all rows regardless
            of lifecycle status.
        status_filter (list[Status | None] | None): Explicit lifecycle statuses
            the caller asked for, if any.
        playbook_status_filter (list[PlaybookStatus] | None): Explicit approval
            statuses the caller asked for, if any.

    Raises:
        StorageError: If ``include_inactive`` is True and either filter is set.
    """
    if not include_inactive:
        return
    conflicting = [
        name
        for name, value in (
            ("status_filter", status_filter),
            ("playbook_status_filter", playbook_status_filter),
        )
        if value is not None
    ]
    if conflicting:
        raise StorageError(
            f"include_inactive=True cannot be combined with {' and '.join(conflicting)}"
            " — include_inactive ignores lifecycle filters. Pass one or the other."
        )
