"""Shared validation for the lifecycle filters on bulk by-id getters.

``include_inactive=True`` returns every owned row regardless of lifecycle
status, expiry, or approval — it is the historical-resolution mode used by the
retrieved-learning evaluator. Combining it with an explicit status filter is
contradictory: the filter would be silently discarded and the caller would get
back rows it asked to exclude. Fail loud instead.
"""

from __future__ import annotations

from reflexio.models.api_schema.domain import PlaybookStatus, Status
from reflexio.server.services.storage.error import StorageError


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
