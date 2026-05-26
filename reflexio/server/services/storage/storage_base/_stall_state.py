"""Stall state interface for storage backends.

Declares the methods every storage implementation MUST implement to expose
the singleton stall_state row used by the credit-stall notification flow.

The default implementations raise :class:`NotImplementedError`. Concrete
backends that support the stall_state feature (currently
:class:`SQLiteStorage`) override these methods via their own mixin. Backends
that don't support it (currently :class:`DiskStorage`) will surface a clear
``NotImplementedError`` at call time rather than ``AttributeError``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.server.services.storage.sqlite_storage._stall_state import (
        StallReason,
        StallState,
    )


class StallStateMixin:
    """Default no-op stall_state interface for storage backends.

    Subclasses that support the stall_state feature must override every method.
    """

    def get_stall_state(self) -> StallState:
        """Return the current stall row.

        Returns:
            StallState: Current snapshot of the singleton stall row.

        Raises:
            NotImplementedError: When the backend does not support stall_state.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support stall_state")

    def upsert_stall_state(
        self,
        *,
        reason: StallReason,
        stalled_at: datetime,
        reset_estimate: datetime | None,
        error_message: str,
    ) -> None:
        """Mark the singleton as stalled with the given reason.

        Args:
            reason (StallReason): The stall reason discriminator.
            stalled_at (datetime): When the stall was first detected.
            reset_estimate (datetime | None): Estimated reset time, if known.
            error_message (str): Raw error text for debugging.

        Raises:
            NotImplementedError: When the backend does not support stall_state.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support stall_state")

    def mark_stall_notified(self) -> None:
        """Set ``notified_in_cc=1`` for the current stall.

        Raises:
            NotImplementedError: When the backend does not support stall_state.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support stall_state")

    def clear_stall_state(self) -> None:
        """Mark the singleton clean — clears all stall fields atomically.

        Raises:
            NotImplementedError: When the backend does not support stall_state.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support stall_state")
