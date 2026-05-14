"""Singleton ``stall_state`` row — tracks whether reflexio's claude -p extractor
is currently stalled by a billing or auth error.

One row per database (``id`` is constrained to ``1``). The row always exists
after schema init; ``stalled=0`` is the clean default. See the credit-stall
notification spec for the state machine.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from reflexio.models.api_schema.stall_state_schema import StallReason

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StallState:
    """Snapshot of the singleton stall row.

    Attributes:
        stalled (bool): True when learning is currently paused.
        reason (StallReason | None): Why it's stalled. None when clean.
        stalled_at (datetime | None): When the stall was first observed.
        reset_estimate (datetime | None): Best-effort reset time parsed from
            the error text. ``None`` for auth errors (no reset semantics).
        notified_in_cc (bool): True once the SessionStart banner has fired
            for this stall event.
        error_message (str | None): Raw terminal-error text for debugging.
    """

    stalled: bool
    reason: StallReason | None
    stalled_at: datetime | None
    reset_estimate: datetime | None
    notified_in_cc: bool
    error_message: str | None


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS stall_state (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    stalled         BOOLEAN NOT NULL DEFAULT 0,
    reason          TEXT,
    stalled_at      TIMESTAMP,
    reset_estimate  TIMESTAMP,
    notified_in_cc  BOOLEAN NOT NULL DEFAULT 0,
    last_attempt_at TIMESTAMP,
    error_message   TEXT
);
"""

_SEED_SQL = "INSERT OR IGNORE INTO stall_state (id, stalled) VALUES (1, 0);"


def init_stall_state_table(conn: sqlite3.Connection) -> None:
    """Create the table and seed the singleton row. Idempotent.

    Args:
        conn (sqlite3.Connection): An open SQLite connection.
    """
    with conn:
        conn.execute(_CREATE_SQL)
        conn.execute(_SEED_SQL)


def get_stall_state(conn: sqlite3.Connection) -> StallState:
    """Return the current stall row.

    Args:
        conn (sqlite3.Connection): An open SQLite connection.

    Returns:
        StallState: Current snapshot of the singleton stall row.
    """
    row = conn.execute(
        "SELECT stalled, reason, stalled_at, reset_estimate, "
        "notified_in_cc, error_message FROM stall_state WHERE id = 1"
    ).fetchone()
    if row is None:
        return StallState(False, None, None, None, False, None)
    return StallState(
        stalled=bool(row[0]),
        reason=row[1],
        stalled_at=_parse_ts(row[2]),
        reset_estimate=_parse_ts(row[3]),
        notified_in_cc=bool(row[4]),
        error_message=row[5],
    )


def upsert_stall_state(
    conn: sqlite3.Connection,
    *,
    reason: StallReason,
    stalled_at: datetime,
    reset_estimate: datetime | None,
    error_message: str,
) -> None:
    """Mark the singleton as stalled with the given reason. Re-arms notified flag.

    Args:
        conn (sqlite3.Connection): An open SQLite connection.
        reason (StallReason): The stall reason discriminator (``billing_error``
            or ``auth_error``).
        stalled_at (datetime): When the stall was first detected.
        reset_estimate (datetime | None): Estimated reset time, if known.
        error_message (str): Raw error text for debugging.
    """
    with conn:
        conn.execute(
            "UPDATE stall_state SET stalled=1, reason=?, stalled_at=?, "
            "reset_estimate=?, notified_in_cc=0, last_attempt_at=?, "
            "error_message=? WHERE id=1",
            (
                reason,
                stalled_at.isoformat(),
                reset_estimate.isoformat() if reset_estimate else None,
                stalled_at.isoformat(),
                error_message,
            ),
        )


def mark_stall_notified(conn: sqlite3.Connection) -> None:
    """Set ``notified_in_cc=1`` for the current stall. No-op when clean.

    Args:
        conn (sqlite3.Connection): An open SQLite connection.
    """
    with conn:
        conn.execute(
            "UPDATE stall_state SET notified_in_cc=1 WHERE id=1 AND stalled=1"
        )


def clear_stall_state(conn: sqlite3.Connection) -> None:
    """Mark the singleton clean — clears all stall fields atomically.

    Args:
        conn (sqlite3.Connection): An open SQLite connection.
    """
    with conn:
        conn.execute(
            "UPDATE stall_state SET stalled=0, reason=NULL, stalled_at=NULL, "
            "reset_estimate=NULL, notified_in_cc=0, error_message=NULL "
            "WHERE id=1"
        )


def _parse_ts(raw: str | None) -> datetime | None:
    """Parse ISO-format TIMESTAMP, tolerating None and naive strings.

    Args:
        raw (str | None): Raw ISO datetime string from SQLite, or None.

    Returns:
        datetime | None: Parsed datetime, or None if input is None or invalid.
    """
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        _LOGGER.warning("Malformed timestamp in stall_state: %r", raw)
        return None


class SQLiteStallStateMixin:
    """SQLite-backed stall_state operations exposed as storage methods."""

    # Type hints for instance attributes provided by SQLiteStorageBase via MRO
    conn: Any

    def get_stall_state(self) -> StallState:
        """Return the current stall row.

        Returns:
            StallState: Current snapshot of the singleton stall row.
        """
        return get_stall_state(self.conn)

    def upsert_stall_state(
        self,
        *,
        reason: StallReason,
        stalled_at: datetime,
        reset_estimate: datetime | None,
        error_message: str,
    ) -> None:
        """Mark the singleton as stalled with the given reason. Re-arms notified flag.

        Args:
            reason (StallReason): The stall reason discriminator (``billing_error``
                or ``auth_error``).
            stalled_at (datetime): When the stall was first detected.
            reset_estimate (datetime | None): Estimated reset time, if known.
            error_message (str): Raw error text for debugging.
        """
        upsert_stall_state(
            self.conn,
            reason=reason,
            stalled_at=stalled_at,
            reset_estimate=reset_estimate,
            error_message=error_message,
        )

    def mark_stall_notified(self) -> None:
        """Set ``notified_in_cc=1`` for the current stall. No-op when clean."""
        mark_stall_notified(self.conn)

    def clear_stall_state(self) -> None:
        """Mark the singleton clean — clears all stall fields atomically."""
        clear_stall_state(self.conn)
