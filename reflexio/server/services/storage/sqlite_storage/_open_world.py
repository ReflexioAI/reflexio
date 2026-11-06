"""Open-world tuner tables on SQLite, and the sweep-progress half of the port.

The open-world tuner reaches storage through ``OpenWorldStoragePort``. This
module holds the SQLite schema for the tables that port needs and the mixin
implementing its sweep-progress member, ``OpenWorldSweepProgressStore``:

* ``open_world_sweep_progress`` -- one row per playbook per UTC day, the
  record a sweep tick writes and the queue read drains.
* ``user_playbook_exposure_events`` -- the published exposure ledger. The
  queue's activity predicate reads it. Nothing on SQLite writes it yet.

The semantics mirror the Postgres functions ``record_open_world_sweep_progress``
and the two-argument ``load_open_world_playbooks_without_progress`` row for
row, minus ``project_id``: a SQLite database is one tenant and one project.
The admission constants come from ``open_world_records``. Postgres restates
them as SQL literals, so the shared contract test
(``reflexio.test_support.open_world_port_contract``) holds both backends to
the same answers.

One deliberate difference: each method here takes ONE clock reading for the
whole statement, where the Postgres queue read calls ``clock_timestamp()``
several times. Every result the contract can observe is the same.
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from reflexio.server.services.offline_tuner.open_world_records import (
    EVIDENCE_REATTEMPT_COOLDOWN_SECONDS,
    EVIDENCE_REATTEMPT_LOOKBACK_SECONDS,
    EVIDENCE_REATTEMPT_PREFIX,
    MAX_SAME_DAY_SWEEP_ATTEMPTS,
    NEVER_ATTEMPTED_EPOCH,
    RATE_LIMIT_RETRY_COOLDOWN_SECONDS,
    RETRY_ELIGIBLE_OUTCOME,
    SweepProgressRecord,
    SweepQueueEntry,
)
from reflexio.server.services.storage.error import StorageError

from ._base import SQLiteStorageBase

_SECONDS_PER_DAY = 86_400

#: The exposure-event id prefix the published-exposure path writes. Only these
#: rows count as activity, as in the Postgres read.
_PUBLISHED_EXPOSURE_PREFIX = "publish-v1:"

_OPEN_WORLD_DDL = """
CREATE TABLE IF NOT EXISTS open_world_sweep_progress (
    user_playbook_id INTEGER NOT NULL CHECK (user_playbook_id > 0),
    progress_day TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (trim(outcome) != ''),
    job_id INTEGER CHECK (job_id IS NULL OR job_id > 0),
    attempted_at INTEGER NOT NULL CHECK (attempted_at >= 0),
    attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count > 0),
    PRIMARY KEY (user_playbook_id, progress_day)
);
CREATE INDEX IF NOT EXISTS idx_open_world_sweep_progress_attempted_at
    ON open_world_sweep_progress(user_playbook_id, attempted_at DESC);

CREATE TABLE IF NOT EXISTS user_playbook_exposure_events (
    exposure_event_id TEXT PRIMARY KEY,
    request_id TEXT,
    session_id TEXT,
    user_id TEXT,
    playbook_owner_user_id TEXT,
    user_playbook_id INTEGER,
    served_semantic_digest TEXT,
    served_full_version_fingerprint TEXT,
    exposed_at INTEGER,
    ingested_at INTEGER NOT NULL,
    governance_subject_ref TEXT,
    playbook_owner_governance_subject_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_up_exposure_events_playbook_window
    ON user_playbook_exposure_events(
        user_playbook_id, COALESCE(exposed_at, ingested_at)
    );
CREATE INDEX IF NOT EXISTS idx_up_exposure_events_request_id
    ON user_playbook_exposure_events(request_id);
"""


def migrate_open_world(conn: sqlite3.Connection) -> None:
    """Create the open-world tables and indexes when absent. Idempotent.

    Args:
        conn (sqlite3.Connection): An open SQLite connection.
    """
    with conn:
        conn.executescript("BEGIN;\n" + _OPEN_WORLD_DDL + "\nCOMMIT;")


_RECORD_SQL = """
INSERT INTO open_world_sweep_progress AS progress (
    user_playbook_id, progress_day, outcome, job_id, attempted_at, attempt_count
) VALUES (:playbook_id, :today, :outcome, :job_id, :now, 1)
ON CONFLICT (user_playbook_id, progress_day) DO UPDATE
    SET outcome = excluded.outcome,
        job_id = COALESCE(excluded.job_id, progress.job_id),
        attempted_at = excluded.attempted_at,
        attempt_count = progress.attempt_count + 1
RETURNING user_playbook_id, progress_day, outcome, job_id, attempted_at
"""

# The window predicate on one exposure row, shared by the candidate filter and
# both re-admission arms exactly as the Postgres function repeats it.
_EXPOSURE_IN_WINDOW = """(
    substr({e}.exposure_event_id, 1, length(:published_prefix)) = :published_prefix
    AND COALESCE({e}.exposed_at, {e}.ingested_at) >= :day_start - :window_seconds
)"""

_QUEUE_SQL = f"""
WITH candidates AS (
    SELECT playbooks.user_playbook_id
    FROM user_playbooks AS playbooks
    WHERE playbooks.status IS NULL
      AND (
          :window_seconds IS NULL
          OR EXISTS (
              SELECT 1
              FROM user_playbook_exposure_events AS exposure
              WHERE exposure.user_playbook_id = playbooks.user_playbook_id
                AND {_EXPOSURE_IN_WINDOW.format(e="exposure")}
          )
      )
),
-- Today's ineligible rows that the re-attempt could still admit.
spent AS MATERIALIZED (
    SELECT progress.user_playbook_id, progress.attempted_at
    FROM open_world_sweep_progress AS progress
    WHERE progress.progress_day = :today
      AND substr(progress.outcome, 1, length(:evidence_prefix)) = :evidence_prefix
      AND progress.attempt_count < :attempt_cap
),
-- Sessions with activity since the earliest such attempt. A request counts as
-- of created_at + lookback: until then its session may still have been
-- waiting on the judge when the attempt ran.
active AS MATERIALIZED (
    SELECT activity.session_id, activity.user_id,
           max(activity.at_epoch) AS last_epoch
    FROM (
        SELECT later.session_id, later.user_id,
               unixepoch(later.created_at) + :lookback AS at_epoch
        FROM requests AS later
        WHERE unixepoch(later.created_at)
              > (SELECT min(spent.attempted_at) FROM spent) - :lookback
        UNION ALL
        SELECT outcome.session_id, outcome.user_id, outcome.created_at
        FROM session_outcomes AS outcome
        WHERE EXISTS (SELECT 1 FROM spent)
          AND outcome.created_at > (SELECT min(spent.attempted_at) FROM spent)
    ) AS activity
    GROUP BY activity.session_id, activity.user_id
),
-- Today's ineligible playbooks with an exposure inside the window placed on
-- an active session whose activity is newer than the attempt.
-- Request-carrying exposures are placed through the request's CURRENT
-- session, session-only ones through their own.
readmittable AS MATERIALIZED (
    SELECT spent.user_playbook_id
    FROM spent
    JOIN user_playbook_exposure_events AS exposure
      ON exposure.user_playbook_id = spent.user_playbook_id
    JOIN requests AS placed
      ON placed.request_id = exposure.request_id
    JOIN active
      ON active.session_id = placed.session_id
     AND active.user_id = placed.user_id
     AND active.last_epoch > spent.attempted_at
    WHERE :window_seconds IS NULL OR {_EXPOSURE_IN_WINDOW.format(e="exposure")}
    UNION
    SELECT spent.user_playbook_id
    FROM spent
    JOIN user_playbook_exposure_events AS exposure
      ON exposure.user_playbook_id = spent.user_playbook_id
     AND exposure.request_id IS NULL
    JOIN active
      ON active.session_id = exposure.session_id
     AND active.user_id = exposure.user_id
     AND active.last_epoch > spent.attempted_at
    WHERE :window_seconds IS NULL OR {_EXPOSURE_IN_WINDOW.format(e="exposure")}
)
SELECT candidates.user_playbook_id,
       (
           SELECT max(history.attempted_at)
           FROM open_world_sweep_progress AS history
           WHERE history.user_playbook_id = candidates.user_playbook_id
       ) AS last_attempted_at
FROM candidates
WHERE NOT EXISTS (
    SELECT 1
    FROM open_world_sweep_progress AS today
    WHERE today.user_playbook_id = candidates.user_playbook_id
      AND today.progress_day = :today
      AND NOT (
          today.outcome = :retry_outcome
          AND today.attempt_count < :attempt_cap
          AND today.attempted_at <= :now - :rate_limit_cooldown
      )
      AND NOT (
          substr(today.outcome, 1, length(:evidence_prefix)) = :evidence_prefix
          AND today.attempt_count < :attempt_cap
          AND today.attempted_at <= :now - :evidence_cooldown
          AND EXISTS (
              SELECT 1
              FROM readmittable
              WHERE readmittable.user_playbook_id = today.user_playbook_id
          )
      )
)
ORDER BY last_attempted_at ASC NULLS FIRST,
         candidates.user_playbook_id ASC
LIMIT :limit
"""


def _clock_reading(clock: Callable[[], float]) -> tuple[int, str]:
    """The epoch second and its UTC day, both from ONE reading of ``clock``.

    The day and the instant must agree, or a row written across 00:00 UTC
    carries yesterday's instant on today's day.
    """
    now = math.floor(clock())
    return now, datetime.fromtimestamp(now, tz=UTC).date().isoformat()


class SQLiteOpenWorldMixin:
    """``OpenWorldSweepProgressStore`` on SQLite."""

    conn: sqlite3.Connection
    _lock: Any
    _own_transaction: Any

    #: The clock both sweep-progress methods read, once per call. Tests replace
    #: it on an instance to place the call at a chosen instant.
    _sweep_progress_clock: Callable[[], float] = time.time

    @SQLiteStorageBase.handle_exceptions
    def record_open_world_sweep_progress(
        self, *, user_playbook_id: int, outcome: str, job_id: int | None
    ) -> SweepProgressRecord:
        """Record that a playbook had its sweep tick today, whatever the outcome.

        Args:
            user_playbook_id (int): The playbook the tick was spent on.
            outcome (str): Tick outcome label; stored with surrounding spaces
                trimmed.
            job_id (int | None): The optimization job, when the tick created
                one. A re-record without one keeps the earlier job.

        Returns:
            SweepProgressRecord: The row as stored, including today's UTC day.

        Raises:
            StorageError: ``user_playbook_id`` or ``job_id`` is not positive,
                or ``outcome`` is blank. Nothing is written.
        """
        # Spaces only, as Postgres ``btrim`` does: the two backends must agree
        # on which labels are blank and on the label they store.
        label = outcome.strip(" ")
        if user_playbook_id <= 0 or not label or (job_id is not None and job_id <= 0):
            raise StorageError(message="sweep progress identity is invalid")
        now, today = _clock_reading(self._sweep_progress_clock)
        with self._lock:
            # Drained before the commit: a RETURNING statement is still in
            # progress until its rows are read.
            (row,) = self.conn.execute(
                _RECORD_SQL,
                {
                    "playbook_id": int(user_playbook_id),
                    "today": today,
                    "outcome": label,
                    "job_id": job_id,
                    "now": now,
                },
            ).fetchall()
            # Inside a caller's commit_scope the scope owns the commit.
            if self._own_transaction():
                self.conn.commit()
        return SweepProgressRecord(
            user_playbook_id=int(row["user_playbook_id"]),
            progress_day=str(row["progress_day"]),
            outcome=str(row["outcome"]),
            job_id=None if row["job_id"] is None else int(row["job_id"]),
            attempted_at=int(row["attempted_at"]),
        )

    @SQLiteStorageBase.handle_exceptions
    def list_open_world_playbooks_without_progress(
        self, *, limit: int, window_seconds: int | None
    ) -> tuple[SweepQueueEntry, ...]:
        """Current playbooks whose day is not spent, oldest attempt first.

        Args:
            limit (int): Ceiling on rows returned; values below 1 read as 1.
            window_seconds (int | None): Admit only playbooks with a published
                exposure no older than this many seconds before the start of
                today (UTC). ``None`` admits every current playbook.

        Returns:
            tuple[SweepQueueEntry, ...]: Never-attempted playbooks first (with
            ``NEVER_ATTEMPTED_EPOCH``), then oldest last attempt first, ties by
            id.
        """
        now, today = _clock_reading(self._sweep_progress_clock)
        params = {
            "now": now,
            "today": today,
            "day_start": now - now % _SECONDS_PER_DAY,
            "window_seconds": None if window_seconds is None else int(window_seconds),
            "limit": max(int(limit), 1),
            "published_prefix": _PUBLISHED_EXPOSURE_PREFIX,
            "evidence_prefix": EVIDENCE_REATTEMPT_PREFIX,
            "retry_outcome": RETRY_ELIGIBLE_OUTCOME,
            "attempt_cap": MAX_SAME_DAY_SWEEP_ATTEMPTS,
            "rate_limit_cooldown": RATE_LIMIT_RETRY_COOLDOWN_SECONDS,
            "evidence_cooldown": EVIDENCE_REATTEMPT_COOLDOWN_SECONDS,
            "lookback": EVIDENCE_REATTEMPT_LOOKBACK_SECONDS,
        }
        with self._lock:
            rows = self.conn.execute(_QUEUE_SQL, params).fetchall()
        return tuple(
            SweepQueueEntry(
                playbook_id=int(row["user_playbook_id"]),
                last_attempt_epoch=(
                    NEVER_ATTEMPTED_EPOCH
                    if row["last_attempted_at"] is None
                    else int(row["last_attempted_at"])
                ),
            )
            for row in rows
        )
