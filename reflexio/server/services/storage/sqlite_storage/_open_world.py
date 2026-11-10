"""Open-world tuner tables on SQLite, and the sweep-progress half of the port.

The open-world tuner reaches storage through ``OpenWorldStoragePort``. This
module holds the SQLite schema for the tables that port needs and the mixin
implementing its sweep-progress member, ``OpenWorldSweepProgressStore``:

* ``open_world_sweep_progress`` -- one row per playbook per UTC day, the
  record a sweep tick writes and the queue read drains.
* ``user_playbook_exposure_events`` -- the published exposure ledger. The
  queue's activity predicate reads it. Publishing an interaction writes it,
  inside the interaction's transaction (:meth:`SQLiteOpenWorldMixin.
  _record_published_exposures_locked`); clearing a user deletes the rows that
  name them as viewer or playbook owner. ``integrity_state`` and
  ``integrity_reasons`` are stored at write time from
  :func:`~reflexio.server.services.search_exposure.exposure_event_integrity`,
  the one definition; Postgres derives them in generated columns.

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

import json
import math
import sqlite3
import time
from collections.abc import Callable
from dataclasses import astuple
from datetime import UTC, datetime
from typing import Any

from reflexio.models.api_schema.domain import Interaction
from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
    ROW_BYTES,
)
from reflexio.server.services.offline_tuner.exposure_receipts import (
    PUBLISHED_EXPOSURE_PREFIX,
    build_published_exposure_events,
    published_playbook_id,
    receipt_project_id,
)
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
from reflexio.server.services.search_exposure import (
    UserPlaybookExposureEvent,
    exposure_event_integrity,
)
from reflexio.server.services.storage.error import StorageError

from ._base import SQLiteStorageBase
from ._open_world_publication import migrate_open_world_publication

_SECONDS_PER_DAY = 86_400


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

"""

# Only the activity predicate's (playbook, window) read and the request join
# are indexed; a single-tenant ledger needs no per-subject indexes.
_EXPOSURE_DDL = """
CREATE TABLE IF NOT EXISTS user_playbook_exposure_events (
    exposure_event_id TEXT PRIMARY KEY CHECK (trim(exposure_event_id, ' ') != ''),
    request_id TEXT,
    session_id TEXT,
    user_id TEXT,
    playbook_owner_user_id TEXT,
    user_playbook_id INTEGER,
    served_semantic_digest TEXT,
    served_full_version_fingerprint TEXT,
    exposed_at INTEGER,
    ingested_at INTEGER NOT NULL CHECK (ingested_at >= 0),
    governance_subject_ref TEXT,
    playbook_owner_governance_subject_ref TEXT,
    integrity_state TEXT NOT NULL,
    integrity_reasons TEXT NOT NULL,
    CHECK (
        (integrity_state = 'complete' AND integrity_reasons = '[]')
        OR (integrity_state = 'incomplete' AND integrity_reasons != '[]')
    )
);
CREATE INDEX IF NOT EXISTS idx_up_exposure_events_playbook_window
    ON user_playbook_exposure_events(
        user_playbook_id, COALESCE(exposed_at, ingested_at)
    );
CREATE INDEX IF NOT EXISTS idx_up_exposure_events_request_id
    ON user_playbook_exposure_events(request_id);
"""


def _hex64(column: str) -> str:
    """A CHECK fragment: ``column`` is 64 lowercase hex characters."""
    return f"(length({column}) = 64 AND {column} NOT GLOB '*[^0-9a-f]*')"


# The durable attempt state. Each child row cascades from its optimizer job.
# The qualification authority is a singleton with no job. The checks restate
# the PostgreSQL ones, except that no 'confirmation' role is admitted: only the
# online confirmation analyst writes that role, and SQLite has none.
_ATTEMPT_DDL = f"""
CREATE TABLE IF NOT EXISTS offline_tuner_open_world_qualification_authority (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    epoch INTEGER NOT NULL CHECK (epoch > 0),
    state TEXT NOT NULL,
    authority_digest TEXT NOT NULL CHECK {_hex64("authority_digest")},
    discovery_component_identity_digest TEXT,
    discovery_qualification_suite_digest TEXT,
    discovery_qualification_result_digest TEXT,
    held_out_component_identity_digest TEXT,
    held_out_qualification_suite_digest TEXT,
    held_out_qualification_result_digest TEXT,
    candidate_generator_identity_digest TEXT,
    candidate_generator_authorization_digest TEXT,
    unavailable_reason TEXT,
    updated_at INTEGER NOT NULL CHECK (updated_at >= 0),
    method_authority_json TEXT,
    CHECK (
        (state = 'active' AND unavailable_reason IS NULL
         AND {_hex64("discovery_component_identity_digest")}
         AND {_hex64("discovery_qualification_suite_digest")}
         AND {_hex64("discovery_qualification_result_digest")}
         AND {_hex64("candidate_generator_identity_digest")}
         AND {_hex64("candidate_generator_authorization_digest")}
         AND method_authority_json IS NOT NULL
         AND held_out_component_identity_digest IS NULL
         AND held_out_qualification_suite_digest IS NULL
         AND held_out_qualification_result_digest IS NULL)
        OR (state = 'unavailable' AND trim(unavailable_reason) != ''
         AND discovery_component_identity_digest IS NULL
         AND discovery_qualification_suite_digest IS NULL
         AND discovery_qualification_result_digest IS NULL
         AND held_out_component_identity_digest IS NULL
         AND held_out_qualification_suite_digest IS NULL
         AND held_out_qualification_result_digest IS NULL
         AND candidate_generator_identity_digest IS NULL
         AND candidate_generator_authorization_digest IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS offline_tuner_open_world_attempt_context (
    job_id INTEGER PRIMARY KEY
        REFERENCES playbook_optimization_jobs(job_id) ON DELETE CASCADE,
    bundle_digest TEXT NOT NULL CHECK {_hex64("bundle_digest")},
    outcome_contract_digest TEXT NOT NULL CHECK {_hex64("outcome_contract_digest")},
    manifest_json TEXT NOT NULL CHECK (json_type(manifest_json) = 'array'),
    discovery_component_identity_digest TEXT NOT NULL,
    discovery_qualification_suite_digest TEXT NOT NULL,
    discovery_qualification_result_digest TEXT NOT NULL,
    held_out_component_identity_digest TEXT,
    held_out_qualification_suite_digest TEXT,
    held_out_qualification_result_digest TEXT,
    candidate_generator_identity_digest TEXT NOT NULL,
    candidate_generator_authorization_digest TEXT NOT NULL,
    context_json TEXT NOT NULL,
    context_digest TEXT NOT NULL CHECK {_hex64("context_digest")},
    created_at INTEGER NOT NULL,
    qualification_authority_epoch INTEGER NOT NULL
        CHECK (qualification_authority_epoch > 0),
    qualification_authority_digest TEXT NOT NULL
        CHECK {_hex64("qualification_authority_digest")}
);

CREATE TABLE IF NOT EXISTS offline_tuner_open_world_invocations (
    analyst_input_identity TEXT PRIMARY KEY
        CHECK {_hex64("analyst_input_identity")},
    job_id INTEGER NOT NULL
        REFERENCES playbook_optimization_jobs(job_id) ON DELETE CASCADE,
    component_identity_digest TEXT NOT NULL
        CHECK {_hex64("component_identity_digest")},
    expected_provider TEXT NOT NULL,
    expected_model_revision TEXT NOT NULL,
    input_digest TEXT NOT NULL CHECK {_hex64("input_digest")},
    component_role TEXT NOT NULL,
    manifest_role TEXT,
    user_id TEXT,
    session_id TEXT,
    state TEXT NOT NULL DEFAULT 'prepared',
    output_json TEXT,
    output_digest TEXT,
    observed_provider TEXT,
    observed_model_revision TEXT,
    called_at INTEGER,
    latency_ms INTEGER,
    usage_json TEXT,
    prepared_at INTEGER NOT NULL,
    completed_at INTEGER,
    qualification_result_digest TEXT NOT NULL
        CHECK {_hex64("qualification_result_digest")},
    trajectory_digest TEXT,
    candidate_artifact_digest TEXT,
    candidate_analyst_input_identity TEXT,
    CHECK (
        (component_role IN ('discovery', 'candidate') AND manifest_role IS NULL
         AND user_id IS NULL AND session_id IS NULL
         AND trajectory_digest IS NULL AND candidate_artifact_digest IS NULL
         AND candidate_analyst_input_identity IS NULL)
        OR (component_role = 'discovery' AND manifest_role = 'discovery'
         AND user_id IS NOT NULL AND session_id IS NOT NULL
         AND {_hex64("trajectory_digest")}
         AND candidate_artifact_digest IS NULL
         AND candidate_analyst_input_identity IS NULL)
        OR (component_role = 'held_out'
         AND manifest_role IN ('held_out', 'safety_only')
         AND user_id IS NOT NULL AND session_id IS NOT NULL
         AND {_hex64("trajectory_digest")}
         AND {_hex64("candidate_artifact_digest")}
         AND {_hex64("candidate_analyst_input_identity")})
    ),
    CHECK (
        (state = 'prepared' AND output_json IS NULL AND output_digest IS NULL
         AND observed_provider IS NULL AND observed_model_revision IS NULL
         AND called_at IS NULL AND latency_ms IS NULL AND usage_json IS NULL
         AND completed_at IS NULL)
        OR (state = 'completed' AND output_json IS NOT NULL
         AND {_hex64("output_digest")}
         AND observed_provider IS NOT NULL AND observed_model_revision IS NOT NULL
         AND called_at IS NOT NULL AND latency_ms >= 0 AND completed_at IS NOT NULL)
    )
);
-- One row per question slot of a job: NULLs compare equal, as the PostgreSQL
-- index's NULLS NOT DISTINCT makes them.
CREATE UNIQUE INDEX IF NOT EXISTS uq_open_world_invocations_manifest_slot
    ON offline_tuner_open_world_invocations(
        job_id, component_role, COALESCE(manifest_role, ''),
        COALESCE(user_id, ''), COALESCE(session_id, '')
    );
CREATE INDEX IF NOT EXISTS idx_open_world_invocations_user
    ON offline_tuner_open_world_invocations(user_id);

CREATE TABLE IF NOT EXISTS offline_tuner_session_compactions (
    job_id INTEGER NOT NULL
        REFERENCES playbook_optimization_jobs(job_id) ON DELETE CASCADE,
    compaction_identity TEXT NOT NULL CHECK {_hex64("compaction_identity")},
    batches INTEGER NOT NULL CHECK (batches BETWEEN 1 AND 128),
    complete INTEGER NOT NULL DEFAULT 0 CHECK (complete IN (0, 1)),
    content_json TEXT NOT NULL
        CHECK (length(CAST(content_json AS BLOB)) <= {ROW_BYTES}),
    content_digest TEXT NOT NULL CHECK {_hex64("content_digest")},
    PRIMARY KEY (job_id, compaction_identity)
);
CREATE INDEX IF NOT EXISTS idx_session_compactions_reuse
    ON offline_tuner_session_compactions(compaction_identity, job_id DESC)
    WHERE complete = 1;
"""

#: Every stored column of an exposure row, in ``UserPlaybookExposureEvent``
#: field order, then the two integrity columns.
_EXPOSURE_COLUMNS = (
    "exposure_event_id",
    "request_id",
    "session_id",
    "user_id",
    "playbook_owner_user_id",
    "user_playbook_id",
    "served_semantic_digest",
    "served_full_version_fingerprint",
    "exposed_at",
    "ingested_at",
    "governance_subject_ref",
    "playbook_owner_governance_subject_ref",
    "integrity_state",
    "integrity_reasons",
)

_INSERT_EXPOSURE_SQL = (
    f"INSERT INTO user_playbook_exposure_events ({', '.join(_EXPOSURE_COLUMNS)})"
    f" VALUES ({', '.join('?' * len(_EXPOSURE_COLUMNS))})"
    " ON CONFLICT (exposure_event_id) DO NOTHING"
)

#: Columns an existing event must match for a re-record to be a no-op. The two
#: timestamps are exempt, as the Postgres writer exempts them when the publish
#: path asks it to reuse existing timestamps.
_EXPOSURE_IDENTITY_COLUMNS = tuple(
    name for name in _EXPOSURE_COLUMNS[:12] if name not in {"exposed_at", "ingested_at"}
)


def _exposure_row(event: UserPlaybookExposureEvent) -> tuple[object, ...]:
    state, reasons = exposure_event_integrity(event)
    return (*astuple(event), state, json.dumps(list(reasons)))


def migrate_open_world(conn: sqlite3.Connection) -> None:
    """Create the open-world tables and indexes when absent. Idempotent.

    Args:
        conn (sqlite3.Connection): An open SQLite connection.
    """
    with conn:
        conn.executescript("BEGIN;\n" + _OPEN_WORLD_DDL + "\nCOMMIT;")
    _rebuild_exposure_ledger_without_integrity(conn)
    with conn:
        conn.executescript("BEGIN;\n" + _EXPOSURE_DDL + "\nCOMMIT;")
    with conn:
        conn.executescript("BEGIN;\n" + _ATTEMPT_DDL + "\nCOMMIT;")
    migrate_open_world_publication(conn)


def _rebuild_exposure_ledger_without_integrity(conn: sqlite3.Connection) -> None:
    """Bring a ledger created before the integrity columns to the current shape.

    The table first shipped without ``integrity_state`` / ``integrity_reasons``
    (and without its CHECKs). SQLite cannot add a ``NOT NULL`` column without a
    default, so the table is rebuilt: rows are read, the old table and its
    indexes dropped, and every row re-inserted with its integrity computed by
    the one shared function. A no-op when the table is absent or current.
    """
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(user_playbook_exposure_events)")
    }
    if not columns or "integrity_state" in columns:
        return
    with conn:
        conn.execute("BEGIN")
        rows = conn.execute(
            f"SELECT {', '.join(_EXPOSURE_COLUMNS[:12])}"
            " FROM user_playbook_exposure_events"
        ).fetchall()
        conn.execute("DROP TABLE user_playbook_exposure_events")
        for statement in _EXPOSURE_DDL.split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.executemany(
            _INSERT_EXPOSURE_SQL,
            [_exposure_row(UserPlaybookExposureEvent(*tuple(row))) for row in rows],
        )


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
    """``OpenWorldSweepProgressStore`` on SQLite, and the published-exposure ledger."""

    conn: sqlite3.Connection
    org_id: str
    _lock: Any
    _own_transaction: Any
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any

    def _record_published_exposures_locked(self, interaction: Interaction) -> None:
        """Record a NEWLY inserted interaction's served user playbooks.

        Runs inside the interaction's own transaction, under the storage lock,
        after its row is written, so its exposure rows commit with it or not
        at all. Events come from :func:`build_published_exposure_events`;
        served digests only from a receipt that verifies.

        A reference this store cannot attribute records nothing and does not
        fail the publish: an id naming no playbook, a playbook without an
        owner, or an interaction whose request (or its session) is not
        stored. Postgres refuses the whole publish instead; the shared port
        contract records the difference (``refuses_unattributable_exposure``).

        Raises:
            SubjectWriteBarrierError: A playbook owner is under an erasure
                barrier; the interaction rolls back with it.
            StorageError: An event id is already stored with different content.
        """
        playbook_ids = sorted(
            {
                number
                for ref in interaction.retrieved_learnings
                if ref.kind == "user_playbook"
                and (number := published_playbook_id(ref.learning_id)) is not None
            }
        )
        if not playbook_ids:
            return
        request = self.conn.execute(
            "SELECT session_id FROM requests WHERE request_id = ? AND user_id = ?",
            (interaction.request_id, interaction.user_id),
        ).fetchone()
        if request is None or not request["session_id"]:
            return
        owners = {
            int(row["user_playbook_id"]): str(row["user_id"])
            for row in self.conn.execute(
                "SELECT user_playbook_id, user_id FROM user_playbooks"
                f" WHERE user_playbook_id IN ({', '.join('?' * len(playbook_ids))})",
                playbook_ids,
            ).fetchall()
            if row["user_id"]
        }
        for owner_ref in sorted(
            {self._subject_ref_for_user_id(owner) for owner in owners.values()}
        ):
            self._assert_subject_writable_locked(owner_ref)
        events = build_published_exposure_events(
            interaction,
            org_id=self.org_id,
            project_id=receipt_project_id(),
            session_id=str(request["session_id"]),
            owners=owners,
            subject_ref=self._subject_ref_for_user_id,
            ingested_at=int(time.time()),
        )
        for event in events:
            if self.conn.execute(_INSERT_EXPOSURE_SQL, _exposure_row(event)).rowcount:
                continue
            existing = self.conn.execute(
                f"SELECT {', '.join(_EXPOSURE_IDENTITY_COLUMNS)}"
                " FROM user_playbook_exposure_events WHERE exposure_event_id = ?",
                (event.exposure_event_id,),
            ).fetchone()
            if existing is None or tuple(existing) != tuple(
                getattr(event, name) for name in _EXPOSURE_IDENTITY_COLUMNS
            ):
                raise StorageError(
                    message="user playbook exposure event conflict: "
                    + event.exposure_event_id
                )

    def _delete_exposure_events_for_user_locked(self, user_id: str) -> int:
        """Delete every exposure naming ``user_id`` as viewer or playbook owner.

        Matches by user id and by governance subject ref, as the Postgres
        clear does. The caller owns the transaction.

        Returns:
            int: Rows deleted.
        """
        subject_ref = self._subject_ref_for_user_id(user_id)
        return self.conn.execute(
            "DELETE FROM user_playbook_exposure_events"
            " WHERE governance_subject_ref = ? OR user_id = ?"
            " OR playbook_owner_governance_subject_ref = ?"
            " OR playbook_owner_user_id = ?",
            (subject_ref, user_id, subject_ref, user_id),
        ).rowcount

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
            "published_prefix": PUBLISHED_EXPOSURE_PREFIX,
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
