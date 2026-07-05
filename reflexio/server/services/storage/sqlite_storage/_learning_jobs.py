"""SQLite implementation of the durable learning-job queue (Task 3)."""

import sqlite3
import time
import uuid
from typing import Any

# Matches the upper bound of the done-row retention window (24–72 h).
# Err toward "not done" until we are sure a done row would have been GC'd.
_ABSENCE_DONE_AFTER_SECONDS = 72 * 3600

from reflexio.server.services.storage.storage_base._learning_jobs import (
    LearningJob,
    LearningJobStoreABC,
)

from ._base import SQLiteStorageBase, _epoch_to_iso, _iso_to_epoch


def _row_to_learning_job(row: sqlite3.Row) -> LearningJob:
    """Convert a sqlite3.Row from learning_jobs to a LearningJob dataclass."""
    d = dict(row)
    ct = d.get("covers_through")
    return LearningJob(
        job_id=d["job_id"],
        org_id=d["org_id"],
        user_id=d["user_id"],
        job_type=d["job_type"],
        latest_request_id=d.get("latest_request_id"),
        status=d["status"],
        attempts=d["attempts"],
        claim_token=d.get("claim_token"),
        covers_through=float(_iso_to_epoch(ct)) if ct else None,
    )


class SQLiteLearningJobStoreMixin(LearningJobStoreABC):
    """SQLite implementation of the learning-job queue.

    Relies on instance attributes provided by SQLiteStorageBase via MRO:
    ``conn``, ``_lock``, ``_own_transaction``, ``org_id``.
    """

    # Type annotations for attributes/methods supplied by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    org_id: str
    _own_transaction: Any
    _fetchall: Any

    @SQLiteStorageBase.handle_exceptions
    def enqueue_learning_job(
        self,
        *,
        org_id: str,
        user_id: str,
        request_id: str,
        covers_through: float,
        job_type: str = "learning",
    ) -> str:
        """Coalescing upsert — safe to call inside a commit_scope."""
        job_id = str(uuid.uuid4())
        # int() truncates sub-second precision — intentional (second-precision epochs).
        iso_covers = _epoch_to_iso(int(covers_through))
        with self._lock:
            own_txn = self._own_transaction()
            try:
                if own_txn:
                    self.conn.execute("BEGIN IMMEDIATE")
                row = self.conn.execute(
                    """
                    INSERT INTO learning_jobs
                        (job_id, org_id, user_id, job_type, latest_request_id,
                         covers_through, status,
                         created_at, updated_at)
                    VALUES
                        (?, ?, ?, ?, ?,
                         ?, 'pending',
                         strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                         strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                    ON CONFLICT (org_id, user_id, job_type) WHERE status = 'pending'
                    DO UPDATE SET
                        latest_request_id = excluded.latest_request_id,
                        covers_through = CASE
                            WHEN learning_jobs.covers_through > excluded.covers_through
                            THEN learning_jobs.covers_through
                            ELSE excluded.covers_through
                        END,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    RETURNING job_id
                    """,
                    (job_id, org_id, user_id, job_type, request_id, iso_covers),
                ).fetchone()
                if own_txn:
                    self.conn.commit()
            except Exception:
                if own_txn:
                    self.conn.rollback()
                raise
        if row is None:
            raise RuntimeError("enqueue_learning_job RETURNING job_id returned no row")
        return str(row["job_id"])

    @SQLiteStorageBase.handle_exceptions
    def claim_learning_jobs(
        self,
        *,
        claimed_by: str,
        limit: int,
        lease_seconds: int,
    ) -> list[LearningJob]:
        """BEGIN IMMEDIATE + SELECT + UPDATE to atomically claim jobs."""
        with self._lock:
            own_txn = self._own_transaction()
            try:
                if own_txn:
                    self.conn.execute("BEGIN IMMEDIATE")

                # Find candidate job_ids using DB's now() to avoid clock skew
                candidate_rows = self.conn.execute(
                    """
                    SELECT job_id FROM learning_jobs
                    WHERE org_id = ?
                      AND (
                            status = 'pending'
                            OR (status = 'claimed'
                                AND claim_expires_at < strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                          )
                    ORDER BY created_at
                    LIMIT ?
                    """,
                    (self.org_id, limit),
                ).fetchall()

                claimed: list[LearningJob] = []
                for cand in candidate_rows:
                    job_id = cand["job_id"]
                    claim_token = str(uuid.uuid4())
                    updated = self.conn.execute(
                        """
                        UPDATE learning_jobs SET
                            status = 'claimed',
                            claimed_by = ?,
                            claim_token = ?,
                            claim_expires_at = strftime(
                                '%Y-%m-%dT%H:%M:%fZ', 'now',
                                ? || ' seconds'
                            ),
                            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                            attempts = attempts + 1
                        WHERE job_id = ?
                        RETURNING *
                        """,
                        (claimed_by, claim_token, str(lease_seconds), job_id),
                    ).fetchone()
                    if updated is not None:
                        claimed.append(_row_to_learning_job(updated))

                if own_txn:
                    self.conn.commit()
            except Exception:
                if own_txn:
                    self.conn.rollback()
                raise

        return claimed

    @SQLiteStorageBase.handle_exceptions
    def heartbeat_learning_job(
        self,
        *,
        job_id: str,
        claim_token: str,
        lease_seconds: int,
    ) -> bool:
        """Extend the lease; return True if the token is still live."""
        with self._lock:
            own_txn = self._own_transaction()
            try:
                if own_txn:
                    self.conn.execute("BEGIN IMMEDIATE")
                cur = self.conn.execute(
                    """
                    UPDATE learning_jobs SET
                        claim_expires_at = strftime(
                            '%Y-%m-%dT%H:%M:%fZ', 'now',
                            ? || ' seconds'
                        ),
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    WHERE job_id = ? AND claim_token = ? AND status = 'claimed'
                    """,
                    (str(lease_seconds), job_id, claim_token),
                )
                updated = cur.rowcount == 1
                if own_txn:
                    self.conn.commit()
            except Exception:
                if own_txn:
                    self.conn.rollback()
                raise

        return updated

    @SQLiteStorageBase.handle_exceptions
    def complete_learning_job(
        self,
        *,
        job_id: str,
        claim_token: str,
    ) -> int:
        """Fenced completion — returns rowcount (0=superseded, 1=success).

        Safe to call inside a commit_scope — no own BEGIN/COMMIT issued.
        """
        with self._lock:
            own_txn = self._own_transaction()
            try:
                if own_txn:
                    self.conn.execute("BEGIN IMMEDIATE")
                cur = self.conn.execute(
                    """
                    UPDATE learning_jobs SET
                        status = 'done',
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    WHERE job_id = ? AND claim_token = ? AND status = 'claimed'
                    """,
                    (job_id, claim_token),
                )
                rowcount = cur.rowcount
                if own_txn:
                    self.conn.commit()
            except Exception:
                if own_txn:
                    self.conn.rollback()
                raise

        return rowcount

    @SQLiteStorageBase.handle_exceptions
    def fail_learning_job(
        self,
        *,
        job_id: str,
        claim_token: str,
        dead: bool,
    ) -> None:
        """Fenced fail/dead transition — increments attempts, clears token for retry."""
        new_status = "dead" if dead else "failed"
        # Clear claim_token and claim_expires_at only for 'failed' so it's reclaimable.
        # For 'dead', we keep claim_token set for auditability (won't be reclaimed anyway).
        with self._lock:
            own_txn = self._own_transaction()
            try:
                if own_txn:
                    self.conn.execute("BEGIN IMMEDIATE")
                self.conn.execute(
                    """
                    UPDATE learning_jobs SET
                        status = ?,
                        attempts = attempts + 1,
                        claim_token = CASE WHEN ? THEN claim_token ELSE NULL END,
                        claim_expires_at = CASE WHEN ? THEN claim_expires_at ELSE NULL END,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    WHERE job_id = ? AND claim_token = ? AND status = 'claimed'
                    """,
                    (new_status, dead, dead, job_id, claim_token),
                )
                if own_txn:
                    self.conn.commit()
            except Exception:
                if own_txn:
                    self.conn.rollback()
                raise

    @SQLiteStorageBase.handle_exceptions
    def get_learning_status_for_request(
        self,
        *,
        user_id: str,
        request_created_at: float,
    ) -> str:
        """Coverage-based status lookup (§3.6 rule).

        Converts request_created_at epoch to ISO for lexicographic comparison
        with stored covers_through ISO strings (same format, both UTC).
        """
        req_iso = _epoch_to_iso(int(request_created_at))
        rows = self._fetchall(
            "SELECT status, covers_through FROM learning_jobs "
            "WHERE org_id = ? AND user_id = ?",
            (self.org_id, user_id),
        )

        has_pending = False
        has_claimed_covering = False
        has_dead_covering = False

        for row in rows:
            status = row["status"]
            ct: str | None = row["covers_through"]
            covers = ct is not None and ct >= req_iso

            if covers and status == "done":
                return "done"
            if covers and status == "claimed":
                has_claimed_covering = True
            if covers and status == "dead":
                has_dead_covering = True
            if status == "failed":
                # 'failed' is reclaimable (attempts < max_attempts); treat as pending.
                return "pending"
            if status == "pending":
                has_pending = True
            if status == "claimed":
                # Deliberately treats any claimed job as covering, regardless of its
                # covers_through value — it will extend the window once it completes.
                has_claimed_covering = True

        if has_claimed_covering:
            return "processing"
        if has_pending:
            return "pending"
        if has_dead_covering:
            return "failed"
        # Absence semantics: terminal rows (done/dead) are GC'd after 24–72 h.
        # Only treat absence as "done" once the request is old enough that a done
        # row would have been reaped; a recent request with no rows is still pending.
        if time.time() - request_created_at >= _ABSENCE_DONE_AFTER_SECONDS:
            return "done"
        return "pending"
