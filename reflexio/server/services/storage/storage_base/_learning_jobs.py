"""ABC, dataclass, and enum for the durable learning-job queue (Task 3)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

# Upper bound of the done-row retention window.
# Shared by all backends so the value cannot drift between implementations.
# Err toward "not done" until we are sure a done row would have been GC'd.
_ABSENCE_DONE_AFTER_SECONDS = 72 * 3600


class LearningJobStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    DONE = "done"
    FAILED = "failed"
    DEAD = "dead"


@dataclass(frozen=True)
class LearningJob:
    job_id: str
    org_id: str
    user_id: str
    job_type: str
    latest_request_id: str | None
    status: str
    attempts: int
    claim_token: str | None
    covers_through: (
        float | None
    )  # epoch seconds (converted from stored ISO/timestamptz)
    force_extraction: bool = False
    skip_aggregation: bool = False
    max_attempts: int = 3


class LearningJobStoreABC(ABC):
    """Abstract interface for durable learning-job queue operations.

    All methods are scoped to the storage instance's own org (``self.org_id``).
    ``enqueue_learning_job`` and ``complete_learning_job`` are safe to call
    inside a ``commit_scope`` — they issue plain SQL on the scope's connection
    without owning a separate BEGIN/COMMIT.
    """

    @abstractmethod
    def enqueue_learning_job(
        self,
        *,
        org_id: str,
        user_id: str,
        request_id: str,
        covers_through: float,
        job_type: str = "learning",
        force_extraction: bool = False,
        skip_aggregation: bool = False,
    ) -> str:
        """Coalescing upsert into the learning_jobs queue.

        If a pending job for ``(org_id, user_id, job_type)`` already exists,
        update its ``latest_request_id`` and keep the max ``covers_through``;
        otherwise insert a new ``pending`` row.  Returns the ``job_id`` of the
        pending row (existing or newly inserted).

        Safe to call inside a ``commit_scope`` — no own BEGIN/COMMIT issued.
        """
        raise NotImplementedError

    @abstractmethod
    def claim_learning_jobs(
        self,
        *,
        claimed_by: str,
        limit: int,
        lease_seconds: int,
    ) -> list[LearningJob]:
        """Atomically claim up to ``limit`` of this org's claimable jobs.

        A job is claimable when it is ``pending`` OR (``claimed`` AND
        ``claim_expires_at < now``).  Each claimed job receives a fresh
        ``claim_token``, ``claimed_by``, and ``claim_expires_at``.

        Postgres/Supabase: uses ``FOR UPDATE SKIP LOCKED`` via the
        ``claim_learning_jobs`` SQL function.  SQLite: ``BEGIN IMMEDIATE``.
        The claim predicate uses the DB's ``now()`` to avoid app/DB clock skew.
        """
        raise NotImplementedError

    @abstractmethod
    def heartbeat_learning_job(
        self,
        *,
        job_id: str,
        claim_token: str,
        lease_seconds: int,
    ) -> bool:
        """Extend the lease on a claimed job.

        Updates ``claim_expires_at = now + lease_seconds`` only when
        ``claim_token`` matches and status is still ``claimed``.

        Returns:
            True if the lease was extended (rowcount == 1), False if the token
            was superseded or the job is no longer claimed.
        """
        raise NotImplementedError

    @abstractmethod
    def complete_learning_job(
        self,
        *,
        job_id: str,
        claim_token: str,
    ) -> int:
        """Fenced transition to ``done``.

        Executes::

            UPDATE learning_jobs
               SET status='done', updated_at=now()
             WHERE job_id=:job_id AND claim_token=:claim_token AND status='claimed'

        Returns:
            rowcount — 0 if the token was superseded or the job is already
            done/dead/failed; 1 on success.  Callers MUST raise on 0 to roll
            back the enclosing ``commit_scope``.

        Safe to call inside a ``commit_scope`` — no own BEGIN/COMMIT issued.
        """
        raise NotImplementedError

    @abstractmethod
    def fail_learning_job(
        self,
        *,
        job_id: str,
        claim_token: str,
        dead: bool,
    ) -> None:
        """Fenced fail/dead transition.

        Fenced on ``claim_token``.  Increments ``attempts``.  Sets
        ``status='dead'`` when ``dead=True``, else ``status='failed'`` (and
        clears ``claim_token``/``claim_expires_at`` so the job is reclaimable).
        """
        raise NotImplementedError

    @abstractmethod
    def list_org_ids_with_pending_learning_jobs(self) -> list[str]:
        """Return distinct org_ids that have actionable learning jobs.

        Cross-org discovery query for the :class:`DurableLearningScheduler`:
        surfaces every org with at least one job that is ``pending``, ``failed``
        (reclaimable), or ``claimed`` with an expired lease (a stolen/abandoned
        claim). Terminal ``done``/``dead`` rows and live claims are excluded.

        Mirrors ``list_resumable_work_org_ids``: NOT scoped to ``self.org_id`` —
        on the shared global table (Postgres ``public.learning_jobs``, or a
        SQLite DB shared across orgs) it returns every org with work in one
        query so the scheduler need not enumerate all orgs.

        Returns:
            list[str]: Distinct org_ids with actionable work, ordered ascending.
        """
        raise NotImplementedError

    @abstractmethod
    def get_learning_status_for_request(
        self,
        *,
        user_id: str,
        request_created_at: float,
    ) -> str:
        """Coverage-based status for a single request (§3.6 rule).

        Returns one of ``"done" | "processing" | "pending" | "failed"``.

        Coverage rule (evaluated in priority order):

        1. Any ``done`` job with ``covers_through >= request_created_at``
           → ``"done"`` (correct even for zero-yield windows).
        2. Any ``claimed`` job (any ``covers_through``)
           → ``"processing"``.
        3. Any ``pending`` job for this user
           → ``"pending"``.
        4. Any ``failed`` (reclaimable) job
           → ``"pending"``.
        5. Any ``dead`` job with ``covers_through >= request_created_at``
           → ``"failed"``.
        6. No rows AND ``now - request_created_at >= 72 h`` (retention window)
           → ``"done"`` (done row would have been GC'd by now).
        7. No rows AND request is recent
           → ``"pending"`` (err toward not-done until retention window passes).
        """
        raise NotImplementedError


__all__ = [
    "_ABSENCE_DONE_AFTER_SECONDS",
    "LearningJob",
    "LearningJobStatus",
    "LearningJobStoreABC",
]
