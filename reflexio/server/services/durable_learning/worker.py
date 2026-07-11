"""DurableLearningWorker: claim jobs from the durable queue, split each into
compute (no writer transaction) -> persist (short fenced commit_scope) ->
post-commit side-effects so exactly one of N racing workers commits its outputs.

Design constraints (gate b — do not remove):
- LLM extraction + dedup + embeddings run in compute_deferred_learning, OUTSIDE
  the commit_scope. Only the fence-critical writes (profile/playbook rows +
  bookmark advances + complete_learning_job) run inside the scope; billing /
  telemetry / tagging / off-thread schedulers + the per-user lock release run
  AFTER the scope commits (emit_deferred_learning_side_effects). Holding the
  writer transaction across the ~30-120s LLM window is exactly what this split
  removes.
- No heartbeat thread (deferred; would deadlock a SQLite scope anyway).
- Exactly-once is guaranteed by the fenced complete_learning_job:
    rowcount == 0  -> lease was stolen -> _SupersededError -> rollback.
    rowcount == 1  -> we own the lease -> commit succeeds.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.cache.reflexio_cache import get_reflexio
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.services.storage.storage_base._learning_jobs import LearningJob

logger = logging.getLogger(__name__)


class _SupersededError(Exception):
    """Raised inside commit_scope when complete_learning_job returns 0.

    Propagating this exception causes commit_scope to roll back all writes
    (profiles, playbooks, bookmark advances) made by persist_deferred_learning
    for the superseded worker — guaranteeing exactly-once side effects. The
    post-commit emit is skipped, so a superseded job phantom-bills nothing.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(f"learning job {job_id} superseded (lease stolen)")


class DurableLearningWorker:
    """Claim durable learning jobs and process each in a fenced commit_scope.

    Two instances racing the same job produce profile/playbook side effects
    identical to a single run: the loser's complete_learning_job returns 0,
    which raises _SupersededError inside the scope, rolling back its writes.

    Args:
        request_context_factory: Callable that returns a RequestContext for a
            given org_id.  Called once per drain_org invocation.
        instance_id: Stable label written to claimed_by on each claim.
            Defaults to a random short UUID suffix.
    """

    def __init__(
        self,
        request_context_factory: Callable[[str], RequestContext],
        *,
        instance_id: str | None = None,
    ) -> None:
        self._factory = request_context_factory
        self._instance_id = instance_id or str(uuid.uuid4())[:8]

    def drain_org(self, org_id: str, batch_size: int, lease_seconds: int) -> int:
        """Claim up to batch_size of this org's jobs and process each.

        Args:
            org_id: Organisation to drain.
            batch_size: Maximum number of jobs to claim in one call.
            lease_seconds: Lease duration passed to claim_learning_jobs.

        Returns:
            Number of jobs successfully completed (not counting superseded or failed).
        """
        ctx = self._factory(org_id)
        storage = ctx.storage
        if storage is None:
            logger.error("event=learning_worker_no_storage org_id=%s", org_id)
            return 0
        jobs = storage.claim_learning_jobs(
            claimed_by=self._instance_id,
            limit=batch_size,
            lease_seconds=lease_seconds,
        )
        processed = 0
        for job in jobs:
            if self._process_job(ctx, job):
                processed += 1
        return processed

    def _process_job(self, ctx: RequestContext, job: LearningJob) -> bool:
        """Process a single claimed job.

        Args:
            ctx: RequestContext for the job's org.
            job: The claimed LearningJob (claim_token set).

        Returns:
            True if the job was successfully completed; False if superseded or failed.
        """
        storage = ctx.storage
        if storage is None:
            logger.error(
                "event=learning_job_no_storage job_id=%s org_id=%s",
                job.job_id,
                job.org_id,
            )
            return False

        # claim_learning_jobs always sets claim_token on a returned job.
        claim_token = job.claim_token
        if claim_token is None:
            logger.error(
                "event=learning_job_no_claim_token job_id=%s org_id=%s",
                job.job_id,
                job.org_id,
            )
            return False

        dead = job.attempts >= job.max_attempts

        # The job does not carry agent_version / session_id / source.
        # These must be read from the durably-persisted Request.
        if job.latest_request_id is None:
            logger.error(
                "event=learning_job_no_request_id job_id=%s org_id=%s user_id=%s",
                job.job_id,
                job.org_id,
                job.user_id,
            )
            storage.fail_learning_job(
                job_id=job.job_id, claim_token=claim_token, dead=dead
            )
            return False

        request = storage.get_request(job.latest_request_id)
        if request is None:
            logger.error(
                "event=learning_job_missing_request job_id=%s request_id=%s",
                job.job_id,
                job.latest_request_id,
            )
            storage.fail_learning_job(
                job_id=job.job_id, claim_token=claim_token, dead=dead
            )
            return False

        gen: GenerationService | None = None
        try:
            reflexio = get_reflexio(
                org_id=ctx.org_id, storage_base_dir=ctx.storage_base_dir
            )
            gen = GenerationService(llm_client=reflexio.llm_client, request_context=ctx)

            # COMPUTE — LLM extraction + dedup + embeddings, NO writer transaction
            # held. Acquires the per-user F4 lock; issues no learning DB write.
            plan = gen.compute_deferred_learning(
                user_id=job.user_id,
                request_id=job.latest_request_id,
                session_id=request.session_id,
                source=request.source,
                agent_version=request.agent_version,
                force_extraction=job.force_extraction,
                skip_aggregation=job.skip_aggregation,
            )

            # Same-user contention (F4): another same-user durable job holds the
            # per-user lock. Leave THIS job reclaimable (dead=False) — do NOT
            # complete it — so the queue re-claims it once the holder finishes.
            # REFUND the attempt (refund_attempt=True): claim_learning_jobs did
            # attempts += 1 on this claim, but no real work ran, and the ~2s poll
            # would otherwise re-claim (and re-increment) every couple of seconds
            # while the holder is in its ~60s compute — inflating attempts past
            # max_attempts in seconds so the eventual winner dead-letters on its
            # first transient error with zero real retries. The refund nets each
            # contention cycle (claim +1, release -1) to zero. No lock to release
            # (compute never acquired it).
            if not plan.lock_acquired:
                storage.fail_learning_job(
                    job_id=job.job_id,
                    claim_token=claim_token,
                    dead=False,
                    refund_attempt=True,
                )
                return False

            # PERSIST — short fenced scope: fence-critical writes + bookmark
            # advances only, then the claim-token fence. rows == 0 -> lease
            # stolen -> _SupersededError -> the scope rolls persist back.
            with storage.commit_scope():
                gen.persist_deferred_learning(plan)
                rows = storage.complete_learning_job(
                    job_id=job.job_id, claim_token=claim_token
                )
                if rows == 0:
                    raise _SupersededError(job.job_id)

            # POST-COMMIT — billing / telemetry / tagging / off-thread schedulers
            # + the per-user lock release, only for the winning worker.
            gen.emit_deferred_learning_side_effects(plan)
            logger.info(
                "event=learning_job_done job_id=%s org_id=%s user_id=%s",
                job.job_id,
                job.org_id,
                job.user_id,
            )
            return True
        except _SupersededError:
            logger.info(
                "event=learning_job_superseded job_id=%s org_id=%s",
                job.job_id,
                job.org_id,
            )
            # The persist rolled back and emit never ran, so the per-user lock is
            # still held by this compute — release it so the reclaim isn't blocked.
            self._release_user_lock(gen, job)
            return False
        except Exception:
            logger.exception(
                "event=learning_job_failed job_id=%s org_id=%s user_id=%s",
                job.job_id,
                job.org_id,
                job.user_id,
            )
            # emit (which releases the lock) never ran on this path — release the
            # per-user lock so a failed job doesn't strand it.
            self._release_user_lock(gen, job)
            # attempts was incremented by claim_learning_jobs; go dead when we've
            # exhausted max_attempts total claim attempts.
            storage.fail_learning_job(
                job_id=job.job_id, claim_token=claim_token, dead=dead
            )
            return False

    def _release_user_lock(
        self, gen: GenerationService | None, job: LearningJob
    ) -> None:
        """Best-effort release of the per-user F4 lock after a rolled-back/failed
        job, so a job whose emit (the normal release site) never ran does not
        strand the lock and block the same user's re-claim.

        Safe to call even when compute never acquired the lock: the release is a
        CAS on the holder (``clear_in_progress_lock_if_owner``) — a no-op unless
        this compute's ``request_id`` still owns it. Never raises.
        """
        if gen is None or job.latest_request_id is None:
            return
        try:
            gen._release_durable_learning_lock(
                user_id=job.user_id, request_id=job.latest_request_id
            )
        except Exception:
            logger.exception(
                "event=learning_job_lock_release_failed job_id=%s org_id=%s",
                job.job_id,
                job.org_id,
            )
