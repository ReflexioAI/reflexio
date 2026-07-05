"""DurableLearningWorker: claim jobs from the durable queue, process in a fenced
commit_scope so exactly one of N racing workers commits its outputs.

Design constraints (v1 — do not remove):
- LLM calls are inside the commit_scope (compute/persist separation is deferred).
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
    (profiles, playbooks) made by run_deferred_learning for the superseded
    worker — guaranteeing exactly-once side effects.
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

        try:
            reflexio = get_reflexio(
                org_id=ctx.org_id, storage_base_dir=ctx.storage_base_dir
            )
            gen = GenerationService(llm_client=reflexio.llm_client, request_context=ctx)

            with storage.commit_scope():
                gen.run_deferred_learning(
                    user_id=job.user_id,
                    request_id=job.latest_request_id,
                    session_id=request.session_id,
                    source=request.source,
                    agent_version=request.agent_version,
                    force_extraction=job.force_extraction,
                    skip_aggregation=job.skip_aggregation,
                    sequential=True,  # prevent ThreadPoolExecutor deadlock on commit_scope RLock
                )
                rows = storage.complete_learning_job(
                    job_id=job.job_id, claim_token=claim_token
                )
                if rows == 0:
                    raise _SupersededError(job.job_id)
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
            return False
        except Exception:
            logger.exception(
                "event=learning_job_failed job_id=%s org_id=%s user_id=%s",
                job.job_id,
                job.org_id,
                job.user_id,
            )
            # attempts was incremented by claim_learning_jobs; go dead when we've
            # exhausted max_attempts total claim attempts.
            storage.fail_learning_job(
                job_id=job.job_id, claim_token=claim_token, dead=dead
            )
            return False
