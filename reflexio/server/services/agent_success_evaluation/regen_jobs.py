"""Regenerate job registry + worker for replaying the LLM judge.

In-memory only; process-local. Survives the worker thread but not a
backend restart. v2 will move job state to storage.
"""

from __future__ import annotations

import logging
import random
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

from reflexio.models.api_schema.internal_schema import SessionDescriptor
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)
from reflexio.server.services.evaluation_overview.eval_sampler import (
    SampleCandidate,
    sample_candidates,
)
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)

JobStatus = Literal["running", "completed", "cancelled", "error"]
"""Lifecycle states for a regenerate job.

``"completed"`` means the worker loop finished iterating, regardless of whether
every session succeeded — per-session pass/fail counts are in the ``completed``
and ``failed`` counters. ``"error"`` means the worker itself crashed before or
during iteration (e.g. storage was unavailable). ``"cancelled"`` means a caller
set the cancel event and the worker observed it between sessions.
"""

DEFAULT_TTL_SECONDS = 3600
_FAILURE_CAP = 50


@dataclass
class JobFailure:
    session_id: str
    reason: str


@dataclass
class RegenJob:
    job_id: str
    org_id: str
    from_ts: int
    to_ts: int
    status: JobStatus
    total: int
    completed: int = 0
    failed: int = 0
    failures: list[JobFailure] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started_at: float = field(default_factory=time.time)
    """Unix seconds (wall clock) at job creation — returned to API clients."""
    finished_at: float | None = None
    """Unix seconds (wall clock) when the worker exited; ``None`` while running."""
    total_candidates: int = 0
    """Total candidate sessions discovered in the (from_ts, to_ts) window before sampling."""
    sampled_count: int = 0
    """Number of sessions actually sampled from the candidate pool for this job."""
    concurrency_limit: int = 0
    """Worker concurrency cap applied to this job (0 = unset/sequential)."""


class RegenJobRegistry:
    """Process-local job registry. One active singleton job per org."""

    def __init__(self) -> None:
        self._jobs: dict[str, RegenJob] = {}
        self._by_org: dict[str, str] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        org_id: str,
        from_ts: int,
        to_ts: int,
        total: int,
    ) -> RegenJob:
        """Register a new running job. Raises RuntimeError when an actively-running
        job exists for the org.

        Only *running* jobs block; a previous completed/cancelled/errored job for
        the same key is replaced. Eviction by TTL is a separate cleanup concern —
        we don't want users to wait an hour after a completed run before they can
        regenerate again.
        """
        with self._lock:
            self._evict_completed_locked(DEFAULT_TTL_SECONDS)
            key = org_id
            active = self._active_job_for_locked(key)
            if active is not None:
                raise RuntimeError("A regenerate is already running for this org")
            job = RegenJob(
                job_id=uuid.uuid4().hex,
                org_id=org_id,
                from_ts=from_ts,
                to_ts=to_ts,
                status="running",
                total=total,
            )
            self._jobs[job.job_id] = job
            self._by_org[key] = job.job_id
            return job

    def get(self, job_id: str) -> RegenJob | None:
        with self._lock:
            self._evict_completed_locked(DEFAULT_TTL_SECONDS)
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            job.cancel_event.set()

    def has_active(self, org_id: str) -> bool:
        with self._lock:
            self._evict_completed_locked(DEFAULT_TTL_SECONDS)
            return self._active_job_for_locked(org_id) is not None

    def _active_job_for_locked(self, key: str) -> RegenJob | None:
        """Return the running job for an org, or None when no live job exists.

        A registry entry whose job has already finished
        (``completed`` / ``cancelled`` / ``error``) is treated as having no active
        job — the previous run finished and the user can start a new one. The
        registry still holds the finished job until TTL eviction so status polls
        keep working, but it doesn't block new submissions.
        """
        job_id = self._by_org.get(key)
        if job_id is None:
            return None
        job = self._jobs.get(job_id)
        if job is None or job.status != "running":
            return None
        return job

    def evict_completed_older_than(self, ttl_seconds: int) -> None:
        with self._lock:
            self._evict_completed_locked(ttl_seconds)

    def _evict_completed_locked(self, ttl_seconds: int) -> None:
        now = time.time()
        drop = [
            jid
            for jid, j in self._jobs.items()
            if j.status in ("completed", "cancelled", "error")
            and j.finished_at is not None
            and (now - j.finished_at) > ttl_seconds
        ]
        for jid in drop:
            j = self._jobs.pop(jid)
            self._by_org.pop(j.org_id, None)


# Single process-local instance.
REGEN_JOBS = RegenJobRegistry()


def _load_first_request(
    storage: BaseStorage,
    user_id: str,
    session_id: str,
    cache: dict[str, tuple[int, dict[str, Any]]],
) -> tuple[int, dict[str, Any]]:
    """Return the (created_at, metadata) of a session's earliest request, memoized.

    The regen worker can see multiple SessionDescriptors for the same
    session_id (one per distinct agent_version/source tuple). This
    helper guarantees each session_id triggers at most one storage call,
    mirroring the amortized pattern F2's evaluation overview service
    uses.

    Args:
        storage (BaseStorage): Storage backend bound to the request_context.
        user_id (str): Owner of the session's requests.
        session_id (str): Session whose first-request data to return.
        cache (dict[str, tuple[int, dict[str, Any]]]): Memoization dict
            keyed by session_id; updated in place.

    Returns:
        tuple[int, dict[str, Any]]: ``(first_created_at, first_metadata)``.
        Falls back to ``(0, {})`` when the session has no requests (the
        descriptor is still kept so the regen worker can attempt
        evaluation; the sampler simply treats it as the epoch-zero day
        bucket and the untagged group).
    """
    cached = cache.get(session_id)
    if cached is not None:
        return cached
    try:
        requests = storage.get_requests_by_session(user_id, session_id)
    except Exception as e:  # noqa: BLE001 — per-session resilience boundary
        logger.warning(
            "Failed to fetch requests for session %s during F3 sampler candidate "
            "discovery (%s); falling back to (created_at=0, metadata={}). The "
            "session will be retried in the per-session loop below.",
            session_id,
            e,
        )
        cache[session_id] = (0, {})
        return cache[session_id]
    if not requests:
        logger.warning(
            "Session %s has no requests in storage despite being returned by "
            "get_session_ids_in_window (possible race or contract violation); "
            "falling back to (created_at=0, metadata={}).",
            session_id,
        )
        cache[session_id] = (0, {})
        return cache[session_id]
    first = min(requests, key=lambda r: r.created_at)
    cache[session_id] = (first.created_at, first.metadata or {})
    return cache[session_id]


def _build_sample_candidates(
    storage: BaseStorage,
    descriptors: list[SessionDescriptor],
) -> list[SampleCandidate]:
    """Convert raw SessionDescriptors into SampleCandidates with metadata.

    Reads the first-request data for each distinct session_id at most
    once via ``_load_first_request``'s cache, then assembles one
    ``SampleCandidate`` per descriptor. ``created_at`` is sourced from
    the first request's timestamp so the day-bucket stratum aligns with
    the session's wall clock (not the regen window's edges).

    Args:
        storage (BaseStorage): Storage backend used to load per-session data.
        descriptors (list[SessionDescriptor]): Raw descriptors emitted by
            ``storage.get_session_ids_in_window``.

    Returns:
        list[SampleCandidate]: One candidate per descriptor, ready for the
        pure ``sample_candidates`` function.
    """
    cache: dict[str, tuple[int, dict[str, Any]]] = {}
    candidates: list[SampleCandidate] = []
    for sd in descriptors:
        created_at, metadata = _load_first_request(
            storage, sd.user_id, sd.session_id, cache
        )
        candidates.append(
            SampleCandidate(
                session_id=sd.session_id,
                user_id=sd.user_id,
                agent_version=sd.agent_version,
                source=sd.source,
                created_at=created_at,
                first_request_metadata=metadata,
            )
        )
    return candidates


def _dispatch_one(
    sc: SampleCandidate,
    *,
    job: RegenJob,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
) -> None:
    """Run the LLM judge for one sampled candidate.

    Honors ``job.cancel_event`` as an early-exit by raising a sentinel
    ``_CancelledError`` so the dispatcher can distinguish cancellations from
    real failures.

    Args:
        sc (SampleCandidate): Session to grade.
        job (RegenJob): Owning job; the cancel event is read here so
            workers already dequeued from the pool can bail out fast
            instead of running the full judge after cancellation.
        request_context (RequestContext): Storage/config/prompt carrier
            forwarded to ``run_group_evaluation``.
        llm_client (LiteLLMClient): Shared LLM client forwarded to
            ``run_group_evaluation``.

    Raises:
        _CancelledError: When the cancel event is set before this worker
            starts its judge call.
        Exception: Whatever ``run_group_evaluation`` itself raised.
    """
    if job.cancel_event.is_set():
        raise _CancelledError
    run_group_evaluation(
        org_id=job.org_id,
        user_id=sc.user_id,
        session_id=sc.session_id,
        agent_version=sc.agent_version,
        source=sc.source,
        request_context=request_context,
        llm_client=llm_client,
        force_regenerate=True,
    )


class _CancelledError(Exception):
    """Sentinel raised by a worker that observed ``job.cancel_event``.

    Distinguishes a clean cancellation (must not count as a failure)
    from any other exception ``run_group_evaluation`` might raise.
    """


def run_regen(
    *,
    job: RegenJob,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
    rng: random.Random | None = None,
) -> None:
    """Worker body. Samples candidates per stratum, then drives ``run_group_evaluation``.

    Step 1 enumerates all candidate sessions in ``[from_ts, to_ts]`` via
    ``storage.get_session_ids_in_window``. Step 2 stratifies them by
    (day-bucket x F2 group) and samples up to
    ``config.eval_sample_n_per_stratum`` per stratum — giving the regen
    pipeline predictable cost regardless of traffic volume. Step 3
    dispatches the sampled subset through a ``ThreadPoolExecutor`` whose
    ``max_workers`` is bound by ``config.eval_concurrency_limit``,
    capping LLM provider rate-limit pressure.

    On normal completion of the sampled work, sets
    ``job.status = "completed"`` even if individual sessions raised —
    per-session failures are recorded in ``job.failed`` and
    ``job.failures`` (capped at ``_FAILURE_CAP`` to bound memory). Sets
    ``job.status = "error"`` only if the worker itself crashes (e.g.
    storage misconfigured).

    Cancellation semantics:
      - ``job.cancel_event`` may be set at any time during the worker run.
      - Futures already executing ``run_group_evaluation`` finish naturally
        (Python threads cannot be safely interrupted mid-LLM-call).
      - Futures the pool dequeues AFTER the event is set short-circuit at
        the dispatch boundary by raising a private ``_CancelledError``,
        which is silently dropped — not counted as a success or failure.
      - The post-loop ``job.cancel_event.is_set()`` check is authoritative:
        even if every submitted future happened to complete before the
        user clicked cancel, the job is reported as ``"cancelled"``.
      - Consequence: when the job is cancelled,
        ``job.completed + job.failed`` may be strictly less than
        ``job.sampled_count``. Consumers reading
        ``RegenerateStatusResponse`` must accept this invariant.

    Args:
        job (RegenJob): Pre-registered job whose counters
            (``total_candidates``, ``sampled_count``, ``concurrency_limit``,
            ``completed``, ``failed``) and ``status`` this worker mutates in
            place.
        request_context (RequestContext): Carrier for storage, config, and
            prompt manager. The configurator's current Config is read to
            pick up ``eval_sample_n_per_stratum`` and ``eval_concurrency_limit``.
        llm_client (LiteLLMClient): Shared LLM client used by
            ``run_group_evaluation`` per session.
        rng (random.Random | None): Optional seeded RNG for reproducible
            sampling. Defaults to an unseeded ``random.Random()`` so callers
            that don't care about reproducibility don't need to pass anything.
    """
    try:
        storage = request_context.storage
        if storage is None:
            raise RuntimeError("storage is not configured")
        config = request_context.configurator.get_config()
        job.concurrency_limit = config.eval_concurrency_limit

        descriptors = storage.get_session_ids_in_window(
            from_ts=job.from_ts, to_ts=job.to_ts
        )
        job.total_candidates = len(descriptors)

        candidates = _build_sample_candidates(storage, descriptors)
        sampled = sample_candidates(
            candidates,
            n_per_stratum=config.eval_sample_n_per_stratum,
            rng=rng or random.Random(),  # noqa: S311 — sampling, not crypto
        )
        job.sampled_count = len(sampled)

        observed_cancel = _run_pool(
            sampled,
            job=job,
            request_context=request_context,
            llm_client=llm_client,
        )
        job.status = "cancelled" if observed_cancel else "completed"
    except Exception:  # noqa: BLE001 — worker boundary
        job.status = "error"
        logger.exception("Regen worker crashed for job=%s", job.job_id)
    finally:
        job.finished_at = time.time()


def _run_pool(
    sampled: list[SampleCandidate],
    *,
    job: RegenJob,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
) -> bool:
    """Dispatch ``sampled`` through a ThreadPoolExecutor and aggregate results.

    Updates ``job.completed`` / ``job.failed`` / ``job.failures`` (capped
    at ``_FAILURE_CAP``) as futures resolve. When ``job.cancel_event``
    fires the pool stops submitting new work; futures already running
    finish, but their cancellation sentinels are silently dropped.

    Args:
        sampled (list[SampleCandidate]): Candidates already chosen by
            the stratified sampler.
        job (RegenJob): Owning job whose counters this function mutates.
        request_context (RequestContext): Forwarded to each per-session
            worker via ``_dispatch_one``.
        llm_client (LiteLLMClient): Forwarded to each per-session worker
            via ``_dispatch_one``.

    Returns:
        bool: ``True`` if cancellation was observed at any point
        (caller should set ``job.status = "cancelled"``); ``False`` if
        every submitted future ran to completion (success or failure)
        without cancellation.
    """
    if not sampled:
        return False
    max_workers = max(1, job.concurrency_limit)
    observed_cancel = False
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_sc: dict[Future[None], SampleCandidate] = {
            pool.submit(
                _dispatch_one,
                sc,
                job=job,
                request_context=request_context,
                llm_client=llm_client,
            ): sc
            for sc in sampled
        }
        for fut in as_completed(future_to_sc):
            sc = future_to_sc[fut]
            try:
                fut.result()
                job.completed += 1
            except _CancelledError:
                # cancel_event was set before this future started; the
                # post-loop check below is the single source of truth for
                # the cancelled state. Don't touch counters here.
                continue
            except Exception as e:  # noqa: BLE001 — worker boundary
                job.failed += 1
                if len(job.failures) < _FAILURE_CAP:
                    job.failures.append(
                        JobFailure(session_id=sc.session_id, reason=str(e)[:200])
                    )
                logger.warning("Regen failed for session=%s: %s", sc.session_id, e)
    if job.cancel_event.is_set():
        observed_cancel = True
    return observed_cancel
