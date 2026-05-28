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
    evaluation_name: str
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
    """Process-local job registry. One active job per (org_id, evaluation_name)."""

    def __init__(self) -> None:
        self._jobs: dict[str, RegenJob] = {}
        self._by_org_evaluator: dict[tuple[str, str], str] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        org_id: str,
        evaluation_name: str,
        from_ts: int,
        to_ts: int,
        total: int,
    ) -> RegenJob:
        """Register a new running job. Raises RuntimeError when an actively-running
        job exists for (org, evaluator).

        Only *running* jobs block; a previous completed/cancelled/errored job for
        the same key is replaced. Eviction by TTL is a separate cleanup concern —
        we don't want users to wait an hour after a completed run before they can
        regenerate again.
        """
        with self._lock:
            self._evict_completed_locked(DEFAULT_TTL_SECONDS)
            key = (org_id, evaluation_name)
            active = self._active_job_for_locked(key)
            if active is not None:
                raise RuntimeError(
                    f"A regenerate is already running for evaluator '{evaluation_name}'"
                )
            job = RegenJob(
                job_id=uuid.uuid4().hex,
                org_id=org_id,
                evaluation_name=evaluation_name,
                from_ts=from_ts,
                to_ts=to_ts,
                status="running",
                total=total,
            )
            self._jobs[job.job_id] = job
            self._by_org_evaluator[key] = job.job_id
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

    def has_active(self, org_id: str, evaluation_name: str) -> bool:
        with self._lock:
            self._evict_completed_locked(DEFAULT_TTL_SECONDS)
            return self._active_job_for_locked((org_id, evaluation_name)) is not None

    def _active_job_for_locked(self, key: tuple[str, str]) -> RegenJob | None:
        """Return the running job for (org, evaluator), or None when no live job exists.

        A registry entry whose job has already finished
        (``completed`` / ``cancelled`` / ``error``) is treated as having no active
        job — the previous run finished and the user can start a new one. The
        registry still holds the finished job until TTL eviction so status polls
        keep working, but it doesn't block new submissions.
        """
        job_id = self._by_org_evaluator.get(key)
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
            self._by_org_evaluator.pop((j.org_id, j.evaluation_name), None)


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
    requests = storage.get_requests_by_session(user_id, session_id)
    if not requests:
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
    iterates the sampled subset sequentially; Task 6 will replace this
    loop with a bounded ``ThreadPoolExecutor``.

    On normal completion of the sampled loop, sets ``job.status = "completed"``
    even if individual sessions raised — per-session failures are recorded in
    ``job.failed`` and ``job.failures``. Sets ``job.status = "error"`` only if
    the worker itself crashes (e.g. storage misconfigured). Sets
    ``job.status = "cancelled"`` if the cancel event is observed between
    sessions.

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

        for sc in sampled:
            if job.cancel_event.is_set():
                job.status = "cancelled"
                break
            try:
                run_group_evaluation(
                    org_id=job.org_id,
                    user_id=sc.user_id,
                    session_id=sc.session_id,
                    agent_version=sc.agent_version,
                    source=sc.source,
                    request_context=request_context,
                    llm_client=llm_client,
                    force_regenerate=True,
                    evaluation_name=job.evaluation_name,
                )
                job.completed += 1
            except Exception as e:  # noqa: BLE001 — worker boundary
                job.failed += 1
                if len(job.failures) < _FAILURE_CAP:
                    job.failures.append(
                        JobFailure(session_id=sc.session_id, reason=str(e)[:200])
                    )
                logger.warning("Regen failed for session=%s: %s", sc.session_id, e)
        else:
            job.status = "completed"
    except Exception:  # noqa: BLE001 — worker boundary
        job.status = "error"
        logger.exception("Regen worker crashed for job=%s", job.job_id)
    finally:
        job.finished_at = time.time()
