"""Regenerate job registry + worker for replaying the LLM judge.

In-memory only; process-local. Survives the worker thread but not a
backend restart. v2 will move job state to storage.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)

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


def run_regen(
    *,
    job: RegenJob,
    request_context: RequestContext,
    llm_client: LiteLLMClient,
) -> None:
    """Worker body. Drives run_group_evaluation per session in the window.

    On normal completion of the session loop, sets ``job.status = "completed"``
    even if individual sessions raised — per-session failures are recorded in
    ``job.failed`` and ``job.failures``. Sets ``job.status = "error"`` only if
    the worker itself crashes (e.g. storage misconfigured). Sets
    ``job.status = "cancelled"`` if the cancel event is observed between
    sessions.

    Args:
        job (RegenJob): Pre-registered job whose counters and status this
            worker mutates in place.
        request_context (RequestContext): Carrier for storage, config, and
            prompt manager.
        llm_client (LiteLLMClient): Shared LLM client used by
            ``run_group_evaluation`` per session.
    """
    try:
        storage = request_context.storage
        if storage is None:
            raise RuntimeError("storage is not configured")
        descriptors = storage.get_session_ids_in_window(
            from_ts=job.from_ts, to_ts=job.to_ts
        )
        for sd in descriptors:
            if job.cancel_event.is_set():
                job.status = "cancelled"
                break
            try:
                run_group_evaluation(
                    org_id=job.org_id,
                    user_id=sd.user_id,
                    session_id=sd.session_id,
                    agent_version=sd.agent_version,
                    source=sd.source,
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
                        JobFailure(session_id=sd.session_id, reason=str(e)[:200])
                    )
                logger.warning("Regen failed for session=%s: %s", sd.session_id, e)
        else:
            job.status = "completed"
    except Exception:  # noqa: BLE001 — worker boundary
        job.status = "error"
        logger.exception("Regen worker crashed for job=%s", job.job_id)
    finally:
        job.finished_at = time.time()
