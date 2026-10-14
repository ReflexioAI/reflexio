"""Health-check endpoint and per-process metrics.

Exposes ``GET /healthz`` returning a small JSON payload useful for verifying
worker recycling (each worker has its own PID + request count). The middleware
that increments ``request_count`` is per-process — counts are NOT synchronized
across workers, which is intentional: it lets operators see load distribution
across the worker pool.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, Request, Response

from reflexio.server.services.agent_success_evaluation import _eval_health

_STARTED_AT = time.monotonic()
_REQUEST_COUNT = 0


def _read_rss_mb() -> float | None:
    """Return the current process's resident set size in MiB, or None.

    Called on the event loop by ``/healthz``; on Linux psutil reads
    ``/proc/self/stat`` and ``/proc/self/statm`` (procfs, no disk I/O).

    Returns:
        float | None: RSS in MiB; None when psutil is unavailable.
    """
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1024.0 / 1024.0


_AMBER_SECONDS = 5 * 60
_RED_SECONDS = 30 * 60


def _liveness_from_tick(last_tick_monotonic: float | None) -> str:
    """Map scheduler-tick age to a green/amber/red liveness color.

    Args:
        last_tick_monotonic (float | None): The last recorded scheduler tick,
            from `time.monotonic()`; None if the scheduler has not yet ticked.

    Returns:
        str: One of "green", "amber", "red".
    """
    if last_tick_monotonic is None:
        return "red"
    age = time.monotonic() - last_tick_monotonic
    if age > _RED_SECONDS:
        return "red"
    if age > _AMBER_SECONDS:
        return "amber"
    return "green"


def install(app: FastAPI) -> None:
    """Install the /healthz route and request-count middleware on ``app``.

    Args:
        app (FastAPI): The FastAPI application to attach to.
    """

    @app.middleware("http")
    async def _count_requests(request: Request, call_next: Any) -> Response:
        global _REQUEST_COUNT
        _REQUEST_COUNT += 1
        return await call_next(request)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Return per-worker process metrics for liveness/observability.

        ``async`` on purpose: this is the container liveness probe, and a sync
        handler runs on the AnyIO threadpool, where it queues behind every
        handler blocked on a stalled database. The probe then times out and
        the orchestrator kills a process that is merely waiting. The body is
        loop-safe: module globals plus a tiny ``/proc`` read for RSS.
        """
        return {
            "pid": os.getpid(),
            "uptime_sec": time.monotonic() - _STARTED_AT,
            "request_count": _REQUEST_COUNT,
            "rss_mb": _read_rss_mb(),
        }

    @app.get("/healthz/eval")
    def healthz_eval() -> dict[str, Any]:
        """Return evaluation-pipeline health: skip counts, failure counts, liveness."""
        status = _eval_health.get_status()
        return {
            **status,
            "liveness": _liveness_from_tick(status["last_tick_monotonic"]),
        }
