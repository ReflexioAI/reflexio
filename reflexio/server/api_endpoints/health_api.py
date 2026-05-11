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

_STARTED_AT = time.monotonic()
_REQUEST_COUNT = 0


def _read_rss_mb() -> float | None:
    """Return the current process's resident set size in MiB, or None.

    Returns:
        float | None: RSS in MiB; None when psutil is unavailable.
    """
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 1024.0 / 1024.0


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
    def healthz() -> dict[str, Any]:
        """Return per-worker process metrics for liveness/observability."""
        return {
            "pid": os.getpid(),
            "uptime_sec": time.monotonic() - _STARTED_AT,
            "request_count": _REQUEST_COUNT,
            "rss_mb": _read_rss_mb(),
        }
