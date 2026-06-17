"""Integration test: a backend spawned with low --max-requests actually recycles.

Marked slow because it spawns a real subprocess + HTTP traffic.

Uses ``--workers 2`` because uvicorn's request-count recycling only respawns when
a manager process is supervising worker children. With ``--workers 1`` uvicorn
exits cleanly on ``--max-requests`` exhaustion without respawning.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time

import httpx
import pytest

pytestmark = pytest.mark.integration


def _free_port() -> int:
    """Allocate a free TCP port on loopback.

    Returns:
        int: A port that is free at call time (race with re-binding is possible
            but acceptable for a test harness).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_healthz(url: str, timeout: float = 30.0) -> None:
    """Poll ``url`` until it returns 200 or ``timeout`` seconds elapse.

    Args:
        url (str): Full URL to GET.
        timeout (float): Maximum seconds to wait. Default 30.

    Raises:
        RuntimeError: If the URL never returns 200 within the deadline.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {url} did not become ready in {timeout}s")


def test_daemon_mode_recycles_worker_after_max_requests() -> None:
    """Spawn the backend with --max-requests 5 and verify a worker PID rolls.

    With ``--workers 2`` we observe the set of worker PIDs serving ``/healthz``
    across many requests. After enough traffic to trip ``--max-requests`` on at
    least one worker, the supervisor should respawn it under a new PID. The
    assertion is "at least one new worker PID was observed", which is the
    recycle behavior we care about (not "every worker rolled").
    """
    port = _free_port()
    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "reflexio.server",
            "--port",
            str(port),
            "--workers",
            "2",
            "--max-requests",
            "5",
            "--max-requests-jitter",
            "0",
        ]
    )
    try:
        url = f"http://127.0.0.1:{port}/healthz"
        # Cold start loads the local embedder + cross-encoder reranker into each
        # worker. With --workers 2 the manager brings workers up roughly serially,
        # so a clean cold start already takes ~70s even with models cached; a
        # worker death+respawn re-triggers a full model reload, and a contended box
        # (this test runs at the tail of the parallel suite) stretches it further.
        # Give it a generous budget so the recycle assertion isn't masked by a
        # too-tight readiness deadline.
        _wait_for_healthz(url, timeout=180.0)
        # Collect an initial set of worker PIDs by hammering the endpoint a few times.
        initial_pids: set[int] = set()
        for _ in range(10):
            with contextlib.suppress(httpx.HTTPError):
                initial_pids.add(httpx.get(url, timeout=2.0).json()["pid"])
        assert initial_pids, "no initial worker PIDs observed"

        # Drive enough requests to recycle every worker many times over.
        # Brief connection blips during recycle windows are expected.
        for _ in range(60):
            with contextlib.suppress(httpx.HTTPError):
                httpx.get(url, timeout=2.0)

        # Let the supervisor respawn any recycled workers. A respawned worker
        # reloads the embedder + reranker models from scratch, so it can take far
        # longer than a warm request to start serving again — wait accordingly.
        time.sleep(3.0)
        _wait_for_healthz(url, timeout=180.0)

        # Now collect post-traffic PIDs.
        post_pids: set[int] = set()
        for _ in range(20):
            with contextlib.suppress(httpx.HTTPError):
                post_pids.add(httpx.get(url, timeout=2.0).json()["pid"])
        assert post_pids, "no post-traffic worker PIDs observed"

        new_pids = post_pids - initial_pids
        assert new_pids, (
            "expected at least one new worker PID after exceeding --max-requests; "
            f"initial_pids={sorted(initial_pids)} post_pids={sorted(post_pids)}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
