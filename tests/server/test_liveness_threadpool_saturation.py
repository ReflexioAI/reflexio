"""The liveness probes answer while the AnyIO threadpool is exhausted.

``/livez`` is the container liveness probe and ``/healthz`` is its fallback on
older images. During a database stall every sync handler holds a threadpool
token while it waits; a sync probe would queue behind them, time out, and get
a merely-waiting process killed.
"""

from __future__ import annotations

import threading

import anyio
import httpx
import pytest
from anyio.to_thread import current_default_thread_limiter

from reflexio.server.api import create_app


@pytest.mark.parametrize("path", ["/livez", "/healthz"])
@pytest.mark.asyncio
async def test_probe_answers_while_every_threadpool_token_is_held(path: str) -> None:
    app = create_app(mount_data_plane=False)
    release = threading.Event()
    entered = threading.Event()

    @app.get("/stalled")
    def stalled() -> dict[str, bool]:  # sync: holds a threadpool token
        entered.set()
        release.wait(timeout=30)
        return {"ok": True}

    limiter = current_default_thread_limiter()
    original = limiter.total_tokens
    limiter.total_tokens = 1  # one stalled request now exhausts the pool
    headers = {"User-Agent": "pytest-agent"}
    transport = httpx.ASGITransport(app=app)
    try:
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://t") as client,
            anyio.create_task_group() as tg,
        ):

            async def _stall() -> None:
                await client.get("/stalled", headers=headers)

            tg.start_soon(_stall)
            try:
                with anyio.fail_after(5):
                    while not entered.is_set():
                        await anyio.sleep(0.01)
                assert limiter.borrowed_tokens == limiter.total_tokens

                with anyio.fail_after(2):
                    response = await client.get(path, headers=headers)
                assert response.status_code == 200
            finally:
                # Unblock the stalled thread before the task group waits on it.
                release.set()
    finally:
        release.set()
        limiter.total_tokens = original
