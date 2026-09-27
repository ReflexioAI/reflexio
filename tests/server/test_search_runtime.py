"""Search HTTP deadline and ownership regressions (real worker threads)."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI

from reflexio.server import search_runtime as runtime
from reflexio.server.tracing import profile_step


async def call(app, *, disconnect=None):
    messages = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        if disconnect is not None:
            await disconnect.wait()
            return {"type": "http.disconnect"}
        return await asyncio.Future()

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/search",
            "root_path": "",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 1),
            "http_version": "1.1",
        },
        receive,
        send,
    )
    return messages


@pytest.mark.asyncio
async def test_timeout_does_not_release_worker_capacity_or_send_late_success():
    app = FastAPI()
    entered, release, ended = threading.Event(), threading.Event(), threading.Event()
    side_effects = []

    @app.post("/api/search")
    def search():
        entered.set()
        try:
            release.wait(2)
            runtime.checkpoint()
            side_effects.append("exposure")
            return {"success": True}
        finally:
            ended.set()

    middleware = runtime.SearchRuntimeMiddleware(app, timeout=0.04, capacity=1)
    try:
        start = time.monotonic()
        messages = await call(middleware)
        assert time.monotonic() - start < 0.5
        assert entered.is_set() and not ended.is_set()
        assert messages[0]["status"] == 504
        assert json.loads(messages[1]["body"])["reason"] == "search_deadline"
        assert len(middleware.tasks) == 1
        rejected = await call(middleware)
        assert rejected[0]["status"] == 503
    finally:
        release.set()
        await asyncio.gather(*middleware.tasks, return_exceptions=True)
    assert ended.is_set()
    assert side_effects == []
    assert len(messages) == 2
    await asyncio.sleep(0)
    assert not middleware.tasks


@pytest.mark.asyncio
async def test_success_and_business_failure_are_distinct_metrics(monkeypatch):
    events = []
    monkeypatch.setattr(
        runtime, "record_health", lambda name, *_args, **kw: events.append((name, kw))
    )
    app = FastAPI()

    @app.post("/api/search")
    def search():
        with profile_step("search.embedding"):
            runtime.set_outcome(False)
        return {"success": False}

    messages = await call(runtime.SearchRuntimeMiddleware(app))
    assert messages[0]["status"] == 200
    assert (
        "search.requests",
        {"outcome": "application_failure", "status": "200"},
    ) in events


@pytest.mark.asyncio
async def test_disconnect_stops_followup_work():
    disconnect, started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    effects = []

    async def app(scope, receive, send):
        await receive()
        started.set()
        await release.wait()
        runtime.checkpoint()
        effects.append("late")

    middleware = runtime.SearchRuntimeMiddleware(app)
    request = asyncio.create_task(call(middleware, disconnect=disconnect))
    await started.wait()
    disconnect.set()
    assert await request == []
    release.set()
    await asyncio.gather(*middleware.tasks, return_exceptions=True)
    assert effects == []


def test_nested_timing_uses_union_without_tracing():
    scope = runtime.SearchScope(started=10)
    scope.intervals = [("search.endpoint", 10, 20), ("a", 11, 15), ("b", 12, 16)]
    result = scope.snapshot(20)
    assert result["unattributed_ms"] == 5000
    assert result["phases_ms"] == {"search.endpoint": 10000, "a": 4000, "b": 4000}


def test_queue_cancellation_does_not_run_queued_job():
    scope = runtime.SearchScope()
    token = runtime._scope.set(scope)
    release = threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(release.wait, 1)
            future = executor.submit(lambda: pytest.fail("cancelled work ran"))
            runtime.track(future)
            scope.cancel("timeout")
            assert future.cancelled()
            release.set()
            active.result()
    finally:
        release.set()
        runtime._scope.reset(token)


def test_cancellation_cannot_reach_released_connection():
    called = threading.Event()
    lease = runtime.ConnectionLease(called.set)
    lease.release()
    lease.interrupt()
    assert not called.wait(0.05)


def test_shared_retry_budget_is_atomic():
    scope = runtime.SearchScope()
    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(lambda _: scope.claim_retry(), range(8)))
    assert outcomes.count(True) == 1


@pytest.mark.asyncio
async def test_rollback_switch_preserves_measurement(monkeypatch):
    monkeypatch.setenv("REFLEXIO_SEARCH_DEADLINE_ENABLED", "false")

    async def app(scope, receive, send):
        await asyncio.sleep(0.02)
        runtime.set_outcome(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    messages = await call(runtime.SearchRuntimeMiddleware(app, timeout=0.001))
    assert messages[0]["status"] == 200


def test_embedded_calls_have_no_deadline():
    assert runtime.current() is None
    assert runtime.remaining(30) == 30


@pytest.mark.asyncio
async def test_body_wait_is_woken_on_timeout():
    ended = asyncio.Event()

    async def app(scope, receive, send):
        await receive()
        assert (await receive())["type"] == "http.disconnect"
        ended.set()

    middleware = runtime.SearchRuntimeMiddleware(app, timeout=0.01)
    assert (await call(middleware))[0]["status"] == 504
    await asyncio.wait_for(ended.wait(), 0.2)
    await asyncio.gather(*middleware.tasks)
    await asyncio.sleep(0)
    assert not middleware.tasks


@pytest.mark.asyncio
async def test_queue_timeout_stays_owned_and_attributed():
    from reflexio.server.services.unified_search_service import (
        _submit_with_current_context,
    )

    release = threading.Event()
    states = []
    executed = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release.wait, 2)

        async def app(scope, receive, send):
            states.append(runtime.current())
            _submit_with_current_context(executor, lambda: executed.append(True))
            await asyncio.sleep(0.04)

        middleware = runtime.SearchRuntimeMiddleware(app, timeout=0.01, capacity=1)
        try:
            assert (await call(middleware))[0]["status"] == 504
            await asyncio.sleep(0.05)
            assert len(middleware.tasks) == 1
            snapshot = states[0].snapshot(time.monotonic())
            assert snapshot["phases_ms"]["search.worker_queue"] >= 10
            assert "search.worker_queue" in snapshot["active_phases"]
            assert (await call(middleware))[0]["status"] == 503
        finally:
            release.set()
            blocker.result()
            await asyncio.gather(*middleware.tasks)
    assert not executed


@pytest.mark.asyncio
async def test_downstream_failure_overrides_success(monkeypatch):
    events = []
    monkeypatch.setattr(
        runtime, "record_health", lambda name, *_args, **kw: events.append((name, kw))
    )

    async def app(scope, receive, send):
        runtime.set_outcome(True)
        raise ValueError("serialization failed")

    with pytest.raises(ValueError):
        await call(runtime.SearchRuntimeMiddleware(app))
    assert ("search.requests", {"outcome": "http_failure", "status": "500"}) in events


def test_config_version_memo_is_only_per_request():
    from unittest.mock import Mock

    from reflexio.server.cache.reflexio_cache import _probe_version_safe

    reflexio = Mock()
    reflexio.current_config_version.side_effect = [("db", 1), ("db", 2)]
    for version in [1, 2]:
        token = runtime._scope.set(runtime.SearchScope())
        try:
            assert _probe_version_safe(reflexio) == ("db", version)
            assert _probe_version_safe(reflexio) == ("db", version)
        finally:
            runtime._scope.reset(token)
    assert reflexio.current_config_version.call_count == 2


def test_http_timeout_hook_is_request_local_and_preserves_stricter_policy():
    import httpx

    observed = []

    def transport(request):
        observed.append(dict(request.extensions["timeout"]))
        return httpx.Response(200, json={})

    with httpx.Client(
        transport=httpx.MockTransport(transport),
        timeout=30,
        event_hooks={"request": [runtime.http_request_deadline]},
    ) as client:
        client.get("https://example.test")
        assert observed[-1]["read"] == 30
        state = runtime.SearchScope(deadline=time.monotonic() + 1)
        token = runtime._scope.set(state)
        try:
            client.get(
                "https://example.test", timeout=httpx.Timeout(30, connect=0, pool=0.01)
            )
            assert observed[-1]["connect"] == 0
            assert observed[-1]["pool"] == 0.01
            assert 0 < observed[-1]["read"] <= 1
            assert client.timeout.read == 30
            state.deadline = None
            client.get("https://example.test")
            assert observed[-1]["read"] == 30
            state.cancel("timeout")
            with pytest.raises(runtime.SearchDeadlineError):
                client.get("https://example.test")
            assert len(observed) == 3
        finally:
            runtime._scope.reset(token)
        client.get("https://example.test")
        assert observed[-1]["read"] == 30


@pytest.mark.asyncio
async def test_stalled_auth_http_call_is_bounded_and_releases_admission():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from typing import Annotated

    import httpx
    from fastapi import Depends

    release, entered = threading.Event(), threading.Event()

    class StalledAuth(BaseHTTPRequestHandler):
        def do_GET(self):
            entered.set()
            release.wait(2)

    server = ThreadingHTTPServer(("127.0.0.1", 0), StalledAuth)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    client = httpx.Client(
        timeout=30, event_hooks={"request": [runtime.http_request_deadline]}
    )
    app = FastAPI()
    reached = []

    def authenticate():
        client.get(f"http://127.0.0.1:{server.server_port}/auth")

    @app.post("/api/search")
    def search(_auth: Annotated[None, Depends(authenticate)]):
        reached.append(True)
        return {"success": True}

    middleware = runtime.SearchRuntimeMiddleware(app, timeout=0.05)
    try:
        start = time.monotonic()
        messages = await call(middleware)
        assert entered.is_set()
        assert messages[0]["status"] == 504
        assert time.monotonic() - start < 0.5
        await asyncio.wait_for(
            asyncio.gather(*middleware.tasks, return_exceptions=True), 0.5
        )
        await asyncio.sleep(0)
        assert not middleware.tasks
        assert not reached
        assert client.timeout.read == 30
    finally:
        release.set()
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(1)


def test_parallel_phase_wall_time_does_not_double_count_work():
    state = runtime.SearchScope(started=10)
    state.intervals = [("storage.query", 11, 15), ("storage.query", 12, 16)]
    snapshot = state.snapshot(20)
    assert snapshot["phases_ms"]["storage.query"] == 5000
    assert snapshot["phase_work_ms"]["storage.query"] == 8000
    assert snapshot["unattributed_ms"] == 5000


def test_cancellation_does_not_skip_transaction_cleanup():
    state = runtime.SearchScope()
    token = runtime._scope.set(state)
    cleaned = []
    try:
        state.cancel("timeout")
        with runtime.phase("storage.rollback", cleanup=True):
            cleaned.append(True)
        with pytest.raises(runtime.SearchDeadlineError), runtime.phase("storage.query"):
            pytest.fail("new query after cancellation")
    finally:
        runtime._scope.reset(token)
    assert cleaned == [True]
    assert "storage.rollback" in state.snapshot(time.monotonic())["phases_ms"]
