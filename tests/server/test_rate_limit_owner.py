"""Quota ownership at the actual SDK app and configuration boundaries."""

import asyncio
import os
import threading

import httpx
import pytest
from fastapi import APIRouter, Request
from fastapi.testclient import TestClient

from reflexio.server import api, rate_limit


@pytest.fixture(autouse=True)
def clean_rate_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("RATELIMIT_"):
            monkeypatch.delenv(name)


def sdk_app(monkeypatch, limiter, router):
    monkeypatch.setattr(api, "limiter", limiter)
    return api.create_app(
        get_org_id=lambda: "rate-owner-test",
        additional_routers=[router],
        mount_data_plane=False,
    )


def test_actual_sdk_rejection_has_advisory_rule_window(monkeypatch):
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    router = APIRouter()
    calls = []

    @router.get("/quota-owner")
    @limiter.limit("1 per 2 hours")
    async def limited(request: Request):
        calls.append("handler")
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    with TestClient(app) as client:
        assert client.get("/quota-owner").status_code == 200
        response = client.get("/quota-owner")
    assert response.status_code == 429
    assert response.json() == {"error": "Rate limit exceeded: 1 per 2 hour"}
    assert response.headers["Retry-After"] == "7200"
    assert calls == ["handler"]


def test_dependency_cannot_load_cwd_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("RATELIMIT_ENABLED=false\n")
    monkeypatch.chdir(tmp_path)
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    assert limiter.enabled is True


def test_fallback_field_can_select_original_rules(monkeypatch):
    monkeypatch.setenv("RATELIMIT_IN_MEMORY_FALLBACK", "inherit")
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    assert limiter._in_memory_fallback_enabled is True
    assert limiter._in_memory_fallback == []


@pytest.mark.asyncio
async def test_actual_sdk_async_precheck_leaves_loop_responsive(monkeypatch):
    entered = threading.Event()
    heartbeat = threading.Event()
    released = threading.Event()
    observed = []

    def key(request):
        assert isinstance(request, Request)
        entered.set()
        observed.append(heartbeat.wait(2))
        released.set()
        return "test"

    limiter = type(rate_limit.limiter)(key_func=key)
    router = APIRouter()

    @router.get("/quota-owner")
    @limiter.limit("1/minute")
    async def limited(request: Request):
        return {"ok": True}

    @router.get("/quota-heartbeat")
    async def pulse():
        heartbeat.set()
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        limited_task = asyncio.create_task(client.get("/quota-owner"))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert (await client.get("/quota-heartbeat")).status_code == 200
            assert (await limited_task).status_code == 200
        finally:
            heartbeat.set()
            await limited_task
    assert released.is_set()
    assert observed == [True], "quota key callback blocked the event loop"


@pytest.mark.parametrize("name", ["RATELIMIT_ENABLED", "RATELIMIT_SWALLOW_ERRORS"])
@pytest.mark.parametrize(
    "value, expected", [("false", False), ("0", False), ("true", True), ("1", True)]
)
def test_boolean_values_are_real_booleans(monkeypatch, name, value, expected):
    monkeypatch.setenv(name, value)
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    assert (
        limiter.enabled if name == "RATELIMIT_ENABLED" else limiter._swallow_errors
    ) is expected


@pytest.mark.parametrize("fallback, accepted", [("inherit", 1), ("2/minute", 2)])
def test_outage_fallback_keeps_selected_rules(monkeypatch, fallback, accepted):
    monkeypatch.setenv("RATELIMIT_IN_MEMORY_FALLBACK", fallback)
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")

    def unavailable(*_args, **_kwargs):
        raise OSError("synthetic rate-store outage")

    monkeypatch.setattr(limiter._limiter, "hit", unavailable)
    router = APIRouter()

    @router.get("/fallback-owner")
    @limiter.limit("1/minute")
    async def limited(request: Request):
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    with TestClient(app) as client:
        for _ in range(accepted):
            assert client.get("/fallback-owner").status_code == 200
        response = client.get("/fallback-owner")
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"


@pytest.mark.parametrize("swallow", [False, True])
def test_outage_without_fallback_does_not_accidentally_fail_open(monkeypatch, swallow):
    monkeypatch.setenv("RATELIMIT_SWALLOW_ERRORS", str(swallow))
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")

    def unavailable(*_args, **_kwargs):
        raise OSError("synthetic rate-store outage")

    monkeypatch.setattr(limiter._limiter, "hit", unavailable)
    router = APIRouter()
    calls = []

    @router.get("/unavailable-owner")
    @limiter.limit("1/minute")
    async def limited(request: Request):
        calls.append(True)
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/unavailable-owner")
    assert response.status_code == (200 if swallow else 500)
    assert calls == ([True] if swallow else [])


def test_sdk_middleware_keeps_default_application_and_override_scopes(monkeypatch):
    monkeypatch.setenv("RATELIMIT_APPLICATION", "2/minute")
    monkeypatch.setenv("RATELIMIT_DEFAULT", "1/minute")
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    router = APIRouter()

    @router.get("/middleware-default")
    async def default(request: Request):
        return {"ok": True}

    @router.get("/middleware-override")
    @limiter.limit("2/minute")
    async def override(request: Request):
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    app.add_middleware(rate_limit.RateLimitMiddleware)
    with TestClient(app) as client:
        assert client.get("/middleware-default").status_code == 200
        assert client.get("/middleware-default").status_code == 429
        assert client.get("/middleware-override").status_code == 200
        assert client.get("/middleware-override").status_code == 200
        assert client.get("/middleware-override").status_code == 429


@pytest.mark.asyncio
async def test_cancelled_quota_worker_cannot_start_actual_sdk_handler(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    calls = []

    def key(request):
        assert isinstance(request, Request)
        entered.set()
        try:
            assert release.wait(3)
            return "test"
        finally:
            completed.set()

    limiter = type(rate_limit.limiter)(key_func=key)
    router = APIRouter()

    @router.get("/cancelled-quota")
    @limiter.limit("1/minute")
    async def limited(request: Request):
        calls.append("handler")
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(client.get("/cancelled-quota"))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            assert not completed.is_set(), "physical quota work was not held"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.to_thread(completed.wait, 3)
    assert completed.is_set()
    assert calls == []


@pytest.mark.asyncio
async def test_middleware_quota_check_leaves_loop_responsive(monkeypatch):
    monkeypatch.setenv("RATELIMIT_DEFAULT", "1/minute")
    entered = threading.Event()
    heartbeat = threading.Event()
    observed = []

    def key(request):
        assert isinstance(request, Request)
        entered.set()
        observed.append(heartbeat.wait(2))
        return "test"

    limiter = type(rate_limit.limiter)(key_func=key)
    router = APIRouter()

    @router.get("/middleware-quota")
    async def limited(request: Request):
        return {"ok": True}

    @router.get("/middleware-heartbeat")
    @limiter.exempt
    async def pulse(request: Request):
        heartbeat.set()
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    app.add_middleware(rate_limit.RateLimitMiddleware)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(client.get("/middleware-quota"))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert (await client.get("/middleware-heartbeat")).status_code == 200
            assert (await task).status_code == 200
        finally:
            heartbeat.set()
            await task
    assert observed == [True]


def test_live_discarded_registration_and_stacked_limits_share_one_check(monkeypatch):
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    router = APIRouter()
    checks = []
    upstream_check = limiter._check_request_limit

    def counted(*args, **kwargs):
        checks.append(True)
        return upstream_check(*args, **kwargs)

    monkeypatch.setattr(limiter, "_check_request_limit", counted)

    @router.get("/stacked-owner")
    @limiter.limit("3/minute")
    @limiter.limit("10/hour")
    async def limited(request: Request):
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    # Enterprise adds QPS rules after router inclusion and discards the wrapper.
    limiter.limit("2/minute")(limited)
    with TestClient(app) as client:
        assert client.get("/stacked-owner").status_code == 200
        assert client.get("/stacked-owner").status_code == 200
        assert client.get("/stacked-owner").status_code == 429
    assert len(checks) == 3


def test_shared_cost_and_sync_route_preserve_budget(monkeypatch):
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    router = APIRouter()

    @router.get("/shared-sync-owner")
    @limiter.shared_limit("3/minute", "owner", cost=2)
    def first(request: Request):
        return {"ok": True}

    @router.get("/shared-async-owner")
    @limiter.shared_limit("3/minute", "owner", cost=lambda _request: 2)
    async def second(request: Request):
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    with TestClient(app) as client:
        assert client.get("/shared-sync-owner").status_code == 200
        assert client.get("/shared-async-owner").status_code == 429


def test_runtime_policy_rejects_bad_values_without_echoing_credentials(monkeypatch):
    monkeypatch.setenv("RATELIMIT_STORAGE_URL", "invalid://secret-password@host")
    with pytest.raises(ValueError) as error:
        type(rate_limit.limiter)(key_func=lambda: "test")
    assert "secret-password" not in str(error.value)


@pytest.mark.parametrize("value", ["true", "false", "1", "0"])
def test_retired_header_gate_is_never_accepted(monkeypatch, value):
    monkeypatch.setenv("RATELIMIT_HEADERS_ENABLED", value)
    with pytest.raises(ValueError, match="RATELIMIT_HEADERS_ENABLED"):
        type(rate_limit.limiter)(key_func=lambda: "test")


@pytest.mark.asyncio
async def test_cancel_after_counter_hit_cannot_enter_handler(monkeypatch):
    import anyio

    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    original_hit = limiter._limiter.hit
    counter_calls = []
    handler_calls = []
    cancel_scope = anyio.CancelScope()

    def cancel_after_hit(*args, **kwargs):
        result = original_hit(*args, **kwargs)
        counter_calls.append(True)
        anyio.from_thread.run_sync(cancel_scope.cancel)
        return result

    monkeypatch.setattr(limiter._limiter, "hit", cancel_after_hit)
    router = APIRouter()

    @router.get("/cancel-after-quota")
    @limiter.limit("1/minute")
    async def limited(request: Request):
        handler_calls.append(True)
        return {"ok": True}

    app = sdk_app(monkeypatch, limiter, router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        with cancel_scope:
            await client.get("/cancel-after-quota")
    assert cancel_scope.cancel_called
    assert counter_calls == [True]
    assert handler_calls == []
    # The completed physical hit is not undone by cancellation.
    monkeypatch.setattr(limiter._limiter, "hit", original_hit)
    with TestClient(app) as client:
        assert client.get("/cancel-after-quota").status_code == 429


@pytest.mark.asyncio
async def test_cancelled_worker_queue_never_checks_quota_or_enters_handler(monkeypatch):
    import anyio

    entered, release = threading.Event(), threading.Event()
    waiting = asyncio.Event()
    calls = []
    limiter = type(rate_limit.limiter)(key_func=lambda: "test")
    router = APIRouter()

    @router.get("/queued-quota")
    @limiter.limit("1/minute")
    async def limited(request: Request):
        calls.append("handler")
        return {"ok": True}

    original_check = limiter._check_request_limit
    original_worker = rate_limit.run_in_threadpool

    def counted(*args, **kwargs):
        calls.append("check")
        return original_check(*args, **kwargs)

    async def observed_worker(*args, **kwargs):
        waiting.set()
        return await original_worker(*args, **kwargs)

    def hold_worker():
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(limiter, "_check_request_limit", counted)
    monkeypatch.setattr(rate_limit, "run_in_threadpool", observed_worker)
    app = sdk_app(monkeypatch, limiter, router)
    workers = anyio.to_thread.current_default_thread_limiter()
    previous = workers.total_tokens
    workers.total_tokens = 1
    holder = asyncio.create_task(anyio.to_thread.run_sync(hold_worker))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(client.get("/queued-quota"))
            await asyncio.wait_for(waiting.wait(), 3)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
    finally:
        release.set()
        await holder
        workers.total_tokens = previous
    assert calls == []
