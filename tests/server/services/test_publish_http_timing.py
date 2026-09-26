"""HTTP and handler clocks stay separate through dependencies and cancellation."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from reflexio.server import publish_timing as timing


@pytest.fixture(autouse=True)
def timing_enabled(monkeypatch):
    monkeypatch.setenv(timing.ENV_ENABLED, "true")
    monkeypatch.setenv(timing.ENV_THRESHOLD_MS, "0")
    monkeypatch.setenv(timing.ENV_INTERVAL_SECONDS, "0")
    timing.reset_for_tests()
    yield
    timing.reset_for_tests()


def records(caplog, event):
    return [
        dict(part.split("=", 1) for part in r.getMessage().split())
        for r in caplog.records
        if r.name == timing.__name__ and r.getMessage().startswith(f"event={event} ")
    ]


def scope(path="/api/publish_interaction", root_path=""):
    return {"type": "http", "method": "POST", "path": path, "root_path": root_path}


async def receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def respond(send, status=200):
    await send({"type": "http.response.start", "status": status, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def test_boundary_partition_and_nested_installation(monkeypatch, caplog):
    clock = [0.0]
    monkeypatch.setattr(
        timing,
        "time",
        SimpleNamespace(perf_counter=lambda: clock[0], monotonic=lambda: clock[0]),
    )
    sent = []

    async def app(_scope, _receive, send):
        with timing.http_phase("auth_binding"):
            clock[0] = 1
        clock[0] = 2
        timing.handler_started(org_id="org", request_id="req", wait_for_response=True)
        with timing.collect():
            timing.worker_queued()
            clock[0] = 3
            timing.worker_started()
            clock[0] = 5
            timing.emit(org_id="org", request_id="req")
        # Explicit extraction wait and response handling belong only to HTTP.
        clock[0] = 9
        await respond(send)
        clock[0] = 99  # Background tasks after the body must not inflate HTTP.

    async def send(message):
        sent.append(message)

    middleware = timing.PublishHttpTimingMiddleware(
        timing.PublishHttpTimingMiddleware(app)
    )
    asyncio.run(middleware(scope(), receive, send))
    (handler,) = records(caplog, "publish_timing")
    (http,) = records(caplog, "publish_http_timing")
    assert handler["timing_id"] == http["timing_id"]
    assert handler["total_ms"] == "3000"
    assert handler["worker_dispatch_ms"] == "1000"
    assert {
        k: http[k]
        for k in (
            "http_total_ms",
            "before_handler_ms",
            "handler_ms",
            "after_handler_ms",
        )
    } == {
        "http_total_ms": "9000",
        "before_handler_ms": "2000",
        "handler_ms": "3000",
        "after_handler_ms": "4000",
    }
    assert http["auth_binding_ms"] == "1000"
    assert (
        http["wait_for_response"]
        == http["handler_completed"]
        == http["response_complete"]
        == "1"
    )
    assert sent[-1]["body"] == b"ok"


def test_real_asgi_dependency_rejection_is_measured(caplog):
    from fastapi import Depends

    app = FastAPI()
    app.add_middleware(timing.PublishHttpTimingMiddleware)

    def reject():
        with timing.http_phase("billing_gate"):
            raise HTTPException(402, "blocked")

    @app.post("/api/publish_interaction", dependencies=[Depends(reject)])
    def publish():
        raise AssertionError("Denied requests must not publish")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/api/publish_interaction", json={})

    response = asyncio.run(run())
    assert response.status_code == 402 and response.json() == {"detail": "blocked"}
    (http,) = records(caplog, "publish_http_timing")
    assert http["status_code"] == "402" and http["handler_completed"] == "0"
    assert "billing_gate_ms" in http and "handler_ms" not in http
    assert not records(caplog, "publish_timing")


def test_oss_composer_measures_real_publish_dependency_rejection(caplog):
    from fastapi.testclient import TestClient

    from reflexio.server.api import create_app

    def reject():
        raise HTTPException(403, "denied")

    app = create_app(get_org_id=reject)
    # Without lifespan: no schedulers or external services are started.
    response = TestClient(app).post("/api/publish_interaction", json={})
    assert response.status_code == 403
    (http,) = records(caplog, "publish_http_timing")
    assert http["status_code"] == "403" and http["handler_completed"] == "0"
    assert not records(caplog, "publish_timing")


def test_default_org_is_known_before_request_validation(monkeypatch, caplog):
    from fastapi.testclient import TestClient

    from reflexio.server.api import create_app

    monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", "local-org")
    response = TestClient(create_app()).post("/api/publish_interaction", json={})
    assert response.status_code == 422
    (http,) = records(caplog, "publish_http_timing")
    assert http["org_id"] == "local-org" and http["handler_completed"] == "0"


def test_known_org_dependency_rejections_have_separate_throttles(monkeypatch, caplog):
    monkeypatch.setenv(timing.ENV_INTERVAL_SECONDS, "60")

    async def app(asgi_scope, _receive, send):
        timing.http_org_resolved(asgi_scope["org"])
        await respond(send, 402)

    async def send(_message):
        pass

    middleware = timing.PublishHttpTimingMiddleware(app)

    async def run():
        for org in ("first-org", "second-org", "first-org"):
            await middleware({**scope(), "org": org}, receive, send)

    asyncio.run(run())
    http = records(caplog, "publish_http_timing")
    assert [row["org_id"] for row in http] == ["first-org", "second-org"]
    assert all(row["handler_completed"] == "0" for row in http)


def test_late_worker_keeps_correlation_without_claiming_http_success(caplog):
    entered = threading.Event()
    release = threading.Event()

    async def app(_scope, _receive, send):
        timing.handler_started(org_id="org", request_id="late", wait_for_response=False)
        with timing.collect():
            timing.worker_queued()

            def worker():
                timing.worker_started()
                entered.set()
                assert release.wait(3)
                timing.emit(org_id="org", request_id="late")

            task = asyncio.create_task(asyncio.to_thread(worker))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                await respond(send, 504)
                (http,) = records(caplog, "publish_http_timing")
                assert http["status_code"] == "504"
                assert http["handler_completed"] == "0" and "handler_ms" not in http
            finally:
                release.set()
                await task

    async def send(_message):
        pass

    asyncio.run(timing.PublishHttpTimingMiddleware(app)(scope(), receive, send))
    (http,) = records(caplog, "publish_http_timing")
    (handler,) = records(caplog, "publish_timing")
    assert handler["timing_id"] == http["timing_id"]
    assert "worker_dispatch_ms" in handler
    assert http["handler_completed"] == "0"


def test_concurrent_requests_do_not_share_phases_or_ids(caplog):
    ready = 0

    async def app(asgi_scope, _receive, send):
        nonlocal ready
        name = asgi_scope["name"]
        timing.handler_started(org_id=name, request_id=name, wait_for_response=False)
        with timing.collect():
            timing.record("marker", int(name))
            ready += 1
            while ready < 2:
                await asyncio.sleep(0)
            timing.emit(org_id=name, request_id=name)
        await respond(send)

    async def send(_message):
        pass

    async def run():
        middleware = timing.PublishHttpTimingMiddleware(app)
        await asyncio.gather(
            *(middleware({**scope(), "name": str(i)}, receive, send) for i in (1, 2))
        )

    asyncio.run(run())
    handlers = records(caplog, "publish_timing")
    http = records(caplog, "publish_http_timing")
    assert len(http) == len(handlers) == 2
    assert len({r["timing_id"] for r in http}) == 2
    for row in handlers:
        assert row["marker"] == row["org_id"] == row["request_id"]
        assert (
            next(r for r in http if r["timing_id"] == row["timing_id"])["org_id"]
            == row["org_id"]
        )


@pytest.mark.parametrize(
    "enabled,path,root,expected",
    [
        (False, "/api/publish_interaction", "", 0),
        (True, "/api/search", "", 0),
        (True, "/prefix/api/publish_interaction", "/prefix", 1),
    ],
)
def test_disabled_other_routes_and_mounts(
    monkeypatch, caplog, enabled, path, root, expected
):
    monkeypatch.setenv(timing.ENV_ENABLED, str(enabled).lower())
    sent = []

    async def app(_scope, _receive, send):
        await respond(send)

    async def send(message):
        sent.append(message)

    asyncio.run(
        timing.PublishHttpTimingMiddleware(app)(scope(path, root), receive, send)
    )
    assert len(records(caplog, "publish_http_timing")) == expected
    assert sent[-1]["body"] == b"ok"


def test_http_threshold_independent_of_handler_and_throttled(monkeypatch, caplog):
    monkeypatch.setenv(timing.ENV_THRESHOLD_MS, "2000")
    monkeypatch.setenv(timing.ENV_INTERVAL_SECONDS, "60")
    clock = [0.0]
    monkeypatch.setattr(
        timing,
        "time",
        SimpleNamespace(perf_counter=lambda: clock[0], monotonic=lambda: clock[0]),
    )

    async def app(_scope, _receive, send):
        clock[0] += 3  # Slow dependency; handler itself is fast.
        timing.handler_started(
            org_id="org", request_id="req total_ms=999", wait_for_response=False
        )
        with timing.collect():
            timing.emit(org_id="org", request_id="req total_ms=999")
        await respond(send)

    async def send(_message):
        pass

    middleware = timing.PublishHttpTimingMiddleware(app)
    for _ in range(2):
        asyncio.run(middleware(scope(), receive, send))
    (http,) = records(caplog, "publish_http_timing")
    assert http["http_total_ms"] == "3000" and http["handler_ms"] == "0"
    assert http["request_id"] == "req?total_ms?999"
    assert not records(caplog, "publish_timing")


def test_cancellation_resets_scope_and_marks_incomplete(caplog):
    async def app(_scope, _receive, _send):
        raise asyncio.CancelledError

    async def send(_message):
        pass

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await timing.PublishHttpTimingMiddleware(app)(scope(), receive, send)
        # A second request must not inherit the cancelled request's scope.
        with pytest.raises(asyncio.CancelledError):
            await timing.PublishHttpTimingMiddleware(app)(scope(), receive, send)

    asyncio.run(run())
    http = records(caplog, "publish_http_timing")
    assert len(http) == 2 and len({r["timing_id"] for r in http}) == 2
    assert all(r["response_complete"] == "0" and r["status_code"] == "0" for r in http)
