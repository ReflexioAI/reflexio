import asyncio

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from reflexio.server.api import create_app
from reflexio.server.correlation import correlation_id_var
from reflexio.server.middleware import (
    REQUEST_TIMEOUT_SECONDS,
    ROUTE_BACKSTOP_SECONDS,
    SYNC_REQUEST_TIMEOUT_SECONDS,
    BodySizeLimitMiddleware,
    TimeoutMiddleware,
    backstop_for,
)


def test_cors_uses_frontend_url_allowlist(monkeypatch):
    monkeypatch.delenv("REFLEXIO_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("FRONTEND_URL", "https://app.example.com")

    # The credentialed allowlist is an enterprise concern — only hosts that
    # require auth lock down browser origins.
    client = TestClient(create_app(require_auth=True))

    allowed = client.options(
        "/health",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
    assert "access-control-allow-credentials" in allowed.headers
    assert "access-control-allow-origin" not in denied.headers


def test_cors_allowed_origins_override_frontend_url(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "https://app.example.com")
    monkeypatch.setenv(
        "REFLEXIO_ALLOWED_ORIGINS",
        "https://admin.example.com, https://console.example.com/",
    )

    client = TestClient(create_app(require_auth=True))
    response = client.options(
        "/health",
        headers={
            "Origin": "https://console.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["access-control-allow-origin"] == (
        "https://console.example.com"
    )


def test_cors_local_mode_allows_any_origin(monkeypatch):
    """OSS/local mode (no auth) does not restrict browser origins.

    The bundled docs playground is served cross-origin (a different port from
    the backend), so the local server must echo an allow-origin header for any
    requester. CORS lockdown is an enterprise-only concern.
    """
    monkeypatch.setenv("FRONTEND_URL", "https://app.example.com")

    client = TestClient(create_app())
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8062",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["access-control-allow-origin"] == "*"
    # No credentials are issued in local mode, so the wildcard is spec-clean.
    assert "access-control-allow-credentials" not in response.headers


def test_body_size_limit_rejects_large_declared_body(monkeypatch):
    monkeypatch.setenv("REFLEXIO_MAX_BODY_BYTES", "4")

    client = TestClient(create_app())
    response = client.post("/", content=b"12345")

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}


def test_body_size_limit_rejects_streamed_body_without_content_length(monkeypatch):
    monkeypatch.setenv("REFLEXIO_MAX_BODY_BYTES", "4")

    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware)

    @app.post("/consume")
    async def consume_body(request: Request):
        return {"size": len(await request.body())}

    messages = [
        {"type": "http.request", "body": b"12", "more_body": True},
        {"type": "http.request", "body": b"345", "more_body": False},
    ]
    sent = []

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/consume",
        "raw_path": b"/consume",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"user-agent", b"testclient")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    asyncio.run(app(scope, receive, send))

    response_start = next(m for m in sent if m["type"] == "http.response.start")
    response_body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert response_start["status"] == 413
    assert response_body == b'{"detail":"Request body too large"}'


def test_playbook_review_uses_synchronous_request_timeout(monkeypatch):
    observed: dict[str, float | None] = {}

    async def fake_wait_for(awaitable, *, timeout=None):
        observed["timeout"] = timeout
        return await awaitable

    async def call_next(_request):
        from starlette.responses import Response

        return Response()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/api/review_user_playbooks",
            "raw_path": b"/api/review_user_playbooks",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )

    asyncio.run(TimeoutMiddleware(FastAPI()).dispatch(request, call_next))

    assert observed["timeout"] == SYNC_REQUEST_TIMEOUT_SECONDS


def test_playbook_aggregation_post_uses_synchronous_request_timeout_without_wait_query(
    monkeypatch,
):
    observed: dict[str, float | None] = {}

    async def fake_wait_for(awaitable, *, timeout=None):
        observed["timeout"] = timeout
        return await awaitable

    async def call_next(_request):
        from starlette.responses import Response

        return Response()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/api/run_playbook_aggregation",
            "raw_path": b"/api/run_playbook_aggregation",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )

    asyncio.run(TimeoutMiddleware(FastAPI()).dispatch(request, call_next))

    assert observed["timeout"] == SYNC_REQUEST_TIMEOUT_SECONDS


def test_security_headers_are_added(monkeypatch):
    monkeypatch.delenv("REFLEXIO_ALLOWED_ORIGINS", raising=False)

    client = TestClient(create_app())
    response = client.get("/health", headers={"X-Forwarded-Proto": "https"})

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert response.headers["strict-transport-security"] == (
        "max-age=31536000; includeSubDomains"
    )


def _route_own_deadlines() -> dict[str, float]:
    """Each backstop path paired with the deadline the route itself enforces.

    Read at call time rather than at import so the pairing resolves against the
    live module. Keys must match ``ROUTE_BACKSTOP_SECONDS`` exactly — the guard
    below fails in BOTH directions, so neither table can grow a row alone and
    sit unchecked under a comment claiming it is covered.

    Returns:
        dict[str, float]: Path -> the route's own deadline, in seconds.
    """
    from reflexio.server.routes import interactions

    return {
        "/api/publish_interaction": interactions.PUBLISH_REQUEST_TIMEOUT_SECONDS,
    }


def test_every_backstop_row_exceeds_its_routes_own_deadline():
    """The route must reach its own deadline before the middleware fires.

    This is the whole bug: a 60s middleware budget under a 240s route budget
    made the route's request_id-bearing 504 unreachable, so every timed-out
    publish came back as an uncorrelatable ``{"detail": "Request timeout"}``.
    """
    own = _route_own_deadlines()
    unpaired = sorted(set(ROUTE_BACKSTOP_SECONDS) - set(own))
    assert not unpaired, (
        f"backstop rows with no route deadline to compare against: {unpaired}. "
        "Add the route's own constant to _route_own_deadlines()."
    )
    stale = sorted(set(own) - set(ROUTE_BACKSTOP_SECONDS))
    assert not stale, f"paired routes with no backstop row: {stale}"
    for path, backstop in ROUTE_BACKSTOP_SECONDS.items():
        assert backstop > own[path], (
            f"{path}: backstop {backstop}s must exceed the route's own "
            f"{own[path]}s deadline"
        )


def test_publish_interaction_dispatch_uses_the_backstop(monkeypatch):
    """``dispatch`` must USE the table, not merely define it beside the old code.

    Asserting on ``backstop_for()`` alone cannot catch a ``dispatch`` still
    running the inline ``REQUEST_TIMEOUT_SECONDS`` / ``SYNC_REQUEST_PATHS``
    logic while the table sits unused — and that absence IS the production bug.
    So this drives the real dispatch and reads the timeout ``asyncio.wait_for``
    was actually handed.
    """
    observed: dict[str, float | None] = {}

    async def fake_wait_for(awaitable, *, timeout=None):
        observed["timeout"] = timeout
        return await awaitable

    async def call_next(_request):
        from starlette.responses import Response

        return Response()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/api/publish_interaction",
            "raw_path": b"/api/publish_interaction",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )

    asyncio.run(TimeoutMiddleware(FastAPI()).dispatch(request, call_next))

    assert observed["timeout"] == ROUTE_BACKSTOP_SECONDS["/api/publish_interaction"]
    assert observed["timeout"] != REQUEST_TIMEOUT_SECONDS


def test_backstop_precedence_ignores_wait_for_response_for_table_paths():
    """``wait_for_response`` must not move publish onto a shorter budget.

    The route's own deadline covers the coverage wait too, so the publish
    backstop wins in both modes. Lose that precedence and an unwaited publish
    drops back to the 60s default — the original bug. The two non-table
    assertions pin the fallback rule the table must not have disturbed.
    """
    waited = backstop_for("/api/publish_interaction", wait_for_response=True)
    unwaited = backstop_for("/api/publish_interaction", wait_for_response=False)
    assert waited == unwaited == ROUTE_BACKSTOP_SECONDS["/api/publish_interaction"]
    assert backstop_for("/api/other", wait_for_response=False) == (
        REQUEST_TIMEOUT_SECONDS
    )
    assert backstop_for("/api/other", wait_for_response=True) == (
        SYNC_REQUEST_TIMEOUT_SECONDS
    )


def test_backstop_timeout_body_carries_a_correlation_id():
    """A generic 504 carrying no identifier is unactionable.

    This handler runs outside the route, so it cannot name a request_id; the
    correlation id is the only handle it has, and without one neither the
    client nor support can find the request. ``correlation_id_var`` defaults to
    ``""``, so the assertion is on a non-empty value — asserting the key is
    merely present would pass against an empty one.
    """
    import json

    async def call_next(_request):
        raise TimeoutError

    correlation_id_var.set("cid-under-test")
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/api/some_slow_route",
            "raw_path": b"/api/some_slow_route",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )

    response = asyncio.run(TimeoutMiddleware(FastAPI()).dispatch(request, call_next))

    assert response.status_code == 504
    body = json.loads(bytes(response.body))
    # Unchanged on purpose: existing clients read this exact string.
    assert body["detail"] == "Request timeout"
    assert body["correlation_id"] == "cid-under-test"
    assert body["reason"] == "backstop_timeout"
