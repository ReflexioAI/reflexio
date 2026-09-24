import asyncio
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from reflexio.server.api import create_app
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

    Read at call time rather than at import, so a monkeypatched constant is
    seen. Keys must match ROUTE_BACKSTOP_SECONDS exactly -- the guard below
    fails in BOTH directions, so neither table can grow a row alone.
    """
    from reflexio.server.routes import interactions

    return {
        "/api/publish_interaction": interactions.PUBLISH_REQUEST_TIMEOUT_SECONDS,
    }


def test_every_backstop_row_exceeds_its_routes_own_deadline():
    """The route must reach its own deadline before the middleware fires.

    This is the whole bug: a 60s middleware budget under a 240s route budget
    made the route's request_id-bearing exit unreachable.

    Iterating the table rather than indexing one hardcoded path is deliberate.
    middleware.py claims the ordering holds "for every path here"; a guard that
    read one path would let a second row ship unguarded under a comment that
    says it is covered -- the same shape as a check that cannot fail.
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


def test_the_publish_deadline_chain_is_strictly_ordered():
    """response deadline < worker deadline < middleware backstop.

    Enumerated as a SET in one place rather than as three scattered pairwise
    assertions, because an ordering invariant is only visible as a whole: the
    worker clock was added precisely because nobody had written down that the
    response deadline was doing two jobs.

    This asserts the DECLARED constants. The behavioural guard that the worker
    actually outlives the response is
    test_durable_window_pipeline.py::test_publish_past_deadline_commits_the_
    work_the_202_promised, which drives a real publish and reads the row back.

    Non-goal, stated beside the set so its absence is visible: the EDGE budgets
    are AWS configuration and cannot be asserted from here. Measured read-only
    on 2026-09-22, both sit BELOW the response deadline -- CloudFront
    E15WBN9QYYCSND alb-origin OriginReadTimeout 30s, agenticmem-alb
    idle_timeout.timeout_seconds 60s -- so the 202 exit does not currently
    reach a client through https://reflexio.ai/api/*. See the design's §2a.
    """
    from reflexio.server.routes import interactions

    response_deadline = interactions.PUBLISH_REQUEST_TIMEOUT_SECONDS
    worker_deadline = response_deadline + interactions.PUBLISH_WORKER_GRACE_SECONDS
    backstop = ROUTE_BACKSTOP_SECONDS["/api/publish_interaction"]

    assert worker_deadline > response_deadline, (
        "the worker must outlive the response, or the 202 says 'processing' "
        "about work whose next checkpoint aborts"
    )
    assert backstop > response_deadline, (
        "the middleware must not pre-empt the route's own exit"
    )


def test_slow_dependencies_do_not_let_the_backstop_pre_empt_the_route(monkeypatch):
    """The ordering above is only real if both clocks share an ORIGIN.

    The constants can be strictly ordered while the wall-clock deadlines are
    not. The backstop starts in ``TimeoutMiddleware.dispatch``; the route's own
    deadline could only start once FastAPI had parsed the body and resolved the
    sync auth/billing dependencies, which run in the threadpool and can queue.
    Burn the gap between the two constants there and the MIDDLEWARE expires
    first, answering its generic 504 -- no request_id, no retry instruction --
    for a publish the route was still shepherding.

    Driven behaviourally rather than by reading the source, and scaled down so
    it runs in under a second: backstop 1.0s, route deadline 0.5s, and a
    dependency that sleeps 0.7s.

    - Shared origin: the route aims at 0.5s, which has already passed when the
      handler starts at 0.7s, so ``acquire_ingestion`` refuses at once and the
      ROUTE answers -- a 503 whose detail carries the request_id. Refusing is
      the correct call: the budget really is spent, and nothing was admitted.
    - Independent clocks: the route aims at 0.7 + 0.5 = 1.2s, PAST the 1.0s
      backstop, so the MIDDLEWARE answers first with ``{"detail": "Request
      timeout"}`` -- no request_id, no retry instruction.

    The assertion is therefore on IDENTITY -- which layer produced the body --
    not on a status code, because both layers can answer 5xx.
    """
    import time as _time

    from reflexio.server.routes import interactions

    monkeypatch.setitem(ROUTE_BACKSTOP_SECONDS, "/api/publish_interaction", 1.0)
    monkeypatch.setattr(interactions, "PUBLISH_REQUEST_TIMEOUT_SECONDS", 0.5)

    def slow_org_id() -> str:
        # Stands in for auth/billing resolution queueing on the threadpool.
        _time.sleep(0.7)
        return "clock-origin-test"

    app = create_app(get_org_id=slow_org_id)
    client = TestClient(app, raise_server_exceptions=False)

    with patch(
        "reflexio.server.api_endpoints.publisher_api.add_user_interaction",
        side_effect=lambda **_: _time.sleep(5),
    ):
        response = client.post(
            "/api/publish_interaction",
            json={
                "user_id": "u",
                "session_id": "s",
                "interaction_data_list": [{"role": "User", "content": "hi"}],
            },
        )

    detail = response.json().get("detail")
    assert detail != "Request timeout", (
        "the middleware backstop pre-empted the route: the two publish clocks "
        f"are not measured from one arrival (got {response.status_code}: "
        f"{response.text[:200]})"
    )
    assert isinstance(detail, dict) and detail.get("request_id"), (
        "a route-owned exit must carry the request_id the generic backstop "
        f"cannot (got {response.status_code}: {response.text[:200]})"
    )


def test_publish_backstop_wins_over_the_default():
    assert backstop_for("/api/publish_interaction", wait_for_response=False) > (
        REQUEST_TIMEOUT_SECONDS
    )


def test_publish_backstop_does_not_shorten_the_coverage_wait():
    """wait_for_response=true must not land publish on a SHORTER budget.

    The route's own 240s deadline covers the coverage wait too, so the publish
    backstop has to sit above it in both modes. If this ever returns the
    default 60s, a waited publish is cut off mid-wait.
    """
    waited = backstop_for("/api/publish_interaction", wait_for_response=True)
    unwaited = backstop_for("/api/publish_interaction", wait_for_response=False)
    assert waited == unwaited == ROUTE_BACKSTOP_SECONDS["/api/publish_interaction"]


def test_publish_interaction_dispatch_uses_the_backstop(monkeypatch):
    """The middleware must actually USE the backstop table, not just define it.

    The three tests above only exercise backstop_for() and module-level
    constants — none of them calls TimeoutMiddleware.dispatch(), so none of
    them can catch a dispatch() that still runs the old inline REQUEST_
    TIMEOUT_SECONDS/SYNC_REQUEST_PATHS logic while ROUTE_BACKSTOP_SECONDS and
    backstop_for sit unused beside it. That absence is exactly the production
    bug: the middleware cutting a publish off at 60s.
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

    assert observed["timeout"] == 300.0


def test_backstop_timeout_body_carries_the_correlation_id(monkeypatch):
    """A 504 with no identifier is unactionable.

    The client cannot correlate it with anything, and support cannot find the
    request in logs. correlation_id_var is already set by the time the
    timeout fires.
    """
    import asyncio
    import json

    from fastapi import FastAPI
    from starlette.requests import Request

    from reflexio.server.correlation import correlation_id_var
    from reflexio.server.middleware import TimeoutMiddleware

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
    assert body["correlation_id"] == "cid-under-test"
    assert body["reason"] == "server_deadline"
