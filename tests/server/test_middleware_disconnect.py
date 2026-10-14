"""Client disconnects must not surface as ``RuntimeError: No response returned.``

On ``POST /api/search`` a client that hung up while the endpoint was still
working made ``SearchRuntimeMiddleware`` return without sending
``http.response.start`` (deliberately -- nobody is listening). Any
Starlette ``BaseHTTPMiddleware`` stacked above it then raised
``RuntimeError("No response returned.")`` from ``call_next``, logged as a 500.
These tests drive the real ``create_app`` middleware stack with raw ASGI so a
disconnect is delivered exactly as uvicorn delivers one.
"""

import asyncio
import threading

import pytest

from reflexio.server import search_runtime as runtime
from reflexio.server.api import create_app


def _scope(path: str, method: str = "POST", headers=None) -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "headers": headers
        if headers is not None
        else [(b"user-agent", b"pytest-agent"), (b"content-type", b"application/json")],
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("127.0.0.1", 1),
        "http_version": "1.1",
    }


async def _drain_search_tasks(app) -> None:
    middleware = app.middleware_stack
    while middleware is not None and not isinstance(
        middleware, runtime.SearchRuntimeMiddleware
    ):
        middleware = getattr(middleware, "app", None)
    if middleware is not None:
        await asyncio.gather(*middleware.tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_search_client_disconnect_does_not_raise_no_response_returned():
    app = create_app(mount_data_plane=False)
    started = threading.Event()
    release = threading.Event()

    @app.post("/api/search")
    def search():
        started.set()
        release.wait(3)
        return {"ok": True}

    disconnected = asyncio.Event()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    request = asyncio.create_task(app(_scope("/api/search"), receive, send))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        disconnected.set()
        # Before the fix this raised RuntimeError("No response returned.").
        await asyncio.wait_for(request, timeout=5)
        assert sent == []
    finally:
        release.set()
        await _drain_search_tasks(app)


def _client_app():
    from fastapi import Request
    from fastapi.testclient import TestClient

    from reflexio.server.correlation import correlation_id_var

    app = create_app(mount_data_plane=False)

    @app.get("/")
    @app.get("/probe")
    def probe(request: Request):
        return {
            "cid": correlation_id_var.get(),
            "tp": [
                getattr(request.state, name, "missing")
                for name in ("tp_borrowed", "tp_total", "tp_waiting")
            ],
        }

    return TestClient(app)


def test_normal_response_keeps_security_headers_correlation_id_and_tp_state():
    with _client_app() as client:  # also drives lifespan through the stack
        plain = client.get("/probe", headers={"User-Agent": "pytest-agent"})
        https = client.get(
            "/probe",
            headers={"User-Agent": "pytest-agent", "X-Forwarded-Proto": "HTTPS"},
        )

    assert plain.status_code == 200
    for response in (plain, https):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
        # The id the route logged under is the id the client is handed.
        assert response.headers["x-correlation-id"] == response.json()["cid"]
        borrowed, total, waiting = response.json()["tp"]
        assert isinstance(borrowed, int) and isinstance(waiting, int)
        assert total > 0
    assert "strict-transport-security" not in plain.headers
    assert https.headers["strict-transport-security"] == (
        "max-age=31536000; includeSubDomains"
    )
    assert plain.headers["x-correlation-id"] != https.headers["x-correlation-id"]


@pytest.mark.parametrize(
    ("headers", "path", "status", "detail"),
    [
        ({"User-Agent": ""}, "/probe", 403, "Forbidden: Missing user agent"),
        ({"User-Agent": "curl/8.0"}, "/probe", 403, "Forbidden: Suspicious user agent"),
        (
            {"User-Agent": "GoogleBot"},
            "/probe",
            403,
            "Forbidden: Suspicious user agent",
        ),
        ({"User-Agent": ""}, "/", 200, None),
        # Health checks hit the probe paths with curl.
        ({"User-Agent": "curl/8.0"}, "/health/live", 200, None),
        ({"User-Agent": ""}, "/health/live", 200, None),
        ({"User-Agent": "curl/8.0"}, "/healthz", 200, None),
        ({"User-Agent": ""}, "/healthz", 200, None),
        # The exemption is an exact path match, not a prefix.
        (
            {"User-Agent": "curl/8.0"},
            "/healthz/eval",
            403,
            "Forbidden: Suspicious user agent",
        ),
        ({"User-Agent": "pytest-agent"}, "/probe", 200, None),
    ],
)
def test_bot_protection_rejections_are_unchanged(headers, path, status, detail):
    with _client_app() as client:
        response = client.get(path, headers=headers)

    assert response.status_code == status
    if detail is not None:
        assert response.json() == {"detail": detail}
    # Rejections still pass back out through the outer middleware.
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-correlation-id"]
