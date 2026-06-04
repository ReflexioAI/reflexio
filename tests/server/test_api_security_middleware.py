import asyncio

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from reflexio.server.api import BodySizeLimitMiddleware, create_app


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
            "Origin": "http://localhost:8082",
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
