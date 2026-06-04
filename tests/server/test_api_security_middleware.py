from fastapi.testclient import TestClient

from reflexio.server.api import create_app


def test_cors_uses_frontend_url_allowlist(monkeypatch):
    monkeypatch.delenv("REFLEXIO_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("FRONTEND_URL", "https://app.example.com")

    client = TestClient(create_app())

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

    client = TestClient(create_app())
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


def test_body_size_limit_rejects_large_declared_body(monkeypatch):
    monkeypatch.setenv("REFLEXIO_MAX_BODY_BYTES", "4")

    client = TestClient(create_app())
    response = client.post("/", content=b"12345")

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}


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
