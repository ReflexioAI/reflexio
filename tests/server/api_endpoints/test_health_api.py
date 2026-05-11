"""Tests for the /healthz endpoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reflexio.server.api_endpoints import health_api


def _build_app() -> FastAPI:
    app = FastAPI()
    health_api.install(app)
    return app


def test_healthz_returns_required_fields() -> None:
    app = _build_app()
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) >= {"pid", "uptime_sec", "request_count"}
    assert isinstance(body["pid"], int)
    assert isinstance(body["uptime_sec"], float)
    assert body["uptime_sec"] >= 0.0
    assert isinstance(body["request_count"], int)


def test_healthz_request_count_increments() -> None:
    app = _build_app()
    client = TestClient(app)
    first = client.get("/healthz").json()["request_count"]
    second = client.get("/healthz").json()["request_count"]
    assert second > first


def test_healthz_rss_mb_optional() -> None:
    """rss_mb is present when psutil is available; None otherwise."""
    app = _build_app()
    client = TestClient(app)
    body = client.get("/healthz").json()
    assert "rss_mb" in body
    if body["rss_mb"] is not None:
        assert isinstance(body["rss_mb"], float)
        assert body["rss_mb"] > 0.0
