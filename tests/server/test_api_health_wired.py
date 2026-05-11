"""Verify /healthz is wired up on the production FastAPI app factory."""

from __future__ import annotations

from fastapi.testclient import TestClient

from reflexio.server.api import create_app


def test_create_app_exposes_healthz() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "pid" in response.json()
