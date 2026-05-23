"""Smoke tests for the Braintrust connector endpoints.

Validates only that the routes are registered with the expected
shape — service-level behavior is covered by tests/server/services/braintrust/.
"""

from fastapi.testclient import TestClient

from reflexio.server.api import create_app


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_braintrust_status_returns_disconnected_on_fresh_app() -> None:
    """A fresh in-memory app reports `connected=False`."""
    response = _client().get(
        "/api/braintrust/status", headers={"User-Agent": "test"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["connected"] is False
    # Sensitive fields never appear
    assert "api_key" not in body
    assert "api_key_enc" not in body


def test_braintrust_sync_returns_failure_when_not_connected() -> None:
    """POST /sync against a disconnected org returns success=False with a message."""
    response = _client().post(
        "/api/braintrust/sync", headers={"User-Agent": "test"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "Not connected" in body["msg"]
    assert body["scored_count"] == 0


def test_braintrust_disconnect_returns_success() -> None:
    """DELETE on a disconnected org is a no-op that still returns success."""
    response = _client().delete(
        "/api/braintrust/connection", headers={"User-Agent": "test"}
    )
    assert response.status_code == 200
    assert response.json() == {"success": True}


def test_braintrust_connect_validates_api_key_field() -> None:
    """An empty api_key fails Pydantic validation (422)."""
    response = _client().post(
        "/api/braintrust/connect",
        json={"api_key": ""},
        headers={"User-Agent": "test"},
    )
    assert response.status_code == 422
