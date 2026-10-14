"""Verify /healthz is wired up on the production FastAPI app factory."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reflexio.server.api import create_app


def test_create_app_exposes_healthz() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "pid" in response.json()


@pytest.mark.parametrize("path", ["/health/live", "/health/ready"])
@pytest.mark.parametrize("agent", ["curl/8.5.0", ""])
def test_probe_paths_answer_a_health_checker_on_the_full_app(
    path: str, agent: str
) -> None:
    """Health checkers send ``curl`` or no user agent; bot protection must not
    turn either probe into a 403 (the standalone app is ready, so 200)."""
    response = TestClient(create_app()).get(path, headers={"User-Agent": agent})
    assert response.status_code == 200
