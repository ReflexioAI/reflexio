"""``/health`` warm-before-ready behaviour and its dormancy anti-regression.

The 200 path is the ALB + container health target and MUST stay byte-for-byte
identical to the pre-Phase-2 response whenever the in-process gate is inactive.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reflexio.server.api import create_app
from reflexio.server.llm.providers.embedder_warmup import (
    mark_embedder_ready,
    reset_warmup_state_for_test,
)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    reset_warmup_state_for_test()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health_200_unchanged_when_gate_inactive(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default (no provider) → exactly the historical 200 body.

    This is the dormancy anti-regression: a change that makes the pre-flip
    ``/health`` anything other than ``200 {"status": "healthy"}`` fails here.
    """
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.parametrize("mode", ["cloud", "off"])
def test_health_200_when_non_inprocess_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """A non-inprocess provider keeps /health on the unchanged 200 path."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", mode)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_503_when_gate_active_and_not_warm(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate active + embedder not yet loaded → 503 not-ready."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "starting"}


def test_health_200_after_ready_event_set(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the embedder is warm, /health returns to the unchanged 200 body."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    assert client.get("/health").status_code == 503

    mark_embedder_ready()  # simulate warmup completion (no real model loaded)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
