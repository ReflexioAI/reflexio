"""Verify POST /api/get_evaluation_overview wires through to the service."""

from fastapi.testclient import TestClient

from reflexio.server.api import create_app


def test_endpoint_returns_empty_state_on_fresh_app() -> None:
    """A fresh in-memory app with no evaluations returns the empty hero state."""
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/get_evaluation_overview",
        json={"from_ts": 0, "to_ts": 1_000_000_000_000, "bucket": "week"},
        headers={"User-Agent": "test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["hero"]["state"] in ("empty", "shadow_off")
    assert "context_tiles" in body
    assert "rule_attribution" in body
    assert body["score_distribution"]["labels"] == ["0", "1", "2", "3", "4", "5+"]
    assert isinstance(body["source_set_comparison"]["available_sources"], list)
    assert body["source_set_comparison"]["sets"] == []
    assert body["source_set_comparison"]["unmatched_session_count"] == 0
