"""Lifecycle contract tests for retrieval experiments."""

from uuid import uuid4

from fastapi.testclient import TestClient

from reflexio.server.api import create_app
from reflexio.server.cache.reflexio_cache import invalidate_reflexio_cache


def test_experiment_lifecycle_keeps_history_and_enforces_single_active() -> None:
    org_id = f"retrieval-experiment-{uuid4().hex}"
    client = TestClient(create_app(get_org_id=lambda: org_id))
    try:
        started = client.post(
            "/api/retrieval_experiments",
            json={"experiment_id": "exp-1", "holdout_percentage": 20},
        )
        assert started.status_code == 200, started.text
        assert started.json()["active_experiment"]["experiment_id"] == "exp-1"

        concurrent = client.post(
            "/api/retrieval_experiments",
            json={"experiment_id": "exp-2", "holdout_percentage": 20},
        )
        assert concurrent.status_code == 409, concurrent.text

        stopped = client.post(
            "/api/retrieval_experiments/stop",
            json={"experiment_id": "exp-1"},
        )
        assert stopped.status_code == 200, stopped.text
        assert "active_experiment" not in stopped.json()
        assert (
            stopped.json()["experiments"][0]["ended_at"]
            >= stopped.json()["experiments"][0]["started_at"]
        )

        reused = client.post(
            "/api/retrieval_experiments",
            json={"experiment_id": "exp-1", "holdout_percentage": 30},
        )
        assert reused.status_code == 409, reused.text

        results = client.get("/api/retrieval_experiments/exp-1/results")
        assert results.status_code == 200, results.text
        assert results.json()["experiment"]["experiment_id"] == "exp-1"
        assert results.json()["treatment"]["evaluated_session_count"] == 0
        assert results.json()["holdout"]["evaluated_session_count"] == 0
    finally:
        invalidate_reflexio_cache(org_id=org_id)
