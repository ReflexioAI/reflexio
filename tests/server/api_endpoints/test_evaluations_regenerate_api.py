"""Integration tests for the three regenerate endpoints.

The fixtures (see conftest.py) wire a fresh-per-test FastAPI app to a
unique SQLite-backed Reflexio org and tear down both the cache entry
and any registry state created during the test.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration


def test_post_accepts_deprecated_evaluation_name(client_with_org):
    """evaluation_name is a deprecated, accepted-but-ignored compatibility input.

    Evaluation is singleton (one evaluator per org), so an arbitrary name no
    longer triggers name-based validation — the request is accepted.
    """
    client, _org_id = client_with_org
    resp = client.post(
        "/api/evaluations/regenerate",
        json={
            "evaluation_name": "does_not_exist",
            "from_ts": 0,
            "to_ts": 9_999_999_999,
        },
    )
    assert resp.status_code == 200


def test_get_unknown_job_id_returns_404(client_with_org):
    """GET on an unknown job_id returns 404."""
    client, _ = client_with_org
    resp = client.get("/api/evaluations/regenerate/nonexistent")
    assert resp.status_code == 404


def test_post_409_when_job_already_running(client_with_org_and_evaluator):
    """A second POST while a job is still running returns 409."""
    client, _ = client_with_org_and_evaluator
    body = {
        "evaluation_name": "overall_success",
        "from_ts": 0,
        "to_ts": 9_999_999_999,
    }
    # Patch threading.Thread so the worker never starts and the first job
    # stays "running" — otherwise the empty-storage worker would finish
    # immediately and the second POST would succeed.
    with patch("reflexio.server.api.threading.Thread") as thread_cls:
        thread_cls.return_value.start = lambda: None
        first = client.post("/api/evaluations/regenerate", json=body)
        assert first.status_code == 200, first.text
        second = client.post("/api/evaluations/regenerate", json=body)
        assert second.status_code == 409


def test_full_lifecycle_running_then_completed(client_with_org_and_evaluator):
    """POST then poll GET until status transitions out of "running"."""
    client, _ = client_with_org_and_evaluator
    body = {
        "evaluation_name": "overall_success",
        "from_ts": 0,
        "to_ts": 9_999_999_999,
    }
    resp = client.post("/api/evaluations/regenerate", json=body)
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    deadline = time.time() + 30
    last_status = None
    while time.time() < deadline:
        r = client.get(f"/api/evaluations/regenerate/{job_id}")
        assert r.status_code == 200
        last_status = r.json()["status"]
        if last_status in ("completed", "cancelled", "error"):
            break
        time.sleep(0.3)
    # Empty storage → no descriptors → worker exits cleanly via the for-else.
    assert last_status == "completed"


def test_delete_cancels_running_job(client_with_org_and_evaluator):
    """DELETE sets the cancel flag and returns {"status": "cancelled"}."""
    client, _ = client_with_org_and_evaluator
    body = {
        "evaluation_name": "overall_success",
        "from_ts": 0,
        "to_ts": 9_999_999_999,
    }
    with patch("reflexio.server.api.threading.Thread") as thread_cls:
        thread_cls.return_value.start = lambda: None
        resp = client.post("/api/evaluations/regenerate", json=body)
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

        delete_resp = client.delete(f"/api/evaluations/regenerate/{job_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["status"] == "cancelled"
