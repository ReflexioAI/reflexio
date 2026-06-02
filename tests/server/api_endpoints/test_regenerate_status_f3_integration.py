"""Verify the regenerate status endpoint surfaces F3 counters.

Tasks 3+4 added ``total_candidates``, ``sampled_count`` and
``concurrency_limit`` on both ``RegenerateStatusResponse`` (response
schema) and ``RegenJob`` (dataclass). Task 7 is responsible for wiring
them together inside ``get_regenerate_status`` so the values populated
by the worker (or pre-populated for tests) actually reach API clients.

This integration test exercises the full FastAPI route — it inserts a
``RegenJob`` directly into the process-local registry (the same store
the handler reads from), then issues a GET against the status endpoint
and asserts the F3 fields round-trip.
"""

from __future__ import annotations

import pytest

from reflexio.server.services.agent_success_evaluation.regen_jobs import (
    REGEN_JOBS,
    RegenJob,
)

pytestmark = pytest.mark.integration


def test_status_endpoint_surfaces_f3_counters(client_with_org):
    """A RegenJob with populated F3 counters round-trips via the
    GET /api/evaluations/regenerate/{job_id} endpoint.

    Pre-populates the registry with a job owned by the test org so the
    handler's org-scoping check passes, then asserts the three F3
    fields appear in the JSON response with the values we set.
    """
    client, org_id = client_with_org

    # Insert the job manually so we don't need a live worker — Task 7 is
    # specifically about the handler -> response conversion, not the
    # write side of the registry.
    job = RegenJob(
        job_id="f3-status-roundtrip",
        org_id=org_id,
        from_ts=0,
        to_ts=9_999_999_999,
        status="completed",
        total=2_000,
        completed=2_000,
        failed=0,
        total_candidates=6_200,
        sampled_count=2_000,
        concurrency_limit=10,
    )
    REGEN_JOBS._jobs[job.job_id] = job  # noqa: SLF001 — test seam
    try:
        resp = client.get(f"/api/evaluations/regenerate/{job.job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total_candidates"] == 6_200
        assert body["sampled_count"] == 2_000
        assert body["concurrency_limit"] == 10
        # Existing counters still surface unchanged.
        assert body["total"] == 2_000
        assert body["completed"] == 2_000
        assert body["failed"] == 0
        assert body["status"] == "completed"
    finally:
        REGEN_JOBS._jobs.pop(job.job_id, None)  # noqa: SLF001 — test seam


def test_status_endpoint_defaults_when_f3_counters_unset(client_with_org):
    """A RegenJob created without F3 fields exposes zero-valued defaults.

    Guards backwards-compatibility for in-flight jobs that started before
    the worker began populating these counters (or for the very brief
    window inside ``run_regen`` before sampling completes).
    """
    client, org_id = client_with_org

    job = RegenJob(
        job_id="f3-status-defaults",
        org_id=org_id,
        from_ts=0,
        to_ts=9_999_999_999,
        status="running",
        total=0,
    )
    REGEN_JOBS._jobs[job.job_id] = job  # noqa: SLF001 — test seam
    try:
        resp = client.get(f"/api/evaluations/regenerate/{job.job_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total_candidates"] == 0
        assert body["sampled_count"] == 0
        assert body["concurrency_limit"] == 0
    finally:
        REGEN_JOBS._jobs.pop(job.job_id, None)  # noqa: SLF001 — test seam
