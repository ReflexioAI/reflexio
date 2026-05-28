"""Verify RegenJob's new F3 counters."""

from typing import Any

from reflexio.server.services.agent_success_evaluation.regen_jobs import (
    RegenJob,
)


def _minimal_job(**overrides: Any) -> RegenJob:
    """Build a RegenJob using the discovered required-field set.

    Required fields on RegenJob: job_id, org_id, evaluation_name, from_ts,
    to_ts, status, total. Everything else has a default.
    """
    base: dict[str, Any] = {
        "job_id": "j1",
        "org_id": "o",
        "evaluation_name": "e",
        "from_ts": 0,
        "to_ts": 1,
        "status": "running",
        "total": 0,
    }
    base.update(overrides)
    return RegenJob(**base)


def test_regen_job_has_total_candidates_default_zero() -> None:
    assert _minimal_job().total_candidates == 0


def test_regen_job_has_sampled_count_default_zero() -> None:
    assert _minimal_job().sampled_count == 0


def test_regen_job_has_concurrency_limit_default_zero() -> None:
    assert _minimal_job().concurrency_limit == 0


def test_regen_job_accepts_explicit_values() -> None:
    job = _minimal_job(
        total_candidates=6_200,
        sampled_count=2_000,
        concurrency_limit=10,
    )
    assert job.total_candidates == 6_200
    assert job.sampled_count == 2_000
    assert job.concurrency_limit == 10
