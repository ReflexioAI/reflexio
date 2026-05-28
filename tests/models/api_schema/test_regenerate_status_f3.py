"""Verify F3 informational fields on RegenerateStatusResponse."""

from typing import Any

from reflexio.models.api_schema.eval_overview_schema import (
    RegenerateStatusResponse,
)


def _minimal(**overrides: Any) -> RegenerateStatusResponse:
    base: dict[str, Any] = {
        "job_id": "j1",
        "status": "running",
        "total": 0,
        "completed": 0,
        "failed": 0,
        "failures": [],
        "started_at": 0.0,
        "finished_at": None,
    }
    base.update(overrides)
    return RegenerateStatusResponse(**base)


def test_total_candidates_defaults_to_zero():
    r = _minimal()
    assert r.total_candidates == 0


def test_sampled_count_defaults_to_zero():
    r = _minimal()
    assert r.sampled_count == 0


def test_concurrency_limit_defaults_to_zero():
    r = _minimal()
    assert r.concurrency_limit == 0


def test_informational_fields_round_trip():
    r = _minimal(total_candidates=6_200, sampled_count=2_000, concurrency_limit=10)
    re = RegenerateStatusResponse(**r.model_dump())
    assert re.total_candidates == 6_200
    assert re.sampled_count == 2_000
    assert re.concurrency_limit == 10
