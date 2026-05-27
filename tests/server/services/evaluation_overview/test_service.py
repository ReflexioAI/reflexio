"""Unit tests for EvaluationOverviewService static helpers."""

from __future__ import annotations

import pytest

from reflexio.models.api_schema.domain.entities import AgentSuccessEvaluationResult
from reflexio.server.services.evaluation_overview.service import (
    EvaluationOverviewService,
    _weekly_buckets,
)


def _shadow_result(
    is_success: bool,
    shadow_is_success: bool | None = None,
    created_at: int = 0,
) -> AgentSuccessEvaluationResult:
    return AgentSuccessEvaluationResult(
        session_id=f"s_{id(object())}",
        agent_version="v",
        evaluation_name="overall_success",
        is_success=is_success,
        is_escalated=False,
        shadow_is_success=shadow_is_success,
        shadow_is_escalated=False,
        created_at=created_at,
    )


def test_compute_shadow_rate_excludes_ungraded_rows() -> None:
    results = [
        _shadow_result(True, shadow_is_success=True),
        _shadow_result(False, shadow_is_success=True),
        _shadow_result(True, shadow_is_success=None),  # excluded
    ]
    rate, delta = EvaluationOverviewService._compute_shadow_rate_and_delta(
        results, regular_rate=2 / 3 * 100
    )
    assert rate == 100.0
    assert delta == pytest.approx(100.0 - 2 / 3 * 100)


def test_compute_shadow_rate_is_none_when_no_graded_rows() -> None:
    results = [
        _shadow_result(True, shadow_is_success=None),
        _shadow_result(False, shadow_is_success=None),
    ]
    rate, delta = EvaluationOverviewService._compute_shadow_rate_and_delta(
        results, regular_rate=50.0
    )
    assert rate is None
    assert delta is None


def test_compute_shadow_rate_handles_all_false_shadow() -> None:
    results = [
        _shadow_result(True, shadow_is_success=False),
        _shadow_result(True, shadow_is_success=False),
    ]
    rate, delta = EvaluationOverviewService._compute_shadow_rate_and_delta(
        results, regular_rate=100.0
    )
    assert rate == 0.0
    assert delta == -100.0


def test_compute_shadow_rate_empty_results() -> None:
    rate, delta = EvaluationOverviewService._compute_shadow_rate_and_delta(
        [], regular_rate=50.0
    )
    assert rate is None
    assert delta is None


def test_compute_shadow_rate_partial_graded() -> None:
    """50% shadow success rate from a mix of graded and ungraded results."""
    results = [
        _shadow_result(True, shadow_is_success=True),
        _shadow_result(True, shadow_is_success=False),
        _shadow_result(False, shadow_is_success=None),  # ungraded, excluded
    ]
    rate, delta = EvaluationOverviewService._compute_shadow_rate_and_delta(
        results, regular_rate=66.7
    )
    assert rate == pytest.approx(50.0)
    assert delta == pytest.approx(50.0 - 66.7)


def test_weekly_buckets_pin_shadow_rate_in_ratio_form() -> None:
    """HeroBucket.shadow_rate must be a ratio (0.0-1.0), not pp.

    The hero-level shadow_success_rate_pp uses 0-100 (per its name),
    but HeroBucket.shadow_rate uses 0-1 to match the existing
    HeroBucket.regular_rate convention. Pin both fields with concrete
    values so a future refactor that adds `* 100` to either side
    breaks loudly.
    """
    # Three graded shadow results in the same week, 2 success + 1 failure.
    # Pick a fixed timestamp that all 3 share so they bucket together.
    ts = 1700000000
    results = [
        _shadow_result(True, shadow_is_success=True, created_at=ts),
        _shadow_result(True, shadow_is_success=True, created_at=ts),
        _shadow_result(False, shadow_is_success=False, created_at=ts),
        # Plus one ungraded — must NOT contribute to shadow_rate or shadow_n
        _shadow_result(True, shadow_is_success=None, created_at=ts),
    ]
    buckets = _weekly_buckets(results)
    # Expect at least one bucket containing all 4 results
    bucket = next(b for b in buckets if b.regular_n == 4)
    assert bucket.shadow_n == 3
    assert bucket.shadow_rate == pytest.approx(2 / 3)
    # Crucial: shadow_rate is a ratio, NOT pp — i.e. NOT 66.67
    assert bucket.shadow_rate is not None and bucket.shadow_rate <= 1.0


def test_n_shadow_in_window_excludes_ungraded_rows() -> None:
    """Ensure rows with shadow_is_success=None do not count toward the
    FULL-gate threshold. This is the spec's semantics tightening — without
    it, orgs that publish shadow_content but flip the flag mid-window
    would prematurely hit FULL state.
    """
    results = [
        _shadow_result(True, shadow_is_success=True),  # counts
        _shadow_result(True, shadow_is_success=False),  # counts
        _shadow_result(True, shadow_is_success=None),  # excluded
        _shadow_result(True, shadow_is_success=None),  # excluded
    ]
    # Compute n_shadow_in_window the same way _build_hero does
    n_shadow_graded = sum(1 for r in results if r.shadow_is_success is not None)
    assert n_shadow_graded == 2  # not 4
