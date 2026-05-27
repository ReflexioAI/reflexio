"""Unit tests for EvaluationOverviewService static helpers."""

from __future__ import annotations

import pytest

from reflexio.models.api_schema.domain.entities import AgentSuccessEvaluationResult
from reflexio.server.services.evaluation_overview.service import (
    EvaluationOverviewService,
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
