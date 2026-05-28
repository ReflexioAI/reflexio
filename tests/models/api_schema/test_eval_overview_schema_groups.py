"""Tests for F2's new TrendPoint, SuccessRateTrendByGroup, and the
extended GetEvaluationOverviewResponse field."""

from reflexio.models.api_schema.eval_overview_schema import (
    ContextTile,
    GetEvaluationOverviewResponse,
    HeroBlock,
    NumberWithDelta,
    PercentWithDelta,
    ScoreDistribution,
    SuccessRateTrendByGroup,
    TrendPoint,
)


def _minimal_response_kwargs() -> dict:
    """Build a minimal valid response payload kwargs dict to test that
    success_rate_trend_by_group is optional with a sensible default.

    Returns:
        dict: kwargs ready to splat into GetEvaluationOverviewResponse(...).
    """
    return {
        "hero": HeroBlock(
            state="empty",
            regular_success_rate_pp=0.0,
            shadow_success_rate_pp=None,
            delta_pp=None,
            buckets=[],
        ),
        "context_tiles": ContextTile(
            success=PercentWithDelta(current=0.0, delta_pp=0.0),
            corrections=NumberWithDelta(current=0.0, delta=0.0),
            turns=NumberWithDelta(current=0.0, delta=0.0),
            escalation=PercentWithDelta(current=0.0, delta_pp=0.0),
        ),
        "rule_attribution": [],
        "score_distribution": ScoreDistribution(
            current_bins=[], baseline_bins=[], labels=[]
        ),
    }


def test_trend_point_shape():
    p = TrendPoint(ts=1_700_000_000, rate=0.72, n=900)
    assert p.ts == 1_700_000_000
    assert p.rate == 0.72
    assert p.n == 900


def test_trend_point_n_must_be_non_negative():
    """n is a session count — negative values are nonsensical."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrendPoint(ts=1, rate=0.5, n=-1)


def test_success_rate_trend_by_group_defaults_to_empty_arrays():
    g = SuccessRateTrendByGroup()
    assert g.treatment == []
    assert g.control == []
    assert g.untagged == []


def test_overview_response_defaults_to_empty_group_trend():
    """Existing customers (who don't send anything new) get backward-compatible
    empty arrays — never None, never missing field."""
    r = GetEvaluationOverviewResponse(**_minimal_response_kwargs())
    assert r.success_rate_trend_by_group.treatment == []
    assert r.success_rate_trend_by_group.control == []
    assert r.success_rate_trend_by_group.untagged == []


def test_overview_response_carries_provided_group_trend():
    payload = _minimal_response_kwargs()
    payload["success_rate_trend_by_group"] = SuccessRateTrendByGroup(
        treatment=[TrendPoint(ts=1, rate=0.7, n=10)],
        control=[TrendPoint(ts=1, rate=0.5, n=5)],
        untagged=[],
    )
    r = GetEvaluationOverviewResponse(**payload)
    assert len(r.success_rate_trend_by_group.treatment) == 1
    assert r.success_rate_trend_by_group.treatment[0].rate == 0.7
    assert len(r.success_rate_trend_by_group.control) == 1
    assert r.success_rate_trend_by_group.untagged == []
