"""Unit tests for the pure F2 group aggregation helpers."""

import math

import pytest

from reflexio.models.api_schema.eval_overview_schema import TrendPoint
from reflexio.server.services.evaluation_overview.group_aggregation import (
    GroupAssignment,
    assign_group_from_metadata,
    bucket_outcomes,
    compute_lift_with_ci,
    compute_trend_by_group,
    group_session_outcomes_by_metadata,
)

# --- assign_group_from_metadata -------------------------------------


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"reflexio_retrieval_enabled": True}, GroupAssignment.TREATMENT),
        ({"reflexio_retrieval_enabled": False}, GroupAssignment.CONTROL),
        ({}, GroupAssignment.UNTAGGED),
        ({"other_key": "v"}, GroupAssignment.UNTAGGED),
        ({"reflexio_retrieval_enabled": "true"}, GroupAssignment.UNTAGGED),
        ({"reflexio_retrieval_enabled": 1}, GroupAssignment.UNTAGGED),
        ({"reflexio_retrieval_enabled": None}, GroupAssignment.UNTAGGED),
    ],
)
def test_assign_group_from_metadata(metadata, expected):
    assert assign_group_from_metadata(metadata) == expected


def test_assign_group_from_metadata_non_dict():
    """Defensive: non-dict input maps to UNTAGGED rather than raising."""
    assert assign_group_from_metadata(None) == GroupAssignment.UNTAGGED  # type: ignore[arg-type]
    assert assign_group_from_metadata("not a dict") == GroupAssignment.UNTAGGED  # type: ignore[arg-type]


# --- group_session_outcomes_by_metadata ----------------------------


def test_group_outcomes_buckets_by_first_request_metadata():
    """The grouped dict always has 3 keys, even when one bucket is empty."""
    outcomes = [
        ("s1", 1000, True, {"reflexio_retrieval_enabled": True}),
        ("s2", 1000, False, {"reflexio_retrieval_enabled": True}),
        ("s3", 1000, True, {"reflexio_retrieval_enabled": False}),
        ("s4", 1000, False, {}),
    ]
    groups = group_session_outcomes_by_metadata(outcomes)
    assert {o[0] for o in groups[GroupAssignment.TREATMENT]} == {"s1", "s2"}
    assert {o[0] for o in groups[GroupAssignment.CONTROL]} == {"s3"}
    assert {o[0] for o in groups[GroupAssignment.UNTAGGED]} == {"s4"}


def test_group_outcomes_always_has_all_three_keys():
    """Even when input is empty, the output has all 3 group keys present."""
    groups = group_session_outcomes_by_metadata([])
    assert set(groups.keys()) == set(GroupAssignment)


# --- bucket_outcomes -----------------------------------------------


def test_bucket_outcomes_groups_by_week():
    outcomes = [
        ("a", 1_700_000_000, True),
        ("b", 1_700_001_000, False),
        ("c", 1_700_700_000, True),
    ]
    buckets = bucket_outcomes(outcomes, bucket_seconds=7 * 24 * 3600)
    assert len(buckets) == 2
    assert all("ts" in b and "n" in b and "successes" in b for b in buckets)
    assert sum(b["n"] for b in buckets) == 3
    # Buckets must be sorted by ts ascending.
    assert buckets[0]["ts"] < buckets[1]["ts"]


def test_bucket_outcomes_empty_input_returns_empty_list():
    assert bucket_outcomes([], bucket_seconds=86400) == []


def test_bucket_outcomes_rejects_non_positive_bucket_seconds():
    with pytest.raises(ValueError):
        bucket_outcomes([("s", 1, True)], bucket_seconds=0)
    with pytest.raises(ValueError):
        bucket_outcomes([("s", 1, True)], bucket_seconds=-10)


def test_bucket_outcomes_aligns_to_bucket_boundary():
    """ts // bucket_seconds * bucket_seconds → bucket start is aligned."""
    bucket_seconds = 86400
    outcomes = [("a", bucket_seconds + 5, True), ("b", bucket_seconds + 100, False)]
    buckets = bucket_outcomes(outcomes, bucket_seconds=bucket_seconds)
    assert len(buckets) == 1
    assert buckets[0]["ts"] == bucket_seconds  # aligned start


# --- compute_trend_by_group ----------------------------------------


def test_compute_trend_by_group_returns_three_curves():
    outcomes = [
        ("s1", 1_700_000_000, True, {"reflexio_retrieval_enabled": True}),
        ("s2", 1_700_000_000, True, {"reflexio_retrieval_enabled": True}),
        ("s3", 1_700_000_000, False, {"reflexio_retrieval_enabled": True}),
        ("s4", 1_700_000_000, True, {"reflexio_retrieval_enabled": False}),
        ("s5", 1_700_000_000, False, {"reflexio_retrieval_enabled": False}),
        ("s6", 1_700_000_000, True, {}),
    ]
    trend = compute_trend_by_group(outcomes, bucket_seconds=7 * 24 * 3600)

    # Treatment: 2/3 success
    assert len(trend.treatment) == 1
    assert trend.treatment[0].n == 3
    assert abs(trend.treatment[0].rate - (2 / 3)) < 1e-9

    # Control: 1/2 success
    assert len(trend.control) == 1
    assert trend.control[0].n == 2
    assert trend.control[0].rate == 0.5

    # Untagged: 1/1 success
    assert len(trend.untagged) == 1
    assert trend.untagged[0].n == 1
    assert trend.untagged[0].rate == 1.0


def test_compute_trend_by_group_empty_input():
    trend = compute_trend_by_group([], bucket_seconds=86400)
    assert trend.treatment == []
    assert trend.control == []
    assert trend.untagged == []


def test_compute_trend_by_group_omits_empty_buckets():
    """A bucket with n=0 should not appear in the trend point list."""
    # All in treatment, in a single bucket; no control sessions at all.
    outcomes = [
        ("s1", 1_700_000_000, True, {"reflexio_retrieval_enabled": True}),
    ]
    trend = compute_trend_by_group(outcomes, bucket_seconds=86400)
    assert len(trend.treatment) == 1
    assert trend.control == []  # no empty bucket entry
    assert trend.untagged == []


def test_compute_trend_by_group_trend_points_are_pydantic():
    """TrendPoint instances are returned, not raw dicts."""
    outcomes = [
        ("s1", 1_700_000_000, True, {"reflexio_retrieval_enabled": True}),
    ]
    trend = compute_trend_by_group(outcomes, bucket_seconds=86400)
    assert isinstance(trend.treatment[0], TrendPoint)


# --- compute_lift_with_ci ------------------------------------------


def test_lift_basic():
    """+14pp lift with reasonable n, finite CI."""
    lift, ci_pp = compute_lift_with_ci(n_t=900, p_t=0.72, n_c=100, p_c=0.58)
    assert lift is not None
    assert abs(lift - 0.14) < 1e-9
    assert ci_pp is not None
    assert ci_pp > 0  # nonzero CI half-width
    # Bounded — dominated by the smaller control arm (n=100), CI ≈ ±10pp.
    assert ci_pp < 0.15


def test_lift_zero_control_returns_none_pair():
    lift, ci_pp = compute_lift_with_ci(n_t=900, p_t=0.72, n_c=0, p_c=0.0)
    assert lift is None
    assert ci_pp is None


def test_lift_zero_treatment_returns_none_pair():
    lift, ci_pp = compute_lift_with_ci(n_t=0, p_t=0.0, n_c=100, p_c=0.58)
    assert lift is None
    assert ci_pp is None


def test_lift_n_eq_1_each_side_returns_finite_lift():
    """Edge case n=1 each side — lift computable; CI very wide but finite."""
    lift, ci_pp = compute_lift_with_ci(n_t=1, p_t=1.0, n_c=1, p_c=0.0)
    assert lift == 1.0
    assert ci_pp is not None
    assert math.isfinite(ci_pp)


def test_lift_negative_when_control_outperforms():
    lift, _ = compute_lift_with_ci(n_t=100, p_t=0.5, n_c=100, p_c=0.6)
    assert lift is not None
    assert abs(lift - (-0.1)) < 1e-9


def test_lift_ci_is_capped_at_50pp_for_tiny_samples():
    """With n=2 each side and opposite-extreme rates, the raw Wald half-width
    exceeds 0.5; the cap clamps it to 0.5 so the dashboard doesn't show false
    precision.
    """
    # Raw Wald CI for n_t=n_c=2, p_t=1.0, p_c=0.0:
    #   var = 1*0/2 + 0*1/2 = 0 → ci_half = 0, which is fine (not capped).
    # Use a case where var > 0 but n is tiny:
    #   n_t=n_c=2, p_t=0.5, p_c=0.5 → var = 0.25/2 + 0.25/2 = 0.25
    #   sqrt(0.25) = 0.5; 1.96 * 0.5 = 0.98 → would render ±98pp without the cap.
    lift, ci_pp = compute_lift_with_ci(n_t=2, p_t=0.5, n_c=2, p_c=0.5)
    assert lift == 0.0
    assert ci_pp == 0.5  # clamped from ~0.98 to the cap


def test_lift_ci_uncapped_when_below_threshold():
    """Sanity-check that the cap only fires when raw CI exceeds 0.5."""
    lift, ci_pp = compute_lift_with_ci(n_t=900, p_t=0.72, n_c=100, p_c=0.58)
    # Raw CI ≈ 0.1011 — well below the 0.5 cap, so returned as-is.
    assert lift is not None
    assert ci_pp is not None
    assert 0.10 < ci_pp < 0.15
    assert ci_pp < 0.5  # uncapped
