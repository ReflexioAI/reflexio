"""Unit tests for F1's shadow_win_rate_trend aggregator (pure)."""

from __future__ import annotations

from datetime import UTC, datetime

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.server.services.evaluation_overview.components.shadow_aggregation import (
    compute_shadow_win_rate_trend,
)


def _make(better: str, reflexio_is_r1: bool, ts: int) -> ShadowComparisonVerdict:
    """Build a minimal verdict at a UTC timestamp ``ts`` (epoch seconds)."""
    return ShadowComparisonVerdict(
        verdict_id=0,
        interaction_id="i",
        session_id="s",
        agent_version="v1",
        reflexio_is_request_1=reflexio_is_r1,
        output=ShadowComparisonOutput(
            better_request=better,  # type: ignore[arg-type]
            is_significantly_better=False,
        ),
        judge_prompt_version="v1.0.0",
        created_at=datetime.fromtimestamp(ts, tz=UTC),
    )


def test_empty_input_returns_empty_trend() -> None:
    trend = compute_shadow_win_rate_trend([])
    assert trend.daily == []
    assert trend.window_total.n == 0
    assert trend.window_total.wins == 0
    assert trend.window_total.losses == 0
    assert trend.window_total.ties == 0
    assert trend.window_total.win_rate == 0.0
    assert trend.window_total.net_win == 0.0


def test_single_bucket_with_mixed_outcomes() -> None:
    ts = 1_700_000_000
    verdicts = [
        # 2 wins
        _make("1", True, ts),
        _make("2", False, ts),
        # 1 loss
        _make("2", True, ts),
        # 1 tie
        _make("tie", True, ts),
    ]
    trend = compute_shadow_win_rate_trend(verdicts)
    assert len(trend.daily) == 1
    day = trend.daily[0]
    assert day.n == 4
    assert day.wins == 2
    assert day.losses == 1
    assert day.ties == 1
    assert trend.window_total.n == 4
    assert trend.window_total.wins == 2
    assert trend.window_total.losses == 1
    assert trend.window_total.ties == 1
    assert trend.window_total.win_rate == 0.5
    assert trend.window_total.net_win == 0.25


def test_multi_day_bucketing() -> None:
    ts_day_1 = 1_700_000_000
    ts_day_2 = ts_day_1 + 86_400 + 1
    verdicts = [
        _make("1", True, ts_day_1),
        _make("1", True, ts_day_1),
        _make("2", True, ts_day_2),
    ]
    trend = compute_shadow_win_rate_trend(verdicts)
    assert len(trend.daily) == 2
    day_one, day_two = trend.daily
    assert day_one.wins == 2
    assert day_one.losses == 0
    assert day_one.ties == 0
    assert day_one.n == 2
    assert day_two.wins == 0
    assert day_two.losses == 1
    assert day_two.ties == 0
    assert day_two.n == 1
    # Bucket dates returned in ascending order.
    assert day_one.date < day_two.date


def test_daily_buckets_use_utc_date_keys() -> None:
    # 2023-11-14 22:13:20 UTC
    ts = 1_700_000_000
    trend = compute_shadow_win_rate_trend([_make("1", True, ts)])
    assert len(trend.daily) == 1
    assert trend.daily[0].date == "2023-11-14"


def test_judge_prompt_version_passthrough() -> None:
    trend = compute_shadow_win_rate_trend([], judge_prompt_version="v2.0.0")
    assert trend.judge_prompt_version == "v2.0.0"


def test_default_judge_prompt_version_is_v1() -> None:
    trend = compute_shadow_win_rate_trend([])
    assert trend.judge_prompt_version == "v1.0.0"


def test_all_ties_yields_zero_rates() -> None:
    ts = 1_700_000_000
    verdicts = [
        _make("tie", True, ts),
        _make("tie", False, ts),
    ]
    trend = compute_shadow_win_rate_trend(verdicts)
    assert trend.window_total.n == 2
    assert trend.window_total.ties == 2
    assert trend.window_total.win_rate == 0.0
    assert trend.window_total.net_win == 0.0
