"""F1 shadow win-rate aggregator: ShadowComparisonVerdict[] → ShadowWinRateTrend.

Pure: DB-free, LLM-free. Bucketed by UTC calendar date; the window total is
computed by summing the daily buckets, so it is consistent with what the
dashboard renders by construction.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonVerdict,
    ShadowWinRateTrend,
    ShadowWinRateTrendPoint,
    ShadowWinRateTrendWindowTotal,
)
from reflexio.server.services.shadow_comparison.outcome import (
    Outcome,
    derive_reflexio_outcome,
)

# Map outcome enum → bucket key used in the per-day accumulator below.
_OUTCOME_TO_KEY: dict[Outcome, str] = {
    Outcome.WIN: "wins",
    Outcome.LOSS: "losses",
    Outcome.TIE: "ties",
}


def compute_shadow_win_rate_trend(
    verdicts: Iterable[ShadowComparisonVerdict],
    judge_prompt_version: str = "v1.0.0",
) -> ShadowWinRateTrend:
    """
    Bucket verdicts by UTC date, derive win/loss/tie per verdict, sum.

    Args:
        verdicts (Iterable[ShadowComparisonVerdict]): Position-randomized
            verdicts in the trend window. Caller is expected to have
            already filtered to the pinned prompt version.
        judge_prompt_version (str): Informational; echoed on the response
            so the dashboard knows which rubric epoch the numbers came
            from. Defaults to ``"v1.0.0"``.

    Returns:
        ShadowWinRateTrend: Daily buckets sorted ascending by date, plus
            the window total computed from those same buckets.
    """
    by_day: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "losses": 0, "ties": 0}
    )
    for verdict in verdicts:
        date_key = verdict.created_at.astimezone(UTC).strftime("%Y-%m-%d")
        bucket = by_day[date_key]
        bucket["n"] += 1
        bucket[_OUTCOME_TO_KEY[derive_reflexio_outcome(verdict)]] += 1

    daily = [
        ShadowWinRateTrendPoint(
            date=date,
            n=bucket["n"],
            wins=bucket["wins"],
            losses=bucket["losses"],
            ties=bucket["ties"],
        )
        for date, bucket in sorted(by_day.items())
    ]

    total_n = sum(p.n for p in daily)
    total_wins = sum(p.wins for p in daily)
    total_losses = sum(p.losses for p in daily)
    total_ties = sum(p.ties for p in daily)
    win_rate = total_wins / total_n if total_n > 0 else 0.0
    net_win = (total_wins - total_losses) / total_n if total_n > 0 else 0.0

    return ShadowWinRateTrend(
        daily=daily,
        window_total=ShadowWinRateTrendWindowTotal(
            n=total_n,
            wins=total_wins,
            losses=total_losses,
            ties=total_ties,
            win_rate=win_rate,
            net_win=net_win,
        ),
        judge_prompt_version=judge_prompt_version,
    )
