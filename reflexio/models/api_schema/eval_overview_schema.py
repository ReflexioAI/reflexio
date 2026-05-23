"""Request/response models for POST /api/get_evaluation_overview.

The endpoint returns everything the redesigned /evaluations page needs in a
single round-trip so the frontend renders, never computes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HeroStateLiteral = Literal["full", "early", "shadow_off", "empty"]
BucketLiteral = Literal["day", "week"]


class HeroBucket(BaseModel):
    """One point on the trend chart in the hero block."""

    ts: int
    regular_rate: float
    shadow_rate: float | None
    regular_n: int
    shadow_n: int


class HeroBlock(BaseModel):
    """The "answer" band — trend + headline delta."""

    state: HeroStateLiteral
    regular_success_rate_pp: float
    shadow_success_rate_pp: float | None
    delta_pp: float | None
    buckets: list[HeroBucket]


class NumberWithDelta(BaseModel):
    current: float
    delta: float


class PercentWithDelta(BaseModel):
    current: float
    delta_pp: float


class ContextTile(BaseModel):
    """Wrapper for the four mini-tiles in the context band.

    Each tile is rendered with an absolute value + a delta vs the previous
    7d window. Percent-shaped values carry `delta_pp` (percentage points);
    raw counts carry `delta` (absolute difference).
    """

    success: PercentWithDelta
    corrections: NumberWithDelta
    turns: NumberWithDelta
    escalation: PercentWithDelta


class RuleAttributionRow(BaseModel):
    """One row in the "rules that moved the needle" panel."""

    rule_id: str
    kind: Literal["playbook", "profile"]
    title: str = ""
    successes_with: int = Field(ge=0)
    failures_with: int = Field(ge=0)
    net_sessions: int


class ScoreDistribution(BaseModel):
    """Corrections-per-session histogram, current window + baseline."""

    current_bins: list[int]
    baseline_bins: list[int]
    labels: list[str]


class GetEvaluationOverviewRequest(BaseModel):
    """Input for the overview endpoint.

    Args:
        from_ts (int): Window start, unix epoch seconds.
        to_ts (int): Window end, unix epoch seconds.
        bucket (BucketLiteral): Granularity of the hero trend buckets.
        include_shadow (bool): When False, skip the shadow-side aggregations
            (cheaper) — the hero will degrade to shadow_off state.
    """

    from_ts: int = Field(ge=0)
    to_ts: int = Field(ge=0)
    bucket: BucketLiteral = "week"
    include_shadow: bool = True


class GetEvaluationOverviewResponse(BaseModel):
    hero: HeroBlock
    context_tiles: ContextTile
    rule_attribution: list[RuleAttributionRow]
    score_distribution: ScoreDistribution
