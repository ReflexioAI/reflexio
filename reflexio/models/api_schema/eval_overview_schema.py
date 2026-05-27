"""Request/response models for POST /api/get_evaluation_overview.

The endpoint returns everything the redesigned /evaluations page needs in a
single round-trip so the frontend renders, never computes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from reflexio.models.api_schema.validators import NonEmptyStr

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


class BraintrustTileRow(BaseModel):
    """One imported-scorer aggregate for the context band (Plan C-overview).

    Args:
        scorer_name (str): The Braintrust scorer name.
        current (float): Mean value across the current window.
        n (int): Number of imported scores backing `current`.
        delta (float): Mean current − mean prior-window. Equals `current`
            when no baseline (the frontend renders "no baseline").
    """

    scorer_name: str
    current: float
    n: int = Field(ge=0)
    delta: float


class GetEvaluationOverviewResponse(BaseModel):
    hero: HeroBlock
    context_tiles: ContextTile
    rule_attribution: list[RuleAttributionRow]
    score_distribution: ScoreDistribution
    braintrust_tiles: list[BraintrustTileRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# /api/evaluations/regenerate — replay-the-judge endpoints
# ---------------------------------------------------------------------------


class RegenerateRequest(BaseModel):
    """Input for POST /api/evaluations/regenerate.

    Args:
        evaluation_name (NonEmptyStr): Name of the evaluator to replay.
            Must match one of the ``agent_success_configs[*].evaluation_name``
            entries in the caller's config.
        from_ts (int): Inclusive lower bound of the window (Unix seconds).
        to_ts (int): Inclusive upper bound of the window (Unix seconds).
            Must be strictly greater than ``from_ts``.
    """

    evaluation_name: NonEmptyStr
    from_ts: int = Field(ge=0)
    to_ts: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_window(self) -> RegenerateRequest:
        if self.from_ts >= self.to_ts:
            raise ValueError("from_ts must be strictly before to_ts")
        return self


class RegenerateStartResponse(BaseModel):
    """Returned by POST /api/evaluations/regenerate.

    Args:
        job_id (str): Opaque handle used to poll status or cancel.
        total (int): Number of distinct (user, session, agent_version, source)
            tuples the worker will replay.
    """

    job_id: str
    total: int


class RegenerateFailure(BaseModel):
    """One failed session in a regenerate job's failure list.

    Args:
        session_id (str): The session whose replay failed.
        reason (str): Truncated exception message (worker boundary).
    """

    session_id: str
    reason: str


class RegenerateStatusResponse(BaseModel):
    """Returned by GET /api/evaluations/regenerate/{job_id}.

    Args:
        job_id (str): Opaque job handle.
        status (Literal["running", "completed", "cancelled", "error"]):
            Current lifecycle state. ``"completed"`` means the worker loop
            finished iterating regardless of whether every session
            succeeded — per-session pass/fail is in ``completed`` vs.
            ``failed``. ``"error"`` means the worker itself crashed before
            or during iteration.
        total (int): Total tuples queued at job creation.
        completed (int): Number of successfully replayed tuples.
        failed (int): Number of tuples that raised in the worker.
        failures (list[RegenerateFailure]): Per-session failure rows, capped
            inside the worker so the response stays small.
        started_at (float): Unix-seconds wall-clock timestamp at job creation.
        finished_at (float | None): Unix-seconds wall-clock timestamp at
            worker exit; ``None`` while ``status == "running"``.
    """

    job_id: str
    status: Literal["running", "completed", "cancelled", "error"]
    total: int
    completed: int
    failed: int
    failures: list[RegenerateFailure]
    started_at: float
    finished_at: float | None
