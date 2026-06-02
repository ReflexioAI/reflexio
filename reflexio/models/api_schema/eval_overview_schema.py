"""Request/response models for POST /api/get_evaluation_overview.

The endpoint returns everything the redesigned /evaluations page needs in a
single round-trip so the frontend renders, never computes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflexio.models.api_schema.validators import NonEmptyStr

HeroStateLiteral = Literal["full", "early", "shadow_off", "empty"]
BucketLiteral = Literal["day", "week"]


class HeroBucket(BaseModel):
    """One point on the trend chart in the hero block.

    ``avg_corrections`` is the mean of ``number_of_correction_per_session``
    across this bucket's evaluation results. Surfaced so the frontend can
    plot a "corrections over time" line beside the success-rate trend.
    Lower is better.

    ``escalation_rate`` is the fraction of sessions in this bucket whose
    eval result had ``is_escalated=True``. Range 0.0 – 1.0. Surfaced so
    the frontend can plot an "escalations over time" mini-trend beside
    the absolute escalation-rate metric tile.
    """

    ts: int
    regular_rate: float
    shadow_rate: float | None
    regular_n: int
    shadow_n: int
    avg_corrections: float = 0.0
    escalation_rate: float = 0.0


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
    cited_session_ids: list[str] = Field(default_factory=list)
    """Session IDs (within the trend window) that cited this rule. The
    frontend uses this to filter the detail band to the sessions where the
    rule actually fired — answering 'which sessions did this rule help or
    hurt?' without a second roundtrip."""


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

    @model_validator(mode="after")
    def validate_time_window(self) -> Self:
        """Ensure the requested time window is ordered."""
        if self.from_ts > self.to_ts:
            raise ValueError("from_ts must be <= to_ts")
        return self


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


class TrendPoint(BaseModel):
    """One point on a grouped success-rate trend curve (F2).

    Args:
        ts (int): Unix epoch seconds — bucket start.
        rate (float): Success rate for sessions in this bucket, [0.0, 1.0].
        n (int): Session count backing the rate. Must be non-negative.
    """

    ts: int
    rate: float
    n: int = Field(ge=0)


class SuccessRateTrendByGroup(BaseModel):
    """Group-split trend data for the dashboard's dual-curve chart (F2).

    Grouping is by ``Request.metadata.reflexio_retrieval_enabled``, read from
    the first request of each session in the window. Sessions whose first
    request has the key absent OR a non-bool value land in ``untagged``.

    Args:
        treatment (list[TrendPoint]): Curve for sessions where the first
            request had ``metadata.reflexio_retrieval_enabled = True``.
        control (list[TrendPoint]): Curve for ``... = False``.
        untagged (list[TrendPoint]): Curve for sessions where the key is
            absent or non-bool — surfaced (not silently coerced) so
            customers can see how many of their sessions are untagged.
    """

    treatment: list[TrendPoint] = Field(default_factory=list)
    control: list[TrendPoint] = Field(default_factory=list)
    untagged: list[TrendPoint] = Field(default_factory=list)


class ShadowWinRateTrendPoint(BaseModel):
    """One daily bucket of per-turn shadow-comparison verdicts (F1).

    Args:
        date (str): ISO date for the bucket start (``YYYY-MM-DD``), UTC.
        n (int): Total verdicts in this bucket.
        wins (int): Reflexio wins.
        losses (int): Reflexio losses.
        ties (int): Ties.
    """

    date: str
    n: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    ties: int = Field(ge=0)


class ShadowWinRateTrendWindowTotal(BaseModel):
    """Aggregate of all shadow verdicts in the trend window (F1).

    Args:
        n (int): Total verdicts in the window.
        wins (int): Reflexio wins.
        losses (int): Reflexio losses.
        ties (int): Ties.
        win_rate (float): ``wins / n``; ``0.0`` when ``n == 0``.
        net_win (float): ``(wins - losses) / n``; ``0.0`` when ``n == 0``.
    """

    n: int = Field(ge=0)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    ties: int = Field(ge=0)
    win_rate: float = Field(ge=0.0, le=1.0)
    net_win: float = Field(ge=-1.0, le=1.0)


class ShadowWinRateTrend(BaseModel):
    """F1 shadow win-rate trend payload for the evaluation overview.

    Daily buckets are UTC-aligned and presented in ascending date order.
    ``judge_prompt_version`` is echoed so the dashboard can show which
    rubric epoch produced the numbers — verdicts from older rubrics are
    filtered out at storage time, never silently mixed in.

    Args:
        daily (list[ShadowWinRateTrendPoint]): Daily buckets (UTC), sorted
            ascending. Empty when no verdicts exist in the window.
        window_total (ShadowWinRateTrendWindowTotal): Aggregate over all
            daily buckets.
        judge_prompt_version (str): Pinned prompt version the verdicts in
            this payload were graded under.
    """

    daily: list[ShadowWinRateTrendPoint] = Field(default_factory=list)
    window_total: ShadowWinRateTrendWindowTotal = Field(
        default_factory=lambda: ShadowWinRateTrendWindowTotal(
            n=0,
            wins=0,
            losses=0,
            ties=0,
            win_rate=0.0,
            net_win=0.0,
        )
    )
    judge_prompt_version: str = Field(default="v1.0.0")


class GetEvaluationOverviewResponse(BaseModel):
    hero: HeroBlock
    context_tiles: ContextTile
    rule_attribution: list[RuleAttributionRow]
    score_distribution: ScoreDistribution
    braintrust_tiles: list[BraintrustTileRow] = Field(default_factory=list)
    success_rate_trend_by_group: SuccessRateTrendByGroup = Field(
        default_factory=SuccessRateTrendByGroup
    )
    shadow_win_rate_trend: ShadowWinRateTrend = Field(
        default_factory=ShadowWinRateTrend
    )


# ---------------------------------------------------------------------------
# /api/evaluations/regenerate — replay-the-judge endpoints
# ---------------------------------------------------------------------------


class RegenerateRequest(BaseModel):
    """Input for POST /api/evaluations/regenerate.

    Args:
        evaluation_name (NonEmptyStr | None): Deprecated compatibility field.
            Singleton evaluation ignores name-based selection.
        from_ts (int): Inclusive lower bound of the window (Unix seconds).
        to_ts (int): Inclusive upper bound of the window (Unix seconds).
            Must be strictly greater than ``from_ts``.
    """

    evaluation_name: NonEmptyStr | None = None
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
        total_candidates (int): F3: count of distinct (session, agent_version)
            candidate tuples discovered in the regen window BEFORE stratified
            sampling. Defaults to 0 for jobs created before F3 shipped.
        sampled_count (int): F3: count of candidates retained after stratified
            sampling. Equal to total_candidates when no stratum exceeded
            Config.eval_sample_n_per_stratum.
        concurrency_limit (int): F3: max simultaneous worker threads. Mirrors
            Config.eval_concurrency_limit at job start.
    """

    job_id: str
    status: Literal["running", "completed", "cancelled", "error"]
    total: int
    completed: int
    failed: int
    failures: list[RegenerateFailure]
    started_at: float
    finished_at: float | None

    total_candidates: int = Field(default=0, ge=0)
    """F3: count of distinct (session, agent_version) candidate tuples
    discovered in the regen window BEFORE stratified sampling."""

    sampled_count: int = Field(default=0, ge=0)
    """F3: count of candidates retained after stratified sampling. Equal
    to total_candidates when no stratum exceeded `Config.eval_sample_n_per_stratum`."""

    concurrency_limit: int = Field(default=0, ge=0)
    """F3: max simultaneous worker threads. Mirrors
    `Config.eval_concurrency_limit` at job start; reported so the dashboard
    can show 'n_sampled / concurrency_limit' status legibly."""


# ---------------------------------------------------------------------------
# /api/evaluations/grade_on_demand — single-session click-through grading
# ---------------------------------------------------------------------------


class GradeOnDemandRequest(BaseModel):
    """Input for POST /api/evaluations/grade_on_demand.

    Args:
        session_id (NonEmptyStr): Target session to grade.
        agent_version (NonEmptyStr): Agent version filter (must be set — eval
            results are versioned).
        evaluation_name (NonEmptyStr | None): Deprecated compatibility field.
            Singleton evaluation ignores name-based selection.
    """

    session_id: NonEmptyStr
    agent_version: NonEmptyStr
    evaluation_name: NonEmptyStr | None = None


class GradeOnDemandResponse(BaseModel):
    """Returned by POST /api/evaluations/grade_on_demand.

    Args:
        session_id (str): Echo of the requested session.
        result_id (int | None): The eval result row id, or None if grading
            was skipped (e.g., session not found, no interactions).
        cached (bool): True when the response came from the 24h cache
            window. False on a fresh grade.
        skipped_reason (str | None): If grading was skipped, the reason
            (e.g., "NO_REQUESTS"). None on success.
    """

    session_id: str
    result_id: int | None = None
    cached: bool = False
    skipped_reason: str | None = None


# ---------------------------------------------------------------------------
# Per-turn shadow comparison verdicts (F1)
# ---------------------------------------------------------------------------


class ShadowComparisonOutput(BaseModel):
    """LLM judge verdict for a per-turn Reflexio-vs-Shadow comparison (F1).

    Args:
        better_request (Literal["1", "2", "tie"]): Which side the judge
            picked. Position is randomized per call so "1" and "2" are
            blind to the judge; the mapping is recorded on
            ShadowComparisonVerdict.reflexio_is_request_1.
        is_significantly_better (bool): True if the better side is clearly
            better; False if marginal/close-but-edges-it. Used to filter
            the "Top 10 disagreements" widget down to actionable cases.
        comparison_reason (str | None): 1-2 sentence rationale. Displayed
            in the drill-down drawer.
    """

    better_request: Literal["1", "2", "tie"]
    is_significantly_better: bool
    comparison_reason: str | None = None

    # Dual-defense extras policy:
    # - extra="allow" at runtime so the server doesn't crash if the LLM
    #   returns an unexpected field. We log what we recognize and ignore
    #   the rest.
    # - additionalProperties=False in the JSON schema sent to the LLM
    #   so the structured-output constraint tells the model NOT to add
    #   extra fields in the first place.
    # This matches the convention from the (now-removed) session-level
    # comparison schema; do not change one without changing the other.
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class ShadowComparisonVerdict(BaseModel):
    """One per-turn comparison verdict, stored in shadow_comparison_verdicts (F1).

    Args:
        verdict_id (int): Storage-assigned autoincrement primary key.
        interaction_id (str): The interaction this verdict grades. Joins
            to the Interaction.interaction_id for drill-down display.
        session_id (str): The session containing the interaction.
        agent_version (str): Pinned for trend-by-version slicing.
        reflexio_is_request_1 (bool): Position-randomization record. True
            when the Reflexio response was shown as Request 1 to the judge.
            The dashboard derives win/loss/tie via:
                derived_win = (better == "1") == reflexio_is_request_1
        output (ShadowComparisonOutput): The judge's structured verdict.
        judge_prompt_version (str): Semver of shadow_comparison prompt
            used. The dashboard filters to the org's current pinned
            version (Config.shadow_comparison_judge_prompt_version) so
            verdicts from a prior rubric never mix into the headline.
        created_at (datetime): When the judge call returned.
    """

    verdict_id: int
    interaction_id: str
    session_id: str
    agent_version: str
    reflexio_is_request_1: bool
    output: ShadowComparisonOutput
    judge_prompt_version: NonEmptyStr
    created_at: datetime
    """When the judge call returned. Storage layers assume UTC — callers
    must pass a tz-aware datetime (typically `datetime.now(UTC)`)."""


class GetRecentShadowComparisonsResponse(BaseModel):
    """Returned by GET /api/evaluations/shadow_comparisons/recent (F1).

    Args:
        verdicts (list[ShadowComparisonVerdict]): Recent verdicts for the
            org's current pinned ``shadow_comparison`` prompt version,
            newest first. Capped at the ``limit`` query param (default 10,
            max 100). Empty when the backend does not support the
            ``shadow_comparison_verdicts`` storage feature.
    """

    verdicts: list[ShadowComparisonVerdict] = Field(default_factory=list)
