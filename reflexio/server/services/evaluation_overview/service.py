"""Aggregator that composes hero, tiles, rule attribution, and distribution.

The service bulk-loads the requested evaluation window plus related source,
citation, Braintrust, and optional shadow verdict rows, then returns a
GetEvaluationOverviewResponse. It's invoked from the FastAPI route handler; the
storage is the same BaseStorage the rest of the server uses, so the same
instance is reused via request_context.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from reflexio.models.api_schema.braintrust_schema import ImportedScore
from reflexio.models.api_schema.domain.entities import (
    AgentSuccessEvaluationResult,
)
from reflexio.models.api_schema.eval_overview_schema import (
    BraintrustTileRow,
    BucketLiteral,
    ContextTile,
    EvaluationSourceSetRequest,
    GetEvaluationOverviewRequest,
    GetEvaluationOverviewResponse,
    HeroBlock,
    HeroBucket,
    NumberWithDelta,
    PercentWithDelta,
    RuleAttributionRow,
    ScoreDistribution,
    ShadowWinRateTrend,
    SourceSetComparison,
    SourceSetEvaluationMetrics,
)
from reflexio.models.config_schema import Config
from reflexio.server.services.evaluation_overview.components.distribution import (
    BUCKET_LABELS,
    bucket_corrections,
)
from reflexio.server.services.evaluation_overview.components.hero_state import (
    compute_hero_state,
)
from reflexio.server.services.evaluation_overview.components.rule_attribution import (
    compute_net_sessions,
)
from reflexio.server.services.evaluation_overview.components.shadow_aggregation import (
    compute_shadow_win_rate_trend,
)

_DAY_SECONDS = 24 * 60 * 60
_WEEK_SECONDS = 7 * 24 * 60 * 60
_TOP_N_RULES = 5
_LOGGER = logging.getLogger(__name__)
ResultKey = tuple[str, str]


@dataclass
class EvaluationOverviewService:
    """Builds the full /api/get_evaluation_overview payload.

    The service holds a storage handle and the org's current Config. Each
    call to `run` performs the three reads and returns a fresh response.
    Stateless across calls — safe to reuse the instance across requests.
    """

    storage: Any
    config: Config

    def run(
        self, request: GetEvaluationOverviewRequest
    ) -> GetEvaluationOverviewResponse:
        """Build the overview payload for the requested window.

        Time windows used here:
          - ``[request.from_ts, request.to_ts]`` is the *trend* window. The hero
            chart, rule attribution, and the rule-attribution session set are
            all computed over it. The frontend sends an 8-week range so the
            trend chart has shape.
          - The *tile baseline* and *distribution baseline* are tighter:
            ``last_7d`` (≤ to_ts) vs ``prior_7d`` (the 7d before that).
            Tying these to ``request.from_ts`` is wrong — with an 8-week
            request, ``prior`` would land 9 weeks back and always be empty,
            so every tile would display "no baseline" regardless of how much
            data the org has.
        """
        # Tile + distribution baselines are always last-7d-vs-prior-7d,
        # anchored to ``request.to_ts`` (which is "now" from the frontend's
        # perspective).
        cur_7d_from = max(request.from_ts, request.to_ts - _WEEK_SECONDS)
        prev_to = cur_7d_from
        prev_from = max(0, prev_to - _WEEK_SECONDS)
        load_from = min(request.from_ts, prev_from)

        start = time.perf_counter()
        all_results = self.storage.get_agent_success_evaluation_results_in_window(  # type: ignore[attr-defined]
            from_ts=load_from,
            to_ts=request.to_ts,
            agent_version=None,
        )
        self._log_phase("eval_results", start, rows=len(all_results))

        results = [
            r for r in all_results if request.from_ts <= r.created_at <= request.to_ts
        ]
        results_current_7d = [
            r for r in all_results if cur_7d_from <= r.created_at <= request.to_ts
        ]
        results_prev_7d = [
            r for r in all_results if prev_from <= r.created_at < prev_to
        ]

        earliest_eval_ts = min((r.created_at for r in all_results), default=None)
        result_keys = list(
            dict.fromkeys((r.user_id, r.session_id) for r in results if r.session_id)
        )
        if request.source_sets:
            source_keys = list(
                dict.fromkeys(
                    (r.user_id, r.session_id)
                    for r in (*results, *results_current_7d, *results_prev_7d)
                    if r.session_id
                )
            )
        else:
            source_keys = result_keys
        start = time.perf_counter()
        session_sources = self._build_first_request_sources(source_keys)
        self._log_phase(
            "session_sources",
            start,
            sessions=len(set(source_keys)),
            rows=len(session_sources),
        )

        start = time.perf_counter()
        citations_by_session, rule_titles = self._load_citations(result_keys)
        self._log_phase(
            "citations",
            start,
            sessions=len(set(result_keys)),
            rows=sum(len(v) for v in citations_by_session.values()),
        )

        hero = self._build_hero(request, results, earliest_eval_ts)
        tiles = self._build_tiles(results_current_7d, results_prev_7d)
        attribution = self._build_attribution(
            results, citations_by_session, rule_titles
        )
        distribution = self._build_distribution(results_current_7d, results_prev_7d)
        start = time.perf_counter()
        current_scores, prior_scores = self._load_braintrust_scores(
            cur_7d_from, request.to_ts, prev_from, prev_to
        )
        self._log_phase(
            "braintrust",
            start,
            current_rows=len(current_scores),
            prior_rows=len(prior_scores),
        )
        braintrust_tiles = self._build_braintrust_tiles_from_scores(
            current_scores, prior_scores
        )
        start = time.perf_counter()
        shadow_win_rate_trend = (
            self._build_shadow_win_rate_trend(request.from_ts, request.to_ts)
            if request.include_shadow
            else ShadowWinRateTrend(
                judge_prompt_version=self.config.shadow_comparison_judge_prompt_version
            )
        )
        self._log_phase("shadow", start, enabled=request.include_shadow)
        source_set_comparison = self._build_source_set_comparison(
            source_sets=request.source_sets,
            results=results,
            current=results_current_7d,
            previous=results_prev_7d,
            session_sources=session_sources,
            bucket=request.bucket,
            citations_by_session=citations_by_session,
            rule_titles=rule_titles,
            current_scores=current_scores,
            prior_scores=prior_scores,
        )

        return GetEvaluationOverviewResponse(
            hero=hero,
            context_tiles=tiles,
            rule_attribution=attribution,
            score_distribution=distribution,
            braintrust_tiles=braintrust_tiles,
            shadow_win_rate_trend=shadow_win_rate_trend,
            source_set_comparison=source_set_comparison,
        )

    # --- private helpers ---

    def _build_hero(
        self,
        request: GetEvaluationOverviewRequest,
        results: list[AgentSuccessEvaluationResult],
        earliest_eval_ts: int | None,
    ) -> HeroBlock:
        if earliest_eval_ts is None:
            days_since = None
        else:
            days_since = (
                int(datetime.now(UTC).timestamp()) - earliest_eval_ts
            ) // 86_400
        # Shadow direct-grade has been removed; counterfactual measurement is
        # pending a methodologically sound replacement (see the validity spec).
        # The state machine still runs so the EMPTY / SHADOW_OFF surfaces work,
        # but the shadow-side fields are always None.
        state = compute_hero_state(
            shadow_enabled=False,
            days_since_first_eval=days_since,
            n_shadow_in_window=0,
            total_results=len(results),
        )
        success_rate = _success_rate(results) * 100
        return HeroBlock(
            state=state.value,  # type: ignore[arg-type]
            regular_success_rate_pp=success_rate,
            shadow_success_rate_pp=None,
            delta_pp=None,
            buckets=_buckets(results, request.bucket),
        )

    def _build_tiles(
        self,
        current: list[AgentSuccessEvaluationResult],
        previous: list[AgentSuccessEvaluationResult],
    ) -> ContextTile:
        cur_success = _success_rate(current) * 100
        prev_success = _success_rate(previous) * 100
        cur_corr = _mean(r.number_of_correction_per_session for r in current)
        prev_corr = _mean(r.number_of_correction_per_session for r in previous)
        cur_turns = _mean(
            r.user_turns_to_resolution
            for r in current
            if r.user_turns_to_resolution is not None
        )
        prev_turns = _mean(
            r.user_turns_to_resolution
            for r in previous
            if r.user_turns_to_resolution is not None
        )
        cur_esc = _escalation_rate(current) * 100
        prev_esc = _escalation_rate(previous) * 100
        return ContextTile(
            success=PercentWithDelta(
                current=cur_success, delta_pp=cur_success - prev_success
            ),
            corrections=NumberWithDelta(current=cur_corr, delta=cur_corr - prev_corr),
            turns=NumberWithDelta(current=cur_turns, delta=cur_turns - prev_turns),
            escalation=PercentWithDelta(current=cur_esc, delta_pp=cur_esc - prev_esc),
        )

    def _build_attribution(
        self,
        results: list[AgentSuccessEvaluationResult],
        citations_by_session: dict[ResultKey, list[tuple[str, str]]],
        rule_titles: dict[tuple[str, str], str],
    ) -> list[RuleAttributionRow]:
        is_success_by_session = {
            (r.user_id, r.session_id): r.is_success for r in results
        }
        rows = compute_net_sessions(
            citations_by_session=citations_by_session,
            is_success_by_session=is_success_by_session,
            rule_titles=rule_titles,
            top_n=_TOP_N_RULES,
        )
        return [
            RuleAttributionRow(
                rule_id=r.rule_id,
                kind=r.kind,  # type: ignore[arg-type]
                title=r.title,
                successes_with=r.successes_with,
                failures_with=r.failures_with,
                net_sessions=r.net_sessions,
                cited_session_ids=list(r.cited_session_ids),
            )
            for r in rows
        ]

    def _load_citations(
        self, result_keys: list[ResultKey]
    ) -> tuple[dict[ResultKey, list[tuple[str, str]]], dict[tuple[str, str], str]]:
        """Pull `Interaction.citations` keyed by user/session, with title lookup.

        Falls back to empty data when the underlying storage method returns
        no interactions (default behavior on backends that haven't yet
        implemented `get_interactions_by_session`).
        """
        wanted = set(result_keys)
        session_ids = sorted({session_id for _, session_id in result_keys})
        citations_by_session: dict[ResultKey, list[tuple[str, str]]] = defaultdict(list)
        rule_titles: dict[tuple[str, str], str] = {}
        for citation in self.storage.get_citations_by_session_ids(session_ids):  # type: ignore[attr-defined]
            result_key = (citation.user_id, citation.session_id)
            if result_key not in wanted:
                continue
            key = (citation.kind, citation.real_id)
            citations_by_session[result_key].append(key)
            if citation.title and key not in rule_titles:
                rule_titles[key] = citation.title
        return citations_by_session, rule_titles

    def _load_braintrust_scores(
        self,
        from_ts: int,
        to_ts: int,
        prev_from: int,
        prev_to: int,
    ) -> tuple[list[ImportedScore], list[ImportedScore]]:
        org_id = self._org_id()
        if not org_id:
            return [], []
        current = self.storage.get_imported_scores(org_id, from_ts, to_ts)  # type: ignore[attr-defined]
        if not current:
            return [], []
        prior = self.storage.get_imported_scores(org_id, prev_from, prev_to)  # type: ignore[attr-defined]
        return current, prior

    def _build_braintrust_tiles_from_scores(
        self,
        current: list[ImportedScore],
        prior: list[ImportedScore],
        session_ids: set[str] | None = None,
    ) -> list[BraintrustTileRow]:
        """Aggregate imported_score rows per scorer_name for current + prior windows."""
        if session_ids is not None:
            current = [s for s in current if s.session_id in session_ids]
        if not current:
            return []
        if session_ids is not None:
            prior = [s for s in prior if s.session_id in session_ids]
        cur_agg = _aggregate_imported_scores(current)
        prior_agg = _aggregate_imported_scores(prior)
        rows: list[BraintrustTileRow] = []
        for scorer_name, (mean, n) in sorted(cur_agg.items()):
            # No prior data → set delta = current so the frontend's
            # `delta == current` check renders "no baseline" honestly.
            # With prior data → real difference.
            prior_entry = prior_agg.get(scorer_name)
            delta = mean if prior_entry is None else mean - prior_entry[0]
            rows.append(
                BraintrustTileRow(
                    scorer_name=scorer_name,
                    current=mean,
                    n=n,
                    delta=delta,
                )
            )
        return rows

    def _build_source_set_comparison(
        self,
        *,
        source_sets: list[EvaluationSourceSetRequest],
        results: list[AgentSuccessEvaluationResult],
        current: list[AgentSuccessEvaluationResult],
        previous: list[AgentSuccessEvaluationResult],
        session_sources: dict[ResultKey, str],
        bucket: BucketLiteral,
        citations_by_session: dict[ResultKey, list[tuple[str, str]]],
        rule_titles: dict[tuple[str, str], str],
        current_scores: list[ImportedScore],
        prior_scores: list[ImportedScore],
    ) -> SourceSetComparison:
        """Build request-source cohort metrics for the evaluation page."""
        available_sources = sorted(
            {session_sources.get((r.user_id, r.session_id), "") for r in results}
        )
        if not source_sets:
            return SourceSetComparison(available_sources=available_sources)

        requested_sources = {source for s in source_sets for source in s.sources}
        unmatched = sum(
            1
            for r in results
            if session_sources.get((r.user_id, r.session_id), "")
            not in requested_sources
        )
        rows: list[SourceSetEvaluationMetrics] = []
        for source_set in source_sets:
            source_values = set(source_set.sources)
            set_results = [
                r
                for r in results
                if session_sources.get((r.user_id, r.session_id), "") in source_values
            ]
            set_current = [
                r
                for r in current
                if session_sources.get((r.user_id, r.session_id), "") in source_values
            ]
            set_previous = [
                r
                for r in previous
                if session_sources.get((r.user_id, r.session_id), "") in source_values
            ]
            session_ids = {r.session_id for r in set_results if r.session_id}
            rows.append(
                SourceSetEvaluationMetrics(
                    label=source_set.label,
                    sources=list(source_set.sources),
                    session_count=len(set_results),
                    session_ids=sorted(session_ids),
                    success_rate_pp=_success_rate(set_results) * 100,
                    buckets=_buckets(set_results, bucket),
                    context_tiles=self._build_tiles(set_current, set_previous),
                    score_distribution=self._build_distribution(
                        set_current, set_previous
                    ),
                    rule_attribution=self._build_attribution(
                        set_results, citations_by_session, rule_titles
                    ),
                    braintrust_tiles=self._build_braintrust_tiles_from_scores(
                        current_scores,
                        prior_scores,
                        session_ids=session_ids,
                    ),
                )
            )
        return SourceSetComparison(
            available_sources=available_sources,
            sets=rows,
            unmatched_session_count=unmatched,
        )

    def _org_id(self) -> str:
        """Resolve org_id from request_context when available; else empty string."""
        # The service is constructed with `storage` + `config`; org_id isn't a
        # direct attribute. For Plan C-overview, we read it via the storage
        # instance's org_id when present (every BaseStorage carries one).
        return str(getattr(self.storage, "org_id", "") or "")

    def _build_shadow_win_rate_trend(
        self,
        from_ts: int,
        to_ts: int,
    ) -> ShadowWinRateTrend:
        """Fetch shadow verdicts in the window and aggregate them per day.

        Filters verdicts to the org's pinned ``shadow_comparison`` prompt
        version (``Config.shadow_comparison_judge_prompt_version``) so a
        future rubric bump never silently mixes epochs into the headline.
        Backends that don't yet implement the verdicts table raise
        ``NotImplementedError``; we degrade to the empty trend default so
        the dashboard still renders the rest of the overview.

        Args:
            from_ts (int): Window start, Unix epoch seconds (UTC).
            to_ts (int): Window end, Unix epoch seconds (UTC).

        Returns:
            ShadowWinRateTrend: Daily buckets + window total. Empty when
                the window has no verdicts or the backend doesn't support
                verdict storage.
        """
        pinned_version = self.config.shadow_comparison_judge_prompt_version
        try:
            verdicts = self.storage.get_shadow_comparison_verdicts(  # type: ignore[attr-defined]
                from_ts=from_ts,
                to_ts=to_ts,
                judge_prompt_version=pinned_version,
            )
        except NotImplementedError:
            return ShadowWinRateTrend(judge_prompt_version=pinned_version)
        return compute_shadow_win_rate_trend(
            verdicts, judge_prompt_version=pinned_version
        )

    def _build_first_request_sources(
        self, result_keys: list[ResultKey]
    ) -> dict[ResultKey, str]:
        """Map each user/session slice to its earliest request's source."""
        first_requests = self.storage.get_first_requests_by_user_session_pairs(
            result_keys
        )  # type: ignore[attr-defined]
        return {key: row.source or "" for key, row in first_requests.items()}

    def _build_distribution(
        self,
        current: list[AgentSuccessEvaluationResult],
        previous: list[AgentSuccessEvaluationResult],
    ) -> ScoreDistribution:
        cur_bins = bucket_corrections(
            r.number_of_correction_per_session for r in current
        )
        prev_bins = bucket_corrections(
            r.number_of_correction_per_session for r in previous
        )
        return ScoreDistribution(
            current_bins=list(cur_bins),
            baseline_bins=list(prev_bins),
            labels=list(BUCKET_LABELS),
        )

    def _log_phase(self, phase: str, started_at: float, **metadata: object) -> None:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        fields = " ".join(f"{key}={value}" for key, value in metadata.items())
        _LOGGER.info(
            "evaluation_overview phase=%s duration_ms=%d %s",
            phase,
            duration_ms,
            fields,
        )


# --- module-level helpers (pure) ---


def _success_rate(results: list[AgentSuccessEvaluationResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.is_success) / len(results)


def _escalation_rate(results: list[AgentSuccessEvaluationResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.is_escalated) / len(results)


def _mean(values: Iterable[float | int | None]) -> float:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def _aggregate_imported_scores(
    scores: list[ImportedScore],
) -> dict[str, tuple[float, int]]:
    """Group imported scores by scorer_name → (mean, count)."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in scores:
        bucket[s.scorer_name].append(s.value)
    return {name: (sum(vs) / len(vs), len(vs)) for name, vs in bucket.items() if vs}


def _buckets(
    results: list[AgentSuccessEvaluationResult], bucket: BucketLiteral
) -> list[HeroBucket]:
    """Build day- or week-sized buckets across the given results.

    Bucket granularity follows the request: ``"day"`` for the frontend's
    daily trend mode (tooltip activates on every X position, smoother
    curves on narrow ranges), ``"week"`` otherwise. Each bucket carries
    success rate, average corrections, and escalation rate so the metric
    mini-trends stay aligned with the headline numbers.
    """
    if not results:
        return []
    buckets: dict[int, list[AgentSuccessEvaluationResult]] = defaultdict(list)
    step = _DAY_SECONDS if bucket == "day" else _WEEK_SECONDS
    for r in results:
        # Anchor each result to the START of its bucket (epoch-aligned).
        bucket_start = (r.created_at // step) * step
        buckets[bucket_start].append(r)
    out: list[HeroBucket] = []
    for ts in sorted(buckets):
        bucket_results = buckets[ts]
        avg_corr = _mean(r.number_of_correction_per_session for r in bucket_results)
        # is_escalated may be None on legacy rows; coerce to False so the
        # bucket mean reflects "fraction of sessions we know escalated".
        escalation_rate = _mean(
            (1.0 if (r.is_escalated is True) else 0.0) for r in bucket_results
        )
        out.append(
            HeroBucket(
                ts=ts,
                regular_rate=_success_rate(bucket_results),
                shadow_rate=None,
                regular_n=len(bucket_results),
                shadow_n=0,
                avg_corrections=avg_corr,
                escalation_rate=escalation_rate,
            )
        )
    return out
