"""Aggregator that composes hero, tiles, rule attribution, and distribution.

The service does four reads (results, citations, playbook stats, shadow count)
and returns a GetEvaluationOverviewResponse. It's invoked from the FastAPI
route handler; the storage is the same BaseStorage the rest of the server
uses, so the same instance is reused via request_context.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from reflexio.models.api_schema.braintrust_schema import ImportedScore
from reflexio.models.api_schema.domain.entities import (
    AgentSuccessEvaluationResult,
)
from reflexio.models.api_schema.eval_overview_schema import (
    BraintrustTileRow,
    ContextTile,
    GetEvaluationOverviewRequest,
    GetEvaluationOverviewResponse,
    HeroBlock,
    HeroBucket,
    NumberWithDelta,
    PercentWithDelta,
    RuleAttributionRow,
    ScoreDistribution,
)
from reflexio.models.config_schema import Config
from reflexio.server.services.evaluation_overview.distribution import (
    BUCKET_LABELS,
    bucket_corrections,
)
from reflexio.server.services.evaluation_overview.hero_state import (
    compute_hero_state,
)
from reflexio.server.services.evaluation_overview.rule_attribution import (
    compute_net_sessions,
)

_WEEK_SECONDS = 7 * 24 * 60 * 60
_TOP_N_RULES = 5


@dataclass
class EvaluationOverviewService:
    """Builds the full /api/get_evaluation_overview payload.

    The service holds a storage handle and the org's current Config. Each
    call to `run` performs the four reads and returns a fresh response.
    Stateless across calls — safe to reuse the instance across requests.
    """

    storage: object
    config: Config

    def run(
        self, request: GetEvaluationOverviewRequest
    ) -> GetEvaluationOverviewResponse:
        """Build the overview payload for the requested window."""
        all_results = self.storage.get_agent_success_evaluation_results(  # type: ignore[attr-defined]
            agent_version=None, limit=10_000
        )
        results = [
            r for r in all_results if request.from_ts <= r.created_at <= request.to_ts
        ]
        prev_to = request.from_ts
        prev_from = max(0, prev_to - _WEEK_SECONDS)
        results_prev = [r for r in all_results if prev_from <= r.created_at < prev_to]

        hero = self._build_hero(request, results)
        tiles = self._build_tiles(results, results_prev)
        attribution = self._build_attribution(results)
        distribution = self._build_distribution(results, results_prev)
        braintrust_tiles = self._build_braintrust_tiles(
            request.from_ts, request.to_ts, prev_from, prev_to
        )

        return GetEvaluationOverviewResponse(
            hero=hero,
            context_tiles=tiles,
            rule_attribution=attribution,
            score_distribution=distribution,
            braintrust_tiles=braintrust_tiles,
        )

    # --- private helpers ---

    def _build_hero(
        self,
        request: GetEvaluationOverviewRequest,
        results: list[AgentSuccessEvaluationResult],
    ) -> HeroBlock:
        if not results:
            days_since = None
        else:
            earliest = min(r.created_at for r in results)
            days_since = (int(datetime.now(UTC).timestamp()) - earliest) // 86_400
        n_shadow = self.storage.count_sessions_with_shadow_content(  # type: ignore[attr-defined]
            request.from_ts, request.to_ts
        )
        state = compute_hero_state(
            shadow_enabled=self.config.shadow_mode_enabled,
            days_since_first_eval=days_since,
            n_shadow_in_window=n_shadow,
            total_results=len(results),
        )
        success_rate = _success_rate(results) * 100
        # shadow_rate and delta are None until real shadow-content evaluations
        # exist. When count_sessions_with_shadow_content > 0 AND a future
        # storage method exposes shadow-side outcomes, populate from there.
        # Today both stay None and the frontend renders state-1 vs state-3
        # purely from the `state` enum.
        shadow_rate: float | None = None
        delta: float | None = None
        return HeroBlock(
            state=state.value,  # type: ignore[arg-type]
            regular_success_rate_pp=success_rate,
            shadow_success_rate_pp=shadow_rate,
            delta_pp=delta,
            buckets=_weekly_buckets(results),
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
        self, results: list[AgentSuccessEvaluationResult]
    ) -> list[RuleAttributionRow]:
        is_success_by_session = {r.session_id: r.is_success for r in results}
        citations_by_session, rule_titles = self._load_citations(
            list(is_success_by_session.keys())
        )
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
        self, session_ids: list[str]
    ) -> tuple[dict[str, list[tuple[str, str]]], dict[tuple[str, str], str]]:
        """Pull `Interaction.citations` keyed by session, with title lookup.

        Falls back to empty data when the underlying storage method returns
        no interactions (default behavior on backends that haven't yet
        implemented `get_interactions_by_session`).
        """
        citations_by_session: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for sid in session_ids:
            interactions = self.storage.get_interactions_by_session(sid)  # type: ignore[attr-defined]
            for interaction in interactions:
                for cite in getattr(interaction, "citations", []) or []:
                    # Citations may arrive as Pydantic Citation objects (from
                    # the normal storage path) or as plain dicts (e.g. when
                    # tests stub the storage). Handle both shapes.
                    if isinstance(cite, dict):
                        kind = cite.get("kind")
                        rid = cite.get("real_id")
                    else:
                        kind = getattr(cite, "kind", None)
                        rid = getattr(cite, "real_id", None)
                    if kind and rid:
                        citations_by_session[sid].append((kind, str(rid)))
        # Titles via existing playbook_application_stats lookup
        stats = self.storage.get_playbook_application_stats(days_back=30)  # type: ignore[attr-defined]
        rule_titles = {(s.kind, s.real_id): s.title for s in stats}
        return citations_by_session, rule_titles

    def _build_braintrust_tiles(
        self, from_ts: int, to_ts: int, prev_from: int, prev_to: int
    ) -> list[BraintrustTileRow]:
        """Aggregate imported_score rows per scorer_name for current + prior windows.

        Returns [] when the org has no Braintrust connection (default no-op
        storage returns []). The frontend treats an empty list as "not
        connected" and hides the Braintrust strip entirely.
        """
        org_id = self._org_id()
        if not org_id:
            return []
        current = self.storage.get_imported_scores(org_id, from_ts, to_ts)  # type: ignore[attr-defined]
        if not current:
            return []
        prior = self.storage.get_imported_scores(org_id, prev_from, prev_to)  # type: ignore[attr-defined]
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

    def _org_id(self) -> str:
        """Resolve org_id from request_context when available; else empty string."""
        # The service is constructed with `storage` + `config`; org_id isn't a
        # direct attribute. For Plan C-overview, we read it via the storage
        # instance's org_id when present (every BaseStorage carries one).
        return str(getattr(self.storage, "org_id", "") or "")

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


def _weekly_buckets(
    results: list[AgentSuccessEvaluationResult],
) -> list[HeroBucket]:
    """Build week-sized buckets across the given results."""
    if not results:
        return []
    buckets: dict[int, list[AgentSuccessEvaluationResult]] = defaultdict(list)
    for r in results:
        # Anchor each result to the START of its week (epoch-aligned).
        week_start = (r.created_at // _WEEK_SECONDS) * _WEEK_SECONDS
        buckets[week_start].append(r)
    out: list[HeroBucket] = []
    for ts in sorted(buckets):
        bucket = buckets[ts]
        out.append(
            HeroBucket(
                ts=ts,
                regular_rate=_success_rate(bucket),
                shadow_rate=None,
                regular_n=len(bucket),
                shadow_n=0,
            )
        )
    return out
