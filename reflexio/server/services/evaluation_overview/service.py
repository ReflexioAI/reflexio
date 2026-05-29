"""Aggregator that composes hero, tiles, rule attribution, and distribution.

The service does three reads (results, citations, playbook stats)
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
    Request,
)
from reflexio.models.api_schema.eval_overview_schema import (
    BraintrustTileRow,
    BucketLiteral,
    ContextTile,
    GetEvaluationOverviewRequest,
    GetEvaluationOverviewResponse,
    HeroBlock,
    HeroBucket,
    NumberWithDelta,
    PercentWithDelta,
    RuleAttributionRow,
    ScoreDistribution,
    ShadowWinRateTrend,
    SuccessRateTrendByGroup,
)
from reflexio.models.config_schema import Config
from reflexio.server.services.evaluation_overview.distribution import (
    BUCKET_LABELS,
    bucket_corrections,
)
from reflexio.server.services.evaluation_overview.group_aggregation import (
    compute_trend_by_group,
)
from reflexio.server.services.evaluation_overview.hero_state import (
    compute_hero_state,
)
from reflexio.server.services.evaluation_overview.rule_attribution import (
    compute_net_sessions,
)
from reflexio.server.services.evaluation_overview.shadow_aggregation import (
    compute_shadow_win_rate_trend,
)

_DAY_SECONDS = 24 * 60 * 60
_WEEK_SECONDS = 7 * 24 * 60 * 60
_TOP_N_RULES = 5


@dataclass
class EvaluationOverviewService:
    """Builds the full /api/get_evaluation_overview payload.

    The service holds a storage handle and the org's current Config. Each
    call to `run` performs the three reads and returns a fresh response.
    Stateless across calls — safe to reuse the instance across requests.
    """

    storage: object
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
        all_results = self.storage.get_agent_success_evaluation_results(  # type: ignore[attr-defined]
            agent_version=None, limit=10_000
        )
        results = [
            r for r in all_results if request.from_ts <= r.created_at <= request.to_ts
        ]

        # Tile + distribution baselines are always last-7d-vs-prior-7d, anchored
        # to ``request.to_ts`` (which is "now" from the frontend's perspective).
        cur_7d_from = max(request.from_ts, request.to_ts - _WEEK_SECONDS)
        prev_to = cur_7d_from
        prev_from = max(0, prev_to - _WEEK_SECONDS)
        results_current_7d = [
            r for r in all_results if cur_7d_from <= r.created_at <= request.to_ts
        ]
        results_prev_7d = [
            r for r in all_results if prev_from <= r.created_at < prev_to
        ]

        earliest_eval_ts = min((r.created_at for r in all_results), default=None)
        hero = self._build_hero(request, results, earliest_eval_ts)
        tiles = self._build_tiles(results_current_7d, results_prev_7d)
        attribution = self._build_attribution(results)
        distribution = self._build_distribution(results_current_7d, results_prev_7d)
        braintrust_tiles = self._build_braintrust_tiles(
            cur_7d_from, request.to_ts, prev_from, prev_to
        )
        # F2: group-aware success-rate trend curves. Joins each eval result
        # with its session's first request metadata to bucket sessions into
        # treatment / control / untagged, then time-buckets each curve at
        # the same granularity as the hero chart.
        success_rate_trend_by_group = self._build_group_trend(results, request.bucket)
        # F1: per-turn shadow win-rate trend. Filters verdicts to the org's
        # pinned judge prompt version so a future rubric bump doesn't
        # silently mix epochs into the headline.
        shadow_win_rate_trend = self._build_shadow_win_rate_trend(
            request.from_ts, request.to_ts
        )

        return GetEvaluationOverviewResponse(
            hero=hero,
            context_tiles=tiles,
            rule_attribution=attribution,
            score_distribution=distribution,
            braintrust_tiles=braintrust_tiles,
            success_rate_trend_by_group=success_rate_trend_by_group,
            shadow_win_rate_trend=shadow_win_rate_trend,
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

    def _build_group_trend(
        self,
        results: list[AgentSuccessEvaluationResult],
        bucket: str,
    ) -> SuccessRateTrendByGroup:
        """Build the dual+untagged-curve trend payload for the window.

        Joins each eval result with the first request of its session to read
        ``metadata.reflexio_retrieval_enabled``, then delegates to the pure
        ``compute_trend_by_group`` aggregator. Returns the default empty
        ``SuccessRateTrendByGroup`` when ``results`` is empty.

        Args:
            results (list[AgentSuccessEvaluationResult]): Eval results in the
                trend window.
            bucket (str): Bucket granularity literal (``"week"`` or ``"day"``).

        Returns:
            SuccessRateTrendByGroup: Three curves (any of which may be empty).
        """
        if not results:
            return SuccessRateTrendByGroup()
        bucket_seconds = _WEEK_SECONDS if bucket == "week" else _DAY_SECONDS
        outcomes = self._build_group_outcomes(results)
        return compute_trend_by_group(outcomes, bucket_seconds)

    def _build_group_outcomes(
        self,
        results: list[AgentSuccessEvaluationResult],
    ) -> list[tuple[str, int, bool, dict]]:
        """Pair each eval result with its session's first-request metadata.

        Caches ``session_id → first_request_metadata`` so storage is hit
        once per distinct session, not once per result. Sessions without a
        matching request fall through with an empty-dict metadata, which the
        downstream aggregator routes to the UNTAGGED bucket.

        Args:
            results (list[AgentSuccessEvaluationResult]): Eval results to
                annotate with their session's first-request metadata.

        Returns:
            list[tuple[str, int, bool, dict]]: Tuples of
            ``(session_id, created_at, is_success, first_request_metadata)``
            suitable for :func:`compute_trend_by_group`.
        """
        session_ids = {r.session_id for r in results if r.session_id}
        first_request_metadata: dict[str, dict] = {}
        for sid in session_ids:
            reqs = self._get_session_requests(sid)
            if reqs:
                first = min(reqs, key=lambda r: r.created_at)
                first_request_metadata[sid] = first.metadata or {}
            else:
                first_request_metadata[sid] = {}

        return [
            (
                r.session_id or "",
                r.created_at,
                r.is_success,
                first_request_metadata.get(r.session_id or "", {}),
            )
            for r in results
        ]

    def _get_session_requests(self, session_id: str) -> list[Request]:
        """Return every request in ``session_id`` regardless of user.

        Uses ``BaseStorage.get_sessions(session_id=...)`` because the
        per-session, user-id-required ``get_requests_by_session`` doesn't
        fit our caller (eval results don't carry ``user_id``). All locally
        testable backends ignore the ``user_id`` filter when it's ``None``,
        and ``top_k`` is raised well above realistic per-session counts so
        we don't truncate.

        Args:
            session_id (str): The session whose requests to fetch.

        Returns:
            list[Request]: All requests in the session, in storage order.
        """
        grouped = self.storage.get_sessions(  # type: ignore[attr-defined]
            session_id=session_id, top_k=1000
        )
        return [
            entry.request
            for entries in grouped.values()
            for entry in entries
            if entry.request is not None
        ]

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
