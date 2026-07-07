"""Runner + metrics for the golden-set search eval.

Two metric layers per case:

- **Mechanical** (judge-free, deterministic, the CI regression gate):
  ``recall_at_k`` / ``mrr`` of the case's ``expected_top_candidates`` within
  their own arm's ranked list, and ``must_not_first_violated`` when any
  ``must_NOT_rank_first`` item ranks first in its arm.
- **Judged** (optional, gated behind a real LLM): the shared
  :class:`~tests.eval.judge.LLMJudge` scores the ranked lists against the
  gold answer using ``judge_prompts/search_rubric.yaml``.

Every outcome carries the case's ``category`` so results aggregate per
category — temporal regressions (``temporal_current`` / ``temporal_window``
/ ``supersession``) stay visible instead of averaging away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from reflexio.server.services.retrieval.temporal import entity_timestamp
from tests.eval.search.providers import SECONDS_PER_DAY

if TYPE_CHECKING:
    from tests.eval.judge import LLMJudge
    from tests.eval.search.providers import ProviderRun, SearchProvider

SearchCase = dict[str, Any]

_SEED_LIST_TO_ARM = {
    "seeded_profiles": "profiles",
    "seeded_user_playbooks": "user_playbooks",
    "seeded_agent_playbooks": "agent_playbooks",
}


def _arm_of_key(case: SearchCase, key: str) -> str | None:
    """Return which arm a case-local seed key belongs to, or None."""
    for seed_list, arm in _SEED_LIST_TO_ARM.items():
        if any(spec["key"] == key for spec in case.get(seed_list, [])):
            return arm
    return None


def _ranked_ids(response: Any) -> dict[str, list[str]]:
    """Extract per-arm ranked id lists from a ``UnifiedSearchResponse``."""
    return {
        "profiles": [str(p.profile_id) for p in response.profiles or []],
        "user_playbooks": [
            str(p.user_playbook_id) for p in response.user_playbooks or []
        ],
        "agent_playbooks": [
            str(p.agent_playbook_id) for p in response.agent_playbooks or []
        ],
    }


def _age_days(item: Any, now: int) -> float | None:
    """Best-effort entity age in days (None when the item has no timestamp)."""
    ts = entity_timestamp(item)
    if not ts:
        return None
    return round((now - ts) / SECONDS_PER_DAY, 1)


@dataclass
class MechanicalMetrics:
    """Judge-free ranking metrics for one case.

    Attributes:
        recall_at_k: Fraction of expected candidates present anywhere in
            their arm's returned list.
        mrr: Reciprocal rank of the best-ranked expected candidate within
            its arm (0.0 when none returned).
        must_not_first_violated: True when any ``must_NOT_rank_first`` item
            is ranked first in its arm.
    """

    recall_at_k: float
    mrr: float
    must_not_first_violated: bool


def compute_mechanical(
    case: SearchCase, response: Any, key_to_id: dict[str, str]
) -> MechanicalMetrics:
    """Compute the mechanical metrics for one (case, response) pair.

    Args:
        case: The golden case dict.
        response: The backend's ``UnifiedSearchResponse``.
        key_to_id: Case-local seed key → real storage id.

    Returns:
        MechanicalMetrics: The deterministic ranking metrics.
    """
    ranked = _ranked_ids(response)

    expected_keys = case.get("expected_top_candidates", [])
    hits = 0
    best_rank: int | None = None
    for key in expected_keys:
        arm = _arm_of_key(case, key)
        real_id = key_to_id.get(key)
        if arm is None or real_id is None:
            continue
        arm_ids = ranked[arm]
        if real_id in arm_ids:
            hits += 1
            rank = arm_ids.index(real_id) + 1
            best_rank = rank if best_rank is None else min(best_rank, rank)

    violated = False
    for key in case.get("must_NOT_rank_first", []):
        arm = _arm_of_key(case, key)
        real_id = key_to_id.get(key)
        if arm and real_id and ranked[arm][:1] == [real_id]:
            violated = True

    return MechanicalMetrics(
        recall_at_k=(hits / len(expected_keys)) if expected_keys else 1.0,
        mrr=(1.0 / best_rank) if best_rank else 0.0,
        must_not_first_violated=violated,
    )


def _build_expected(case: SearchCase, key_to_id: dict[str, str]) -> dict[str, Any]:
    """Assemble what the judge sees as the gold target for ``case``."""
    return {
        "query": case["query"],
        "category": case.get("category", ""),
        "expected_answer": case.get("expected_answer", ""),
        "expected_top_candidates": [
            key_to_id.get(k, k) for k in case.get("expected_top_candidates", [])
        ],
        "must_NOT_rank_first": [
            key_to_id.get(k, k) for k in case.get("must_NOT_rank_first", [])
        ],
        "notes_for_judge": case.get("notes_for_judge", ""),
    }


def _build_actual(response: Any, now: int) -> dict[str, Any]:
    """Assemble the ranked-results payload the judge scores."""

    def project(items: list[Any], id_attr: str) -> list[dict[str, Any]]:
        return [
            {
                "id": str(getattr(item, id_attr, "")),
                "content": getattr(item, "content", ""),
                "age_days": _age_days(item, now),
            }
            for item in items or []
        ]

    return {
        "profiles": project(response.profiles, "profile_id"),
        "user_playbooks": project(response.user_playbooks, "user_playbook_id"),
        "agent_playbooks": project(response.agent_playbooks, "agent_playbook_id"),
    }


@dataclass
class CaseOutcome:
    """Per-case scoring outcome (mechanical always; judged when available)."""

    case_id: str
    category: str
    backend: str
    recall_at_k: float
    mrr: float
    must_not_first_violated: bool
    latency_ms: float
    answer_correctness: float | None = None
    grounded_rate: float | None = None
    rationale: str = ""


@dataclass
class EvalResults:
    """Aggregate metrics over a run of one backend."""

    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Total number of scored cases."""
        return len(self.outcomes)

    def by_category(self) -> dict[str, list[CaseOutcome]]:
        """Group outcomes by case category (sorted keys)."""
        grouped: dict[str, list[CaseOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.category or "uncategorized", []).append(outcome)
        return dict(sorted(grouped.items()))

    def summary(self) -> str:
        """Render an overall + per-category + per-case summary block."""
        lines = ["Search golden-set eval summary", *self._overall_lines()]
        for category, outcomes in self.by_category().items():
            lines.append(f"  [{category}] ({len(outcomes)} cases)")
            lines.extend(
                f"    {o.case_id} ({o.backend}): recall={o.recall_at_k:.2f} "
                f"mrr={o.mrr:.2f} must_not_violated={o.must_not_first_violated}"
                + (
                    f" correctness={o.answer_correctness:.2f}"
                    if o.answer_correctness is not None
                    else ""
                )
                for o in outcomes
            )
        return "\n".join(lines)

    def _overall_lines(self) -> list[str]:
        if not self.outcomes:
            return ["  cases: 0"]
        mean_recall = sum(o.recall_at_k for o in self.outcomes) / self.n
        mean_mrr = sum(o.mrr for o in self.outcomes) / self.n
        violations = sum(o.must_not_first_violated for o in self.outcomes)
        judged = [o for o in self.outcomes if o.answer_correctness is not None]
        lines = [
            f"  cases:              {self.n}",
            f"  recall@k mean:      {mean_recall:.3f}",
            f"  mrr mean:           {mean_mrr:.3f}",
            f"  must-not violations:{violations}",
        ]
        if judged:
            mean_correct = sum(o.answer_correctness or 0.0 for o in judged) / len(
                judged
            )
            lines.append(
                f"  correctness mean:   {mean_correct:.3f} ({len(judged)} judged)"
            )
        return lines


def score_case(
    *,
    case: SearchCase,
    run: ProviderRun,
    backend: str,
    judge: LLMJudge | Any | None = None,
    now: int | None = None,
) -> CaseOutcome:
    """Score one (case, provider run) pair.

    Args:
        case: The golden case dict.
        run: The provider's response + id map + latency.
        backend: Backend label stamped on the outcome (e.g. ``classic``).
        judge: Optional judge exposing ``score(*, expected, actual)``; when
            None only mechanical metrics are filled.
        now: Epoch used for ``age_days`` projection (defaults to real now).

    Returns:
        CaseOutcome: Combined mechanical (+ judged) outcome.
    """
    now = now or int(datetime.now(UTC).timestamp())
    mechanical = compute_mechanical(case, run.response, run.key_to_id)
    outcome = CaseOutcome(
        case_id=case["id"],
        category=case.get("category", ""),
        backend=backend,
        recall_at_k=mechanical.recall_at_k,
        mrr=mechanical.mrr,
        must_not_first_violated=mechanical.must_not_first_violated,
        latency_ms=run.latency_ms,
    )
    if judge is not None:
        score = judge.score(
            expected=_build_expected(case, run.key_to_id),
            actual=_build_actual(run.response, now),
        )
        outcome.answer_correctness = score.answer_correctness
        outcome.grounded_rate = score.grounded_rate
        outcome.rationale = score.rationale
    return outcome


def run_eval(
    *,
    cases: list[SearchCase],
    provider: SearchProvider,
    backend: str,
    judge: LLMJudge | Any | None = None,
) -> EvalResults:
    """Run one backend over all cases and aggregate outcomes.

    Args:
        cases: The golden cases.
        provider: Backend provider mapping a case to a :class:`ProviderRun`.
        backend: Backend label stamped on every outcome.
        judge: Optional judge; when None the run is mechanical-only.

    Returns:
        EvalResults: Per-case outcomes with category grouping.
    """
    results = EvalResults()
    for case in cases:
        run = provider(case)
        results.outcomes.append(
            score_case(case=case, run=run, backend=backend, judge=judge)
        )
    return results
