"""Runner + metrics for the golden-set extraction eval (AI-judged).

Given a list of curated golden cases and, for each, a *produced*
extraction — a ``(profiles, playbooks)`` pair, either precomputed or
obtained by running a live extractor — this module:

- shapes an ``expected`` payload from the case's gold items
  (expected profiles + playbooks, ``must_NOT_include_profiles``, and any
  ``notes_for_judge``),
- shapes an ``actual`` payload from the produced profiles + playbooks
  (pydantic entities are dumped to dicts; plain dicts pass through),
- asks the shared :class:`~tests.eval.judge.LLMJudge` (loaded with the
  ``extraction_rubric.yaml``) to score the pair, and
- aggregates the per-case ``JudgeScore`` floats into means + a pass-rate.

Unlike the reflection / consolidation *decision*-evals, extraction is
**float-scored, not label-scored**: there is no decision "kind" and no
confusion matrix. The headline metrics are the judge's:

- **signal_f1** — recall of the expected signals (with nuance such as
  TTL, supersession, and rationale folded in for nuance cases), and
- **grounded_rate** — fraction of emitted items whose source spans are
  genuinely verbatim in the session transcript.

This module owns no judge or rubric of its own: it reuses the generic
``LLMJudge`` + ``extraction_rubric.yaml`` and the golden YAML case
*dicts* (the established golden-set convention). Extractions arrive
either as a precomputed parallel list (tests use a perfect-extraction
baseline = the case's own gold items) or via an ``extraction_provider``
callable — the documented extension point for scoring a live extractor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from tests.eval.judge import LLMJudge

# The loaded golden case dict (the established golden-set convention).
ExtractionCase = dict[str, Any]
# A produced extraction: ``(profiles, playbooks)``. Kept loose so callers
# may pass ``UserProfile`` / ``UserPlaybook`` entities OR plain dicts.
Extraction = tuple[list[Any], list[Any]]
# Maps a case to its produced extraction (e.g. a live extractor wrapper).
ExtractionProvider = Callable[[ExtractionCase], Extraction]


def _to_dict(x: Any) -> Any:
    """Dump a pydantic ``BaseModel`` to a dict; pass everything else through."""
    return x.model_dump() if isinstance(x, BaseModel) else x


def _build_expected(case: ExtractionCase) -> dict[str, Any]:
    """Assemble what the judge sees as the gold target for ``case``."""
    return {
        "expected_profiles": case.get("expected_profiles", []),
        "expected_playbooks": case.get("expected_playbooks", []),
        "must_NOT_include_profiles": case.get("must_NOT_include_profiles", []),
        "notes_for_judge": case.get("notes_for_judge", ""),
    }


def _build_actual(profiles: list[Any], playbooks: list[Any]) -> dict[str, Any]:
    """Assemble the produced extraction payload the judge scores.

    Both pydantic entities and plain dicts are accepted; entities are
    normalized via ``model_dump()`` so the judge sees uniform dicts.
    """
    return {
        "profiles": [_to_dict(p) for p in profiles],
        "playbooks": [_to_dict(p) for p in playbooks],
    }


@dataclass
class CaseOutcome:
    """Per-case scoring outcome.

    Attributes:
        case_id: The case's id.
        signal_f1: Judge's expected-signal recall, in [0, 1].
        grounded_rate: Judge's source-grounding fraction, in [0, 1].
        rationale: The judge's one-paragraph explanation.
    """

    case_id: str
    signal_f1: float
    grounded_rate: float
    rationale: str = ""


@dataclass
class EvalResults:
    """Aggregate metrics over a run.

    Attributes:
        outcomes: Per-case outcomes.
    """

    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Total number of scored cases."""
        return len(self.outcomes)

    @property
    def signal_f1_mean(self) -> float:
        """Mean ``signal_f1`` over outcomes (0.0 when empty)."""
        if not self.outcomes:
            return 0.0
        return sum(o.signal_f1 for o in self.outcomes) / self.n

    @property
    def grounded_rate_mean(self) -> float:
        """Mean ``grounded_rate`` over outcomes (0.0 when empty)."""
        if not self.outcomes:
            return 0.0
        return sum(o.grounded_rate for o in self.outcomes) / self.n

    def pass_rate(self, threshold: float = 0.7) -> float:
        """Fraction of cases clearing ``threshold`` on both metrics.

        A case passes when ``signal_f1 >= threshold`` AND
        ``grounded_rate >= threshold``. Returns 0.0 when there are no
        outcomes.

        Args:
            threshold: The per-metric pass bar (default 0.7).
        """
        if not self.outcomes:
            return 0.0
        passed = sum(
            o.signal_f1 >= threshold and o.grounded_rate >= threshold
            for o in self.outcomes
        )
        return passed / self.n

    def summary(self) -> str:
        """Render a short human-readable summary block."""
        lines = [
            "Extraction golden-set eval summary",
            f"  cases:            {self.n}",
            f"  signal_f1 mean:   {self.signal_f1_mean:.3f}",
            f"  grounded mean:    {self.grounded_rate_mean:.3f}",
            f"  pass_rate@0.7:    {self.pass_rate(0.7):.3f}",
            "  per-case:",
        ]
        lines.extend(
            f"    {o.case_id}: f1={o.signal_f1:.3f} grounded={o.grounded_rate:.3f}"
            for o in self.outcomes
        )
        return "\n".join(lines)


def score_case(
    *,
    case: ExtractionCase,
    profiles: list[Any],
    playbooks: list[Any],
    judge: LLMJudge | Any,
) -> CaseOutcome:
    """Score a single (case, produced extraction) pair with the judge.

    The judge is required — extraction has no cheap mechanical-only
    metric, so there is no judge-less path. ``judge`` is duck-typed
    (``LLMJudge`` in practice, a ``MagicMock`` stub in tests).

    Args:
        case: The golden case dict.
        profiles: Produced profiles (entities or dicts).
        playbooks: Produced playbooks (entities or dicts).
        judge: An object exposing ``score(*, expected, actual)``.

    Returns:
        A :class:`CaseOutcome`.
    """
    score = judge.score(
        expected=_build_expected(case),
        actual=_build_actual(profiles, playbooks),
    )
    return CaseOutcome(
        case_id=case["id"],
        signal_f1=score.signal_f1,
        grounded_rate=score.grounded_rate,
        rationale=score.rationale,
    )


def run_eval(
    *,
    cases: list[ExtractionCase],
    extractions: list[Extraction] | None = None,
    extraction_provider: ExtractionProvider | None = None,
    judge: LLMJudge | Any,
) -> EvalResults:
    """Run the eval over a list of cases and aggregate metrics.

    Provide produced extractions either as a parallel ``extractions``
    list or via an ``extraction_provider`` callable (e.g. one that runs
    a live extractor). Exactly one of the two must be given.

    Args:
        cases: The golden cases.
        extractions: Precomputed ``(profiles, playbooks)`` pairs, one per
            case (same order).
        extraction_provider: Callable mapping a case to an extraction.
        judge: The judge used to score every case.

    Returns:
        An :class:`EvalResults` with per-case outcomes.

    Raises:
        ValueError: When neither or both extraction sources are given, or
            when ``extractions`` length does not match ``cases``.
    """
    if (extractions is None) == (extraction_provider is None):
        raise ValueError(
            "provide exactly one of 'extractions' or 'extraction_provider'"
        )
    if extractions is not None and len(extractions) != len(cases):
        raise ValueError(
            f"extractions length {len(extractions)} != cases length {len(cases)}"
        )

    results = EvalResults()
    for idx, case in enumerate(cases):
        profiles, playbooks = (
            extractions[idx] if extractions is not None else extraction_provider(case)  # type: ignore[misc]
        )
        outcome = score_case(
            case=case,
            profiles=profiles,
            playbooks=playbooks,
            judge=judge,
        )
        results.outcomes.append(outcome)
    return results
