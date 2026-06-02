"""Runner + metrics for the consolidation decision-eval harness.

Given a list of cases and, for each, a *produced* consolidation decision
(either precomputed or obtained by running the live consolidator), this
module:

- reads each produced decision's explicit kind,
- compares it to the case's ``gold_kind``,
- optionally asks the AI judge whether the decision matches intent (and,
  for a ``unify``, whether it self-contradicts), and
- computes aggregate metrics: kind accuracy (overall + per-kind confusion),
  over-merge / under-merge rates, and the self-contradiction rate.

Two notions of "correct" are tracked separately:

- **kind accuracy** — does the explicit produced kind equal the gold kind?
  (cheap, deterministic, no LLM)
- **judge agreement** — does the AI judge say the decision matches the gold
  intent? (the headline AI-judged metric)

A case can be kind-wrong but judge-right (e.g. a defensible decision the
case author didn't anticipate), which is exactly why the AI judge exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tests.eval.consolidation.case import ConsolidationEvalCase, kind_for_decision
from tests.eval.consolidation.judge import judge_consolidation_decision

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.playbook.playbook_consolidator import (
        ConsolidationDecision,
    )

# A callable mapping a case to a produced decision (e.g. one that runs the
# live consolidator's decision step).
DecisionProvider = Callable[[ConsolidationEvalCase], "ConsolidationDecision"]

# Gold/produced kinds that mean "keep the rules separate".
_SEPARATE_KINDS = frozenset({"differentiate", "independent"})
# Gold/produced kinds that mean "fold the rules together".
_MERGE_KINDS = frozenset({"unify", "reject_new"})


@dataclass
class CaseOutcome:
    """Per-case scoring outcome.

    Attributes:
        case_id: The case's id.
        gold_kind: The case's intended kind.
        produced_kind: The explicit kind of the produced decision.
        kind_match: Whether ``produced_kind == gold_kind``.
        judge_correct: AI-judge verdict, or None when the judge was not run.
        self_contradiction: AI-judge self-contradiction flag for a produced
            ``unify``, or None when the judge was not run or the produced
            kind was not ``unify`` (the flag is only meaningful for unify).
        judge_reason: The judge's one-line reason, if any.
    """

    case_id: str
    gold_kind: str
    produced_kind: str
    kind_match: bool
    judge_correct: bool | None = None
    self_contradiction: bool | None = None
    judge_reason: str = ""


@dataclass
class EvalResults:
    """Aggregate metrics over a run.

    Attributes:
        outcomes: Per-case outcomes.
        confusion: ``Counter`` keyed by ``(gold_kind, produced_kind)``.
    """

    outcomes: list[CaseOutcome] = field(default_factory=list)
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def n(self) -> int:
        """Total number of scored cases."""
        return len(self.outcomes)

    @property
    def kind_accuracy(self) -> float:
        """Fraction of cases whose produced kind equals the gold kind."""
        if not self.outcomes:
            return 0.0
        return sum(o.kind_match for o in self.outcomes) / self.n

    @property
    def judge_accuracy(self) -> float | None:
        """Fraction of judged cases the AI judge marked correct.

        Returns None when no case was judged.
        """
        judged = [o for o in self.outcomes if o.judge_correct is not None]
        if not judged:
            return None
        return sum(bool(o.judge_correct) for o in judged) / len(judged)

    @property
    def over_merge_rate(self) -> float:
        """Fraction of should-stay-separate cases that were merged.

        An *over-merge* is a case whose gold kind says the rules should stay
        separate (``differentiate`` / ``independent``) but whose produced
        decision folded them together (``unify`` / ``reject_new``).
        Denominator is all should-stay-separate gold cases. Returns 0.0
        when there are no such cases.
        """
        eligible = [o for o in self.outcomes if o.gold_kind in _SEPARATE_KINDS]
        if not eligible:
            return 0.0
        over_merges = sum(o.produced_kind in _MERGE_KINDS for o in eligible)
        return over_merges / len(eligible)

    @property
    def under_merge_rate(self) -> float:
        """Fraction of should-merge cases that were kept separate.

        An *under-merge* is a case whose gold kind says the rules should be
        folded together (``unify`` / ``reject_new``) but whose produced
        decision kept them separate (``differentiate`` / ``independent``).
        Denominator is all should-merge gold cases. Returns 0.0 when there
        are no such cases.
        """
        eligible = [o for o in self.outcomes if o.gold_kind in _MERGE_KINDS]
        if not eligible:
            return 0.0
        under_merges = sum(o.produced_kind in _SEPARATE_KINDS for o in eligible)
        return under_merges / len(eligible)

    @property
    def self_contradiction_rate(self) -> float | None:
        """Fraction of judged ``unify`` decisions flagged self-contradictory.

        Restricted to outcomes whose produced kind is ``unify`` AND which
        were judged (``self_contradiction is not None``). Returns None when
        there are no such outcomes.
        """
        judged_unify = [
            o
            for o in self.outcomes
            if o.produced_kind == "unify" and o.self_contradiction is not None
        ]
        if not judged_unify:
            return None
        return sum(bool(o.self_contradiction) for o in judged_unify) / len(judged_unify)

    def summary(self) -> str:
        """Render a short human-readable summary block."""
        lines = [
            "Consolidation decision-eval summary",
            f"  cases:              {self.n}",
            f"  kind accuracy:      {self.kind_accuracy:.3f}",
        ]
        ja = self.judge_accuracy
        lines.append(
            f"  judge agreement:    {ja:.3f}"
            if ja is not None
            else "  judge agreement:    (not run)"
        )
        lines.append(f"  over-merge rate:    {self.over_merge_rate:.3f}")
        lines.append(f"  under-merge rate:   {self.under_merge_rate:.3f}")
        scr = self.self_contradiction_rate
        lines.append(
            f"  self-contradiction: {scr:.3f}"
            if scr is not None
            else "  self-contradiction: (not run)"
        )
        lines.append("  confusion (gold -> produced):")
        for (gold, produced), count in sorted(self.confusion.items()):
            lines.append(f"    {gold:>13} -> {produced:<13} {count}")
        return "\n".join(lines)


def score_case(
    *,
    case: ConsolidationEvalCase,
    produced_decision: ConsolidationDecision,
    llm_client: LiteLLMClient | Any | None = None,
    rubric: dict[str, Any] | None = None,
    panel_size: int = 1,
) -> CaseOutcome:
    """Score a single (case, produced_decision) pair.

    When ``llm_client`` is provided the AI judge is invoked; otherwise only
    the deterministic kind comparison is computed. The ``self_contradiction``
    flag is recorded only when the produced kind is ``unify`` (it is not
    meaningful otherwise); for any other kind it stays None even when the
    judge ran.

    Args:
        case: The eval case.
        produced_decision: The decision produced for the candidate.
        llm_client: Optional judge client. None skips judging.
        rubric: Optional judge rubric.
        panel_size: Judge panel size (see ``judge_consolidation_decision``).

    Returns:
        A :class:`CaseOutcome`.
    """
    produced_kind = kind_for_decision(produced_decision)
    kind_match = produced_kind == case.gold_kind

    judge_correct: bool | None = None
    self_contradiction: bool | None = None
    judge_reason = ""
    if llm_client is not None:
        verdict = judge_consolidation_decision(
            case=case,
            produced_decision=produced_decision,
            llm_client=llm_client,
            rubric=rubric,
            panel_size=panel_size,
        )
        judge_correct = verdict.correct
        judge_reason = verdict.reason
        if produced_kind == "unify":
            self_contradiction = verdict.self_contradiction

    return CaseOutcome(
        case_id=case.id,
        gold_kind=case.gold_kind,
        produced_kind=produced_kind,
        kind_match=kind_match,
        judge_correct=judge_correct,
        self_contradiction=self_contradiction,
        judge_reason=judge_reason,
    )


def run_eval(
    *,
    cases: list[ConsolidationEvalCase],
    decisions: list[ConsolidationDecision] | None = None,
    decision_provider: DecisionProvider | None = None,
    llm_client: LiteLLMClient | Any | None = None,
    rubric: dict[str, Any] | None = None,
    panel_size: int = 1,
) -> EvalResults:
    """Run the eval over a list of cases and aggregate metrics.

    Provide produced decisions either as a parallel ``decisions`` list or
    via a ``decision_provider`` callable (e.g. one that runs the live
    consolidator's decision step). Exactly one of the two must be given.

    Args:
        cases: The eval cases.
        decisions: Precomputed decisions, one per case (same order).
        decision_provider: Callable mapping a case to a decision.
        llm_client: Optional judge client (None skips judging).
        rubric: Optional judge rubric.
        panel_size: Judge panel size.

    Returns:
        An :class:`EvalResults` with per-case outcomes and confusion.

    Raises:
        ValueError: When neither or both decision sources are given, or when
            ``decisions`` length does not match ``cases``.
    """
    if (decisions is None) == (decision_provider is None):
        raise ValueError("provide exactly one of 'decisions' or 'decision_provider'")
    if decisions is not None and len(decisions) != len(cases):
        raise ValueError(
            f"decisions length {len(decisions)} != cases length {len(cases)}"
        )

    results = EvalResults()
    for idx, case in enumerate(cases):
        produced = (
            decisions[idx] if decisions is not None else decision_provider(case)  # type: ignore[misc]
        )
        outcome = score_case(
            case=case,
            produced_decision=produced,
            llm_client=llm_client,
            rubric=rubric,
            panel_size=panel_size,
        )
        results.outcomes.append(outcome)
        results.confusion[(outcome.gold_kind, outcome.produced_kind)] += 1
    return results
