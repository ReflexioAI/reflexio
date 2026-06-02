"""Runner + metrics for the reflection decision-eval harness.

Given a list of cases and, for each, a *produced* reflection decision
(either precomputed or obtained by running the reflection extractor),
this module:

- maps each produced decision to a coarse label,
- compares it to the case's ``gold_label``,
- optionally asks the AI judge whether the decision matches intent, and
- computes aggregate metrics: decision accuracy (overall + per-label
  confusion), false-tighten rate, and an over-specialization flag.

Two notions of "correct" are tracked separately:

- **label accuracy** — does the mechanically-derived label equal the
  gold label? (cheap, deterministic, no LLM)
- **judge agreement** — does the AI judge say the decision matches the
  gold intent? (the headline AI-judged metric)

A case can be label-wrong but judge-right (e.g. a different-but-valid
trigger), which is exactly why the AI judge exists.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tests.eval.reflection.case import ReflectionEvalCase, label_for_decision
from tests.eval.reflection.judge import judge_reflection_decision

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.reflection.reflection_service_utils import (
        ReflectionDecision,
    )

# A produced decision paired with the case it answers.
DecisionProvider = Callable[[ReflectionEvalCase], "ReflectionDecision"]


@dataclass
class CaseOutcome:
    """Per-case scoring outcome.

    Attributes:
        case_id: The case's id.
        gold_label: The case's intended label.
        produced_label: The mechanically-derived label of the produced
            decision (may be ``"scope"`` for an ambiguous trigger change).
        label_match: Whether ``produced_label == gold_label``.
        judge_correct: AI-judge verdict, or None when the judge was not
            run.
        over_specialized: True when a tighten/widen case produced a
            trigger that collapses to a single instance (heuristic).
        judge_reason: The judge's one-line reason, if any.
    """

    case_id: str
    gold_label: str
    produced_label: str
    label_match: bool
    judge_correct: bool | None = None
    over_specialized: bool = False
    judge_reason: str = ""


@dataclass
class EvalResults:
    """Aggregate metrics over a run.

    Attributes:
        outcomes: Per-case outcomes.
        confusion: ``Counter`` keyed by ``(gold_label, produced_label)``.
    """

    outcomes: list[CaseOutcome] = field(default_factory=list)
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def n(self) -> int:
        """Total number of scored cases."""
        return len(self.outcomes)

    @property
    def label_accuracy(self) -> float:
        """Fraction of cases whose produced label equals the gold label."""
        if not self.outcomes:
            return 0.0
        return sum(o.label_match for o in self.outcomes) / self.n

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
    def false_tighten_rate(self) -> float:
        """Fraction of cases that wrongly tightened.

        A *false tighten* is a case whose gold label is NOT ``tighten``
        but whose produced label IS ``tighten`` — the reflection step
        narrowed scope when it should not have. Denominator is all
        non-tighten gold cases (the population that *could* be falsely
        tightened). Returns 0.0 when there are no such cases.
        """
        eligible = [o for o in self.outcomes if o.gold_label != "tighten"]
        if not eligible:
            return 0.0
        false_tightens = sum(o.produced_label == "tighten" for o in eligible)
        return false_tightens / len(eligible)

    @property
    def over_specialization_rate(self) -> float:
        """Fraction of scored cases flagged as over-specialized."""
        if not self.outcomes:
            return 0.0
        return sum(o.over_specialized for o in self.outcomes) / self.n

    def summary(self) -> str:
        """Render a short human-readable summary block."""
        lines = [
            "Reflection decision-eval summary",
            f"  cases:              {self.n}",
            f"  label accuracy:     {self.label_accuracy:.3f}",
        ]
        ja = self.judge_accuracy
        lines.append(
            f"  judge agreement:    {ja:.3f}"
            if ja is not None
            else "  judge agreement:    (not run)"
        )
        lines.append(f"  false-tighten rate: {self.false_tighten_rate:.3f}")
        lines.append(f"  over-specialization:{self.over_specialization_rate:.3f}")
        lines.append("  confusion (gold -> produced):")
        for (gold, produced), count in sorted(self.confusion.items()):
            lines.append(f"    {gold:>10} -> {produced:<10} {count}")
        return "\n".join(lines)


def _is_single_instance_trigger(trigger: str | None) -> bool:
    """Heuristic: does a trigger collapse to a single concrete instance?

    Over-specialization happens when reflection rewrites a general
    trigger into one that only ever fires for one specific case (e.g.
    pins an exact file path, a quoted literal, or a specific id). We
    approximate this cheaply: a trigger that contains a quoted literal,
    looks like a path, or is very short and contains a digit-bearing
    token is treated as single-instance.

    Args:
        trigger: The proposed replacement trigger.

    Returns:
        True when the trigger looks like it pins a single instance.
    """
    if not trigger:
        return False
    t = trigger.strip()
    if '"' in t or "'" in t:
        return True
    if "/" in t or "\\" in t:  # path-like
        return True
    tokens = t.split()
    # Short trigger that names a specific id/number — likely one instance.
    has_numeric_token = any(any(ch.isdigit() for ch in tok) for tok in tokens)
    return len(tokens) <= 4 and has_numeric_token


def score_case(
    *,
    case: ReflectionEvalCase,
    produced_decision: ReflectionDecision,
    llm_client: LiteLLMClient | Any | None = None,
    rubric: dict[str, Any] | None = None,
    panel_size: int = 1,
) -> CaseOutcome:
    """Score a single (case, produced_decision) pair.

    When ``llm_client`` is provided the AI judge is invoked; otherwise
    only the deterministic label comparison is computed.

    Args:
        case: The eval case.
        produced_decision: The decision produced for the cited item.
        llm_client: Optional judge client. None skips judging.
        rubric: Optional judge rubric.
        panel_size: Judge panel size (see ``judge_reflection_decision``).

    Returns:
        A :class:`CaseOutcome`.
    """
    produced_label = label_for_decision(produced_decision, case.cited_item)
    label_match = produced_label == case.gold_label

    over_specialized = case.gold_label in ("tighten", "widen") and (
        _is_single_instance_trigger(produced_decision.new_trigger)
    )

    judge_correct: bool | None = None
    judge_reason = ""
    if llm_client is not None:
        verdict = judge_reflection_decision(
            case=case,
            produced_decision=produced_decision,
            llm_client=llm_client,
            rubric=rubric,
            panel_size=panel_size,
        )
        judge_correct = verdict.correct
        judge_reason = verdict.reason

    return CaseOutcome(
        case_id=case.id,
        gold_label=case.gold_label,
        produced_label=produced_label,
        label_match=label_match,
        judge_correct=judge_correct,
        over_specialized=over_specialized,
        judge_reason=judge_reason,
    )


def run_eval(
    *,
    cases: list[ReflectionEvalCase],
    decisions: list[ReflectionDecision] | None = None,
    decision_provider: DecisionProvider | None = None,
    llm_client: LiteLLMClient | Any | None = None,
    rubric: dict[str, Any] | None = None,
    panel_size: int = 1,
) -> EvalResults:
    """Run the eval over a list of cases and aggregate metrics.

    Provide produced decisions either as a parallel ``decisions`` list or
    via a ``decision_provider`` callable (e.g. one that runs the live
    reflection extractor). Exactly one of the two must be given.

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
        ValueError: When neither or both decision sources are given, or
            when ``decisions`` length does not match ``cases``.
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
        results.confusion[(outcome.gold_label, outcome.produced_label)] += 1
    return results
