"""Metrics runner for the playbook ask_human invocation eval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from tests.eval.playbook_ask_human.case import PlaybookAskHumanCase


@dataclass
class AskHumanPrediction:
    """Provider output for one case."""

    case_id: str
    tool_names: list[str] = field(default_factory=list)
    question_texts: list[str] = field(default_factory=list)
    playbook_count: int = 0

    @property
    def asked_human(self) -> bool:
        return "ask_human" in self.tool_names


@dataclass
class CaseOutcome:
    """Scored result for one labeled case."""

    case_id: str
    vertical: str
    expected_ask_human: bool
    actual_ask_human: bool
    expected_playbooks_needed: bool
    playbook_count: int
    question_matches: bool
    tool_names: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.expected_ask_human == self.actual_ask_human


@dataclass
class LabelMetrics:
    """Binary classification metrics for ask_human invocation."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0


@dataclass
class EvalResults:
    """Aggregate ask_human eval results."""

    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def metrics(self) -> LabelMetrics:
        metrics = LabelMetrics()
        for outcome in self.outcomes:
            if outcome.expected_ask_human and outcome.actual_ask_human:
                metrics.tp += 1
            elif not outcome.expected_ask_human and outcome.actual_ask_human:
                metrics.fp += 1
            elif not outcome.expected_ask_human and not outcome.actual_ask_human:
                metrics.tn += 1
            else:
                metrics.fn += 1
        return metrics

    @property
    def by_vertical(self) -> dict[str, LabelMetrics]:
        grouped: dict[str, LabelMetrics] = defaultdict(LabelMetrics)
        for outcome in self.outcomes:
            metrics = grouped[outcome.vertical]
            if outcome.expected_ask_human and outcome.actual_ask_human:
                metrics.tp += 1
            elif not outcome.expected_ask_human and outcome.actual_ask_human:
                metrics.fp += 1
            elif not outcome.expected_ask_human and not outcome.actual_ask_human:
                metrics.tn += 1
            else:
                metrics.fn += 1
        return dict(sorted(grouped.items()))

    def summary(self) -> str:
        metrics = self.metrics
        lines = [
            "Playbook ask_human invocation eval summary",
            f"  cases:      {metrics.n}",
            f"  tp/fp/tn/fn: {metrics.tp}/{metrics.fp}/{metrics.tn}/{metrics.fn}",
            f"  precision:  {metrics.precision:.3f}",
            f"  recall:     {metrics.recall:.3f}",
            f"  f1:         {metrics.f1:.3f}",
            "  per-vertical:",
        ]
        lines.extend(
            "    "
            f"{vertical}: n={m.n} p={m.precision:.3f} r={m.recall:.3f} "
            f"f1={m.f1:.3f}"
            for vertical, m in self.by_vertical.items()
        )
        return "\n".join(lines)


PredictionProvider = Callable[[PlaybookAskHumanCase], AskHumanPrediction]


def _question_matches(
    case: PlaybookAskHumanCase, prediction: AskHumanPrediction
) -> bool:
    if not case.expected_question_must_include:
        return True
    combined = " ".join(prediction.question_texts).lower()
    return all(
        fragment.lower() in combined for fragment in case.expected_question_must_include
    )


def score_case(
    *,
    case: PlaybookAskHumanCase,
    prediction: AskHumanPrediction,
) -> CaseOutcome:
    """Score one case against a provider prediction."""

    if prediction.case_id != case.id:
        raise ValueError(
            f"prediction for {prediction.case_id} does not match {case.id}"
        )
    return CaseOutcome(
        case_id=case.id,
        vertical=case.vertical,
        expected_ask_human=case.expected_ask_human,
        actual_ask_human=prediction.asked_human,
        expected_playbooks_needed=case.expected_playbooks_needed,
        playbook_count=prediction.playbook_count,
        question_matches=_question_matches(case, prediction),
        tool_names=prediction.tool_names,
    )


def run_eval(
    *,
    cases: list[PlaybookAskHumanCase],
    predictions: list[AskHumanPrediction] | None = None,
    prediction_provider: PredictionProvider | None = None,
) -> EvalResults:
    """Run the eval from fixed predictions or a live provider."""

    if (predictions is None) == (prediction_provider is None):
        raise ValueError(
            "provide exactly one of 'predictions' or 'prediction_provider'"
        )
    if predictions is not None and len(predictions) != len(cases):
        raise ValueError(
            f"predictions length {len(predictions)} != cases length {len(cases)}"
        )

    outcomes: list[CaseOutcome] = []
    for idx, case in enumerate(cases):
        prediction = (
            predictions[idx] if predictions is not None else prediction_provider(case)  # type: ignore[misc]
        )
        outcomes.append(score_case(case=case, prediction=prediction))
    return EvalResults(outcomes=outcomes)
