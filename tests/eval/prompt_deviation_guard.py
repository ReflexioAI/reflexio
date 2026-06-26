"""Prompt deviation guard — gate a candidate prompt version against a baseline.

When you iterate on a memory prompt and add a new version, this runs that
component's deterministic eval harness under **two pinned prompt versions** —
the prior (baseline) and the new (candidate) — and fails if any guard metric
regresses beyond a tolerance, so a degrading edit is caught before you adopt it.

Two components are covered, each driven by its existing live provider + judge:

- ``consolidation`` -> ``playbook_consolidation`` prompt, via
  ``tests/eval/consolidation``
- ``reflection`` -> ``memory_reflection`` prompt, via ``tests/eval/reflection``

Version pinning needs no core change: ``RequestContext`` builds its own
``PromptManager``, but the providers read ``ctx.prompt_manager``, so we swap in
``PromptManager(version_override={prompt_id: version})`` after constructing the
context. A non-active version file loads fine as long as it exists in the
prompt bank.

This makes real LLM calls (provider + optional judge), so it is a manual /
gated tool — run with API keys. The pure gate logic (:func:`compare`) is unit
tested without any LLM.

Usage (from the repository root)::

    uv run python -m tests.eval.prompt_deviation_guard \\
        --component consolidation \\
        --candidate-version v2.4.0 \\
        --judge-model claude-haiku-4-5

Exit codes: 0 when the candidate holds (PASS), 1 when it regresses (FAIL), and
2 on an eval/infra error (provider or judge failed) — distinct from 1 so an
orchestrator never misreads an outage as a prompt regression.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Literal

Component = Literal["consolidation", "reflection"]
COMPONENTS: tuple[Component, ...] = ("consolidation", "reflection")

# Default per-metric tolerance: a higher-is-better metric may not drop by more
# than this, and a lower-is-better metric may not rise by more than this.
DEFAULT_TOLERANCE = 0.05


@dataclass(frozen=True)
class MetricSpec:
    """A guard metric on ``EvalResults`` and the direction that is "better".

    Attributes:
        name: The ``EvalResults`` property to read (e.g. ``kind_accuracy``).
        higher_is_better: True when a larger value is an improvement (so a
            *drop* is a regression); False when smaller is better (so a *rise*
            is a regression).
    """

    name: str
    higher_is_better: bool


# Guard metrics per component. ``judge_accuracy`` / ``self_contradiction_rate``
# can be None (no judge, or no judged ``unify``); such metrics are skipped, not
# failed (see :func:`compare`).
_METRIC_SPECS: dict[Component, tuple[MetricSpec, ...]] = {
    "consolidation": (
        MetricSpec("kind_accuracy", higher_is_better=True),
        MetricSpec("judge_accuracy", higher_is_better=True),
        MetricSpec("over_merge_rate", higher_is_better=False),
        MetricSpec("under_merge_rate", higher_is_better=False),
        MetricSpec("self_contradiction_rate", higher_is_better=False),
    ),
    "reflection": (
        MetricSpec("label_accuracy", higher_is_better=True),
        MetricSpec("judge_accuracy", higher_is_better=True),
        MetricSpec("false_tighten_rate", higher_is_better=False),
        MetricSpec("over_specialization_rate", higher_is_better=False),
    ),
}

# The per-case produced field used for the baseline<->candidate agreement rate.
_PRODUCED_ATTR: dict[Component, str] = {
    "consolidation": "produced_kind",
    "reflection": "produced_label",
}


@dataclass
class MetricDelta:
    """One metric's baseline/candidate comparison.

    Attributes:
        name: Metric name.
        higher_is_better: Direction of improvement.
        baseline: Baseline value, or None when not computed.
        candidate: Candidate value, or None when not computed.
        delta: ``candidate - baseline`` when both present, else None.
        regressed: True when the candidate regressed beyond tolerance.
    """

    name: str
    higher_is_better: bool
    baseline: float | None
    candidate: float | None
    delta: float | None
    regressed: bool

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of this metric delta."""
        return {
            "name": self.name,
            "higher_is_better": self.higher_is_better,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "regressed": self.regressed,
        }


@dataclass
class GuardReport:
    """The full baseline-vs-candidate comparison and gate verdict.

    Attributes:
        component: Which component was guarded.
        baseline_version: Pinned baseline prompt version (or None for active).
        candidate_version: Pinned candidate prompt version.
        tolerance: Per-metric tolerance applied.
        metrics: Per-metric deltas.
        agreement_rate: Fraction of cases whose produced kind/label is
            unchanged between baseline and candidate (raw deviation signal,
            independent of correctness). None when there are no cases.
        passed: True when no guard metric regressed beyond tolerance.
    """

    component: Component
    baseline_version: str | None
    candidate_version: str
    tolerance: float
    metrics: list[MetricDelta] = field(default_factory=list)
    agreement_rate: float | None = None
    passed: bool = True

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of the whole report."""
        return {
            "component": self.component,
            "baseline_version": self.baseline_version,
            "candidate_version": self.candidate_version,
            "tolerance": self.tolerance,
            "agreement_rate": self.agreement_rate,
            "passed": self.passed,
            "metrics": [m.to_json() for m in self.metrics],
        }

    def render(self) -> str:
        """Render a human-readable table + verdict."""

        def fmt(v: float | None) -> str:
            return "  --  " if v is None else f"{v:.3f}"

        lines = [
            f"Prompt deviation guard — {self.component}",
            f"  baseline: {self.baseline_version or '(active)'}    "
            f"candidate: {self.candidate_version}    tolerance: {self.tolerance:.3f}",
            "",
            f"  {'metric':<24} {'baseline':>9} {'candidate':>10} {'delta':>8}  dir   status",
        ]
        for m in self.metrics:
            arrow = "↑" if m.higher_is_better else "↓"
            status = (
                "REGRESSED" if m.regressed else ("n/a" if m.delta is None else "ok")
            )
            delta = "   --  " if m.delta is None else f"{m.delta:+.3f}"
            lines.append(
                f"  {m.name:<24} {fmt(m.baseline):>9} {fmt(m.candidate):>10} "
                f"{delta:>8}  {arrow}    {status}"
            )
        lines.append("")
        lines.append(f"  agreement_rate: {fmt(self.agreement_rate)}")
        lines.append(f"  VERDICT: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


def _agreement_rate(component: Component, baseline, candidate) -> float | None:
    """Fraction of shared cases whose produced kind/label is unchanged.

    Args:
        component: Which component (selects the produced attribute).
        baseline: Baseline ``EvalResults``.
        candidate: Candidate ``EvalResults``.

    Returns:
        The agreement fraction over cases present in both runs, or None when
        there are no shared cases.
    """
    attr = _PRODUCED_ATTR[component]
    base = {o.case_id: getattr(o, attr) for o in baseline.outcomes}
    cand = {o.case_id: getattr(o, attr) for o in candidate.outcomes}
    shared = base.keys() & cand.keys()
    if not shared:
        return None
    return sum(base[cid] == cand[cid] for cid in shared) / len(shared)


def compare(
    *,
    component: Component,
    baseline,
    candidate,
    tolerance: float = DEFAULT_TOLERANCE,
    baseline_version: str | None = None,
    candidate_version: str = "",
) -> GuardReport:
    """Compare two eval runs and decide whether the candidate holds.

    Pure (no LLM): operates on two already-computed ``EvalResults``. A metric
    with a None value on either side is skipped (not failed) — judge metrics
    are None when no judge ran, and ``self_contradiction_rate`` is None when no
    ``unify`` was produced.

    Args:
        component: Which component's metric specs to apply.
        baseline: Baseline ``EvalResults``.
        candidate: Candidate ``EvalResults``.
        tolerance: Max allowed adverse change per metric.
        baseline_version: Baseline version label (for the report only).
        candidate_version: Candidate version label (for the report only).

    Returns:
        A :class:`GuardReport`; ``passed`` is False if any metric regressed.

    Raises:
        ValueError: If ``tolerance`` is negative (a negative tolerance would
            invert the regression test and flag unchanged metrics).
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    report = GuardReport(
        component=component,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        tolerance=tolerance,
        agreement_rate=_agreement_rate(component, baseline, candidate),
    )
    for spec in _METRIC_SPECS[component]:
        b = getattr(baseline, spec.name)
        c = getattr(candidate, spec.name)
        delta = None if (b is None or c is None) else c - b
        regressed = delta is not None and (
            delta < -tolerance if spec.higher_is_better else delta > tolerance
        )
        report.metrics.append(
            MetricDelta(
                name=spec.name,
                higher_is_better=spec.higher_is_better,
                baseline=b,
                candidate=c,
                delta=delta,
                regressed=regressed,
            )
        )
    report.passed = not any(m.regressed for m in report.metrics)
    return report


def _run_component_eval(
    *,
    component: Component,
    version: str | None,
    model: str,
    judge_model: str | None,
):
    """Run a component's eval harness under a pinned prompt version.

    Builds a real ``LiteLLMClient`` for the provider (and an optional judge
    client), a ``RequestContext`` with a version-pinned ``PromptManager``, then
    runs the component's live provider over its illustrative fixtures. Makes
    real LLM calls.

    Args:
        component: Which component to evaluate.
        version: Prompt version to pin, or None to use the active version.
        model: Model for the provider's LLM client.
        judge_model: Model for the AI judge, or None to skip judging.

    Returns:
        The component's ``EvalResults``.
    """
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager

    prompt_id = _prompt_id(component)
    client = LiteLLMClient(LiteLLMConfig(model=model))
    judge = LiteLLMClient(LiteLLMConfig(model=judge_model)) if judge_model else None

    with tempfile.TemporaryDirectory() as tmp:
        ctx = RequestContext(org_id="prompt-deviation-guard", storage_base_dir=tmp)
        if version is not None:
            ctx.prompt_manager = PromptManager(version_override={prompt_id: version})

        # Each branch is kept fully self-contained (own provider, cases, runner)
        # so the two components' incompatible decision/case types never unify.
        if component == "consolidation":
            from tests.eval.consolidation.fixtures import load_illustrative_cases
            from tests.eval.consolidation.providers import (
                make_consolidation_decision_provider,
            )
            from tests.eval.consolidation.runner import run_eval as run_consolidation

            cons_provider = make_consolidation_decision_provider(
                llm_client=client, request_context=ctx
            )
            return run_consolidation(
                cases=load_illustrative_cases(),
                decision_provider=cons_provider,
                llm_client=judge,
            )

        from tests.eval.reflection.fixtures import load_illustrative_cases
        from tests.eval.reflection.providers import make_reflection_decision_provider
        from tests.eval.reflection.runner import run_eval as run_reflection

        refl_provider = make_reflection_decision_provider(
            llm_client=client, request_context=ctx
        )
        return run_reflection(
            cases=load_illustrative_cases(),
            decision_provider=refl_provider,
            llm_client=judge,
        )


def _prompt_id(component: Component) -> str:
    """Return the prompt-bank id for a component."""
    if component == "consolidation":
        from reflexio.server.services.playbook.components.consolidator import (
            PlaybookConsolidator,
        )

        return PlaybookConsolidator.DEDUPLICATION_PROMPT_ID
    from reflexio.server.services.reflection.components.extractor import (
        REFLECTION_PROMPT_ID,
    )

    return REFLECTION_PROMPT_ID


def _active_version(component: Component) -> str | None:
    """Return the currently-active prompt version for a component."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    return PromptManager().get_active_version(_prompt_id(component))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code.

    Exit codes: ``0`` PASS (candidate holds), ``1`` FAIL (a metric regressed),
    ``2`` eval/infra error (provider or judge failed) — kept distinct from ``1``
    so an orchestrator never misreads an outage as a prompt regression.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, choices=COMPONENTS)
    parser.add_argument(
        "--candidate-version",
        required=True,
        help="Prompt version to gate (e.g. v2.4.0); must exist in the bank.",
    )
    parser.add_argument(
        "--baseline-version",
        default=None,
        help="Baseline version; defaults to the currently-active version.",
    )
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="AI-judge model; omit to skip judge metrics (faster/cheaper).",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Optional path to dump the report JSON.",
    )
    args = parser.parse_args(argv)
    if args.tolerance < 0:
        parser.error("--tolerance must be >= 0")
    component: Component = args.component

    baseline_version = args.baseline_version or _active_version(component)
    # Eval execution (LLM provider + judge) can fail for infra reasons —
    # surface those as exit code 2, distinct from a regression FAIL (1).
    try:
        baseline = _run_component_eval(
            component=component,
            version=baseline_version,
            model=args.model,
            judge_model=args.judge_model,
        )
        candidate = _run_component_eval(
            component=component,
            version=args.candidate_version,
            model=args.model,
            judge_model=args.judge_model,
        )
    except Exception as exc:  # noqa: BLE001 — any eval failure is an infra error
        print(f"[prompt-deviation-guard] eval execution failed: {exc}", file=sys.stderr)
        return 2
    report = compare(
        component=component,
        baseline=baseline,
        candidate=candidate,
        tolerance=args.tolerance,
        baseline_version=baseline_version,
        candidate_version=args.candidate_version,
    )
    print(report.render())
    if args.json_path:
        from pathlib import Path

        Path(args.json_path).write_text(json.dumps(report.to_json(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
