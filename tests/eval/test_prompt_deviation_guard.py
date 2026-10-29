"""Unit tests for the prompt deviation guard's gate logic.

These exercise :func:`compare` against real ``EvalResults`` built from crafted
``CaseOutcome``s — no LLM, no real provider/judge. The real end-to-end path
(``_run_component_eval`` / ``main``) makes paid LLM calls and is not run here;
the registry/wiring is sanity-checked cheaply instead.
"""

from __future__ import annotations

import pytest

from tests.eval import prompt_deviation_guard as guard
from tests.eval.consolidation.runner import CaseOutcome as ConsOutcome
from tests.eval.consolidation.runner import EvalResults as ConsResults


def _cons(rows, *, judged: bool = False) -> ConsResults:
    """Build consolidation ``EvalResults`` from ``(case_id, gold, produced)``."""
    res = ConsResults()
    for cid, gold, produced in rows:
        res.outcomes.append(
            ConsOutcome(
                case_id=cid,
                gold_kind=gold,
                produced_kind=produced,
                kind_match=(gold == produced),
                judge_correct=(gold == produced) if judged else None,
                self_contradiction=False
                if (judged and produced in {"create", "replace", "mixed"})
                else None,
            )
        )
    return res


def _metric(report: guard.GuardReport, name: str) -> guard.MetricDelta:
    return next(m for m in report.metrics if m.name == name)


# ---------------------------------------------------------------------------
# PASS / no-regression
# ---------------------------------------------------------------------------


def test_identical_runs_pass_with_full_agreement() -> None:
    rows = [("a", "unify", "unify"), ("b", "differentiate", "differentiate")]
    report = guard.compare(
        component="consolidation", baseline=_cons(rows), candidate=_cons(rows)
    )
    assert report.passed
    assert report.agreement_rate == 1.0
    assert all(not m.regressed for m in report.metrics)


def test_small_change_within_tolerance_passes() -> None:
    # 4 cases: baseline all correct (1.0); candidate 3/4 correct (0.75).
    # Drop of 0.25 > default 0.05 would fail, so use a bigger tolerance.
    base = _cons([(str(i), "unify", "unify") for i in range(4)])
    cand = _cons(
        [("0", "unify", "reject_new"), *[(str(i), "unify", "unify") for i in (1, 2, 3)]]
    )
    report = guard.compare(
        component="consolidation", baseline=base, candidate=cand, tolerance=0.5
    )
    assert report.passed


# ---------------------------------------------------------------------------
# FAIL — higher-is-better metric drops
# ---------------------------------------------------------------------------


def test_kind_accuracy_drop_fails() -> None:
    base = _cons([(str(i), "unify", "unify") for i in range(4)])  # 1.0
    cand = _cons([(str(i), "unify", "reject_new") for i in range(4)])  # 0.0
    report = guard.compare(component="consolidation", baseline=base, candidate=cand)
    assert not report.passed
    assert _metric(report, "kind_accuracy").regressed
    assert _metric(report, "kind_accuracy").delta == -1.0


# ---------------------------------------------------------------------------
# FAIL — lower-is-better metric rises
# ---------------------------------------------------------------------------


def test_judged_contradiction_rise_fails_without_inferring_it_from_action_names():
    base = _cons([("a", "replace", "replace")], judged=True)
    candidate = _cons([("a", "replace", "replace")], judged=True)
    candidate.outcomes[0].self_contradiction = True
    report = guard.compare(
        component="consolidation", baseline=base, candidate=candidate
    )
    metric = _metric(report, "self_contradiction_rate")
    assert metric.baseline == 0 and metric.candidate == 1 and metric.regressed
    assert not report.passed
    assert not _metric(report, "kind_accuracy").regressed


# ---------------------------------------------------------------------------
# None-valued judge metrics are skipped, not failed
# ---------------------------------------------------------------------------


def test_unjudged_metrics_are_skipped_not_failed() -> None:
    rows = [("a", "unify", "unify")]
    report = guard.compare(
        component="consolidation", baseline=_cons(rows), candidate=_cons(rows)
    )
    judge = _metric(report, "judge_accuracy")
    assert judge.baseline is None
    assert judge.candidate is None
    assert judge.delta is None
    assert not judge.regressed
    assert report.passed


def test_agreement_rate_counts_unchanged_cases() -> None:
    base = _cons([("a", "unify", "unify"), ("b", "unify", "unify")])
    cand = _cons([("a", "unify", "unify"), ("b", "unify", "reject_new")])
    report = guard.compare(component="consolidation", baseline=base, candidate=cand)
    assert report.agreement_rate == 0.5


# ---------------------------------------------------------------------------
# Registry / wiring sanity (cheap; catches typos without an LLM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage", ["reconcile-method", "reconcile-audit", "reconcile-revise"]
)
def test_digest_tracks_each_actual_prompt_asset(stage):
    baseline = guard._load_prompt_bundle()
    candidate = {**baseline, stage: baseline[stage] + "\nCandidate treatment."}
    assert guard._bundle_digest(baseline) != guard._bundle_digest(candidate)


def test_metric_specs_cover_every_component() -> None:
    for component in guard.COMPONENTS:
        specs = guard._METRIC_SPECS[component]
        assert specs
        # Every spec name must be a real EvalResults property.
        for spec in specs:
            assert isinstance(
                getattr(ConsResults(), spec.name, "MISSING"), (float, type(None))
            )


def test_report_json_round_trips() -> None:
    rows = [("a", "unify", "unify")]
    report = guard.compare(
        component="consolidation",
        baseline=_cons(rows),
        candidate=_cons(rows),
        candidate_version="v9.9.9",
    )
    blob = report.to_json()
    assert blob["component"] == "consolidation"
    assert blob["candidate_version"] == "v9.9.9"
    assert blob["passed"] is True
    assert isinstance(blob["metrics"], list)


# ---------------------------------------------------------------------------
# Input validation + exit-code semantics
# ---------------------------------------------------------------------------


def test_negative_tolerance_raises() -> None:
    rows = [("a", "unify", "unify")]
    with pytest.raises(ValueError, match="tolerance must be >= 0"):
        guard.compare(
            component="consolidation",
            baseline=_cons(rows),
            candidate=_cons(rows),
            tolerance=-0.1,
        )


def test_main_rejects_negative_tolerance() -> None:
    # argparse parser.error() exits with SystemExit before any eval runs.
    with pytest.raises(SystemExit):
        guard.main(
            [
                "--component",
                "consolidation",
                "--candidate-prompts",
                "v2.3.0",
                "--tolerance",
                "-0.1",
            ]
        )


def test_main_returns_2_on_eval_infra_error(monkeypatch, tmp_path):
    def fail(**_kwargs):
        raise RuntimeError("provider down")

    for name, content in guard._load_prompt_bundle().items():
        (tmp_path / f"{name}.md").write_text(content)
    monkeypatch.setattr(guard, "_run_component_eval", fail)
    assert (
        guard.main(
            ["--component", "consolidation", "--candidate-prompts", str(tmp_path)]
        )
        == 2
    )


def test_main_passes_both_exact_bundles_to_execution(monkeypatch, tmp_path):
    baseline = guard._load_prompt_bundle()
    candidate = {
        **baseline,
        "reconcile-audit": baseline["reconcile-audit"] + "\nStricter audit treatment.",
    }
    for name, content in candidate.items():
        (tmp_path / f"{name}.md").write_text(content)
    seen = []

    def execute(**kwargs):
        seen.append(kwargs["prompts"])
        return _cons([("case", "replace", "replace")])

    monkeypatch.setattr(guard, "_run_component_eval", execute)
    assert (
        guard.main(
            ["--component", "consolidation", "--candidate-prompts", str(tmp_path)]
        )
        == 0
    )
    assert seen == [baseline, candidate]


def test_missing_candidate_asset_fails_before_any_provider_call(monkeypatch, tmp_path):
    def forbidden(**_kwargs):
        raise AssertionError("missing artifact must fail before paid execution")

    monkeypatch.setattr(guard, "_run_component_eval", forbidden)
    assert (
        guard.main(
            ["--component", "consolidation", "--candidate-prompts", str(tmp_path)]
        )
        == 2
    )
