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
from tests.eval.reflection.runner import CaseOutcome as ReflOutcome
from tests.eval.reflection.runner import EvalResults as ReflResults


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
                self_contradiction=False if (judged and produced == "unify") else None,
            )
        )
    return res


def _refl(rows, *, judged: bool = False) -> ReflResults:
    """Build reflection ``EvalResults`` from ``(case_id, gold, produced)``."""
    res = ReflResults()
    for cid, gold, produced in rows:
        res.outcomes.append(
            ReflOutcome(
                case_id=cid,
                gold_label=gold,
                produced_label=produced,
                label_match=(gold == produced),
                judge_correct=(gold == produced) if judged else None,
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


def test_label_accuracy_drop_fails_reflection() -> None:
    base = _refl([(str(i), "widen", "widen") for i in range(4)])  # 1.0
    cand = _refl([(str(i), "widen", "no_change") for i in range(4)])  # 0.0
    report = guard.compare(component="reflection", baseline=base, candidate=cand)
    assert not report.passed
    assert _metric(report, "label_accuracy").regressed


# ---------------------------------------------------------------------------
# FAIL — lower-is-better metric rises
# ---------------------------------------------------------------------------


def test_over_merge_rate_rise_fails() -> None:
    # kind_accuracy held constant (both mismatch) to isolate over_merge_rate.
    base = _cons([("a", "differentiate", "independent")])  # separate, no over-merge
    cand = _cons([("a", "differentiate", "unify")])  # separate gold, merged -> over
    report = guard.compare(component="consolidation", baseline=base, candidate=cand)
    om = _metric(report, "over_merge_rate")
    assert om.baseline == 0.0
    assert om.candidate == 1.0
    assert om.regressed
    assert not report.passed
    # kind_accuracy unchanged (0.0 -> 0.0), so it must NOT be flagged.
    assert not _metric(report, "kind_accuracy").regressed


def test_false_tighten_rate_rise_fails_reflection() -> None:
    base = _refl([("a", "widen", "no_change")])  # not a false tighten
    cand = _refl([("a", "widen", "tighten")])  # gold != tighten, produced tighten
    report = guard.compare(component="reflection", baseline=base, candidate=cand)
    ft = _metric(report, "false_tighten_rate")
    assert ft.baseline == 0.0
    assert ft.candidate == 1.0
    assert ft.regressed
    assert not report.passed


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


def test_prompt_ids_resolve_for_both_components() -> None:
    assert guard._prompt_id("consolidation") == "playbook_consolidation"
    assert guard._prompt_id("reflection") == "memory_reflection"


def test_metric_specs_cover_every_component() -> None:
    for component in guard.COMPONENTS:
        specs = guard._METRIC_SPECS[component]
        assert specs
        # Every spec name must be a real EvalResults property.
        results_cls = ConsResults if component == "consolidation" else ReflResults
        for spec in specs:
            assert isinstance(
                getattr(results_cls(), spec.name, "MISSING"), (float, type(None))
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
                "--candidate-version",
                "v2.3.0",
                "--tolerance",
                "-0.1",
            ]
        )


def test_main_returns_2_on_eval_infra_error(monkeypatch) -> None:
    # An eval/provider failure must surface as exit code 2, not 1 (regression).
    def _boom(**_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(guard, "_active_version", lambda _c: "v2.3.0")
    monkeypatch.setattr(guard, "_run_component_eval", _boom)
    rc = guard.main(["--component", "consolidation", "--candidate-version", "v2.3.0"])
    assert rc == 2
