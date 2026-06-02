"""Unit tests for the golden-set extraction eval harness.

All LLM interaction is mocked — the judge is a ``MagicMock`` returning a
fixed ``JudgeScore`` and produced extractions are the cases' own gold
items (a perfect-extraction baseline). No real API is hit. The one test
that *would* hit a real judge is decorated with ``@skip_low_priority``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from reflexio.test_support.skip_decorators import skip_low_priority
from tests.eval.conftest import _load, _load_rubric
from tests.eval.extraction.runner import (
    CaseOutcome,
    EvalResults,
    _build_actual,
    _build_expected,
    run_eval,
    score_case,
)
from tests.eval.judge import JudgeScore, LLMJudge


def _gold_extraction(case: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Perfect-extraction baseline: the case's own gold profiles + playbooks."""
    return (case.get("expected_profiles", []), case.get("expected_playbooks", []))


def _stub_judge() -> MagicMock:
    """A judge whose ``score(...)`` returns a fixed deterministic ``JudgeScore``."""
    judge = MagicMock()
    judge.score.return_value = JudgeScore(
        signal_f1=0.5,
        answer_correctness=0.0,
        grounded_rate=1.0,
        rationale="stub",
    )
    return judge


def _cases() -> list[dict[str, Any]]:
    return _load("extraction")


# ---------------------------------------------------------------------------
# run_eval mechanics over the real golden cases with a stub judge.
# ---------------------------------------------------------------------------


def test_run_eval_mechanics_with_stub_judge():
    cases = _cases()
    res = run_eval(
        cases=cases,
        extractions=[_gold_extraction(c) for c in cases],
        judge=_stub_judge(),
    )
    assert isinstance(res, EvalResults)
    assert res.n == 3
    assert res.signal_f1_mean == pytest.approx(0.5)
    assert res.grounded_rate_mean == pytest.approx(1.0)
    # signal_f1 (0.5) clears 0.4 but not 0.7; grounded (1.0) clears both.
    assert res.pass_rate(0.4) == pytest.approx(1.0)
    assert res.pass_rate(0.7) == pytest.approx(0.0)


def test_summary_is_renderable():
    cases = _cases()
    res = run_eval(
        cases=cases,
        extractions=[_gold_extraction(c) for c in cases],
        judge=_stub_judge(),
    )
    summary = res.summary()
    assert "Extraction golden-set eval summary" in summary
    # Per-case lines render with a real case id.
    assert any(c["id"] in summary for c in cases)


# ---------------------------------------------------------------------------
# Validation: exactly-one-of extractions / extraction_provider, and length.
# ---------------------------------------------------------------------------


def test_run_eval_requires_exactly_one_extraction_source():
    cases = _cases()
    with pytest.raises(ValueError):
        run_eval(cases=cases, judge=_stub_judge())
    with pytest.raises(ValueError):
        run_eval(
            cases=cases,
            extractions=[_gold_extraction(c) for c in cases],
            extraction_provider=_gold_extraction,
            judge=_stub_judge(),
        )


def test_run_eval_rejects_mismatched_extractions_length():
    cases = _cases()
    with pytest.raises(ValueError):
        run_eval(cases=cases, extractions=[], judge=_stub_judge())


# ---------------------------------------------------------------------------
# Provider path matches the precomputed path.
# ---------------------------------------------------------------------------


def test_run_eval_with_extraction_provider():
    cases = _cases()
    res = run_eval(
        cases=cases,
        extraction_provider=lambda c: _gold_extraction(c),
        judge=_stub_judge(),
    )
    assert res.n == 3
    assert res.signal_f1_mean == pytest.approx(0.5)
    assert res.grounded_rate_mean == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Payload shaping: entities dumped, dicts passed through; expected carries
# the nuance keys.
# ---------------------------------------------------------------------------


class _Item(BaseModel):
    content: str


def test_build_actual_dumps_models_and_passes_dicts():
    actual = _build_actual([_Item(content="p")], [{"trigger": "t", "content": "c"}])
    assert actual["profiles"] == [{"content": "p"}]
    assert actual["playbooks"] == [{"trigger": "t", "content": "c"}]


def test_build_expected_carries_nuance_keys():
    case = {
        "id": "synthetic",
        "expected_profiles": [{"content": "a"}],
        "expected_playbooks": [],
        "must_NOT_include_profiles": [{"content_contains": "stale"}],
        "notes_for_judge": "watch supersession",
    }
    expected = _build_expected(case)
    assert expected["must_NOT_include_profiles"] == [{"content_contains": "stale"}]
    assert expected["notes_for_judge"] == "watch supersession"
    assert expected["expected_profiles"] == [{"content": "a"}]


# ---------------------------------------------------------------------------
# Parametrized golden case via the shared conftest fixtures.
# ---------------------------------------------------------------------------


def test_score_golden_case(extraction_case, extraction_judge):
    outcome = score_case(
        case=extraction_case,
        profiles=extraction_case.get("expected_profiles", []),
        playbooks=extraction_case.get("expected_playbooks", []),
        judge=extraction_judge,
    )
    assert isinstance(outcome, CaseOutcome)
    assert outcome.case_id == extraction_case["id"]
    assert 0.0 <= outcome.signal_f1 <= 1.0
    assert 0.0 <= outcome.grounded_rate <= 1.0


# ---------------------------------------------------------------------------
# Real-API judge (manual only).
# ---------------------------------------------------------------------------


@skip_low_priority
def test_real_judge_smoke():  # pragma: no cover - manual, costs money
    """Smoke test against a real judge model. Run manually with API keys."""
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    rubric = _load_rubric("extraction_rubric.yaml")
    client = LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5"))
    judge = LLMJudge(client=client, rubric=rubric)

    case = _cases()[0]
    profiles, playbooks = _gold_extraction(case)
    outcome = score_case(case=case, profiles=profiles, playbooks=playbooks, judge=judge)
    assert isinstance(outcome, CaseOutcome)
    assert 0.0 <= outcome.signal_f1 <= 1.0
    assert 0.0 <= outcome.grounded_rate <= 1.0
