"""Unit tests for the reflection decision-eval harness.

All LLM interaction is mocked — the judge client is a ``MagicMock`` and
produced decisions are hand-built ``ReflectionDecision`` objects. No real
API is hit. The one test that *would* hit a real judge is decorated with
``@skip_low_priority``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
)
from reflexio.test_support.skip_decorators import skip_low_priority
from tests.eval.reflection.case import (
    CitedItem,
    GoldLabel,
    ReflectionEvalCase,
    label_for_decision,
)
from tests.eval.reflection.fixtures import load_illustrative_cases
from tests.eval.reflection.judge import (
    ReflectionVerdict,
    judge_reflection_decision,
)
from tests.eval.reflection.runner import EvalResults, run_eval


def _playbook_item(**kw: object) -> CitedItem:
    base: dict[str, object] = {
        "kind": "playbook",
        "target_id": "1",
        "content": "Run the formatter.",
        "trigger": "editing files",
        "polarity": "positive",
    }
    base.update(kw)
    return CitedItem.model_validate(base)


def _profile_item(**kw: object) -> CitedItem:
    base: dict[str, object] = {
        "kind": "profile",
        "target_id": "p1",
        "content": "User is on-call this week.",
        "profile_time_to_live": "one_week",
    }
    base.update(kw)
    return CitedItem.model_validate(base)


# ---------------------------------------------------------------------------
# Label mapping: one assertion per field-presence pattern.
# ---------------------------------------------------------------------------


def test_label_no_change_when_no_fields_set():
    d = ReflectionDecision(target_kind="playbook", target_id="1")
    assert label_for_decision(d, _playbook_item()) == "no_change"


def test_label_flip_when_polarity_differs():
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_polarity="negative",
        new_rationale="user reversed it",
    )
    assert label_for_decision(d, _playbook_item(polarity="positive")) == "flip"


def test_label_not_flip_when_polarity_unchanged():
    d = ReflectionDecision(
        target_kind="playbook", target_id="1", new_polarity="positive"
    )
    # Same polarity => not a flip; nothing else set => no_change.
    assert label_for_decision(d, _playbook_item(polarity="positive")) == "no_change"


def test_label_ttl_when_profile_ttl_set():
    d = ReflectionDecision(
        target_kind="profile",
        target_id="p1",
        new_profile_time_to_live=ProfileTimeToLive.ONE_DAY,
    )
    assert label_for_decision(d, _profile_item()) == "ttl"


def test_label_tighten_when_trigger_narrows():
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_trigger="editing Python source files before showing code",
    )
    # Longer trigger => narrower => tighten.
    assert label_for_decision(d, _playbook_item(trigger="editing files")) == "tighten"


def test_label_widen_when_trigger_broadens():
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_trigger="edits",
    )
    # Shorter trigger => broader => widen.
    assert (
        label_for_decision(d, _playbook_item(trigger="editing Python files")) == "widen"
    )


def test_label_scope_when_trigger_change_ambiguous():
    d = ReflectionDecision(target_kind="playbook", target_id="1", new_trigger="abcde")
    # Same length as cited trigger => ambiguous scope.
    assert label_for_decision(d, _playbook_item(trigger="fghij")) == "scope"


def test_label_rewrite_when_only_content_changes():
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_content="Run the formatter and the linter.",
    )
    assert label_for_decision(d, _playbook_item()) == "rewrite"


def test_label_precedence_flip_over_content():
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_polarity="negative",
        new_rationale="why",
        new_content="totally new",
    )
    assert label_for_decision(d, _playbook_item(polarity="positive")) == "flip"


# ---------------------------------------------------------------------------
# Judge labeler: parses a mocked verdict; panel majority.
# ---------------------------------------------------------------------------


def _case() -> ReflectionEvalCase:
    return ReflectionEvalCase(
        id="c1",
        agent_context="ctx",
        cited_item=_playbook_item(),
        gold_label="tighten",
    )


def test_judge_parses_mocked_verdict():
    client = MagicMock()
    client.generate_chat_response.return_value = ReflectionVerdict(
        correct=True, reason="matches"
    )
    d = ReflectionDecision(
        target_kind="playbook", target_id="1", new_trigger="editing python files only"
    )
    v = judge_reflection_decision(case=_case(), produced_decision=d, llm_client=client)
    assert v.correct is True
    assert v.reason == "matches"
    client.generate_chat_response.assert_called_once()


def test_judge_passes_judge_model_from_rubric():
    client = MagicMock()
    client.generate_chat_response.return_value = ReflectionVerdict(correct=False)
    d = ReflectionDecision(target_kind="playbook", target_id="1")
    judge_reflection_decision(
        case=_case(),
        produced_decision=d,
        llm_client=client,
        rubric={"judge_model": "claude-haiku-4-5", "prompt": "p {produced_label}"},
    )
    assert client.generate_chat_response.call_args.kwargs["model"] == "claude-haiku-4-5"


def test_judge_panel_majority():
    client = MagicMock()
    client.generate_chat_response.side_effect = [
        ReflectionVerdict(correct=True, reason="a"),
        ReflectionVerdict(correct=True, reason="b"),
        ReflectionVerdict(correct=False, reason="c"),
    ]
    d = ReflectionDecision(target_kind="playbook", target_id="1")
    v = judge_reflection_decision(
        case=_case(), produced_decision=d, llm_client=client, panel_size=3
    )
    assert v.correct is True
    assert "2/3" in v.reason


def test_judge_panel_tie_is_incorrect():
    client = MagicMock()
    client.generate_chat_response.side_effect = [
        ReflectionVerdict(correct=True),
        ReflectionVerdict(correct=False),
    ]
    d = ReflectionDecision(target_kind="playbook", target_id="1")
    v = judge_reflection_decision(
        case=_case(), produced_decision=d, llm_client=client, panel_size=2
    )
    assert v.correct is False


def test_judge_raises_on_non_verdict_response():
    client = MagicMock()
    client.generate_chat_response.return_value = "not a verdict"
    d = ReflectionDecision(target_kind="playbook", target_id="1")
    with pytest.raises(TypeError):
        judge_reflection_decision(case=_case(), produced_decision=d, llm_client=client)


def test_judge_rejects_zero_panel():
    client = MagicMock()
    d = ReflectionDecision(target_kind="playbook", target_id="1")
    with pytest.raises(ValueError):
        judge_reflection_decision(
            case=_case(), produced_decision=d, llm_client=client, panel_size=0
        )


# ---------------------------------------------------------------------------
# Metrics on a hand-built set of (gold, produced) pairs.
# ---------------------------------------------------------------------------


def _mk_case(case_id: str, gold: GoldLabel, cited: CitedItem) -> ReflectionEvalCase:
    return ReflectionEvalCase(id=case_id, cited_item=cited, gold_label=gold)


def test_metrics_accuracy_and_confusion():
    cases = [
        _mk_case("a", "no_change", _playbook_item(target_id="1")),
        _mk_case("b", "tighten", _playbook_item(target_id="2", trigger="editing")),
        _mk_case("c", "flip", _playbook_item(target_id="3", polarity="positive")),
    ]
    decisions = [
        # a: correct no_change
        ReflectionDecision(target_kind="playbook", target_id="1"),
        # b: correct tighten (longer trigger)
        ReflectionDecision(
            target_kind="playbook", target_id="2", new_trigger="editing python files"
        ),
        # c: WRONG — produced no_change instead of flip
        ReflectionDecision(target_kind="playbook", target_id="3"),
    ]
    res = run_eval(cases=cases, decisions=decisions)
    assert isinstance(res, EvalResults)
    assert res.n == 3
    assert res.label_accuracy == pytest.approx(2 / 3)
    assert res.judge_accuracy is None  # no judge client passed
    assert res.confusion[("no_change", "no_change")] == 1
    assert res.confusion[("tighten", "tighten")] == 1
    assert res.confusion[("flip", "no_change")] == 1
    assert res.flip_correctness == pytest.approx(0.0)


def test_metrics_false_tighten_rate():
    # Two non-tighten gold cases; one is wrongly tightened.
    cases = [
        _mk_case("a", "no_change", _playbook_item(target_id="1", trigger="x")),
        _mk_case("b", "widen", _playbook_item(target_id="2", trigger="editing files")),
    ]
    decisions = [
        # a: wrongly tightened (longer trigger than cited "x")
        ReflectionDecision(
            target_kind="playbook", target_id="1", new_trigger="editing only"
        ),
        # b: correctly widened (shorter trigger)
        ReflectionDecision(target_kind="playbook", target_id="2", new_trigger="edits"),
    ]
    res = run_eval(cases=cases, decisions=decisions)
    # 1 false tighten out of 2 non-tighten gold cases.
    assert res.false_tighten_rate == pytest.approx(0.5)


def test_metrics_over_specialization_flag():
    cases = [_mk_case("a", "tighten", _playbook_item(trigger="editing files"))]
    decisions = [
        ReflectionDecision(
            target_kind="playbook",
            target_id="1",
            new_trigger='editing "src/app/main.py"',  # quoted + path => single instance
        )
    ]
    res = run_eval(cases=cases, decisions=decisions)
    assert res.over_specialization_rate == pytest.approx(1.0)
    assert res.outcomes[0].over_specialized is True


def test_metrics_no_false_tighten_when_no_eligible_cases():
    cases = [_mk_case("a", "tighten", _playbook_item(trigger="editing files"))]
    decisions = [
        ReflectionDecision(
            target_kind="playbook", target_id="1", new_trigger="editing python files"
        )
    ]
    res = run_eval(cases=cases, decisions=decisions)
    # Only gold-tighten case => denominator empty => 0.0
    assert res.false_tighten_rate == pytest.approx(0.0)


def test_run_eval_with_judge_client():
    client = MagicMock()
    client.generate_chat_response.return_value = ReflectionVerdict(correct=True)
    cases = [_mk_case("a", "no_change", _playbook_item(target_id="1"))]
    decisions = [ReflectionDecision(target_kind="playbook", target_id="1")]
    res = run_eval(cases=cases, decisions=decisions, llm_client=client)
    assert res.judge_accuracy == pytest.approx(1.0)
    assert res.outcomes[0].judge_correct is True


def test_run_eval_requires_exactly_one_decision_source():
    cases = [_mk_case("a", "no_change", _playbook_item())]
    with pytest.raises(ValueError):
        run_eval(cases=cases)
    with pytest.raises(ValueError):
        run_eval(
            cases=cases,
            decisions=[ReflectionDecision(target_kind="playbook", target_id="1")],
            decision_provider=lambda _case: ReflectionDecision(
                target_kind="playbook", target_id="1"
            ),
        )


def test_run_eval_with_decision_provider():
    cases = [_mk_case("a", "no_change", _playbook_item(target_id="9"))]

    def provider(case: ReflectionEvalCase) -> ReflectionDecision:
        return ReflectionDecision(target_kind="playbook", target_id="9")

    res = run_eval(cases=cases, decision_provider=provider)
    assert res.label_accuracy == pytest.approx(1.0)


def test_run_eval_rejects_mismatched_decisions_length():
    cases = [_mk_case("a", "no_change", _playbook_item())]
    with pytest.raises(ValueError):
        run_eval(cases=cases, decisions=[])


def test_summary_is_renderable():
    cases = [_mk_case("a", "no_change", _playbook_item(target_id="1"))]
    decisions = [ReflectionDecision(target_kind="playbook", target_id="1")]
    res = run_eval(cases=cases, decisions=decisions)
    summary = res.summary()
    assert "Reflection decision-eval summary" in summary
    assert "false-tighten rate" in summary


# ---------------------------------------------------------------------------
# Fixture sanity + harness smoke run over the illustrative set.
# ---------------------------------------------------------------------------


def test_illustrative_fixture_loads_and_covers_expected_labels():
    cases = load_illustrative_cases()
    labels = {c.gold_label for c in cases}
    assert {"no_change", "tighten", "widen", "flip"} <= labels
    # Cases parse into real entities (window uses Interaction).
    for c in cases:
        assert c.id
        assert c.cited_item.kind in ("playbook", "profile")


def test_harness_smoke_run_over_fixture_with_mocked_judge():
    cases = load_illustrative_cases()
    # Produce a "correct" decision for each fixture case from its gold label.
    decisions = [_decision_for_gold(c) for c in cases]

    client = MagicMock()
    client.generate_chat_response.return_value = ReflectionVerdict(correct=True)
    res = run_eval(cases=cases, decisions=decisions, llm_client=client)

    assert res.n == len(cases)
    assert res.label_accuracy == pytest.approx(1.0)
    assert res.judge_accuracy == pytest.approx(1.0)


def _decision_for_gold(case: ReflectionEvalCase) -> ReflectionDecision:
    """Build a produced decision that should map back to the case's gold label."""
    item = case.cited_item
    tid = item.target_id
    if case.gold_label == "no_change":
        return ReflectionDecision(target_kind=item.kind, target_id=tid)
    if case.gold_label == "flip":
        flipped = "negative" if item.polarity == "positive" else "positive"
        return ReflectionDecision(
            target_kind=item.kind,
            target_id=tid,
            new_polarity=flipped,
            new_rationale="reversed by user",
        )
    if case.gold_label in ("tighten", "widen"):
        return ReflectionDecision(
            target_kind=item.kind,
            target_id=tid,
            new_trigger=case.gold_new_trigger,
        )
    raise AssertionError(f"unhandled gold label {case.gold_label}")


# ---------------------------------------------------------------------------
# Real-API judge (manual only).
# ---------------------------------------------------------------------------


@skip_low_priority
def test_real_judge_smoke():  # pragma: no cover - manual, costs money
    """Smoke test against a real judge model. Run manually with API keys."""
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5"))
    case = _case()
    d = ReflectionDecision(
        target_kind="playbook", target_id="1", new_trigger="editing python files only"
    )
    v = judge_reflection_decision(
        case=case,
        produced_decision=d,
        llm_client=client,
        rubric={"judge_model": "claude-haiku-4-5"},
    )
    assert isinstance(v, ReflectionVerdict)
