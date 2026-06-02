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
    ReflectionOutput,
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
from tests.eval.reflection.providers import make_reflection_decision_provider
from tests.eval.reflection.runner import EvalResults, run_eval


def _playbook_item(**kw: object) -> CitedItem:
    base: dict[str, object] = {
        "kind": "playbook",
        "target_id": "1",
        "content": "Run the formatter.",
        "trigger": "editing files",
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


def test_label_rewrite_when_content_reverses_orientation():
    # An orientation-reversing rewrite (avoidance wording + failure
    # rationale) is no longer mechanically distinguishable from a plain
    # rewrite — flip is retired, so it maps to "rewrite". Whether it truly
    # reversed the rule is the AI judge's call, not the deterministic mapper.
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_content="Avoid doing X.",
        new_rationale="user pushed back when X was recommended",
    )
    assert label_for_decision(d, _playbook_item()) == "rewrite"


def test_label_rewrite_when_content_keeps_orientation():
    # An affirmative rewrite is likewise a "rewrite".
    d = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_content="Do X, but only on weekdays.",
    )
    assert label_for_decision(d, _playbook_item()) == "rewrite"


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
        _mk_case("c", "rewrite", _playbook_item(target_id="3")),
    ]
    decisions = [
        # a: correct no_change
        ReflectionDecision(target_kind="playbook", target_id="1"),
        # b: correct tighten (longer trigger)
        ReflectionDecision(
            target_kind="playbook", target_id="2", new_trigger="editing python files"
        ),
        # c: WRONG — produced no_change instead of rewrite
        ReflectionDecision(target_kind="playbook", target_id="3"),
    ]
    res = run_eval(cases=cases, decisions=decisions)
    assert isinstance(res, EvalResults)
    assert res.n == 3
    assert res.label_accuracy == pytest.approx(2 / 3)
    assert res.judge_accuracy is None  # no judge client passed
    assert res.confusion[("no_change", "no_change")] == 1
    assert res.confusion[("tighten", "tighten")] == 1
    assert res.confusion[("rewrite", "no_change")] == 1


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
    assert {"no_change", "tighten", "widen", "rewrite"} <= labels
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
    if case.gold_label == "rewrite":
        # A rewrite changes new_content to something other than the cited
        # content (orientation-reversing rewrites are scored here too).
        return ReflectionDecision(
            target_kind=item.kind,
            target_id=tid,
            new_content=item.content + " (revised)",
            new_rationale="user pushed back; the prior rule failed in the window",
        )
    if case.gold_label in ("tighten", "widen"):
        return ReflectionDecision(
            target_kind=item.kind,
            target_id=tid,
            new_trigger=case.gold_new_trigger,
        )
    raise AssertionError(f"unhandled gold label {case.gold_label}")


# ---------------------------------------------------------------------------
# Rule localization: multi-rule content where the window implicates ONE rule.
# ---------------------------------------------------------------------------
#
# Harness-expressiveness note
# ---------------------------
# The harness scores a decision against the *whole* cited ``content``: the
# label mapper only checks ``new_content != cited_item.content`` (plus
# trigger / ttl signals). It has no field-level notion of "which rule
# inside the content was edited", so a surgically-localized edit and a
# wholesale rewrite of every rule both map to the same coarse label
# (``rewrite`` or a trigger-scope label). An orientation-reversing rewrite
# is no exception — it is a ``rewrite`` too.
#
# We therefore localize at the two signals the harness *does* expose,
# without adding any harness infrastructure:
#   1. the deterministic label + (for tighten/widen) the runner's
#      over-specialization / edit-magnitude flag, and
#   2. the AI-judge verdict, which sees the full produced ``new_content``
#      and the case ``notes`` and can reject an answer that rewrote the
#      wrong rule or clobbered the untouched rules.


def _multi_rule_tighten_case() -> ReflectionEvalCase:
    """Load the multi-rule localized-tighten fixture case."""
    cases = {c.id: c for c in load_illustrative_cases()}
    return cases["localized_tighten_multi_rule"]


def _multi_rule_reversal_case() -> ReflectionEvalCase:
    """Load the multi-rule localized orientation-reversal fixture case."""
    cases = {c.id: c for c in load_illustrative_cases()}
    return cases["localized_flip_multi_rule"]


def test_localized_tighten_edits_only_implicated_rule_scores_correct():
    """A localized tighten of the one implicated rule maps to gold + judge-correct.

    The cited content holds three rules; only Rule 2's trigger over-fired.
    The decision narrows the *trigger* (leaving the multi-rule content
    untouched), which is the cleanest localized expression of "edit only
    the implicated rule" the harness can represent.
    """
    case = _multi_rule_tighten_case()
    # Narrow the shared trigger so it only fires for the implicated rule;
    # do NOT touch new_content, so the other two rules are left intact.
    localized = ReflectionDecision(
        target_kind="playbook",
        target_id=case.cited_item.target_id,
        new_trigger=case.gold_new_trigger,
    )

    client = MagicMock()
    client.generate_chat_response.return_value = ReflectionVerdict(
        correct=True, reason="narrowed only the implicated rule's trigger"
    )
    res = run_eval(cases=[case], decisions=[localized], llm_client=client)

    assert res.outcomes[0].produced_label == "tighten"
    assert res.outcomes[0].label_match is True
    assert res.outcomes[0].over_specialized is False
    assert res.judge_accuracy == pytest.approx(1.0)


def test_localized_tighten_overspecialized_rewrite_is_flagged():
    """Pinning the trigger to a single instance trips over-specialization.

    A decision that "edits the wrong way" — collapsing the trigger to one
    concrete instance instead of the implicated rule's class — is caught
    by the runner's over-specialization flag (edit-magnitude signal),
    even though it still maps to a tighten label.
    """
    case = _multi_rule_tighten_case()
    over = ReflectionDecision(
        target_kind="playbook",
        target_id=case.cited_item.target_id,
        # Quoted single literal => single-instance trigger.
        new_trigger='handling item "B-00017"',
    )
    res = run_eval(cases=[case], decisions=[over])

    assert res.outcomes[0].over_specialized is True
    assert res.over_specialization_rate == pytest.approx(1.0)


def test_localized_reversal_of_implicated_rule_scores_correct():
    """Reversing only the implicated 'do' rule maps to gold rewrite + judge-correct.

    Rule 2 (auto-retry) caused a double-commit and explicit pushback; the
    localized decision rewrites it in the opposite orientation as an avoid
    rule with a rationale citing the failure. This is mechanically a
    ``rewrite`` (flip is retired); the judge confirms it reversed the right
    rule.
    """
    case = _multi_rule_reversal_case()
    localized = ReflectionDecision(
        target_kind="playbook",
        target_id=case.cited_item.target_id,
        new_content="Avoid auto-retrying step C; surface the failure and wait.",
        new_rationale="auto-retry double-committed and the user pushed back",
    )

    client = MagicMock()
    client.generate_chat_response.return_value = ReflectionVerdict(
        correct=True, reason="reversed only the implicated retry rule"
    )
    res = run_eval(cases=[case], decisions=[localized], llm_client=client)

    assert res.outcomes[0].produced_label == "rewrite"
    assert res.outcomes[0].label_match is True
    assert res.judge_accuracy == pytest.approx(1.0)


def test_localized_reversal_wrong_rule_rewrite_rejected_by_judge():
    """Editing the wrong rule is caught by the judge, not the coarse label.

    Both a correct localized reversal and an edit of the *wrong* rule
    (Rule 1, ordering) map to the same coarse ``rewrite`` label — the
    deterministic label cannot tell them apart, and with flip retired it
    never could. The AI judge — which sees the full produced content and
    the case notes naming Rule 2 — is the only signal that rejects the
    wrong-rule edit; there is no field-level rule-identity check without a
    harness change.
    """
    case = _multi_rule_reversal_case()
    wrong_rule = ReflectionDecision(
        target_kind="playbook",
        target_id=case.cited_item.target_id,
        # Touches Rule 1 (ordering), NOT the implicated Rule 2 (retry).
        new_content="Process items in reverse order received.",
        new_rationale="reordering felt cleaner",
    )

    client = MagicMock()
    client.generate_chat_response.return_value = ReflectionVerdict(
        correct=False, reason="edited the ordering rule, not the implicated retry rule"
    )
    res = run_eval(cases=[case], decisions=[wrong_rule], llm_client=client)

    # Mechanically a plain rewrite — the coarse label matches gold and so
    # cannot flag the wrong-rule edit; that is exactly the judge's job.
    assert res.outcomes[0].produced_label == "rewrite"
    assert res.outcomes[0].label_match is True  # gold is rewrite; label can't catch it
    # The judge is the signal that catches the wrong-rule edit.
    assert res.outcomes[0].judge_correct is False
    assert res.judge_accuracy == pytest.approx(0.0)


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


# ---------------------------------------------------------------------------
# Live reflection decision provider (mocked LLM seam — CI-covered).
# ---------------------------------------------------------------------------
#
# These exercise the real ``ReflectionExtractor`` construction + entity build
# + ``run`` call path with the LLM seam mocked, so default CI catches provider
# bugs (entity construction, ctor signature, prompt rendering reachability)
# WITHOUT a paid API call. The real-LLM end-to-end run lives in the
# ``@skip_low_priority`` smoke below.


def _profile_cited_case() -> ReflectionEvalCase:
    return ReflectionEvalCase(
        id="prof1",
        agent_context="ctx",
        cited_item=_profile_item(),
        gold_label="ttl",
    )


def _playbook_cited_case() -> ReflectionEvalCase:
    return ReflectionEvalCase(
        id="pb1",
        agent_context="ctx",
        cited_item=_playbook_item(),
        gold_label="tighten",
    )


def test_live_provider_returns_canned_decision_for_playbook(tmp_path):
    """Provider builds the playbook entity, renders the real prompt, and
    returns the LLM's decision — proving the construction + call path under
    a mocked LLM seam (CI-covered)."""
    from reflexio.server.api_endpoints.request_context import RequestContext

    canned = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_trigger="editing python source files only",
    )
    mock = MagicMock()
    mock.generate_chat_response.return_value = ReflectionOutput(decisions=[canned])

    ctx = RequestContext(org_id="eval-prov-pb", storage_base_dir=str(tmp_path))
    provider = make_reflection_decision_provider(llm_client=mock, request_context=ctx)

    decision = provider(_playbook_cited_case())

    assert decision == canned
    # The provider reached the LLM call (entity build + prompt render succeeded).
    mock.generate_chat_response.assert_called_once()


def test_live_provider_returns_canned_decision_for_profile(tmp_path):
    """Same path for a profile-cited case: TTL-bearing entity is built and
    the canned decision flows back."""
    from reflexio.server.api_endpoints.request_context import RequestContext

    canned = ReflectionDecision(
        target_kind="profile",
        target_id="p1",
        new_profile_time_to_live=ProfileTimeToLive.ONE_DAY,
    )
    mock = MagicMock()
    mock.generate_chat_response.return_value = ReflectionOutput(decisions=[canned])

    ctx = RequestContext(org_id="eval-prov-prof", storage_base_dir=str(tmp_path))
    provider = make_reflection_decision_provider(llm_client=mock, request_context=ctx)

    decision = provider(_profile_cited_case())

    assert decision == canned
    mock.generate_chat_response.assert_called_once()


def test_live_provider_empty_output_maps_to_no_change(tmp_path):
    """An empty ``ReflectionOutput`` makes the provider return a no-op
    decision that ``label_for_decision`` maps to ``no_change``."""
    from reflexio.server.api_endpoints.request_context import RequestContext

    mock = MagicMock()
    mock.generate_chat_response.return_value = ReflectionOutput(decisions=[])

    ctx = RequestContext(org_id="eval-prov-noop", storage_base_dir=str(tmp_path))
    provider = make_reflection_decision_provider(llm_client=mock, request_context=ctx)

    case = _playbook_cited_case()
    decision = provider(case)

    # No revision fields set => no-op decision.
    assert decision.new_content is None
    assert decision.new_trigger is None
    assert decision.new_profile_time_to_live is None
    assert label_for_decision(decision, case.cited_item) == "no_change"


@skip_low_priority
def test_live_reflection_provider_real(tmp_path):  # pragma: no cover - manual
    """Real end-to-end smoke: live extractor + real LLM over the fixture.

    Asserts only pipeline mechanics (every case produced a non-empty label),
    never exact decisions. Run manually with API keys + RUN_LOW_PRIORITY=1.
    """
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5"))
    ctx = RequestContext(org_id="eval", storage_base_dir=str(tmp_path))
    provider = make_reflection_decision_provider(
        llm_client=client, request_context=ctx
    )

    cases = load_illustrative_cases()
    res = run_eval(cases=cases, decision_provider=provider, llm_client=None)

    assert res.n == len(cases)
    assert all(o.produced_label for o in res.outcomes)
