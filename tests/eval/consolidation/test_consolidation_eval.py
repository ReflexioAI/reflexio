"""Unit tests for the consolidation decision-eval harness.

All LLM interaction is mocked — the judge client is a ``MagicMock`` and
produced decisions are hand-built ``ConsolidationDecision`` objects. No real
API is hit. The one test that *would* hit a real judge is decorated with
``@skip_low_priority``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reflexio.server.services.playbook.playbook_consolidator import (
    ConsolidationDecision,
    DifferentiateDecision,
    IndependentDecision,
    PlaybookConsolidationOutput,
    RejectNewDecision,
    UnifyDecision,
)
from reflexio.test_support.skip_decorators import skip_low_priority
from tests.eval.consolidation.case import (
    ConsolidationEvalCase,
    kind_for_decision,
)
from tests.eval.consolidation.fixtures import load_illustrative_cases
from tests.eval.consolidation.judge import (
    ConsolidationVerdict,
    judge_consolidation_decision,
)
from tests.eval.consolidation.providers import make_consolidation_decision_provider
from tests.eval.consolidation.runner import EvalResults, run_eval, score_case

# ---------------------------------------------------------------------------
# Builders for hand-made decisions.
# ---------------------------------------------------------------------------


def _unify(new_id: str = "n1", **kw: object) -> UnifyDecision:
    base: dict[str, object] = {
        "new_id": new_id,
        "content": "Always summarize the report.",
        "trigger": "generating a report",
        "rationale": "Readers skim the summary first.",
    }
    base.update(kw)
    return UnifyDecision.model_validate(base)


def _reject(
    new_id: str = "n1", superseded_by_existing_id: int = 1
) -> RejectNewDecision:
    return RejectNewDecision(
        new_id=new_id, superseded_by_existing_id=superseded_by_existing_id
    )


def _differentiate(new_id: str = "n1", existing_id: int = 1) -> DifferentiateDecision:
    return DifferentiateDecision(
        new_id=new_id,
        existing_id=existing_id,
        refined_new_trigger="situation A only",
        refined_existing_trigger="situation B only",
    )


def _independent(new_id: str = "n1") -> IndependentDecision:
    return IndependentDecision(new_id=new_id)


def _case(case_id: str = "c1", gold_kind: str = "unify") -> ConsolidationEvalCase:
    return ConsolidationEvalCase.model_validate(
        {
            "id": case_id,
            "agent_context": "ctx",
            "existing": [
                {"id": 1, "content": "existing rule", "trigger": "t", "rationale": ""}
            ],
            "candidate": {
                "new_id": "n1",
                "content": "candidate rule",
                "trigger": "t",
                "rationale": "",
            },
            "gold_kind": gold_kind,
        }
    )


# ---------------------------------------------------------------------------
# Kind mapping: each decision kind maps to its own literal.
# ---------------------------------------------------------------------------


def test_kind_for_unify():
    assert kind_for_decision(_unify()) == "unify"


def test_kind_for_reject_new():
    assert kind_for_decision(_reject()) == "reject_new"


def test_kind_for_differentiate():
    assert kind_for_decision(_differentiate()) == "differentiate"


def test_kind_for_independent():
    assert kind_for_decision(_independent()) == "independent"


# ---------------------------------------------------------------------------
# run_eval mechanics: precomputed decisions, source validation, length check.
# ---------------------------------------------------------------------------


def test_run_eval_kind_accuracy_and_confusion():
    cases = [
        _case("a", "unify"),
        _case("b", "reject_new"),
        _case("c", "differentiate"),
    ]
    decisions = [
        _unify(),  # a: correct
        _reject(),  # b: correct
        _independent(),  # c: WRONG (produced independent, gold differentiate)
    ]
    res = run_eval(cases=cases, decisions=decisions)
    assert isinstance(res, EvalResults)
    assert res.n == 3
    assert res.kind_accuracy == pytest.approx(2 / 3)
    assert res.judge_accuracy is None  # no judge client passed
    assert res.confusion[("unify", "unify")] == 1
    assert res.confusion[("reject_new", "reject_new")] == 1
    assert res.confusion[("differentiate", "independent")] == 1


def test_run_eval_requires_exactly_one_decision_source():
    cases = [_case("a", "unify")]
    # Neither source given.
    with pytest.raises(ValueError):
        run_eval(cases=cases)
    # Both sources given.
    with pytest.raises(ValueError):
        run_eval(
            cases=cases,
            decisions=[_unify()],
            decision_provider=lambda _case: _unify(),
        )


def test_run_eval_rejects_mismatched_decisions_length():
    cases = [_case("a", "unify")]
    with pytest.raises(ValueError):
        run_eval(cases=cases, decisions=[])


def test_run_eval_with_decision_provider():
    cases = [_case("a", "reject_new")]

    def provider(case: ConsolidationEvalCase) -> RejectNewDecision:
        return _reject()

    res = run_eval(cases=cases, decision_provider=provider)
    assert res.kind_accuracy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Over-merge / under-merge metrics.
# ---------------------------------------------------------------------------


def test_over_merge_when_differentiate_gold_is_unified():
    # gold differentiate (should stay separate) but produced unify (merged).
    cases = [_case("a", "differentiate")]
    decisions: list[ConsolidationDecision] = [_unify()]
    res = run_eval(cases=cases, decisions=decisions)
    assert res.over_merge_rate > 0
    assert res.over_merge_rate == pytest.approx(1.0)
    # No should-merge gold cases => under-merge denominator empty => 0.0.
    assert res.under_merge_rate == pytest.approx(0.0)


def test_under_merge_when_unify_gold_is_kept_independent():
    # gold unify (should merge) but produced independent (kept separate).
    cases = [_case("a", "unify")]
    decisions: list[ConsolidationDecision] = [_independent()]
    res = run_eval(cases=cases, decisions=decisions)
    assert res.under_merge_rate > 0
    assert res.under_merge_rate == pytest.approx(1.0)
    # No should-stay-separate gold cases => over-merge denominator empty => 0.0.
    assert res.over_merge_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Judge path (mocked): judge_accuracy + self_contradiction_rate semantics.
# ---------------------------------------------------------------------------


def test_judge_parses_mocked_verdict():
    client = MagicMock()
    client.generate_chat_response.return_value = ConsolidationVerdict(
        correct=True, reason="matches"
    )
    v = judge_consolidation_decision(
        case=_case(gold_kind="unify"),
        produced_decision=_unify(),
        llm_client=client,
    )
    assert v.correct is True
    assert v.reason == "matches"
    client.generate_chat_response.assert_called_once()


def test_run_eval_with_judge_client_computes_judge_accuracy():
    client = MagicMock()
    client.generate_chat_response.return_value = ConsolidationVerdict(correct=True)
    cases = [_case("a", "unify")]
    decisions: list[ConsolidationDecision] = [_unify()]
    res = run_eval(cases=cases, decisions=decisions, llm_client=client)
    assert res.judge_accuracy == pytest.approx(1.0)
    assert res.outcomes[0].judge_correct is True


def test_self_contradiction_rate_counts_only_produced_unify():
    """The contradiction denominator is produced-``unify`` judged cases only.

    Two judged cases: one produced ``unify`` (eligible) and one produced
    ``differentiate`` (NOT eligible — ``score_case`` records
    ``self_contradiction`` only for ``unify``). The non-unify case must be
    excluded from the contradiction denominator even though it was judged.
    """
    client = MagicMock()
    # Both cases are judged correct=True; the contradiction flag only matters
    # for the unify case. score_case leaves self_contradiction=None on the
    # non-unify outcome regardless of the verdict's flag.
    client.generate_chat_response.return_value = ConsolidationVerdict(
        correct=True, self_contradiction=True
    )
    cases = [_case("u", "unify"), _case("d", "differentiate")]
    decisions = [_unify(), _differentiate()]
    res = run_eval(cases=cases, decisions=decisions, llm_client=client)

    # The differentiate outcome is excluded from the denominator.
    outcomes = {o.case_id: o for o in res.outcomes}
    assert outcomes["u"].self_contradiction is True
    assert outcomes["d"].self_contradiction is None  # non-unify => not recorded
    # Denominator = 1 (the unify case only); numerator = 1 => 1.0.
    assert res.self_contradiction_rate == pytest.approx(1.0)


def test_self_contradiction_rate_surfaces_flagged_unify():
    client = MagicMock()
    client.generate_chat_response.return_value = ConsolidationVerdict(
        correct=True, self_contradiction=True
    )
    res = score_case(
        case=_case(gold_kind="unify"),
        produced_decision=_unify(),
        llm_client=client,
    )
    assert res.self_contradiction is True


def test_self_contradiction_none_for_non_unify_even_when_judged():
    client = MagicMock()
    # Verdict claims a contradiction, but the produced kind is not unify, so
    # score_case must NOT record it.
    client.generate_chat_response.return_value = ConsolidationVerdict(
        correct=True, self_contradiction=True
    )
    res = score_case(
        case=_case(gold_kind="reject_new"),
        produced_decision=_reject(),
        llm_client=client,
    )
    assert res.judge_correct is True
    assert res.self_contradiction is None


# ---------------------------------------------------------------------------
# Panel voting: opposite tie-break directions for correct vs self_contradiction.
# ---------------------------------------------------------------------------


def test_panel_correct_majority():
    client = MagicMock()
    client.generate_chat_response.side_effect = [
        ConsolidationVerdict(correct=True, reason="a"),
        ConsolidationVerdict(correct=True, reason="b"),
        ConsolidationVerdict(correct=False, reason="c"),
    ]
    v = judge_consolidation_decision(
        case=_case(gold_kind="unify"),
        produced_decision=_unify(),
        llm_client=client,
        panel_size=3,
    )
    assert v.correct is True
    assert "2/3 correct" in v.reason


def test_panel_correct_tie_resolves_false():
    """`correct` tie -> False (conservative reject for a regression detector)."""
    client = MagicMock()
    client.generate_chat_response.side_effect = [
        ConsolidationVerdict(correct=True),
        ConsolidationVerdict(correct=False),
    ]
    v = judge_consolidation_decision(
        case=_case(gold_kind="unify"),
        produced_decision=_unify(),
        llm_client=client,
        panel_size=2,
    )
    assert v.correct is False


def test_panel_self_contradiction_tie_resolves_true():
    """`self_contradiction` tie -> True (conservatively flag a bad merge).

    A 1-1 split on the contradiction flag must resolve to True — the
    OPPOSITE direction from the ``correct`` tie-break. We hold ``correct``
    unanimous so only the contradiction tie is under test.
    """
    client = MagicMock()
    client.generate_chat_response.side_effect = [
        ConsolidationVerdict(correct=True, self_contradiction=True),
        ConsolidationVerdict(correct=True, self_contradiction=False),
    ]
    v = judge_consolidation_decision(
        case=_case(gold_kind="unify"),
        produced_decision=_unify(),
        llm_client=client,
        panel_size=2,
    )
    assert v.correct is True  # unanimous
    assert v.self_contradiction is True  # tie -> True (conservative)


def test_panel_three_way_split_minority_loses_both_dimensions():
    """A 1-2 split (no ties) confirms a minority vote loses on both dimensions.

    correct votes: True, False, False  -> minority True => majority False.
    contradiction votes: True, False, False -> 1 vs 2 => below threshold =>
    False. The contradiction `>=` tie-break only flips on an *actual* tie
    (covered by ``test_panel_self_contradiction_tie_resolves_true``), not on a
    losing minority — which this asserts.
    """
    client = MagicMock()
    client.generate_chat_response.side_effect = [
        ConsolidationVerdict(correct=True, self_contradiction=True),
        ConsolidationVerdict(correct=False, self_contradiction=False),
        ConsolidationVerdict(correct=False, self_contradiction=False),
    ]
    v = judge_consolidation_decision(
        case=_case(gold_kind="unify"),
        produced_decision=_unify(),
        llm_client=client,
        panel_size=3,
    )
    assert v.correct is False  # 1/3 correct => majority False
    assert v.self_contradiction is False  # 1/3 contradiction => not a tie => False


def test_panel_rejects_zero_size():
    client = MagicMock()
    with pytest.raises(ValueError):
        judge_consolidation_decision(
            case=_case(gold_kind="unify"),
            produced_decision=_unify(),
            llm_client=client,
            panel_size=0,
        )


def test_judge_raises_on_non_verdict_response():
    client = MagicMock()
    client.generate_chat_response.return_value = "not a verdict"
    with pytest.raises(TypeError):
        judge_consolidation_decision(
            case=_case(gold_kind="unify"),
            produced_decision=_unify(),
            llm_client=client,
        )


# ---------------------------------------------------------------------------
# Fixture loads + scores end-to-end with a mocked all-correct judge.
# ---------------------------------------------------------------------------


def _decision_for_gold(case: ConsolidationEvalCase) -> ConsolidationDecision:
    """Build a produced decision whose kind matches the case's gold kind."""
    new_id = case.candidate.new_id
    if case.gold_kind == "unify":
        return UnifyDecision(
            new_id=new_id,
            archive_existing_ids=[e.id for e in case.existing],
            content=case.candidate.content,
            trigger=case.candidate.trigger or "",
            rationale=case.candidate.rationale,
        )
    if case.gold_kind == "reject_new":
        return RejectNewDecision(
            new_id=new_id,
            superseded_by_existing_id=case.existing[0].id,
        )
    if case.gold_kind == "differentiate":
        return DifferentiateDecision(
            new_id=new_id,
            existing_id=case.existing[0].id,
            refined_new_trigger=(case.candidate.trigger or "candidate context")
            + " (candidate-only)",
            refined_existing_trigger=(case.existing[0].trigger or "existing context")
            + " (existing-only)",
        )
    if case.gold_kind == "independent":
        return IndependentDecision(new_id=new_id)
    raise AssertionError(f"unhandled gold kind {case.gold_kind}")


def test_illustrative_fixture_loads_and_covers_expected_kinds():
    cases = load_illustrative_cases()
    assert len(cases) >= 5
    kinds = {c.gold_kind for c in cases}
    assert {"unify", "reject_new", "differentiate", "independent"} <= kinds
    for c in cases:
        assert c.id
        assert c.candidate.new_id


def test_harness_smoke_run_over_fixture_with_mocked_judge():
    cases = load_illustrative_cases()
    decisions = [_decision_for_gold(c) for c in cases]

    client = MagicMock()
    client.generate_chat_response.return_value = ConsolidationVerdict(correct=True)
    res = run_eval(cases=cases, decisions=decisions, llm_client=client)

    assert res.n == len(cases)
    assert res.kind_accuracy == pytest.approx(1.0)
    assert res.judge_accuracy == pytest.approx(1.0)


def test_summary_is_renderable():
    cases = [_case("a", "unify")]
    decisions: list[ConsolidationDecision] = [_unify()]
    res = run_eval(cases=cases, decisions=decisions)
    summary = res.summary()
    assert "Consolidation decision-eval summary" in summary
    assert "self-contradiction" in summary


# ---------------------------------------------------------------------------
# Live consolidation decision provider: mocked-seam unit tests (CI-covered)
# exercise entity construction + the consolidator decision seam without a real
# LLM; the ``@skip_low_priority`` smoke below runs it end-to-end for real.
# ---------------------------------------------------------------------------


def test_live_provider_returns_canned_decision(tmp_path):
    """Provider builds the EXISTING + candidate entities, renders the real
    ``playbook_consolidation`` prompt, reaches the decision seam, and returns
    the LLM's decision — proving the construction + call path under a mocked
    LLM seam (CI-covered, no real API)."""
    from reflexio.server.api_endpoints.request_context import RequestContext

    canned = UnifyDecision(new_id="NEW-0", content="x", trigger="t", rationale="r")
    mock = MagicMock()
    mock.generate_chat_response.return_value = PlaybookConsolidationOutput(
        decisions=[canned]
    )

    ctx = RequestContext(org_id="eval-cons-prov", storage_base_dir=str(tmp_path))
    provider = make_consolidation_decision_provider(
        llm_client=mock, request_context=ctx
    )

    case = _case(gold_kind="unify")  # has >= 1 existing row
    decision = provider(case)

    assert decision == canned
    assert kind_for_decision(decision) == "unify"
    # The provider reached the LLM call (entity build + prompt render succeeded).
    mock.generate_chat_response.assert_called_once()


def test_live_provider_empty_output_maps_to_independent(tmp_path):
    """An empty ``PlaybookConsolidationOutput`` makes the provider fall back to
    an ``IndependentDecision`` carrying the case's ``candidate.new_id``."""
    from reflexio.server.api_endpoints.request_context import RequestContext

    mock = MagicMock()
    mock.generate_chat_response.return_value = PlaybookConsolidationOutput(decisions=[])

    ctx = RequestContext(org_id="eval-cons-noop", storage_base_dir=str(tmp_path))
    provider = make_consolidation_decision_provider(
        llm_client=mock, request_context=ctx
    )

    case = _case(gold_kind="independent")
    decision = provider(case)

    assert isinstance(decision, IndependentDecision)
    assert kind_for_decision(decision) == "independent"
    assert decision.new_id == case.candidate.new_id


@skip_low_priority
def test_live_consolidation_provider_real(tmp_path):  # pragma: no cover - manual
    """Real end-to-end smoke: live consolidator decision step + real LLM over
    the fixture. Asserts only pipeline mechanics (every case produced one of
    the four kinds), never exact kinds. Run manually with API keys +
    RUN_LOW_PRIORITY=1."""
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5"))
    ctx = RequestContext(org_id="eval", storage_base_dir=str(tmp_path))
    provider = make_consolidation_decision_provider(
        llm_client=client, request_context=ctx
    )

    cases = load_illustrative_cases()
    res = run_eval(cases=cases, decision_provider=provider, llm_client=None)

    assert res.n == len(cases)
    valid_kinds = {"unify", "reject_new", "differentiate", "independent"}
    assert all(o.produced_kind in valid_kinds for o in res.outcomes)


# ---------------------------------------------------------------------------
# Real-API judge (manual only).
# ---------------------------------------------------------------------------


@skip_low_priority
def test_real_judge_smoke():  # pragma: no cover - manual, costs money
    """Smoke test against a real judge model. Run manually with API keys."""
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5"))
    # One compose case + its unify decision (the consolidation-specific path).
    case = ConsolidationEvalCase.model_validate(
        {
            "id": "compose_smoke",
            "agent_context": "Assistant that drafts outbound customer emails.",
            "existing": [
                {
                    "id": 1,
                    "content": "Always greet the recipient by name.",
                    "trigger": "drafting a customer email",
                    "rationale": "A personal greeting improves reply rate.",
                }
            ],
            "candidate": {
                "new_id": "n1",
                "content": "Do not attach internal-only files to a customer email.",
                "trigger": "drafting a customer email",
                "rationale": "Internal files can leak confidential data.",
            },
            "gold_kind": "unify",
            "notes": "Compose: do-rule (greeting) + avoid-rule (attachments) for "
            "different sub-aspects of the same task; no self-contradiction.",
        }
    )
    decision = UnifyDecision(
        new_id="n1",
        archive_existing_ids=[1],
        content=(
            "Always greet the recipient by name; do not attach internal-only "
            "files to a customer email."
        ),
        trigger="drafting a customer email",
        rationale="Personal greeting plus no confidential-file leaks.",
    )
    v = judge_consolidation_decision(
        case=case,
        produced_decision=decision,
        llm_client=client,
        rubric={"judge_model": "claude-haiku-4-5"},
    )
    assert isinstance(v, ConsolidationVerdict)
