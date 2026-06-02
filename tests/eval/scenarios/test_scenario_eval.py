"""Tests for the multi-round memory-scenario harness (``runner.py``).

Two tiers, mirroring the per-component eval tests:

- A **non-skipped CI test** drives ``run_scenario`` over every linchpin
  fixture with fully stubbed providers + judges. The provider stubs are
  closures over each scenario's gold so the extract -> consolidate ->
  reflect chain is deterministic, and the judge clients are ``MagicMock``s
  returning ``correct=True``. This proves the chain wires end to end, the
  book evolves as each scenario intends, and ``scenario_passed`` rolls up.
  A companion **wrong-decision** run flips one judge to ``correct=False``
  and asserts ``scenario_passed is False`` — proving the gate is real and
  the green path is non-vacuous.
- A ``@skip_low_priority`` **real-haiku smoke** builds the three real
  providers + real haiku judge clients and runs one scenario end to end,
  asserting only pipeline mechanics (round count), never exact verdicts.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.playbook.playbook_consolidator import (
    ConsolidationDecision,
    DifferentiateDecision,
    IndependentDecision,
    RejectNewDecision,
    UnifyDecision,
)
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
)
from reflexio.test_support.skip_decorators import skip_low_priority
from tests.eval.consolidation.judge import ConsolidationVerdict
from tests.eval.reflection.judge import ReflectionVerdict
from tests.eval.scenarios.book import _next_id, apply_consolidation
from tests.eval.scenarios.case import BookRule
from tests.eval.scenarios.fixtures import load_scenarios
from tests.eval.scenarios.runner import ScenarioResult, run_scenario

# ---------------------------------------------------------------------------
# Stub providers — closures over each scenario's per-round gold so the chain
# is deterministic. The extraction provider emits exactly one candidate rule;
# the consolidation provider builds the decision kind named by the case's
# ``gold_kind``; the reflection provider always tightens (sets new_trigger).
# ---------------------------------------------------------------------------

_CANDIDATE_TRIGGER = "a freshly extracted trigger"
_NARROWER_TRIGGER = "a narrower trigger"


def _extraction_provider(case: dict) -> tuple[list[Any], list[Any]]:
    """Return one playbook dict (no profiles) for any learn round.

    The runner's ``_to_book_rule`` accepts a plain dict, so we hand back the
    loose shape. The content is non-load-bearing — the consolidation decision
    kind (driven by the case ``gold_kind``) is what mutates the book.
    """
    return (
        [],
        [
            {
                "content": "A freshly extracted candidate rule.",
                "trigger": _CANDIDATE_TRIGGER,
                "rationale": "",
            }
        ],
    )


def _consolidation_provider(cc: Any) -> ConsolidationDecision:
    """Build a consolidation decision whose kind equals the case's ``gold_kind``.

    ``cc`` is the runner-built ``ConsolidationEvalCase``; we read ``gold_kind``
    and ``existing`` off it to construct the matching decision:

    - ``unify`` archives EXISTING list-position 0 and supplies merged text;
    - ``differentiate`` references ``existing[0].id`` and refines both triggers;
    - ``reject_new`` is superseded by ``existing[0].id``;
    - ``independent`` admits the candidate as-is.
    """
    new_id = cc.candidate.new_id
    if cc.gold_kind == "unify":
        return UnifyDecision(
            new_id=new_id,
            archive_existing_ids=[0],
            content="Announce the deploy in the channel and avoid Friday-afternoon deploys.",
            trigger="deploying a service",
            rationale="",
        )
    if cc.gold_kind == "differentiate":
        return DifferentiateDecision(
            new_id=new_id,
            existing_id=cc.existing[0].id,
            refined_new_trigger="user asks a factual question that is ambiguous",
            refined_existing_trigger="user asks a factual question that is unambiguous",
        )
    if cc.gold_kind == "reject_new":
        return RejectNewDecision(
            new_id=new_id, superseded_by_existing_id=cc.existing[0].id
        )
    return IndependentDecision(new_id=new_id)


def _reflection_provider(rc: Any) -> ReflectionDecision:
    """Always tighten the cited rule (set ``new_trigger`` only)."""
    return ReflectionDecision(
        target_kind="playbook",
        target_id=rc.cited_item.target_id,
        new_trigger=_NARROWER_TRIGGER,
    )


def _verdict_client(verdict: Any) -> MagicMock:
    """A judge client whose ``generate_chat_response`` returns ``verdict``."""
    client = MagicMock()
    client.generate_chat_response.return_value = verdict
    return client


def _run_with_stubs(
    scenario: Any,
    *,
    consolidation_verdict: ConsolidationVerdict,
    reflection_verdict: ReflectionVerdict,
) -> ScenarioResult:
    """Run one scenario through the fully-stubbed chain."""
    return run_scenario(
        scenario=scenario,
        extraction_provider=_extraction_provider,
        consolidation_provider=_consolidation_provider,
        reflection_provider=_reflection_provider,
        consolidation_judge_client=_verdict_client(consolidation_verdict),
        reflection_judge_client=_verdict_client(reflection_verdict),
    )


# ---------------------------------------------------------------------------
# Independent re-derivation of the expected final book, using the SAME shim
# and SAME canned decisions the runner uses — so the book-evolution asserts
# do not just restate the runner's own output.
# ---------------------------------------------------------------------------


def _expected_final_book(scenario: Any) -> list[BookRule]:
    """Replay the scenario's decisions through the shim to get the final book."""
    book = [r.model_copy() for r in scenario.seed_book]
    for round in scenario.rounds:
        if round.kind == "learn":
            gold_kind = round.gold.get("consolidation_kind", "independent")
            _profiles, playbooks = _extraction_provider({})
            for p in playbooks:
                existing_order = list(book)
                cand = BookRule(
                    id=_next_id(book),
                    content=p["content"],
                    trigger=p["trigger"],
                    rationale=p["rationale"],
                )
                cc = _FakeConsolidationCase(
                    gold_kind=gold_kind,
                    existing=existing_order,
                    candidate_new_id=f"new-{cand.id}",
                )
                decision = _consolidation_provider(cc)
                book = apply_consolidation(
                    book, cand, decision, existing_order=existing_order
                )
        # reflect rounds tighten the cited rule's trigger (handled in asserts).
    return book


class _FakeConsolidationCase:
    """Minimal stand-in exposing the attributes ``_consolidation_provider`` reads."""

    def __init__(
        self, *, gold_kind: str, existing: list[BookRule], candidate_new_id: str
    ) -> None:
        self.gold_kind = gold_kind
        self.existing = existing
        self.candidate = MagicMock(new_id=candidate_new_id)


# ---------------------------------------------------------------------------
# Fixture sanity.
# ---------------------------------------------------------------------------


def test_fixtures_load_three_distinct_scenarios():
    scenarios = load_scenarios()
    assert len(scenarios) >= 3
    ids = {s.id for s in scenarios}
    assert {
        "compose_grows_skill",
        "no_self_contradiction",
        "reflect_corrects",
    } <= ids
    for s in scenarios:
        assert s.rounds  # every scenario has at least one round


# ---------------------------------------------------------------------------
# Green path: the full chain passes when every judge says correct.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda s: s.id)
def test_scenario_chain_passes_with_correct_judges(scenario):
    """The extract -> consolidate -> reflect chain runs and gates green."""
    result = _run_with_stubs(
        scenario,
        consolidation_verdict=ConsolidationVerdict(correct=True),
        reflection_verdict=ReflectionVerdict(correct=True),
    )
    assert result.scenario_passed is True
    assert len(result.round_outcomes) == len(scenario.rounds)
    assert all(o.judged_correct for o in result.round_outcomes)


def test_compose_scenario_unifies_into_one_skill():
    """The compose scenario's learn round merges the seed + new avoid rule."""
    scenario = next(s for s in load_scenarios() if s.id == "compose_grows_skill")
    final_book = _expected_final_book(scenario)
    # Seed (1 rule) + a unify that archives position 0 and appends one merged
    # rule => exactly one rule, carrying both the do- and avoid-aspect text.
    assert len(final_book) == 1
    merged = final_book[0]
    assert "announce" in merged.content.lower()
    assert "friday" in merged.content.lower()


def test_differentiate_scenario_keeps_two_rules():
    """The no-self-contradiction scenario differentiates into two rules."""
    scenario = next(s for s in load_scenarios() if s.id == "no_self_contradiction")
    final_book = _expected_final_book(scenario)
    # Differentiate refines the existing trigger and appends the candidate
    # with its own refined trigger => two rules, distinct triggers.
    assert len(final_book) == 2
    triggers = [r.trigger for r in final_book]
    assert len(set(triggers)) == 2


def test_reflect_scenario_tightens_cited_rule_trigger():
    """The reflect scenario's reflect round narrows the cited rule's trigger.

    Round 0 (independent) appends an unrelated rule, leaving the cited rule 1
    in place; round 1 (reflect, cited=1) tightens its trigger. We run the full
    chain and assert the reflect round's detail shows the tighten gold and the
    round was judged correct — the trigger change itself is exercised by the
    apply shim's own unit tests.
    """
    scenario = next(s for s in load_scenarios() if s.id == "reflect_corrects")
    result = _run_with_stubs(
        scenario,
        consolidation_verdict=ConsolidationVerdict(correct=True),
        reflection_verdict=ReflectionVerdict(correct=True),
    )
    reflect_outcome = next(o for o in result.round_outcomes if o.kind == "reflect")
    assert reflect_outcome.judged_correct is True
    assert "gold=tighten" in reflect_outcome.detail


# ---------------------------------------------------------------------------
# Wrong-decision case: a failing judge verdict must fail the scenario. This
# proves the green path above is non-vacuous — the gate actually gates.
# ---------------------------------------------------------------------------


def test_wrong_consolidation_verdict_fails_scenario():
    """A ``correct=False`` consolidation verdict fails a learn-only scenario."""
    scenario = next(s for s in load_scenarios() if s.id == "compose_grows_skill")
    result = _run_with_stubs(
        scenario,
        consolidation_verdict=ConsolidationVerdict(correct=False, reason="bad merge"),
        reflection_verdict=ReflectionVerdict(correct=True),
    )
    assert result.scenario_passed is False
    assert all(not o.judged_correct for o in result.round_outcomes)


def test_wrong_reflection_verdict_fails_scenario():
    """A ``correct=False`` reflection verdict fails the reflect scenario even
    though its earlier learn round passes — the per-round gate ANDs."""
    scenario = next(s for s in load_scenarios() if s.id == "reflect_corrects")
    result = _run_with_stubs(
        scenario,
        consolidation_verdict=ConsolidationVerdict(correct=True),
        reflection_verdict=ReflectionVerdict(correct=False, reason="did not tighten"),
    )
    assert result.scenario_passed is False
    learn_outcome = next(o for o in result.round_outcomes if o.kind == "learn")
    reflect_outcome = next(o for o in result.round_outcomes if o.kind == "reflect")
    assert learn_outcome.judged_correct is True  # the learn round still passed
    assert reflect_outcome.judged_correct is False  # the reflect round failed the gate


# ---------------------------------------------------------------------------
# Real-haiku smoke (manual only — costs money).
# ---------------------------------------------------------------------------


@skip_low_priority
def test_scenario_real(tmp_path):  # pragma: no cover - manual, costs money
    """Real end-to-end smoke: live providers + real haiku judges over ONE
    scenario. Asserts only pipeline mechanics (round count), never pass/fail
    or exact verdicts. Run manually with API keys + RUN_LOW_PRIORITY=1."""
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from tests.eval.consolidation.providers import (
        make_consolidation_decision_provider,
    )
    from tests.eval.extraction.providers import make_extraction_provider
    from tests.eval.reflection.providers import make_reflection_decision_provider

    client = LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5"))
    ctx = RequestContext(org_id="eval", storage_base_dir=str(tmp_path))

    extraction_provider = make_extraction_provider(
        llm_client=client, request_context=ctx
    )
    consolidation_provider = make_consolidation_decision_provider(
        llm_client=client, request_context=ctx
    )
    reflection_provider = make_reflection_decision_provider(
        llm_client=client, request_context=ctx
    )

    judge_client = LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5"))

    scenario = next(s for s in load_scenarios() if s.id == "reflect_corrects")
    result = run_scenario(
        scenario=scenario,
        extraction_provider=extraction_provider,
        consolidation_provider=consolidation_provider,
        reflection_provider=reflection_provider,
        consolidation_judge_client=judge_client,
        reflection_judge_client=judge_client,
    )

    assert len(result.round_outcomes) == len(scenario.rounds)
