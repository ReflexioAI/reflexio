"""Tests for the multi-round extraction and consolidation scenario harness."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.playbook.components.consolidator import (
    ConsolidationDecision,
    DifferentiateDecision,
    IndependentDecision,
    RejectNewDecision,
    UnifyDecision,
)
from reflexio.test_support.skip_decorators import skip_low_priority
from tests.eval.consolidation.judge import ConsolidationVerdict
from tests.eval.scenarios.book import _next_id, apply_consolidation
from tests.eval.scenarios.case import BookRule
from tests.eval.scenarios.fixtures import load_scenarios
from tests.eval.scenarios.runner import ScenarioResult, run_scenario

_CANDIDATE_TRIGGER = "a freshly extracted trigger"


def _extraction_provider(case: dict) -> tuple[list[Any], list[Any]]:
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


def _consolidation_provider(case: Any) -> ConsolidationDecision:
    new_id = case.candidate.new_id
    if case.gold_kind == "unify":
        return UnifyDecision(
            new_id=new_id,
            archive_existing_ids=[0],
            content=(
                "Announce the deploy in the channel and avoid Friday-afternoon deploys."
            ),
            trigger="deploying a service",
            rationale="",
        )
    if case.gold_kind == "differentiate":
        return DifferentiateDecision(
            new_id=new_id,
            existing_id=case.existing[0].id,
            refined_new_trigger="user asks an ambiguous factual question",
            refined_existing_trigger="user asks an unambiguous factual question",
        )
    if case.gold_kind == "reject_new":
        return RejectNewDecision(
            new_id=new_id,
            superseded_by_existing_id=case.existing[0].id,
        )
    return IndependentDecision(new_id=new_id)


def _verdict_client(verdict: ConsolidationVerdict) -> MagicMock:
    client = MagicMock()
    client.generate_chat_response.return_value = verdict
    return client


def _run_with_stubs(
    scenario: Any,
    *,
    consolidation_verdict: ConsolidationVerdict,
) -> ScenarioResult:
    return run_scenario(
        scenario=scenario,
        extraction_provider=_extraction_provider,
        consolidation_provider=_consolidation_provider,
        consolidation_judge_client=_verdict_client(consolidation_verdict),
    )


def _expected_final_book(scenario: Any) -> list[BookRule]:
    book = [rule.model_copy() for rule in scenario.seed_book]
    for scenario_round in scenario.rounds:
        gold_kind = scenario_round.gold.get("consolidation_kind", "independent")
        _profiles, playbooks = _extraction_provider({})
        for playbook in playbooks:
            existing_order = list(book)
            candidate = BookRule(
                id=_next_id(book),
                content=playbook["content"],
                trigger=playbook["trigger"],
                rationale=playbook["rationale"],
            )
            case = MagicMock(
                gold_kind=gold_kind,
                existing=existing_order,
                candidate=MagicMock(new_id=f"new-{candidate.id}"),
            )
            decision = _consolidation_provider(case)
            book = apply_consolidation(
                book,
                candidate,
                decision,
                existing_order=existing_order,
            )
    return book


def test_fixtures_load_distinct_learning_scenarios() -> None:
    scenarios = load_scenarios()
    assert {scenario.id for scenario in scenarios} == {
        "compose_grows_skill",
        "no_self_contradiction",
    }
    assert all(scenario.rounds for scenario in scenarios)


@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda scenario: scenario.id)
def test_scenario_chain_passes_with_correct_judges(scenario: Any) -> None:
    result = _run_with_stubs(
        scenario,
        consolidation_verdict=ConsolidationVerdict(correct=True),
    )
    assert result.scenario_passed is True
    assert len(result.round_outcomes) == len(scenario.rounds)
    assert all(outcome.judged_correct for outcome in result.round_outcomes)


def test_compose_scenario_unifies_into_one_skill() -> None:
    scenario = next(
        scenario
        for scenario in load_scenarios()
        if scenario.id == "compose_grows_skill"
    )
    final_book = _expected_final_book(scenario)
    assert len(final_book) == 1
    assert "announce" in final_book[0].content.lower()
    assert "friday" in final_book[0].content.lower()


def test_differentiate_scenario_keeps_two_rules() -> None:
    scenario = next(
        scenario
        for scenario in load_scenarios()
        if scenario.id == "no_self_contradiction"
    )
    final_book = _expected_final_book(scenario)
    assert len(final_book) == 2
    assert len({rule.trigger for rule in final_book}) == 2


def test_wrong_consolidation_verdict_fails_scenario() -> None:
    scenario = next(
        scenario
        for scenario in load_scenarios()
        if scenario.id == "compose_grows_skill"
    )
    result = _run_with_stubs(
        scenario,
        consolidation_verdict=ConsolidationVerdict(
            correct=False,
            reason="bad merge",
        ),
    )
    assert result.scenario_passed is False


@skip_low_priority
def test_scenario_real(tmp_path) -> None:  # pragma: no cover - manual, costs money
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from tests.eval.consolidation.providers import (
        make_consolidation_decision_provider,
    )
    from tests.eval.extraction.providers import make_extraction_provider

    client = LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5"))
    context = RequestContext(org_id="eval", storage_base_dir=str(tmp_path))
    scenario = load_scenarios()[0]
    result = run_scenario(
        scenario=scenario,
        extraction_provider=make_extraction_provider(
            llm_client=client,
            request_context=context,
        ),
        consolidation_provider=make_consolidation_decision_provider(
            llm_client=client,
            request_context=context,
        ),
        consolidation_judge_client=client,
    )
    assert len(result.round_outcomes) == len(scenario.rounds)
