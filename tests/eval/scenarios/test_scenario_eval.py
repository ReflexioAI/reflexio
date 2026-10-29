"""Chained benchmarks preserve all audited outputs and their source windows."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.playbook.components.reconciliation import Decisions
from reflexio.test_support.llm_mock import assert_litellm_unpatched
from reflexio.test_support.skip_decorators import skip_low_priority
from tests.eval.consolidation.judge import ConsolidationVerdict
from tests.eval.scenarios.fixtures import load_scenarios
from tests.eval.scenarios.runner import run_scenario


def extraction_provider(_case):
    return [], [
        {
            "content": "A supported instruction",
            "trigger": "deploying",
            "rationale": "cited",
            "source_interaction_ids": [1],
        }
    ]


def consolidation_provider(case):
    ref = [source.model_dump() for source in case.candidate.evidence_refs]
    outputs = [
        {
            "content": "Announce the deploy in the channel and avoid Friday-afternoon deploys.",
            "trigger": "deploying a service",
            "rationale": "cited",
            "evidence_refs": ref,
        }
    ]
    if case.id.startswith("no_self_contradiction"):
        outputs = [
            {
                "content": content,
                "trigger": trigger,
                "rationale": "cited",
                "evidence_refs": ref,
            }
            for content, trigger in [
                ("Answer immediately.", "unambiguous factual question"),
                ("Ask a clarifying question.", "ambiguous factual question"),
            ]
        ]
    return Decisions.model_validate(
        {
            "decisions": [
                {
                    "kind": "replace",
                    "new_ids": ["NEW-0"],
                    "existing_ids": ["EXISTING-0"],
                    "reason": "synthetic provider result",
                    "outputs": outputs,
                }
            ]
        }
    )


def run(
    scenario, *, correct=True, end_state_correct=True, provider=consolidation_provider
):
    judge = MagicMock()
    judge.generate_chat_response.return_value = ConsolidationVerdict(correct=correct)
    end = MagicMock()
    end.score.return_value = SimpleNamespace(signal_f1=1 if end_state_correct else 0)
    result = run_scenario(
        scenario=scenario,
        extraction_provider=extraction_provider,
        consolidation_provider=provider,
        consolidation_judge_client=judge,
        end_state_judge=end,
    )
    return result, json.loads(end.score.call_args.kwargs["actual"])


@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda row: row.id)
def test_scenario_passes_only_with_correct_round_and_end_state(scenario):
    result, _book = run(scenario)
    assert result.scenario_passed
    assert len(result.round_outcomes) == len(scenario.rounds)
    assert not run(scenario, correct=False)[0].scenario_passed
    assert not run(scenario, end_state_correct=False)[0].scenario_passed


def test_compose_keeps_both_rules_in_one_actual_final_output():
    scenario = next(row for row in load_scenarios() if row.id == "compose_grows_skill")
    _result, book = run(scenario)
    assert len(book) == 1
    assert "announce" in book[0]["content"].lower()
    assert "friday" in book[0]["content"].lower()


def test_split_keeps_both_actual_outputs_and_distinct_conditions():
    scenario = next(
        row for row in load_scenarios() if row.id == "no_self_contradiction"
    )
    _result, book = run(scenario)
    assert len(book) == 2 and len({row["trigger"] for row in book}) == 2
    assert all(row["evidence_refs"] for row in book)


def test_later_round_keeps_earlier_source_coordinates_and_original_messages():
    scenario = load_scenarios()[0].model_copy(deep=True)
    scenario.rounds.append(scenario.rounds[0].model_copy(deep=True))
    captured = []

    def provider(case):
        captured.append(case.model_copy(deep=True))
        return consolidation_provider(case)

    run(scenario, provider=provider)
    assert len(captured[0].sessions) == 1 and len(captured[1].sessions) == 2
    prior = captured[0].candidate.evidence_refs[0]
    assert captured[1].existing[0].evidence_refs == [prior]
    assert captured[1].candidate.evidence_refs[0].session_id != prior.session_id
    assert captured[1].sessions[0] == captured[0].sessions[0]


@skip_low_priority
def test_scenario_real(tmp_path) -> None:  # pragma: no cover - manual, costs money
    # This test lives outside ``tests/e2e_tests/``, so the session-wide
    # ``litellm.completion`` patch is installed for it. Without this it would
    # grade canned mock text while reporting a real model.
    assert_litellm_unpatched()
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
        ),
        consolidation_judge_client=client,
    )
    assert len(result.round_outcomes) == len(scenario.rounds)
