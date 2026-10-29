"""The benchmark runs and scores complete audited batches without paid calls."""

import json
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.services.playbook.components.reconciliation import (
    Decisions,
)
from tests.eval.consolidation.case import kind_for_decision
from tests.eval.consolidation.fixtures import load_illustrative_cases
from tests.eval.consolidation.judge import (
    ConsolidationVerdict,
    judge_consolidation_decision,
)
from tests.eval.consolidation.providers import make_consolidation_decision_provider
from tests.eval.consolidation.runner import run_eval, score_case


def case():
    return load_illustrative_cases()[0]


def batch(kind="replace"):
    item = case()
    draft = {
        key: getattr(item.candidate, key) for key in ("trigger", "content", "rationale")
    }
    draft["evidence_refs"] = [ref.model_dump() for ref in item.candidate.evidence_refs]
    action = {
        "kind": kind,
        "new_ids": ["NEW-0"],
        "reason": "Synthetic supported result",
    }
    if kind == "replace":
        action["existing_ids"] = ["EXISTING-0"]
    if kind == "remove":
        action["evidence_refs"] = draft["evidence_refs"]
    else:
        action["outputs"] = [draft]
    return Decisions.model_validate({"decisions": [action]})


def provider(client):
    client.config.model = "test-model"
    return make_consolidation_decision_provider(llm_client=client)


@pytest.mark.parametrize("kind", ["create", "replace", "remove"])
def test_kind_reads_the_current_action(kind):
    assert kind_for_decision(batch(kind)) == kind


def test_multiple_kinds_are_not_truncated_to_first_action():
    actions = batch("replace").model_dump()["decisions"]
    actions[0]["new_ids"] = []
    actions.extend(batch("remove").model_dump()["decisions"])
    assert (
        kind_for_decision(Decisions.model_validate({"decisions": actions})) == "mixed"
    )


def test_empty_output_is_never_automatic_acceptance():
    with pytest.raises(ValueError, match="nonempty"):
        kind_for_decision(Decisions(decisions=[]))


@pytest.mark.parametrize("source", ["neither", "both", "wrong-length"])
def test_runner_requires_one_complete_result_per_case(source):
    with pytest.raises(ValueError):
        if source == "both":
            run_eval(
                cases=[case()], decisions=[batch()], decision_provider=lambda _: batch()
            )
        elif source == "wrong-length":
            run_eval(cases=[case()], decisions=[])
        else:
            run_eval(cases=[case()])


def test_metrics_separate_action_shape_from_judged_semantics():
    item = case().model_copy(update={"gold_kind": "create"})
    judge = MagicMock()
    judge.generate_chat_response.return_value = ConsolidationVerdict(
        correct=False,
        self_contradiction=True,
        reason="The new content contradicts a retained row",
    )
    result = run_eval(cases=[item], decisions=[batch("create")], llm_client=judge)
    assert result.kind_accuracy == 1 and result.judge_accuracy == 0
    assert result.self_contradiction_rate == 1
    assert result.confusion[("create", "create")] == 1
    assert "self-contradiction" in result.summary()
    prompt = judge.generate_chat_response.call_args.kwargs["messages"][0]["content"]
    assert item.sessions[0].messages[0].content in prompt
    assert '"id": "NEW-0"' in prompt
    assert '"id": "EXISTING-0"' in prompt


@pytest.mark.parametrize("kind", ["create", "replace"])
def test_contradiction_judging_includes_all_authored_output_kinds(kind):
    judge = MagicMock()
    judge.generate_chat_response.return_value = ConsolidationVerdict(
        correct=False, self_contradiction=True
    )
    assert score_case(
        case=case(), produced_decision=batch(kind), llm_client=judge
    ).self_contradiction


def test_remove_does_not_invent_an_output_contradiction_denominator():
    judge = MagicMock()
    judge.generate_chat_response.return_value = ConsolidationVerdict(
        correct=True, self_contradiction=True
    )
    assert (
        score_case(
            case=case(), produced_decision=batch("remove"), llm_client=judge
        ).self_contradiction
        is None
    )


def test_unjudged_metrics_remain_unknown():
    result = run_eval(cases=[case()], decision_provider=lambda _: batch())
    assert result.judge_accuracy is None and result.self_contradiction_rate is None


@pytest.mark.parametrize(
    "votes,correct,contradiction",
    [
        ([(True, False), (False, True)], False, True),
        ([(True, False), (True, False), (False, True)], True, False),
    ],
)
def test_panel_ties_and_majorities_remain_conservative(votes, correct, contradiction):
    judge = MagicMock()
    judge.generate_chat_response.side_effect = [
        ConsolidationVerdict(correct=c, self_contradiction=s) for c, s in votes
    ]
    verdict = judge_consolidation_decision(
        case=case(), produced_decision=batch(), llm_client=judge, panel_size=len(votes)
    )
    assert verdict.correct is correct and verdict.self_contradiction is contradiction


def test_zero_judges_and_unparseable_verdicts_fail():
    with pytest.raises(ValueError):
        judge_consolidation_decision(
            case=case(), produced_decision=batch(), llm_client=MagicMock(), panel_size=0
        )
    judge = MagicMock()
    judge.generate_chat_response.return_value = object()
    with pytest.raises(TypeError):
        judge_consolidation_decision(
            case=case(), produced_decision=batch(), llm_client=judge
        )


def test_live_adapter_runs_the_actual_method_and_independent_audit():
    client = MagicMock()
    expected = batch()
    client.generate_chat_response_with_provenance.side_effect = [
        CompletionResult(expected.model_dump_json(), ModelProvenance()),
        CompletionResult(json.dumps({"findings": []}), ModelProvenance()),
    ]
    item = case()
    original = item.model_dump_json()
    assert provider(client)(item) == expected
    assert item.model_dump_json() == original
    assert client.generate_chat_response_with_provenance.call_count == 2
    draft, audit = client.generate_chat_response_with_provenance.call_args_list
    original_payload = json.loads(draft.kwargs["messages"][1]["content"])
    assert original_payload["new"][0]["id"] == "NEW-0"
    assert original_payload["sessions"] == [
        session.model_dump() for session in item.sessions
    ]
    audited_payload = json.loads(audit.kwargs["messages"][1]["content"])
    assert audited_payload["proposed_memory"]


def test_invalid_empty_provider_output_abstains_instead_of_accepting():
    client = MagicMock()
    client.generate_chat_response_with_provenance.return_value = CompletionResult(
        '{"decisions": []}', ModelProvenance()
    )
    with pytest.raises(ValueError, match="consumed exactly once"):
        provider(client)(case())


@pytest.mark.parametrize(
    "invalid", ["missing-citation", "absent-message", "duplicate-coordinate"]
)
def test_adapter_refuses_unwitnessed_or_ambiguous_evidence_before_dispatch(invalid):
    item = case()
    if invalid == "missing-citation":
        item.candidate.evidence_refs = []
    elif invalid == "absent-message":
        item.sessions = []
    else:
        item.sessions[0].messages.append(item.sessions[0].messages[0].model_copy())
    client = MagicMock()
    with pytest.raises(ValueError):
        provider(client)(item)
    client.generate_chat_response_with_provenance.assert_not_called()


def test_synthetic_fixture_has_independent_source_messages_and_current_kinds():
    cases = load_illustrative_cases()
    assert {item.gold_kind for item in cases} == {"create", "replace", "remove"}
    assert len({item.id for item in cases}) == len(cases) == 5
    assert all(item.candidate.evidence_refs and item.sessions for item in cases)
