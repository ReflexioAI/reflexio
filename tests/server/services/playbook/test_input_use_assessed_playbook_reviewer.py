"""The new original-input contract must fail before coercion or persistence."""

from __future__ import annotations

from copy import deepcopy

import pytest

from reflexio.server.services.playbook.components.reviewer import (
    InputUseAssessedReviewOutput,
    OriginalInputUseAssessment,
)
from tests.server.services.playbook.test_assessed_playbook_reviewer import (
    _decide,
    _parameter_response,
    _real_reviewer,
    _run_real,
)


def _as_input_response(raw):
    """Translate only synthetic test fixtures, never runtime model observations."""
    old = deepcopy(raw)
    original = old["assessments"][0]
    applicable = original.pop("original_rule_kind") == "supplied_parameter_use"
    facts = old.pop("source_parameter_facts")[0]
    assessment = {
        "id": original["id"],
        "original_proposes_task_input_use": applicable,
        "original_rule_span": original["original_rule_span"],
        "input_facts": {key: value for key, value in facts.items() if key != "id"}
        if applicable
        else None,
        "supported_core_basis": original["supported_core_basis"],
        "defect_codes": original["defect_codes"],
    }
    return {**old, "assessments": [assessment]}


def _input_response(**changes):
    return _as_input_response(_parameter_response(**changes))


def _decide_input(raw):
    return _decide(InputUseAssessedReviewOutput.model_validate(raw), version="1.31.0")


@pytest.mark.parametrize("version", ["1.31.0", "1.32.0"])
@pytest.mark.parametrize("value", [None, "true", "false", 0, 1, "missing"])
def test_native_parser_repairs_non_boolean_or_missing_applicability(value, version):
    good = _input_response()
    bad = deepcopy(good)
    if value == "missing":
        del bad["assessments"][0]["original_proposes_task_input_use"]
    else:
        bad["assessments"][0]["original_proposes_task_input_use"] = value
    reviewer, transport, calls = _real_reviewer([bad, good], version=version)
    with transport:
        outcome = _run_real(reviewer)
    assert len(calls) == 2
    assessment = outcome.assessments[0]
    assert isinstance(assessment, OriginalInputUseAssessment)
    assert assessment.original_proposes_task_input_use is True
    assert assessment.model_dump(by_alias=True) == good["assessments"][0]


@pytest.mark.parametrize("invalid", ["true-null", "false-facts", "false-basis"])
def test_native_parser_repairs_applicability_fact_and_basis_inconsistency(invalid):
    good = _input_response()
    bad = deepcopy(good)
    assessment = bad["assessments"][0]
    if invalid == "true-null":
        assessment["input_facts"] = None
    else:
        assessment["original_proposes_task_input_use"] = False
        if invalid == "false-basis":
            assessment["input_facts"] = None
        else:
            # Isolate false/non-null shape from the separate supplied-basis gate.
            assessment["supported_core_basis"] = "future_instruction"
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.31.0")
    with transport:
        outcome = _run_real(reviewer)
    assert len(calls) == 2
    assert outcome.output.decisions[0].decision == "revise"
    assert outcome.assessments[0].model_dump(by_alias=True) == good["assessments"][0]


@pytest.mark.parametrize("field", ["supply_status", "use_correspondence"])
def test_nested_facts_do_not_accept_not_applicable(field):
    good = _input_response()
    bad = deepcopy(good)
    bad["assessments"][0]["input_facts"][field] = "not_applicable"
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.31.0")
    with transport:
        _run_real(reviewer)
    assert len(calls) == 2


@pytest.mark.parametrize("unsupported", ["supply", "partial", "no-action"])
def test_supplied_basis_requires_both_established_supply_and_complete_action(
    unsupported,
):
    good = _input_response()
    bad = deepcopy(good)
    fact = bad["assessments"][0]["input_facts"]
    if unsupported == "supply":
        fact.update(
            supply_status="not_established",
            original_parameter_spans=None,
            supply_excerpts=None,
        )
    else:
        fact.update(
            use_correspondence="partial_or_unrelated"
            if unsupported == "partial"
            else "not_established",
            original_use_span=None,
            use_excerpts=None,
        )
    with pytest.raises(
        ValueError,
        match=r"assessments\[0\].input_facts supplied_parameter requires established supply and complete correspondence",
    ):
        _decide_input(bad)
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.31.0")
    with transport:
        outcome = _run_real(reviewer)
    assert len(calls) == 2
    assert outcome.output.decisions[0].decision == "revise"


@pytest.mark.parametrize("generic", [False, True])
def test_complete_input_cannot_lose_core(generic):
    bad = _input_response(
        basis="none",
        decision="reject",
        defects=["unsupported_causality", "generic"]
        if generic
        else ["unsupported_causality"],
    )
    with pytest.raises(
        ValueError,
        match=r"assessments\[0\].input_facts complete supplied-input action requires an eligible core",
    ):
        _decide_input(bad)


@pytest.mark.parametrize("basis", ["none", "future_instruction"])
def test_partial_inputs_do_not_force_eligibility_for_another_basis(basis):
    raw = _input_response(
        basis=basis, decision="reject" if basis == "none" else "revise"
    )
    raw["assessments"][0]["input_facts"].update(
        use_correspondence="partial_or_unrelated",
        original_use_span=None,
        use_excerpts=None,
    )
    _, _, _, outcome = _decide_input(raw)
    assert outcome.output.decisions[0].decision == (
        "reject" if basis == "none" else "revise"
    )


@pytest.mark.parametrize("fatal", ["not_agent_decision", "absence_inference"])
def test_fatal_input_facts_remain_private_without_a_survivor(fatal):
    raw = _input_response(basis="none", defects=[fatal], decision="reject", code=fatal)
    reviewer, _, originals, outcome = _decide_input(raw)
    assert reviewer.apply_decisions(candidates=originals, outcome=outcome) == []
    assert isinstance(outcome.assessments[0], OriginalInputUseAssessment)
    assert outcome.assessments[0].input_facts is not None
    assert outcome.output.decisions[0].evidence_ids == []
    assert outcome.supporting_evidence == outcome.source_parameter_facts == []


def test_actual_assessment_survives_privately_and_never_becomes_a_legacy_kind():
    raw = _input_response()
    reviewer, _, originals, outcome = _decide_input(raw)
    assert outcome.assessments[0].model_dump(by_alias=True) == raw["assessments"][0]
    assert outcome.source_parameter_facts == []
    saved = reviewer.apply_decisions(candidates=originals, outcome=outcome)[0]
    assert saved.content == raw["decisions"][0]["revision"]["content"]
    assert saved.trigger == raw["decisions"][0]["revision"]["trigger"]
    assert saved.rationale == raw["decisions"][0]["revision"]["rationale"]
    assert "original_rule_kind" not in outcome.assessments[0].model_dump()
    assert "input_facts" not in saved.model_dump()
    assert "Use width 800." not in (saved.notes or "")


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "supply_excerpts",
            [{"evidence_id": "C1-E1", "text": "PRIVATE_FOREIGN_SOURCE_CANARY"}],
        ),
        ("use_excerpts", [{"evidence_id": "C2-E1", "text": "Use width 800."}]),
        ("original_parameter_spans", ["PRIVATE_FOREIGN_SPAN_CANARY"]),
        ("original_use_span", "PRIVATE_FOREIGN_ACTION_CANARY"),
    ],
)
def test_nested_provenance_errors_name_real_fields_without_customer_text(field, value):
    raw = _input_response()
    raw["assessments"][0]["input_facts"][field] = value
    with pytest.raises(ValueError) as error:
        _decide_input(raw)
    assert "assessments[0].input_facts" in str(error.value)
    assert "PRIVATE_FOREIGN" not in str(error.value)
    assert "source_parameter_facts" not in str(error.value)


@pytest.mark.parametrize("version", ["1.31.0", "1.32.0"])
@pytest.mark.parametrize("recover", [False, True])
def test_native_exhaustion_retains_existing_bounded_fallback(recover, version):
    from reflexio.server.llm.litellm_client import StructuredOutputRepairError

    bad = _input_response(basis="none", decision="reject")
    responses = [bad, bad, _input_response()] if recover else [bad] * 4
    reviewer, transport, calls = _real_reviewer(
        responses, version=version, fallback=True
    )
    with transport:
        if recover:
            assert _run_real(reviewer).output.decisions[0].decision == "revise"
        else:
            with pytest.raises(StructuredOutputRepairError):
                _run_real(reviewer)
    assert [call["model"] for call in calls] == ["primary-model", "primary-model"] + [
        "fallback-model"
    ] * (1 if recover else 2)


def test_structural_validity_does_not_claim_true_semantic_applicability():
    raw = _input_response(
        basis="none", decision="reject", defects=["speculative"], code="speculative"
    )
    raw["assessments"][0].update(
        original_proposes_task_input_use=False, input_facts=None
    )
    # The model can still lie about applicability. Independent source-first gates
    # must reject this semantic result; runtime must not invent a true flag.
    _, _, _, outcome = _decide_input(raw)
    assert outcome.output.decisions[0].decision == "reject"
    assert outcome.assessments[0].model_dump(by_alias=True) == raw["assessments"][0]


@pytest.mark.parametrize("version", ["1.31.0", "1.32.0"])
def test_explicit_future_fixed_default_survives_without_input_conditioning(version):
    from reflexio.server.prompt.prompt_manager import PromptManager
    from tests.server.services.playbook.test_playbook_reviewer import (
        _candidate,
        _interaction_model,
        _reviewer,
    )

    source = "For all future image jobs, resize each image to width 800 pixels."
    candidate = _candidate(
        101, source, content="Resize every future image to width 800 pixels."
    )
    candidate.trigger = "When preparing a future image resize job"
    candidate.rationale = "The user requested this fixed width for future jobs."
    raw = _input_response(
        basis="future_instruction",
        defects=[],
        decision="accept",
        code="grounded_useful",
    )
    raw["assessments"][0].update(
        original_proposes_task_input_use=False,
        original_rule_span=candidate.content,
        input_facts=None,
    )
    raw["supporting_evidence"][0]["excerpts"] = [
        {"evidence_id": "C1-E1", "text": source}
    ]
    reviewer, _ = _reviewer(InputUseAssessedReviewOutput.model_validate(raw))
    reviewer.request_context.prompt_manager = PromptManager(
        version_override={reviewer.PROMPT_ID: version}
    )
    outcome = reviewer.decide(
        candidates=[candidate],
        request_interaction_data_models=[_interaction_model(101, source)],
        existing_playbooks=[],
        agent_context="Image assistant",
        playbook_definition="Future guidance",
        tool_context="",
    )
    saved = reviewer.apply_decisions(candidates=[candidate], outcome=outcome)[0]
    assert (saved.content, saved.trigger, saved.rationale) == (
        candidate.content,
        candidate.trigger,
        candidate.rationale,
    )
    assert isinstance(outcome.assessments[0], OriginalInputUseAssessment)
    assert outcome.assessments[0].input_facts is None
