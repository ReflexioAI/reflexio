"""Raw inventory validation must precede any public projection or persistence."""

from copy import deepcopy

import pytest

from reflexio.server.services.playbook.components.reviewer import (
    InventoryAssessedReviewOutput,
    InventoryRuleAssessment,
)
from tests.server.services.playbook.test_assessed_playbook_reviewer import (
    _decide,
    _real_reviewer,
    _run_real,
)
from tests.server.services.playbook.test_input_use_assessed_playbook_reviewer import (
    _input_response,
)


def _inventory_response(**changes):
    """Synthetic fixtures only; runtime observations never undergo translation."""
    raw = _input_response(**changes)
    for assessment in raw["assessments"]:
        del assessment["original_rule_span"]
    for decision in raw["decisions"]:
        del decision["reason_code"]
    return raw


@pytest.mark.parametrize("fatal", ["not_agent_decision", "absence_inference"])
@pytest.mark.parametrize("verb", ["accept", "revise"])
def test_raw_fatal_decision_is_repaired_without_legacy_coercion(fatal, verb):
    good = _inventory_response(basis="none", defects=[fatal], decision="reject")
    bad = deepcopy(good)
    bad["decisions"][0]["decision"] = verb
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.40.0")
    with transport:
        outcome = _run_real(reviewer)
    assert len(calls) == 2
    assert outcome.output.decisions[0].decision == "reject"
    assert outcome.output.decisions[0].reason_code == fatal
    assert outcome.inventory_output is not None
    assert outcome.inventory_output.model_dump(by_alias=True) == good


@pytest.mark.parametrize("reverse", [False, True])
def test_original_inventory_derives_priority_and_preserves_raw_fields(reverse):
    defects = ["unsupported_evidence", "speculative", "unsupported_causality"]
    if reverse:
        defects.reverse()
    raw = _inventory_response(defects=defects)
    reviewer, transport, calls = _real_reviewer([raw], version="1.40.0")
    with transport:
        outcome = _run_real(reviewer)
    assert len(calls) == 1
    assert isinstance(outcome.assessments[0], InventoryRuleAssessment)
    assert outcome.inventory_output is not None
    assert outcome.inventory_output.model_dump(by_alias=True) == raw
    expected = {**raw["decisions"][0], "reason_code": "unsupported_causality"}
    assert outcome.output.decisions[0].model_dump(by_alias=True) == expected
    assert "original_rule_span" not in outcome.assessments[0].model_dump()


@pytest.mark.parametrize(
    "fault",
    [
        "foreign-source",
        "source-only-action",
        "partial-basis",
        "true-null",
        "unknown-id",
    ],
)
def test_invalid_raw_provenance_and_facts_require_native_repair(fault):
    good = _inventory_response()
    bad = deepcopy(good)
    a = bad["assessments"][0]
    if fault == "foreign-source":
        a["input_facts"]["supply_excerpts"][0]["evidence_id"] = "C2E1"
    elif fault == "source-only-action":
        a["input_facts"]["original_use_span"] = "SOURCE_ONLY_NOT_IN_ORIGINAL"
    elif fault == "partial-basis":
        a["input_facts"].update(
            use_correspondence="partial_or_unrelated",
            original_use_span=None,
            use_excerpts=None,
        )
    elif fault == "true-null":
        a["input_facts"] = None
    else:
        a["id"] = "C2"
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.40.0")
    with transport:
        outcome = _run_real(reviewer)
    assert len(calls) == 2
    assert outcome.inventory_output is not None
    assert outcome.inventory_output.model_dump(by_alias=True) == good


def test_no_raw_projection_can_hide_invalid_final_support():
    good = _inventory_response()
    bad = deepcopy(good)
    bad["supporting_evidence"][0]["excerpts"][0]["text"] = "PRIVATE_FOREIGN_QUOTE"
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.40.0")
    with transport:
        outcome = _run_real(reviewer)
    assert len(calls) == 2
    assert outcome.inventory_output is not None
    assert outcome.inventory_output.model_dump(by_alias=True) == good


def test_new_shape_does_not_accept_retired_provider_fields():
    raw = _inventory_response()
    raw["decisions"][0]["reason_code"] = "unsupported_causality"
    with pytest.raises(ValueError):
        InventoryAssessedReviewOutput.model_validate(raw)


def test_complete_supplied_core_cannot_be_discarded():
    raw = _inventory_response(basis="none", decision="reject", defects=["generic"])
    with pytest.raises(
        ValueError, match="complete supplied-input action requires an eligible core"
    ):
        _decide(InventoryAssessedReviewOutput.model_validate(raw), version="1.40.0")


@pytest.mark.parametrize("fault", ["fact", "final-support", "fatal-decision"])
def test_all_raw_checks_precede_public_model_construction(monkeypatch, fault):
    from reflexio.server.services.playbook.components.reviewer import (
        CandidateReviewDecision,
    )

    good = _inventory_response()
    bad = deepcopy(good)
    if fault == "fact":
        bad["assessments"][0]["input_facts"]["supply_excerpts"][0]["text"] = (
            "FOREIGN_PRIVATE_FACT"
        )
    elif fault == "final-support":
        bad["supporting_evidence"][0]["excerpts"][0]["text"] = (
            "FOREIGN_PRIVATE_FINAL_SUPPORT"
        )
    else:
        bad["assessments"][0]["defect_codes"] = ["absence_inference"]
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.40.0")
    original_validate = CandidateReviewDecision.model_validate
    projection_stages = []

    def observe(cls, value, *args, **kwargs):
        projection_stages.append(len(calls))
        return original_validate(value, *args, **kwargs)

    monkeypatch.setattr(CandidateReviewDecision, "model_validate", classmethod(observe))
    with transport:
        outcome = _run_real(reviewer)
    assert len(calls) == 2
    assert projection_stages and set(projection_stages) == {2}
    assert outcome.inventory_output is not None
    assert outcome.inventory_output.model_dump(by_alias=True) == good


@pytest.mark.parametrize("recover", [False, True])
def test_new_raw_contract_retains_bounded_native_fallback(recover):
    from reflexio.server.llm.litellm_client import StructuredOutputRepairError

    bad = _inventory_response(basis="none", decision="reject")
    responses = [bad, bad, _inventory_response()] if recover else [bad] * 4
    reviewer, transport, calls = _real_reviewer(
        responses, version="1.40.0", fallback=True
    )
    with transport:
        if recover:
            outcome = _run_real(reviewer)
            assert outcome.inventory_output is not None
            assert outcome.inventory_output.model_dump(by_alias=True) == responses[-1]
            assert outcome.output.decisions[0].reason_code == "unsupported_causality"
        else:
            with pytest.raises(StructuredOutputRepairError):
                _run_real(reviewer)
    assert [call["model"] for call in calls] == ["primary-model", "primary-model"] + [
        "fallback-model"
    ] * (1 if recover else 2)


@pytest.mark.parametrize("repair", [False, True, "exhaust"])
@pytest.mark.parametrize("version", ["1.40.0", "1.41.0"])
def test_native_validator_observes_actual_raw_type_without_retired_fields(
    monkeypatch, repair, version
):
    from reflexio.server.llm.litellm_client import StructuredOutputRepairError
    from reflexio.server.services.playbook.components.reviewer import (
        CandidateReviewDecision,
    )

    good = _inventory_response()
    bad = deepcopy(good)
    bad["assessments"][0]["input_facts"]["original_use_span"] = (
        "SOURCE_ONLY_FOREIGN_ACTION"
    )
    replies = [bad, bad] if repair == "exhaust" else [bad, good] if repair else [good]
    reviewer, transport, calls = _real_reviewer(replies, version=version)
    generate = reviewer.client.generate_chat_response
    observations = []
    projected = []
    public_validate = CandidateReviewDecision.model_validate

    def observe_public(cls, value, *args, **kwargs):
        projected.append(value)
        return public_validate(value, *args, **kwargs)

    def observe_generate(*args, **options):
        assert options["response_format"] is InventoryAssessedReviewOutput
        validator = options["structured_output_validator"]

        def observe_validation(value):
            assert type(value) is InventoryAssessedReviewOutput
            raw = value.model_dump(mode="json", by_alias=True)
            errors = validator(value)
            observations.append({"stage": len(calls), "raw": raw, "errors": errors})
            assert "original_rule_span" not in raw["assessments"][0]
            assert "reason_code" not in raw["decisions"][0]
            return errors

        return generate(
            *args, **{**options, "structured_output_validator": observe_validation}
        )

    monkeypatch.setattr(reviewer.client, "generate_chat_response", observe_generate)
    monkeypatch.setattr(
        CandidateReviewDecision, "model_validate", classmethod(observe_public)
    )
    with transport:
        if repair == "exhaust":
            with pytest.raises(StructuredOutputRepairError) as failed:
                _run_real(reviewer)
            assert failed.value.validation_errors == observations[-1]["errors"]
            assert len(calls) == 2 and projected == []
        else:
            outcome = _run_real(reviewer)
            assert outcome.inventory_output is not None
            assert outcome.inventory_output.model_dump(by_alias=True) == good
            assert len(calls) == (2 if repair else 1)
            assert observations[-1]["errors"] == ()
            assert projected and projected[0]["reason_code"] == "unsupported_causality"
    assert [o["raw"] for o in observations] == replies
    assert [o["stage"] for o in observations] == list(range(1, len(replies) + 1))
    if repair:
        assert "original_use_span" in observations[0]["errors"][0]
