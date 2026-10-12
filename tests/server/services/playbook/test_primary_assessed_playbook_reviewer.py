"""The primary contract preserves core, provenance and native repair safeguards."""

from copy import deepcopy

import pytest

from reflexio.server.llm.litellm_client import StructuredOutputRepairError
from reflexio.server.services.playbook.components.reviewer import (
    CandidateReviewDecision,
    PrimaryAssessedReviewOutput,
    PrimaryRuleAssessment,
)
from tests.server.services.playbook.test_assessed_playbook_reviewer import (
    _real_reviewer,
)
from tests.server.services.playbook.test_inventory_assessed_playbook_reviewer import (
    _inventory_response,
)
from tests.server.services.playbook.test_playbook_reviewer import (
    _candidate,
    _interaction_model,
)


def _primary_response(
    *, primary: str | None = "unsupported_causality", duplicate=False, **changes
):
    """Translate synthetic fixtures only, never an observed model response."""
    raw = _inventory_response(defects=[] if primary is None else [primary], **changes)
    for assessment in raw["assessments"]:
        del assessment["defect_codes"]
        assessment["primary_defect"] = primary
        assessment["duplicates_existing_core"] = duplicate
    return raw


def _run_primary(
    reviewer,
    *,
    duplicate=False,
    content="Use the supplied width; never ask again because it causes delays.",
    trigger=None,
):
    candidate = _candidate(
        101,
        "Use width 800.",
        content=content,
    )
    if trigger is not None:
        candidate.trigger = trigger
    candidate.notes = "Original note"
    existing = candidate.model_copy(
        update={
            "content": "Use the supplied width if assigned for the current image.",
            "trigger": "When resizing an image with an assigned width",
            "rationale": "The user supplied the image width.",
        }
    )
    outcome = reviewer.decide(
        candidates=[candidate],
        request_interaction_data_models=[_interaction_model(101, "Use width 800.")],
        existing_playbooks=[existing] if duplicate else [],
        agent_context="UNTRUSTED_CONTEXT_CANARY",
        playbook_definition="Future image guidance",
        tool_context="",
    )
    return [candidate], outcome


@pytest.mark.parametrize("repair", [False, True, "exhaust"])
def test_actual_primary_raw_type_repair_and_projection_order(monkeypatch, repair):
    good = _primary_response()
    bad = deepcopy(good)
    bad["assessments"][0]["primary_defect"] = "generic"
    replies = [bad, bad] if repair == "exhaust" else [bad, good] if repair else [good]
    reviewer, transport, calls = _real_reviewer(replies, version="1.42.0")
    generate = reviewer.client.generate_chat_response
    public_validate = CandidateReviewDecision.model_validate
    observed, projected, returned = [], [], []

    def observe_public(cls, value, *args, **kwargs):
        projected.append(len(calls))
        return public_validate(value, *args, **kwargs)

    def observe_generate(*args, **options):
        assert options["response_format"] is PrimaryAssessedReviewOutput
        validator = options["structured_output_validator"]

        def observe_validation(value):
            assert type(value) is PrimaryAssessedReviewOutput
            errors = validator(value)
            observed.append((value.model_dump(mode="json", by_alias=True), errors))
            return errors

        value = generate(
            *args, **{**options, "structured_output_validator": observe_validation}
        )
        returned.append(value)
        return value

    monkeypatch.setattr(reviewer.client, "generate_chat_response", observe_generate)
    monkeypatch.setattr(
        CandidateReviewDecision, "model_validate", classmethod(observe_public)
    )
    with transport:
        if repair == "exhaust":
            with pytest.raises(StructuredOutputRepairError) as failed:
                _run_primary(reviewer)
            assert failed.value.validation_errors == observed[-1][1]
            assert returned == projected == []
        else:
            candidates, outcome = _run_primary(reviewer)
            assert outcome.primary_output is not None
            assert outcome.primary_output is returned[-1]
            assert outcome.primary_output.model_dump(by_alias=True) == good
            assert outcome.inventory_output is None
            assert isinstance(outcome.assessments[0], PrimaryRuleAssessment)
            assert projected and set(projected) == {len(replies)}
            assert outcome.output.decisions[0].model_dump(by_alias=True) == {
                **good["decisions"][0],
                "reason_code": "unsupported_causality",
            }
            saved = reviewer.apply_decisions(candidates=candidates, outcome=outcome)[0]
            assert saved.content == good["decisions"][0]["revision"]["content"]
            assert (saved.notes or "").startswith("Original note")
            assert "primary_defect" not in saved.model_dump()
            candidates[0].content = "Mutated original"
            with pytest.raises(ValueError, match="original candidate batch"):
                reviewer.apply_decisions(candidates=candidates, outcome=outcome)
    assert [raw for raw, _ in observed] == replies
    assert len(calls) == len(replies)
    if repair:
        assert any("eligible core forbids generic" in e for e in observed[0][1])


@pytest.mark.parametrize(
    "fault",
    [
        "fatal",
        "foreign-source",
        "origin",
        "partial",
        "final-support",
        "unknown-id",
        "repeated-id",
    ],
)
def test_raw_errors_require_repair_before_any_public_decision(monkeypatch, fault):
    good = _primary_response()
    bad = deepcopy(good)
    assessment = bad["assessments"][0]
    if fault == "fatal":
        assessment["primary_defect"] = "absence_inference"
    elif fault == "foreign-source":
        assessment["input_facts"]["supply_excerpts"][0]["evidence_id"] = "C2-E1"
    elif fault == "origin":
        assessment["input_facts"]["original_use_span"] = "SOURCE_ONLY_FOREIGN_ACTION"
    elif fault == "partial":
        assessment["input_facts"].update(
            use_correspondence="partial_or_unrelated",
            original_use_span=None,
            use_excerpts=None,
        )
    elif fault == "final-support":
        bad["supporting_evidence"][0]["excerpts"][0]["text"] = "FOREIGN_FINAL_QUOTE"
    elif fault == "unknown-id":
        assessment["id"] = "C2"
    else:
        bad["assessments"].append(deepcopy(assessment))
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.42.0")
    validate = CandidateReviewDecision.model_validate
    stages = []

    def observe(cls, value, *args, **kwargs):
        stages.append(len(calls))
        return validate(value, *args, **kwargs)

    monkeypatch.setattr(CandidateReviewDecision, "model_validate", classmethod(observe))
    with transport:
        _, outcome = _run_primary(reviewer)
    assert len(calls) == 2 and stages and set(stages) == {2}
    assert outcome.primary_output is not None
    assert outcome.primary_output.model_dump(by_alias=True) == good


@pytest.mark.parametrize(
    "primary,duplicate,decision,valid",
    [
        ("unsupported_causality", False, "reject", False),
        ("unsupported_causality", True, "reject", True),
        ("unsupported_causality", True, "revise", True),
        ("redundant", True, "reject", True),
        ("redundant", False, "reject", False),
        (None, True, "accept", False),
        ("late_trigger", True, "revise", False),
    ],
)
def test_duplicate_rejection_preserves_higher_primary(
    primary, duplicate, decision, valid
):
    raw = _primary_response(primary=primary, duplicate=duplicate, decision=decision)
    good = _primary_response()
    reviewer, transport, calls = _real_reviewer(
        [raw] if valid else [raw, good], version="1.42.0"
    )
    with transport:
        _, outcome = _run_primary(reviewer, duplicate=duplicate)
    assert len(calls) == (1 if valid else 2)
    assert outcome.output.decisions[0].reason_code == (
        primary if valid else "unsupported_causality"
    )


@pytest.mark.parametrize("value", [None, "false", 0, "missing"])
def test_duplicate_flag_is_required_and_strict_in_native_parser(value):
    good = _primary_response()
    bad = deepcopy(good)
    if value == "missing":
        del bad["assessments"][0]["duplicates_existing_core"]
    else:
        bad["assessments"][0]["duplicates_existing_core"] = value
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.42.0")
    with transport:
        _, outcome = _run_primary(reviewer)
    assert len(calls) == 2 and outcome.primary_output is not None


@pytest.mark.parametrize("exhaust", [False, True])
@pytest.mark.parametrize("foreign_id", [False, True])
def test_empty_existing_context_cannot_authorize_duplicate_rejection(
    monkeypatch, exhaust, foreign_id
):
    bad = _primary_response(duplicate=True, decision="reject")
    private_canary = "UNTRUSTED_PRIVATE_CANDIDATE_ID_CANARY"
    if foreign_id:
        bad["assessments"][0]["id"] = private_canary
    good = _primary_response()
    reviewer, transport, calls = _real_reviewer(
        [bad, bad] if exhaust else [bad, good], version="1.42.0"
    )
    generate = reviewer.client.generate_chat_response
    public_validate = CandidateReviewDecision.model_validate
    errors_seen, projected = [], []

    def observe_generate(*args, **options):
        validator = options["structured_output_validator"]

        def observe_validation(value):
            errors = validator(value)
            errors_seen.append(errors)
            return errors

        return generate(
            *args, **{**options, "structured_output_validator": observe_validation}
        )

    def observe_public(cls, value, *args, **kwargs):
        projected.append(len(calls))
        return public_validate(value, *args, **kwargs)

    monkeypatch.setattr(reviewer.client, "generate_chat_response", observe_generate)
    monkeypatch.setattr(
        CandidateReviewDecision, "model_validate", classmethod(observe_public)
    )
    expected = ("assessments[0]: duplicate requires existing playbook context",)
    with transport:
        if exhaust:
            with pytest.raises(StructuredOutputRepairError) as failed:
                _run_primary(reviewer)
            assert failed.value.validation_errors == errors_seen[-1]
            assert errors_seen[0] == errors_seen[1]
            assert projected == []
        else:
            _, outcome = _run_primary(reviewer)
            assert outcome.output.decisions[0].decision == "revise"
            assert errors_seen[1] == ()
            assert projected and set(projected) == {2}
    assert expected[0] in errors_seen[0]
    if not foreign_id:
        assert errors_seen[0] == expected
    assert all(
        private_canary not in message for errors in errors_seen for message in errors
    )
    assert len(calls) == 2


@pytest.mark.parametrize("recover", [False, True])
def test_primary_contract_retains_bounded_native_fallback(recover):
    bad = _primary_response(basis="none", decision="reject")
    responses = [bad, bad, _primary_response()] if recover else [bad] * 4
    reviewer, transport, calls = _real_reviewer(
        responses, version="1.42.0", fallback=True
    )
    with transport:
        if recover:
            _, outcome = _run_primary(reviewer)
            assert outcome.primary_output is not None
        else:
            with pytest.raises(StructuredOutputRepairError):
                _run_primary(reviewer)
    assert [call["model"] for call in calls] == ["primary-model"] * 2 + [
        "fallback-model"
    ] * (1 if recover else 2)


def test_partial_operand_supply_does_not_manufacture_a_clipped_action():
    raw = _primary_response(primary="speculative", basis="none", decision="reject")
    raw["assessments"][0]["input_facts"].update(
        original_parameter_spans=["width"],
        use_correspondence="partial_or_unrelated",
        original_use_span=None,
        use_excerpts=None,
    )
    reviewer, transport, calls = _real_reviewer([raw], version="1.42.0")
    with transport:
        candidates, outcome = _run_primary(
            reviewer, content="Use the supplied width and height to resize the image."
        )
    assert len(calls) == 1 and outcome.primary_output is not None
    assert outcome.primary_output.model_dump(by_alias=True) == raw
    assert reviewer.apply_decisions(candidates=candidates, outcome=outcome) == []


def test_null_primary_preserves_an_already_scoped_supported_original():
    raw = _primary_response(primary=None, decision="accept")
    reviewer, transport, calls = _real_reviewer([raw], version="1.42.0")
    with transport:
        candidates, outcome = _run_primary(
            reviewer,
            content="Use the supplied width for the current image when it has been assigned.",
            trigger="When resizing an image with an assigned width",
        )
    saved = reviewer.apply_decisions(candidates=candidates, outcome=outcome)
    assert len(calls) == 1 and len(saved) == 1
    assert saved[0].content == candidates[0].content
    assert saved[0].trigger == candidates[0].trigger
    assert outcome.output.decisions[0].reason_code == "grounded_useful"


@pytest.mark.parametrize(
    "primary,duplicate,error",
    [
        (None, False, "no core requires an ineligibility defect"),
        ("late_trigger", False, "no core requires an ineligibility defect"),
        ("compound", False, "no core requires an ineligibility defect"),
        ("redundant", True, "no core requires an ineligibility defect"),
        ("speculative", True, "duplicate requires an eligible core"),
    ],
)
def test_no_core_primary_guards_repair_before_projection(
    monkeypatch, primary, duplicate, error
):
    good = _primary_response(
        primary="speculative",
        kind="standing_value_preference",
        basis="none",
        decision="reject",
    )
    bad = deepcopy(good)
    bad["assessments"][0].update(
        primary_defect=primary, duplicates_existing_core=duplicate
    )
    reviewer, transport, calls = _real_reviewer([bad, good], version="1.42.0")
    generate = reviewer.client.generate_chat_response
    validate = CandidateReviewDecision.model_validate
    errors, stages = [], []

    def observe_generate(*args, **options):
        validator = options["structured_output_validator"]

        def observe_validation(value):
            result = validator(value)
            errors.append(result)
            return result

        return generate(
            *args, **{**options, "structured_output_validator": observe_validation}
        )

    def observe_public(cls, value, *args, **kwargs):
        stages.append(len(calls))
        return validate(value, *args, **kwargs)

    monkeypatch.setattr(reviewer.client, "generate_chat_response", observe_generate)
    monkeypatch.setattr(
        CandidateReviewDecision, "model_validate", classmethod(observe_public)
    )
    with transport:
        _, outcome = _run_primary(
            reviewer,
            content="Always use the width recorded for this image as every future image's default.",
        )
    assert any(error in message for message in errors[0])
    assert not any("input_facts" in message for message in errors[0])
    assert errors[-1] == () and len(calls) == 2
    assert stages and set(stages) == {2}
    assert outcome.output.decisions[0].reason_code == "speculative"


@pytest.mark.parametrize("primary", ["generic", "speculative"])
def test_no_core_with_explaining_primary_can_reject_without_input_facts(primary):
    raw = _primary_response(
        primary=primary,
        kind="standing_value_preference",
        basis="none",
        decision="reject",
    )
    reviewer, transport, calls = _real_reviewer([raw], version="1.42.0")
    with transport:
        _, outcome = _run_primary(
            reviewer,
            content="Resize the image as requested."
            if primary == "generic"
            else "Always use the width recorded for this image as every future image's default.",
        )
    assert len(calls) == 1 and outcome.primary_output is not None
    assert outcome.primary_output.model_dump(by_alias=True) == raw
