"""Contracts for grounded sparse repairs, without any storage or LLM access."""

import json

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.rule_evidence import (
    run_rule_evidence,
    validate_diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.models import (
    Diagnostic,
    Proposal,
    RuleEvidenceInput,
)


@pytest.fixture
def data() -> RuleEvidenceInput:
    return RuleEvidenceInput.model_validate_json(
        json.dumps(
            {
                "incumbent": {
                    "id": 1,
                    "user_id": "u",
                    "version": "frozen-version",
                    "trigger": "Sending a report",
                    "content": "Send once. Report queued as delivered. Keep the recipient.",
                },
                "sessions": [
                    {
                        "session_id": "s",
                        "user_id": "u",
                        "exposed_version": "frozen-version",
                        "source_limitations": "",
                        "messages": [
                            {"turn": 3, "role": "tool", "content": 'status="queued"'},
                            {"turn": 8, "role": "agent", "content": "Delivered."},
                        ],
                    }
                ],
                "agent_context": "Report sender",
                "success_definition": "Send once and report actual status",
                "available_tools": [{"name": "send", "description": "Returns queued"}],
                "tool_context_status": "captured",
            }
        )
    )


@pytest.fixture
def diagnostic() -> Diagnostic:
    return Diagnostic.model_validate_json(
        json.dumps(
            {
                "session_id": "s",
                "user_id": "u",
                "observations": [
                    {
                        "dimension": "evidence_grounding",
                        "kind": "problem",
                        "observation": "Reports delivery from queue acceptance",
                        "consequence": "User gets an unsupported delivery claim",
                        "evidence_refs": [{"session_id": "s", "turn": 8}],
                    }
                ],
                "rule_assessment": {
                    "disposition": "rule_gap",
                    "incumbent_quote": "Report queued as delivered.",
                    "expected_if_followed": "Queued receipts are reported as delivered",
                    "reason": "Literal compliance causes the observed false claim",
                    "evidence_refs": [
                        {"session_id": "s", "turn": 3},
                        {"session_id": "s", "turn": 8},
                    ],
                },
                "missing_information": [],
            }
        )
    )


def repair() -> Proposal:
    return Proposal(
        decision="hypothesis",
        gap_index=0,
        hypothesis="Report service status faithfully",
        replacement="Report queued as queued, not delivered.",
        uncertainties=[],
        abstention_reason="",
    )


def test_exact_clause_repair_and_inherited_citations(
    data: RuleEvidenceInput, diagnostic: Diagnostic
) -> None:
    calls = []

    def invoke(system, payload, schema):
        calls.append((system, payload, schema))
        return diagnostic if schema is Diagnostic else repair()

    result = run_rule_evidence(data, invoke)
    assert result.revised_content == (
        "Send once. Report queued as queued, not delivered. Keep the recipient."
    )
    assert result.evidence_refs == diagnostic.rule_assessment.evidence_refs
    assert len(calls) == 2
    assert calls[0][1]["messages"][0]["turn"] == 3
    assert calls[1][1]["available_tools"] == [
        {"name": "send", "description": "Returns queued"}
    ]
    assert "is_success" not in result.model_dump_json()


@pytest.mark.parametrize(
    "disposition", ["adherence_failure", "no_issue", "not_applicable", "unknown"]
)
def test_non_gap_never_invokes_tuner(data, diagnostic, disposition):
    value = diagnostic.model_dump(mode="json")
    value["rule_assessment"]["disposition"] = disposition
    value["missing_information"] = (
        ["Need the service receipt"] if disposition == "unknown" else []
    )
    diagnostic = Diagnostic.model_validate(value)
    calls = []

    def invoke(system, payload, schema):
        calls.append(schema)
        return diagnostic

    result = run_rule_evidence(data, invoke)
    assert result.decision == "abstain"
    assert result.revised_content is None
    assert calls == [Diagnostic]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["sessions"][0].update(user_id="other"),
        lambda value: value["sessions"][0].update(exposed_version="stale"),
        lambda value: value["sessions"].append(value["sessions"][0]),
        lambda value: value["sessions"][0]["messages"].append(
            value["sessions"][0]["messages"][0]
        ),
    ],
)
def test_invalid_source_never_calls_model(data, mutate):
    value = data.model_dump(mode="json")
    mutate(value)
    with pytest.raises(ValidationError):
        RuleEvidenceInput.model_validate(value)


@pytest.mark.parametrize("session_id,turn", [("other", 3), ("s", 0), ("s", 9)])
def test_cannot_cite_unseen_or_foreign_turn(data, diagnostic, session_id, turn):
    value = diagnostic.model_dump(mode="json")
    value["rule_assessment"]["evidence_refs"] = [
        {"session_id": session_id, "turn": turn}
    ]
    with pytest.raises(ValueError, match="supplied turns"):
        validate_diagnostic(Diagnostic.model_validate(value), data, data.sessions[0])


def test_repeated_quote_cannot_be_edited(data, diagnostic):
    value = data.model_dump(mode="json")
    value["incumbent"]["content"] += " Report queued as delivered."
    data = RuleEvidenceInput.model_validate(value)
    with pytest.raises(ValueError, match="exactly once"):
        run_rule_evidence(data, lambda *_: diagnostic)


@pytest.mark.parametrize(
    "field,value", [("gap_index", 1), ("replacement", "Report queued as delivered.")]
)
def test_tuner_cannot_select_unadmitted_gap_or_return_noop(
    data, diagnostic, field, value
):
    proposal = repair().model_copy(update={field: value})
    with pytest.raises(ValueError, match="admitted gap|change the quoted"):
        run_rule_evidence(
            data, lambda _, __, schema: diagnostic if schema is Diagnostic else proposal
        )


def test_provider_failure_is_not_abstention(data):
    def invoke(*_):
        raise TimeoutError("provider unavailable")

    with pytest.raises(TimeoutError):
        run_rule_evidence(data, invoke)


def test_large_payload_refused_before_model(data):
    value = data.model_dump(mode="json")
    value["sessions"][0]["messages"][0]["content"] = "large evidence " * 65_000
    calls = []

    def invoke(*args):
        calls.append(args)
        raise AssertionError("must not invoke")

    with pytest.raises(ValueError, match="input budget"):
        run_rule_evidence(RuleEvidenceInput.model_validate(value), invoke)
    assert calls == []


def test_tuner_may_abstain_after_admitted_diagnosis(data, diagnostic):
    def invoke(_, __, schema):
        if schema is Diagnostic:
            return diagnostic
        return Proposal(
            decision="abstain",
            gap_index=None,
            hypothesis="",
            replacement="",
            uncertainties=[],
            abstention_reason="Receipt does not establish the root cause",
        )

    assert run_rule_evidence(data, invoke).decision == "abstain"
