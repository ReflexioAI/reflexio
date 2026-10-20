"""Contracts for grounded sparse repairs, without any storage or LLM access."""

import json

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.rule_evidence import (
    apply_proposal,
    run_rule_evidence,
    validate_diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.models import (
    Diagnostic,
    Proposal,
    RuleEvidenceInput,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    IndexedDiagnostic,
    exact_edit,
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


def wire_diagnostic(diagnostic):
    body = diagnostic.model_dump(mode="json")
    body["rule_assessment"].pop("incumbent_quote")
    body["rule_assessment"].update(clause_start="CLAUSE_2", clause_end="CLAUSE_2")
    return IndexedDiagnostic.model_validate_json(json.dumps(body))


def test_source_first_whole_rule_and_inherited_citations(data, diagnostic):
    calls = []
    revised = data.incumbent.content.replace(
        "Report queued as delivered.", repair().replacement
    )

    def invoke(system, payload, schema):
        calls.append((system, payload, schema))
        if schema is IndexedDiagnostic:
            return wire_diagnostic(diagnostic)
        assert payload["original_source"] == data.model_dump(mode="json")
        assert "untrusted_draft" not in payload
        return schema.model_validate_json(
            json.dumps(
                {
                    "corrections": [],
                    "edit_session_id": "s",
                    "hypothesis": repair().hypothesis,
                    "replacement": revised,
                    "abstention_reason": "",
                }
            )
        )

    result = run_rule_evidence(data, invoke)
    assert result.revised_content == revised
    assert result.evidence_refs == diagnostic.rule_assessment.evidence_refs
    assert len(calls) == 2
    assert calls[0][1]["messages"][0]["turn"] == 3
    assert "is_success" not in result.model_dump_json()


@pytest.mark.parametrize(
    "disposition", ["adherence_failure", "no_issue", "not_applicable", "unknown"]
)
def test_non_gap_is_audited_then_abstains(data, diagnostic, disposition):
    value = wire_diagnostic(diagnostic).model_dump(mode="json")
    value["rule_assessment"]["disposition"] = disposition
    value["missing_information"] = ["Need receipt"] if disposition == "unknown" else []
    calls = []

    def invoke(system, payload, schema):
        calls.append(schema)
        if schema is IndexedDiagnostic:
            return schema.model_validate_json(json.dumps(value))
        return schema.model_validate_json(
            json.dumps(
                {
                    "corrections": [],
                    "edit_session_id": None,
                    "hypothesis": "",
                    "replacement": "",
                    "abstention_reason": "No grounded defect",
                }
            )
        )

    result = run_rule_evidence(data, invoke)
    assert result.decision == "abstain" and result.revised_content is None
    assert len(calls) == 2


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
        apply_proposal(data, [diagnostic], repair())


@pytest.mark.parametrize(
    "field,value", [("gap_index", 1), ("replacement", "Report queued as delivered.")]
)
def test_tuner_cannot_select_unadmitted_gap_or_return_noop(
    data, diagnostic, field, value
):
    proposal = repair().model_copy(update={field: value})
    with pytest.raises(ValueError, match="admitted gap|change the quoted"):
        apply_proposal(data, [diagnostic], proposal)


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
    result = apply_proposal(
        data,
        [diagnostic],
        Proposal(
            decision="abstain",
            gap_index=None,
            hypothesis="",
            replacement="",
            uncertainties=[],
            abstention_reason="Receipt does not establish the root cause",
        ),
    )
    assert result.decision == "abstain"


@pytest.mark.parametrize(
    "original,revised",
    [
        ("prefix " * 200 + "OLD. end", "prefix " * 200 + "NEW. end"),
        ("x" * 1300 + "END", "x" * 1300 + "INSERTEND"),
        ("START" + "x" * 1300, "INSERTSTART" + "x" * 1300),
        ("x" * 1300 + "DELETEEND", "x" * 1300 + "END"),
        ("  DELETE" + "x" * 1300, "  " + "x" * 1300),
        ("x" * 1300 + " end", "x" * 1300 + "  end"),
        ("OLD repeated " * 100 + "OLD END", "OLD repeated " * 100 + "NEW END"),
        ("  rule  ", "  revised rule  "),
    ],
)
def test_full_generation_is_represented_by_exact_bounded_edit(original, revised):
    quote, replacement = exact_edit(original, revised)
    assert 0 < len(quote) <= 1200 and quote == quote.strip()
    assert original.find(quote) == original.rfind(quote)
    assert original.replace(quote, replacement, 1) == revised


@pytest.mark.parametrize(
    "original,revised", [("same", "same"), ("rule", " "), ("x" * 1300, "y" * 1300)]
)
def test_inadmissible_full_edit_is_explicit_failure(original, revised):
    with pytest.raises(ValueError):
        exact_edit(original, revised)


def test_session_enum_is_stable_across_source_order_and_prior_schema_cache(data):
    from reflexio.server.services.playbook.rule_evidence.wire import tune_schema

    second = data.sessions[0].model_copy(update={"session_id": "a"})
    forward = data.model_copy(update={"sessions": [data.sessions[0], second]})
    backward = data.model_copy(update={"sessions": [second, data.sessions[0]]})
    assert (
        tune_schema(forward, []).model_json_schema()
        == tune_schema(backward, []).model_json_schema()
    )
    assert (
        forward.sessions[0].session_id == "s" and backward.sessions[0].session_id == "a"
    )


def test_asymmetric_context_avoids_whitespace_alignment_rejection():
    original = "qOLD " + "a " * 700 + "OLD" + "b " * 700
    revised = (
        original[: original.rfind("OLD")]
        + "NEW"
        + original[original.rfind("OLD") + 3 :]
    )
    quote, replacement = exact_edit(original, revised)
    assert len(quote) <= 5
    assert original.find(quote) == original.rfind(quote)
    assert original.replace(quote, replacement, 1) == revised


@pytest.mark.parametrize("padding", ["before", "after"])
def test_whitespace_boundaries_do_not_force_an_invalid_left_anchor(padding):
    prefix = " " * 1300 if padding == "before" else ""
    suffix = "x" * 1300 if padding == "after" else ""
    original = prefix + " b  bb  b  ba a" + suffix
    revised = prefix + " b abb  b  ba a" + suffix
    quote, replacement = exact_edit(original, revised)
    assert len(quote) <= 12
    assert original.find(quote) == original.rfind(quote)
    assert original.replace(quote, replacement, 1) == revised
