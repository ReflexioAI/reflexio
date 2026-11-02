"""Contracts for grounded sparse repairs, without any storage or LLM access."""

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.playbook.rule_evidence import (
    run_rule_evidence,
    validate_diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.models import (
    Diagnostic,
    RuleEvidenceInput,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic,
    catalog,
    generation_profile,
    judge_schema,
    tune_schema,
)


@pytest.mark.unit
def test_provider_parser_accepts_original_integer_turn_ids(data, diagnostic):
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    schema = judge_schema(data, data.sessions[0])
    client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))
    parsed = client._maybe_parse_structured_output(
        json.dumps(wire_diagnostic(diagnostic)), schema, True
    )
    assert isinstance(parsed, StrictStructuredOutput)
    assert parsed.model_dump(mode="json")["observations"][0]["evidence_refs"]["r1"] == 8
    assert schema.model_json_schema()["$defs"]["OriginalTurn"]["enum"] == [3, 8]
    assert schema.model_json_schema()["$defs"]["SupportTurns"]["properties"]["r1"] == {
        "$ref": "#/$defs/OriginalTurn"
    }


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, False, 1.0, "1", 2, "TURN_1"])
@pytest.mark.parametrize("mode", ["python", "json"])
@pytest.mark.parametrize("counterevidence", [False, True])
def test_provider_turn_slots_reject_coercion_and_unknown_ids(
    data, diagnostic, bad, mode, counterevidence
):
    source = data.sessions[0].model_copy(deep=True)
    source.messages[0] = source.messages[0].model_copy(update={"turn": 1})
    bound_data = data.model_copy(update={"sessions": [source]})
    schema = judge_schema(bound_data, source)
    body = wire_diagnostic(diagnostic)
    body["rule_assessment"]["evidence_refs"]["r1"] = 1
    body["observations"][0]["evidence_refs"]["r1"] = 1
    schema.model_validate(body)
    slots = (
        body["rule_assessment"]["counterevidence_refs"]
        if counterevidence
        else body["observations"][0]["evidence_refs"]
    )
    slots["r1"] = bad
    with pytest.raises(ValidationError):
        if mode == "json":
            schema.model_validate_json(json.dumps(body))
        else:
            schema.model_validate(body)


@pytest.mark.unit
@pytest.mark.parametrize("repair_first_response", [False, True])
def test_provider_boundary_runs_judge_and_tuner(
    data, diagnostic, monkeypatch, repair_first_response
):
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    replies = []
    if repair_first_response:
        invalid = wire_diagnostic(diagnostic)
        invalid["observations"][0]["evidence_refs"]["r1"] = 999
        replies.append(invalid)
    replies.extend([wire_diagnostic(diagnostic), repair()])
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        response = MagicMock()
        response.choices[0].message.content = json.dumps(replies.pop(0))
        response.choices[0].message.refusal = None
        response.choices[0].message.tool_calls = None
        response.choices[0].finish_reason = "stop"
        response.model = "gpt-4o-2024-08-06"
        response._hidden_params = {"custom_llm_provider": "openai"}
        response.usage = None
        return response

    monkeypatch.setattr("litellm.completion", completion)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-2024-08-06"))

    def invoke(system, payload, schema):
        parsed = client.generate_chat_response(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format=schema,
        )
        assert isinstance(parsed, StrictStructuredOutput)
        return parsed

    result = run_rule_evidence(data, invoke)
    assert result.decision == "proposal"
    assert result.revised_content == (
        "Send once. Report queued as queued. Keep the recipient."
    )
    assert result.diagnostics == [diagnostic]
    assert len(calls) == 2 + int(repair_first_response)
    assert not replies


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


def wire_diagnostic(diagnostic, *, attribution="instruction_conflicts_requirement"):
    body = diagnostic.model_dump(mode="json")
    for observation in body["observations"]:
        observation["evidence_refs"] = {
            f"r{i}": r["turn"] for i, r in enumerate(observation["evidence_refs"], 1)
        }
    assessment = body["rule_assessment"]
    assessment.pop("incumbent_quote")
    assessment.pop("disposition")
    assessment["evidence_refs"] = {
        f"r{i}": r["turn"] for i, r in enumerate(assessment["evidence_refs"], 1)
    }
    assessment.update(
        attribution=attribution,
        target_scope="Delivery-state reporting",
        counterevidence_refs={},
        clause_start="CLAUSE_2",
        clause_end="CLAUSE_2",
    )
    return body


def repair(**updates):
    body = {
        "hypothesis": "Report actual status",
        "edits": [
            {
                "clause_start": "CLAUSE_2",
                "clause_end": "CLAUSE_2",
                "replacement": "Report queued as queued.",
                "defect": "Queued receipt does not establish delivery",
                "evidence_refs": {"r1": "REF_1", "r2": "REF_2"},
            }
        ],
        "abstention_reason": "",
    }
    body.update(updates)
    return body


def execute(
    data, diagnostic, audit=None, *, attribution="instruction_conflicts_requirement"
):
    calls = []

    def invoke(system, payload, schema):
        calls.append((system, payload, schema))
        body = (
            wire_diagnostic(diagnostic, attribution=attribution)
            if issubclass(schema, BoundedTargetDiagnostic)
            else (audit if audit is not None else repair())
        )
        if issubclass(schema, BoundedTargetDiagnostic):
            target = next(
                e
                for e in payload["incumbent"]["content_clauses"]
                if diagnostic.rule_assessment.incumbent_quote in e["text"]
            )
            body["rule_assessment"].update(
                clause_start=target["id"], clause_end=target["id"]
            )
        return schema.model_validate_json(json.dumps(body))

    return run_rule_evidence(data, invoke), calls


def test_global_audit_source_first_and_immutable_diagnostics(data, diagnostic):
    result, calls = execute(data, diagnostic)
    assert (
        result.revised_content
        == "Send once. Report queued as queued. Keep the recipient."
    )
    assert result.diagnostics == [diagnostic]
    assert result.evidence_refs == diagnostic.rule_assessment.evidence_refs
    assert len(calls) == 2
    assert calls[1][1]["original_source"] == data.model_dump(mode="json")
    assert "correctable_fields" not in calls[1][1]
    assert not hasattr(result, "proposal")
    assert result.rule_audit.edits[0].defect


@pytest.mark.parametrize(
    "attribution",
    [
        "adequate_guidance_ignored",
        "no_supported_target_defect",
        "trigger_not_established",
    ],
)
def test_global_repair_does_not_require_local_gap(data, diagnostic, attribution):
    result, calls = execute(data, diagnostic, attribution=attribution)
    assert result.decision == "proposal" and len(calls) == 2
    assert result.diagnostics[0].rule_assessment.disposition != "rule_gap"


def test_healthy_abstention_keeps_diagnostics(data, diagnostic):
    result, calls = execute(
        data,
        diagnostic,
        repair(hypothesis="", edits=[], abstention_reason="No supported global defect"),
        attribution="no_supported_target_defect",
    )
    assert result.decision == "abstain" and result.revised_content is None
    assert result.evidence_refs == [] and len(calls) == 2
    assert result.diagnostics[0].observations == diagnostic.observations


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


@pytest.mark.parametrize(
    "bad", ["foreign", "seventh", "overlap", "reversed", "noop", "blank", "overlong"]
)
def test_global_edit_contract_rejects_invalid_atomic_set(data, bad):
    value = repair()
    if bad == "foreign":
        value["edits"][0]["evidence_refs"]["r1"] = "FOREIGN"
    elif bad == "seventh":
        value["edits"][0]["evidence_refs"]["r7"] = "REF_1"
    elif bad == "overlap":
        value["edits"] *= 2
    elif bad == "reversed":
        value["edits"][0].update(clause_start="CLAUSE_3", clause_end="CLAUSE_1")
    elif bad == "noop":
        value["edits"][0]["replacement"] = "Report queued as delivered."
    elif bad == "blank":
        value["edits"][0].update(
            clause_start="CLAUSE_1", clause_end="CLAUSE_3", replacement=""
        )
    else:
        data = data.model_copy(
            update={
                "incumbent": data.incumbent.model_copy(update={"content": "A" * 1300})
            }
        )
        value["edits"][0].update(clause_start="CLAUSE_1", clause_end="CLAUSE_3")
    with pytest.raises(ValueError):
        tune_schema(data).model_validate_json(json.dumps(value))


@pytest.mark.parametrize("padding", [" ", "\n", "\n\n", "\t"])
def test_boundary_whitespace_is_code_preserved(data, diagnostic, padding):
    original = (
        "Send once."
        + padding
        + "Report queued as delivered."
        + padding
        + "Keep recipient."
    )
    data = data.model_copy(
        update={"incumbent": data.incumbent.model_copy(update={"content": original})}
    )
    target = next(
        e for e in catalog(original) if "Report queued as delivered." in e["text"]
    )
    value = repair()
    value["edits"][0].update(clause_start=target["id"], clause_end=target["id"])
    result, _ = execute(data, diagnostic, value)
    assert result.revised_content == original.replace(
        "Report queued as delivered.", "Report queued as queued."
    )


def test_empty_problem_cannot_manufacture_local_gap(data, diagnostic):
    diagnostic = diagnostic.model_copy(update={"observations": []})
    with pytest.raises(ValueError, match="observed problem"):
        execute(data, diagnostic)


def test_generation_uses_measured_compact_profile():
    for stage in ("judge", "tune"):
        assert generation_profile(stage, provider="minimax") == {
            "max_tokens": 4096,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        assert generation_profile(stage, provider="openai") == {"max_tokens": 4096}


@pytest.mark.unit
def test_admission_builders_match_dispatched_consumer_envelopes(data, diagnostic):
    from reflexio.server.services.playbook.rule_evidence import (
        proposal_request,
        request_input_tokens,
    )
    from reflexio.server.services.playbook.rule_evidence.wire import diagnostic_request

    calls = []

    def invoke(system, payload, schema):
        calls.append((system, payload, schema))
        return schema.model_validate_json(
            json.dumps(wire_diagnostic(diagnostic) if len(calls) == 1 else repair())
        )

    result = run_rule_evidence(data, invoke)
    assert result.decision == "proposal"
    judge = diagnostic_request(data, data.sessions[0])
    # The resolved global audit is an output, not the per-session target audit.
    # Compare that input from the actual invocation without synthesizing one.
    tune = proposal_request(data, result.diagnostics, calls[1][1]["target_audits"])
    for actual, rendered in zip(calls, (judge, tune), strict=True):
        assert actual[:2] == rendered[:2]
        assert actual[2].model_json_schema() == rendered[2].model_json_schema()
        assert request_input_tokens(*rendered) > 0


def test_complete_proposal_allows_bounded_generated_context_above_source_ceiling(
    data, diagnostic
):
    from reflexio.server.services.playbook.rule_evidence import (
        proposal_request,
        request_input_tokens,
    )
    from reflexio.server.services.playbook.rule_evidence.wire import diagnostic_request

    value = data.model_dump(mode="json")
    value["sessions"][0]["messages"][0]["content"] += " trace" * 55_000
    data = RuleEvidenceInput.model_validate(value)
    # A valid generated diagnosis adds context to an already large admitted source.
    diagnostic = diagnostic.model_copy(
        update={
            "rule_assessment": diagnostic.rule_assessment.model_copy(
                update={
                    "reason": "Literal compliance reports an unconfirmed delivery. "
                    * 800
                }
            )
        }
    )
    fixed_tokens = request_input_tokens(*proposal_request(data, [], []))
    assert fixed_tokens < 60_000
    assert request_input_tokens(*diagnostic_request(data, data.sessions[0])) < 60_000

    result, calls = execute(data, diagnostic)
    complete_tokens = request_input_tokens(*calls[-1])
    assert 60_000 < complete_tokens < 120_000
    assert complete_tokens - fixed_tokens < 60_000
    assert len(calls) == 2
    assert result.decision == "proposal"
    assert (
        result.revised_content
        == "Send once. Report queued as queued. Keep the recipient."
    )
    assert calls[-1][1]["original_source"] == data.model_dump(mode="json")
    assert result.diagnostics == [diagnostic]


def test_combined_fixed_source_over_budget_rejects_before_first_diagnosis(data):
    from reflexio.server.services.playbook.rule_evidence import (
        proposal_request,
        request_input_tokens,
    )
    from reflexio.server.services.playbook.rule_evidence.wire import diagnostic_request

    value = data.model_dump(mode="json")
    first = value["sessions"][0]
    first["messages"][0]["content"] += " trace" * 31_000
    second = {**first, "session_id": "second"}
    value["sessions"] = [first, second]
    data = RuleEvidenceInput.model_validate(value)
    assert all(
        request_input_tokens(*diagnostic_request(data, session)) < 60_000
        for session in data.sessions
    )
    assert request_input_tokens(*proposal_request(data, [], [])) > 60_000
    invoke = MagicMock(
        side_effect=AssertionError("must not diagnose inadmissible source")
    )
    with pytest.raises(ValueError, match="input budget"):
        run_rule_evidence(data, invoke)
    invoke.assert_not_called()


def test_generated_context_over_budget_fails_before_tuner(data, diagnostic):
    from reflexio.server.services.playbook.rule_evidence import (
        proposal_request,
        request_input_tokens,
    )
    from reflexio.server.services.playbook.rule_evidence.wire import resolve_diagnostic

    body = wire_diagnostic(diagnostic)
    body["rule_assessment"]["reason"] = "Supported finding. " * 25_000
    schema = judge_schema(data, data.sessions[0])
    output = schema.model_validate_json(json.dumps(body))
    resolved, audit = resolve_diagnostic(output, data, data.sessions[0])
    fixed_tokens = request_input_tokens(*proposal_request(data, [], []))
    complete_tokens = request_input_tokens(*proposal_request(data, [resolved], [audit]))
    assert fixed_tokens < 60_000
    assert complete_tokens - fixed_tokens > 60_000
    calls = []

    def invoke(system, payload, response_schema):
        calls.append((system, payload, response_schema))
        assert issubclass(response_schema, BoundedTargetDiagnostic), (
            "must not invoke tuner"
        )
        return output

    with pytest.raises(ValueError, match="generated context input budget"):
        run_rule_evidence(data, invoke)
    assert len(calls) == 1


def test_later_diagnostic_over_budget_rejects_before_any_model_call(data):
    from reflexio.server.services.playbook.rule_evidence import (
        proposal_request,
        request_input_tokens,
    )
    from reflexio.server.services.playbook.rule_evidence.wire import diagnostic_request

    value = data.model_dump(mode="json")
    second = json.loads(json.dumps(value["sessions"][0]))
    second["session_id"] = "second"
    second["messages"][0]["content"] += " trace" * 58_000
    value["sessions"].append(second)
    data = RuleEvidenceInput.model_validate(value)
    assert request_input_tokens(*proposal_request(data, [], [])) < 60_000
    assert request_input_tokens(*diagnostic_request(data, data.sessions[0])) < 60_000
    assert request_input_tokens(*diagnostic_request(data, data.sessions[1])) > 60_000
    invoke = MagicMock(
        side_effect=AssertionError("preflight all sessions before spending")
    )
    with pytest.raises(ValueError, match="input budget"):
        run_rule_evidence(data, invoke)
    invoke.assert_not_called()


@pytest.mark.parametrize("mode", ["python", "json", "native_parser"])
@pytest.mark.parametrize(
    "padding", [{"r5": None}, {"r6": None}, {"r5": None, "r6": None}]
)
@pytest.mark.parametrize("existing_refs", [{}, {"r1": 3}])
def test_counter_reference_null_padding_preserves_exact_diagnosis(
    data, diagnostic, mode, padding, existing_refs
):
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.services.playbook.rule_evidence.wire import resolve_diagnostic

    schema = judge_schema(data, data.sessions[0])
    original = wire_diagnostic(diagnostic)
    original["rule_assessment"]["counterevidence_refs"] = existing_refs
    expected = resolve_diagnostic(
        schema.model_validate_json(json.dumps(original)), data, data.sessions[0]
    )
    padded = json.loads(json.dumps(original))
    padded["rule_assessment"]["counterevidence_refs"].update(padding)
    snapshot = json.loads(json.dumps(padded))
    if mode == "python":
        parsed = schema.model_validate(padded)
    elif mode == "json":
        parsed = schema.model_validate_json(json.dumps(padded))
    else:
        client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))
        parsed = client._maybe_parse_structured_output(json.dumps(padded), schema, True)
    resolved, audit = resolve_diagnostic(parsed, data, data.sessions[0])
    assert resolved == diagnostic == expected[0]
    assert audit == expected[1]
    assert audit["counterevidence_refs"] == (
        [{"session_id": "s", "turn": 3}] if existing_refs else []
    )
    assert padded == snapshot  # Normalization must not mutate retained raw evidence.
    counter_schema = schema.model_json_schema()["$defs"]["CounterTurns"]
    assert set(counter_schema["properties"]) == {"r1", "r2", "r3", "r4"}
    assert counter_schema["additionalProperties"] is False


@pytest.mark.parametrize("mode", ["python", "json", "native_parser"])
@pytest.mark.parametrize(
    "extra",
    [
        {"r5": 3},
        {"r6": 8},
        {"r5": False},
        {"r6": "3"},
        {"r5": None, "r6": 3},
        {"r7": None},
        {"unrecognized": None},
    ],
)
def test_counter_reference_padding_does_not_admit_new_evidence_or_unknown_fields(
    data, diagnostic, mode, extra
):
    from reflexio.server.llm._litellm_types import StructuredOutputParseError
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    schema = judge_schema(data, data.sessions[0])
    body = wire_diagnostic(diagnostic)
    body["rule_assessment"]["counterevidence_refs"] = {"r1": 3, **extra}
    if mode == "native_parser":
        client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))
        with pytest.raises(StructuredOutputParseError):
            client._maybe_parse_structured_output(json.dumps(body), schema, True)
    else:
        with pytest.raises(ValidationError):
            if mode == "python":
                schema.model_validate(body)
            else:
                schema.model_validate_json(json.dumps(body))
