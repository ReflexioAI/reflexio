"""V7 source-bound diagnostic and global-audit wire contracts.

Raw observations stay distinct from immutable resolved diagnostic/application views.
Dynamic input enums and deterministic validators use the existing correction turn.
"""

import json
import re
from enum import IntEnum
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, BeforeValidator, Field, create_model, model_validator

import reflexio.server.services.playbook.rule_evidence as core

from . import models


def catalog(content: str) -> list[dict[str, Any]]:
    """Lossless sentence/line chunks; adjacent IDs can select a larger exact span."""
    entries = []
    offset = 0
    for part in re.split(r"(?<=[.!?])(?=\s)|(?<=\n)", content):
        for start in range(0, len(part), 600):
            text = part[start : start + 600]
            entries.append(
                {"id": f"CLAUSE_{len(entries) + 1}", "start": offset, "text": text}
            )
            offset += len(text)
    return entries


def selected_quote(
    content: str, entries: list[dict[str, Any]], start: str | None, end: str | None
) -> str:
    if start is None and end is None:
        return ""
    ids = {item["id"]: i for i, item in enumerate(entries)}
    if start not in ids or end not in ids or ids[start] > ids[end]:
        raise ValueError("Select an available ordered clause range or two nulls")
    first, last = entries[ids[start]], entries[ids[end]]
    quote = content[first["start"] : last["start"] + len(last["text"])].strip()
    if not quote or len(quote) > 1200:
        raise ValueError("Selected quote must contain text and at most 1200 characters")
    return quote


ATTRIBUTIONS = {
    "instruction_conflicts_requirement": "rule_gap",
    "observed_instruction_failure": "rule_gap",
    "missing_required_step": "rule_gap",
    "adequate_guidance_ignored": "adherence_failure",
    "no_supported_target_defect": "no_issue",
    "trigger_not_established": "not_applicable",
    "deciding_evidence_missing": "unknown",
}


class SupportTurns(models.EvidenceModel):
    r1: int
    r2: int | None = None
    r3: int | None = None
    r4: int | None = None
    r5: int | None = None
    r6: int | None = None


class CounterTurns(models.EvidenceModel):
    r1: int | None = None
    r2: int | None = None
    r3: int | None = None
    r4: int | None = None


class SlotObservation(models.EvidenceModel):
    dimension: Literal[
        "task_completion",
        "constraint_compliance",
        "evidence_grounding",
        "execution_effectiveness",
    ]
    kind: Literal["strength", "problem", "uncertainty"]
    observation: models.Text
    consequence: models.Text
    evidence_refs: SupportTurns


class BoundedSlotAssessment(models.EvidenceModel):
    attribution: Literal[
        "instruction_conflicts_requirement",
        "observed_instruction_failure",
        "missing_required_step",
        "adequate_guidance_ignored",
        "no_supported_target_defect",
        "trigger_not_established",
        "deciding_evidence_missing",
    ]
    expected_if_followed: models.Text
    reason: models.Text
    evidence_refs: SupportTurns
    clause_start: str | None
    clause_end: str | None
    target_scope: models.Text
    counterevidence_refs: CounterTurns


class BoundedTargetDiagnostic(models.EvidenceModel):
    session_id: models.Text
    user_id: models.Text
    observations: list[SlotObservation] = Field(max_length=8)
    missing_information: list[models.Text] = Field(max_length=4)
    rule_assessment: BoundedSlotAssessment


class GlobalSourceSlots(models.EvidenceModel):
    r1: str
    r2: str | None = None
    r3: str | None = None
    r4: str | None = None
    r5: str | None = None
    r6: str | None = None


class GlobalAuditEdit(models.EvidenceModel):
    clause_start: str
    clause_end: str
    replacement: str
    defect: models.Text
    evidence_refs: GlobalSourceSlots


class GlobalRuleAudit(models.EvidenceModel):
    hypothesis: str
    edits: list[GlobalAuditEdit] = Field(max_length=4)
    abstention_reason: str


def references(source: models.Session, slots: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"session_id": source.session_id, "turn": t}
        for t in dict.fromkeys(v for v in slots.values() if v is not None)
    ]


def resolve_diagnostic(
    value: Any, data: models.RuleEvidenceInput, source: models.Session
) -> tuple[models.Diagnostic, dict[str, Any]]:
    body = value.model_dump(mode="json")
    for observation in body["observations"]:
        observation["evidence_refs"] = references(source, observation["evidence_refs"])
    assessment = body["rule_assessment"]
    assessment["evidence_refs"] = references(source, assessment["evidence_refs"])
    attribution = assessment.pop("attribution")
    audit = {
        "session_id": source.session_id,
        "attribution": attribution,
        "target_scope": assessment.pop("target_scope"),
        "counterevidence_refs": references(
            source, assessment.pop("counterevidence_refs")
        ),
    }
    assessment["disposition"] = ATTRIBUTIONS[attribution]
    assessment["incumbent_quote"] = selected_quote(
        data.incumbent.content,
        catalog(data.incumbent.content),
        assessment.pop("clause_start"),
        assessment.pop("clause_end"),
    )
    diagnostic = models.Diagnostic.model_validate_json(json.dumps(body))
    core.validate_diagnostic(diagnostic, data, source)
    coordinates = {(source.session_id, m.turn) for m in source.messages}
    if any(
        (r["session_id"], r["turn"]) not in coordinates
        for r in audit["counterevidence_refs"]
    ):
        raise ValueError("Counterevidence must cite the original session")
    return diagnostic, audit


def judge_schema(data: models.RuleEvidenceInput, source: models.Session) -> Any:
    turn: Any = IntEnum(
        "OriginalTurn", {"TURN_" + str(m.turn): m.turn for m in source.messages}
    )

    def original_turn(value: Any) -> Any:
        # Providers decode JSON before strict Python-mode validation. Convert
        # only exact integers; bools, floats and strings must remain invalid.
        if type(value) is int:
            return turn(value)
        if isinstance(value, turn):
            return value
        raise ValueError("Evidence references must be original integer turn IDs")

    turn_id: Any = Annotated[turn, BeforeValidator(original_turn)]
    fields: dict[str, Any] = {
        "r1": (turn_id, ...),
        **{f"r{i}": (turn_id | None, None) for i in range(2, 7)},
    }
    support = create_model("SupportTurns", __base__=models.EvidenceModel, **fields)
    contrary = create_model(
        "CounterTurns",
        __base__=models.EvidenceModel,
        **cast(dict[str, Any], {f"r{i}": (turn_id | None, None) for i in range(1, 5)}),
    )
    observation = create_model(
        "SlotObservation",
        __base__=models.EvidenceModel,
        **cast(
            dict[str, Any],
            {
                k: (f.annotation, f)
                for k, f in models.Observation.model_fields.items()
                if k != "evidence_refs"
            },
        ),
        evidence_refs=(support, ...),
    )
    entries = catalog(data.incumbent.content)
    clause: Any = cast(Any, Literal)[tuple(e["id"] for e in entries)]
    assessment = create_model(
        "BoundedSlotAssessment",
        __base__=BoundedSlotAssessment,
        evidence_refs=(support, ...),
        counterevidence_refs=(contrary, ...),
        clause_start=(clause | None, ...),
        clause_end=(clause | None, ...),
    )

    def validate(value: Any) -> Any:
        resolve_diagnostic(value, data, source)
        return value

    return create_model(
        "BoundedTargetDiagnostic",
        __base__=BoundedTargetDiagnostic,
        observations=(list[observation], Field(max_length=8)),
        rule_assessment=(assessment, ...),
        __validators__=cast(
            dict[str, Any], {"validate_source": model_validator(mode="after")(validate)}
        ),
    )


def diagnose(
    data: models.RuleEvidenceInput, invoke: core.Invoke
) -> tuple[list[models.Diagnostic], list[dict[str, Any]]]:
    entries = catalog(data.incumbent.content)
    diagnostics, audits = [], []
    for source in data.sessions:
        incumbent = data.incumbent.model_dump(mode="json")
        incumbent.pop("content")
        incumbent["content_clauses"] = [
            {"id": e["id"], "text": e["text"]} for e in entries
        ]
        payload = {**core._source(data, source), "incumbent": incumbent}
        schema = judge_schema(data, source)
        system = core.prompt("judge")
        core._check_budget(system, payload, schema)
        value = invoke(system, payload, schema)
        diagnostic, audit = resolve_diagnostic(
            schema.model_validate_json(value.model_dump_json()), data, source
        )
        diagnostics.append(diagnostic)
        audits.append(audit)
    return diagnostics, audits


def reference_catalog(data: models.RuleEvidenceInput) -> dict[str, dict[str, Any]]:
    return {
        f"REF_{i}": {"session_id": s.session_id, "turn": m.turn}
        for i, (s, m) in enumerate(
            ((s, m) for s in data.sessions for m in s.messages), 1
        )
    }


def resolve_audit(
    data: models.RuleEvidenceInput, value: Any
) -> tuple[models.SourceLinkedAudit, str]:
    body = value.model_dump(mode="json")
    refs = reference_catalog(data)
    entries = catalog(data.incumbent.content)
    by_id = {e["id"]: (i, e) for i, e in enumerate(entries)}
    spans = []
    for edit in body["edits"]:
        selected = list(
            dict.fromkeys(v for v in edit["evidence_refs"].values() if v is not None)
        )
        if not selected or any(k not in refs for k in selected):
            raise ValueError("Every edit requires supplied source references")
        edit["evidence_refs"] = [refs[k] for k in selected]
        selected_quote(
            data.incumbent.content, entries, edit["clause_start"], edit["clause_end"]
        )
        _, first = by_id[edit["clause_start"]]
        _, last = by_id[edit["clause_end"]]
        start = first["start"]
        end = last["start"] + len(last["text"])
        original = data.incumbent.content[start:end]
        leading = original[: len(original) - len(original.lstrip())]
        trailing = original[len(original.rstrip()) :]
        edit["replacement"] = leading + edit["replacement"].strip() + trailing
        if original == edit["replacement"]:
            raise ValueError("Every edit must change its original span")
        spans.append((start, end, edit["replacement"]))
    audit = models.SourceLinkedAudit.model_validate(body)
    if audit.edits:
        if not audit.hypothesis.strip() or audit.abstention_reason.strip():
            raise ValueError("Proposal requires hypothesis and no abstention reason")
    elif audit.hypothesis.strip() or not audit.abstention_reason.strip():
        raise ValueError("Abstention requires a reason and no hypothesis")
    pieces = []
    position = 0
    for start, end, replacement in sorted(spans):
        if start < position:
            raise ValueError("Edits must not overlap or repeat original spans")
        pieces.extend([data.incumbent.content[position:start], replacement])
        position = end
    pieces.append(data.incumbent.content[position:])
    revised = "".join(pieces)
    if audit.edits and not revised.strip():
        raise ValueError("Complete repair must retain nonblank guidance")
    return audit, revised


def tune_schema(data: models.RuleEvidenceInput) -> Any:
    clause: Any = cast(Any, Literal)[
        tuple(e["id"] for e in catalog(data.incumbent.content))
    ]
    reference: Any = cast(Any, Literal)[tuple(reference_catalog(data))]
    slots = create_model(
        "GlobalSourceSlots",
        __base__=models.EvidenceModel,
        r1=(reference, ...),
        **cast(
            dict[str, Any], {f"r{i}": (reference | None, None) for i in range(2, 7)}
        ),
    )
    edit = create_model(
        "GlobalAuditEdit",
        __base__=models.EvidenceModel,
        clause_start=(clause, ...),
        clause_end=(clause, ...),
        replacement=(str, ...),
        defect=(models.Text, ...),
        evidence_refs=(slots, ...),
    )

    def validate(value: Any) -> Any:
        resolve_audit(data, value)
        return value

    return create_model(
        "GlobalRuleAudit",
        __base__=GlobalRuleAudit,
        edits=(list[edit], Field(max_length=4)),
        __validators__=cast(
            dict[str, Any], {"validate_audit": model_validator(mode="after")(validate)}
        ),
    )


def rebuild(
    data: models.RuleEvidenceInput, diagnostics: list[models.Diagnostic], value: Any
) -> models.RuleEvidenceResult:
    audit, revised = resolve_audit(data, value)
    refs = {(r.session_id, r.turn): r for e in audit.edits for r in e.evidence_refs}
    return models.RuleEvidenceResult(
        decision="proposal" if audit.edits else "abstain",
        reason="Source-grounded global rule hypothesis; qualification required"
        if audit.edits
        else audit.abstention_reason,
        diagnostics=diagnostics,
        rule_audit=audit,
        revised_content=revised if audit.edits else None,
        evidence_refs=list(refs.values()),
    )


def output_kind(schema: type[BaseModel]) -> Literal["judge", "tune"]:
    if issubclass(schema, BoundedTargetDiagnostic):
        return "judge"
    if issubclass(schema, GlobalRuleAudit):
        return "tune"
    raise TypeError("unsupported rule-evidence wire schema")


def contract_schema(stage: Literal["judge", "tune"]) -> type[BaseModel]:
    return BoundedTargetDiagnostic if stage == "judge" else GlobalRuleAudit


def generation_profile(
    stage: Literal["judge", "tune"], *, provider: str
) -> dict[str, Any]:
    if stage not in {"judge", "tune"}:
        raise ValueError("unsupported rule-evidence stage")
    profile: dict[str, Any] = {"max_tokens": 4096}
    if "minimax" in provider.casefold().split(":"):
        profile["extra_body"] = {"thinking": {"type": "disabled"}}
    return profile
