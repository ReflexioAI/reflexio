"""Promoted source-first judge/tuner wire contracts and deterministic exact application.

Input-specific tuner enums are code-owned. Raw wire outputs remain distinct from
resolved/corrected diagnostics and the final exact-span Proposal.
"""

import json
import re
from copy import deepcopy
from types import GenericAlias
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, create_model

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


class IndexedRuleAssessment(models.EvidenceModel):
    disposition: Literal[
        "rule_gap", "adherence_failure", "no_issue", "not_applicable", "unknown"
    ]
    expected_if_followed: models.Text
    reason: models.Text
    evidence_refs: list[models.Reference] = Field(min_length=1, max_length=6)
    clause_start: str | None
    clause_end: str | None


class IndexedDiagnostic(models.EvidenceModel):
    session_id: models.Text
    user_id: models.Text
    observations: list[models.Observation] = Field(max_length=8)
    missing_information: list[models.Text] = Field(max_length=4)
    rule_assessment: IndexedRuleAssessment


def selected_schema() -> type[IndexedDiagnostic]:
    return IndexedDiagnostic


def resolve(
    value: BaseModel, content: str, entries: list[dict[str, Any]]
) -> models.Diagnostic:
    body = value.model_dump(mode="json")
    assessment = body["rule_assessment"]
    assessment["incumbent_quote"] = selected_quote(
        content, entries, assessment.pop("clause_start"), assessment.pop("clause_end")
    )
    return models.Diagnostic.model_validate_json(json.dumps(body))


def field_catalog(
    diagnostics: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], str]]:
    """Only these existing scalar fields may change; IDs are code-owned."""
    result = {}
    for si, diagnostic in enumerate(diagnostics, 1):
        for oi, observation in enumerate(diagnostic["observations"], 1):
            for field in ("observation", "consequence"):
                result[f"S{si}.O{oi}.{field}"] = (observation, field)
        for field in ("reason", "expected_if_followed", "disposition"):
            result[f"S{si}.A.{field}"] = (diagnostic["rule_assessment"], field)
        # Missing facts are retained; this contract cannot silently drop them.
    return result


def exact_edit(original: str, revised: str) -> tuple[str, str]:
    """Keep full short-rule edits; bound long-rule edits without changing their result.

    Common prefix/suffix removal locates all changes. Expand context until the
    quote is unique, nonblank and trim-stable and the replacement is nonblank.
    Never truncate an edit or silently apply only some of the generated changes.
    """
    if original == revised:
        raise ValueError("Whole-rule repair must change the incumbent")
    if not revised.strip():
        raise ValueError("Whole-rule repair must retain nonblank guidance")
    if len(original) <= 1200 and original == original.strip():
        return original, revised
    start = 0
    while (
        start < min(len(original), len(revised)) and original[start] == revised[start]
    ):
        start += 1
    suffix = 0
    while (
        suffix < min(len(original), len(revised)) - start
        and original[-suffix - 1] == revised[-suffix - 1]
    ):
        suffix += 1
    end = len(original) - suffix
    new_end = len(revised) - suffix
    best: tuple[str, str] | None = None
    # Enumerate bounded left anchors. Uniqueness is monotone as the right
    # boundary grows, so binary search avoids enumerating every possible span.
    for left in range(start, max(-1, end - 1201), -1):
        if best is not None and end - left >= len(best[0]):
            break
        if left >= len(original) or original[left].isspace():
            continue
        maximum = min(
            len(original), left + (1200 if best is None else len(best[0]) - 1)
        )
        while maximum > left and original[maximum - 1].isspace():
            maximum -= 1
        low = max(end, left + 1)
        if maximum < low:
            continue
        quote = original[left:maximum]
        if original.find(quote) != original.rfind(quote):
            continue
        high = maximum
        while low < high:
            middle = (low + high) // 2
            quote = original[left:middle]
            if original.find(quote) == original.rfind(quote):
                high = middle
            else:
                low = middle + 1
        for right in range(low, maximum + 1):
            if original[right - 1].isspace():
                continue
            replacement = (
                original[left:start] + revised[start:new_end] + original[end:right]
            )
            if replacement.strip():
                best = original[left:right], replacement
                break
    if best is None:
        raise ValueError(
            "Complete generated rule cannot be represented by a unique1200-character edit"
        )
    if original.replace(best[0], best[1], 1) != revised:
        raise ValueError("Derived edit does not reproduce the complete generated rule")
    return best


def rebuild(
    data: models.RuleEvidenceInput, diagnostics: list[models.Diagnostic], value: Any
) -> models.RuleEvidenceResult:
    """Correct bounded evidence, derive one exact edit and rerun source validation."""
    value = tune_schema(data, diagnostics).model_validate_json(value.model_dump_json())
    bodies = deepcopy([item.model_dump(mode="json") for item in diagnostics])
    if [item["session_id"] for item in bodies] != [
        session.session_id for session in data.sessions
    ]:
        raise ValueError("Diagnostics must cover source sessions once, in order")
    fields = field_catalog(bodies)
    seen = set()
    citation_sets = {}
    for correction in value.corrections:
        if correction.target_id not in fields or correction.target_id in seen:
            raise ValueError("Correction must select a unique supplied field ID")
        seen.add(correction.target_id)
        target, field = fields[correction.target_id]
        target[field] = correction.text
        # Replace the item's complete set: merging can retain obsolete references
        # or overflow the native six-reference bound.
        refs = [r.model_dump(mode="json") for r in correction.evidence_refs]
        refs = [r for i, r in enumerate(refs) if r not in refs[:i]]
        canonical = frozenset((r["session_id"], r["turn"]) for r in refs)
        if id(target) in citation_sets and citation_sets[id(target)] != canonical:
            raise ValueError(
                "Corrections to the same item must agree on its complete citations"
            )
        citation_sets[id(target)] = canonical
        target["evidence_refs"] = refs
    admitted = [d for d in bodies if d["rule_assessment"]["disposition"] == "rule_gap"]
    if value.edit_session_id is None:
        if value.replacement:
            raise ValueError("Abstention cannot carry an edit")
        proposal = models.Proposal(
            decision="abstain",
            gap_index=None,
            hypothesis="",
            replacement="",
            uncertainties=[],
            abstention_reason=value.abstention_reason,
        )
    else:
        matches = [
            i
            for i, d in enumerate(admitted)
            if d["session_id"] == value.edit_session_id
        ]
        if len(matches) != 1:
            raise ValueError("Edit requires one admitted source session gap")
        quote, replacement = exact_edit(data.incumbent.content, value.replacement)
        admitted[matches[0]]["rule_assessment"]["incumbent_quote"] = quote
        proposal = models.Proposal(
            decision="hypothesis",
            gap_index=matches[0],
            hypothesis=value.hypothesis,
            replacement=replacement,
            uncertainties=[],
            abstention_reason="",
        )
    resolved = [
        models.Diagnostic.model_validate_json(json.dumps(item)) for item in bodies
    ]
    return core.apply_proposal(data, resolved, proposal)


def diagnose(
    data: models.RuleEvidenceInput, invoke: core.Invoke
) -> list[models.Diagnostic]:
    entries = catalog(data.incumbent.content)
    diagnostics = []
    for session in data.sessions:
        payload = {
            **core._source(data, session),
            "incumbent": data.incumbent.model_dump(mode="json"),
        }
        payload["incumbent"].pop("content")
        payload["incumbent"]["content_clauses"] = [
            {"id": item["id"], "text": item["text"]} for item in entries
        ]
        system = core.prompt("judge")
        output_schema = selected_schema()
        core._check_budget(system, payload, output_schema)
        value = output_schema.model_validate_json(
            invoke(system, payload, output_schema).model_dump_json()
        )
        diagnostic = resolve(value, data.incumbent.content, entries)
        core.validate_diagnostic(diagnostic, data, session)
        diagnostics.append(diagnostic)
    return diagnostics


class FieldCorrection(models.EvidenceModel):
    target_id: str
    text: str = Field(min_length=1, pattern=r"\S")
    evidence_refs: list[models.Reference] = Field(min_length=1, max_length=6)


class WholeRuleTune(models.EvidenceModel):
    """Raw tuner output; input-specific enum constraints bind each invocation."""

    corrections: list[FieldCorrection] = Field(max_length=3)
    edit_session_id: str | None
    hypothesis: str
    replacement: str
    abstention_reason: str


def output_kind(schema: type[BaseModel]) -> Literal["judge", "tune"]:
    if schema is selected_schema():
        return "judge"
    if issubclass(schema, WholeRuleTune):
        return "tune"
    raise TypeError("unsupported rule-evidence wire schema")


def contract_schema(stage: Literal["judge", "tune"]) -> type[BaseModel]:
    """Stable schema template; actual enumerations are bound per invocation."""
    return selected_schema() if stage == "judge" else tune_schema_template()


def tune_schema(
    data: models.RuleEvidenceInput, diagnostics: list[models.Diagnostic]
) -> type[WholeRuleTune]:
    return _tune_schema(
        tuple(field_catalog([item.model_dump(mode="json") for item in diagnostics])),
        tuple(session.session_id for session in data.sessions),
    )


def _tune_schema(
    target_ids: tuple[str, ...], session_ids: tuple[str, ...]
) -> type[WholeRuleTune]:
    def choice(values: tuple[str, ...]) -> Any:
        return cast(Any, Literal)[values] if values else str

    correction = create_model(
        "FieldCorrection",
        __base__=models.EvidenceModel,
        target_id=(choice(target_ids), ...),
        text=(str, Field(min_length=1, pattern=r"\S")),
        evidence_refs=(
            GenericAlias(list, models.Reference),
            Field(min_length=1, max_length=6),
        ),
    )
    fields: dict[str, Any] = {
        "corrections": (GenericAlias(list, correction), Field(max_length=3)),
        "edit_session_id": (choice(tuple(sorted(session_ids))) | None, ...),
        "hypothesis": (str, ...),
        "replacement": (str, ...),
        "abstention_reason": (str, ...),
    }
    return create_model("WholeRuleTune", __base__=WholeRuleTune, **fields)


def tune_schema_template() -> type[WholeRuleTune]:
    return _tune_schema((), ())


def generation_profile(
    stage: Literal["judge", "tune"], *, provider: str
) -> dict[str, Any]:
    """Measured MiniMax stage settings; keep other configured providers usable."""
    profile: dict[str, Any] = {"max_tokens": 12288 if stage == "judge" else 4096}
    if "minimax" in provider.casefold().split(":"):
        profile["extra_body"] = {
            "thinking": {"type": "enabled" if stage == "judge" else "disabled"}
        }
    return profile
