"""The rule-evidence artifact checks, as pure functions over decoded JSON.

SQLite has no stored functions, so the checks the PostgreSQL implementation of
the same port runs inside the database before admitting a discovery memo,
candidate or decision are restated here, one function per SQL function, with
the same refusal codes and messages. They read only what the caller hands
them; the storage mixin (``_open_world_attempt``) reads the rows.

Two copies of one check can drift, so neither is trusted alone: the shared
port contract (``reflexio.test_support.open_world_attempt_contract``) runs
every accepted and every forged artifact through both backends.

Values follow the JSON model: ``dict``, ``list``, ``str``, ``int`` (never
``bool``), ``bool`` and ``None``. A key that is absent and a key holding
``null`` are told apart only where the SQL tells them apart.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any, NoReturn

from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.open_world_invocations import (
    OpenWorldRefusal,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    INVOCATION_ERROR_BY_REFUSAL,
)
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_trajectory_bytes,
)

#: The characters the SQL trim and clause split treat as whitespace: Python's
#: ``str.isspace`` set, spelled out as the SQL spells it.
WHITESPACE = " \t\n\r\f\x0b\x1c\x1d\x1e\x1f\x85\xa0                　"
_CLAUSE_SPLIT = re.compile("(?<=[.!?])(?=[" + re.escape(WHITESPACE) + "])|(?<=\n)")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SPAN_PATH = re.compile(
    r"requests\[([0-9]+)\]\.interactions\[([0-9]+)\]\.(content|tools_used)"
    r"(?:\[([0-9]+):([0-9]+)\])?"
)
_INT32_MAX = 2_147_483_647

#: The keys a component identity carries, and nothing else.
COMPONENT_IDENTITY_KEYS = frozenset(
    {
        "component_role",
        "provider",
        "immutable_model_revision",
        "prompt_id",
        "prompt_version",
        "prompt_template_digest",
        "output_schema_id",
        "output_schema_version",
        "output_schema_digest",
        "generation_settings_digest",
        "qualification_suite_version",
    }
)


# ---------------------------------------------------------------------------
# Refusals and JSON primitives
# ---------------------------------------------------------------------------


def refuse(refusal: OpenWorldRefusal, message: str) -> NoReturn:
    """Raise the one exception class that carries ``refusal``."""
    raise INVOCATION_ERROR_BY_REFUSAL[refusal](message)


def integrity(message: str) -> NoReturn:
    refuse(OpenWorldRefusal.INTEGRITY, message)


def conflict(message: str) -> NoReturn:
    refuse(OpenWorldRefusal.INVOCATION_CONFLICT, message)


def is_hex64(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def json_type(value: object) -> str | None:
    """``jsonb_typeof``; ``None`` for an absent value."""
    if value is _ABSENT:
        return None
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


class _Absent:
    """A key that is not present, as distinct from one holding ``null``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<absent>"


_ABSENT: Any = _Absent()


def get(value: object, key: str | int) -> Any:
    """``value -> key``: the member, ``_ABSENT`` when there is none."""
    if isinstance(value, dict) and isinstance(key, str):
        return value.get(key, _ABSENT)
    if isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
        return value[key]
    return _ABSENT


def path(value: object, *keys: str | int) -> Any:
    """``value #> '{keys}'``."""
    for key in keys:
        value = get(value, key)
        if value is _ABSENT:
            return _ABSENT
    return value


def text(value: object) -> str | None:
    """``->>``: a string as itself, any other value as its JSON text."""
    if value is _ABSENT or value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def text_at(value: object, *keys: str | int) -> str | None:
    """``value #>> '{keys}'``."""
    return text(path(value, *keys))


def null_or_absent(value: object) -> bool:
    return value is _ABSENT or value is None


def canonical_text(value: object) -> str:
    """The canonical JSON text both backends hash."""
    try:
        return canonical_json_bytes(value).decode()
    except (TypeError, ValueError) as exc:
        integrity(f"unsupported canonical JSON value: {exc}")


def json_digest(value: object) -> str | None:
    """``rule_evidence_json_digest``; ``None`` for an absent value."""
    if value is _ABSENT:
        return None
    return sha256(canonical_text(value).encode()).hexdigest()


def text_digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def trim(value: str) -> str:
    """``rule_evidence_trim``."""
    return value.strip(WHITESPACE)


def without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    """``value - ARRAY[keys]``."""
    return {key: item for key, item in value.items() if key not in keys}


def decode(text_value: object, message: str) -> Any:
    """``text::jsonb``: a JSON text, refused as integrity when it is not one."""
    if not isinstance(text_value, str):
        integrity(message)
    try:
        return json.loads(text_value)
    except json.JSONDecodeError:
        integrity(message)


# ---------------------------------------------------------------------------
# Catalogues and fixed citation slots
# ---------------------------------------------------------------------------


def catalog(content: str) -> list[dict[str, str]]:
    """``rule_evidence_catalog``: sentence/line clauses of at most 600 chars."""
    entries: list[dict[str, str]] = []
    for part in _CLAUSE_SPLIT.split(content):
        for start in range(0, len(part), 600):
            entries.append(
                {"id": f"CLAUSE_{len(entries) + 1}", "text": part[start : start + 600]}
            )
    return entries


def choice_schema(choices: Sequence[Any]) -> dict[str, Any]:
    """``rule_evidence_choice_schema``."""
    if not choices:
        return {"type": "string"}
    if len(choices) == 1:
        return {"type": "string", "const": choices[0]}
    return {"type": "string", "enum": list(choices)}


def references(source: object) -> dict[str, dict[str, Any]]:
    """``rule_evidence_references``: ``REF_n`` for every message, in order."""
    result: dict[str, dict[str, Any]] = {}
    for session in _array(get(source, "sessions")):
        for message in _array(get(session, "messages")):
            result[f"REF_{len(result) + 1}"] = {
                "session_id": _member(session, "session_id"),
                "turn": _member(message, "turn"),
            }
    return result


def slots(
    values: object, session: str | None, maximum: int, required: bool
) -> list[dict[str, Any]]:
    """``rule_evidence_slots``: fixed ``r1..rN`` turn slots as references."""
    if not isinstance(values, dict) or maximum not in (4, 6):
        integrity("invalid fixed citation slots")
    allowed = {f"r{index}" for index in range(1, maximum + 1)}
    if set(values) - allowed or (required and null_or_absent(get(values, "r1"))):
        integrity("citation slots require deciding r1 and no extra fields")
    result: list[dict[str, Any]] = []
    for index in range(1, maximum + 1):
        value = get(values, f"r{index}")
        if null_or_absent(value):
            continue
        if json_type(value) != "number" or not re.fullmatch(
            r"[0-9]+", text(value) or ""
        ):
            integrity("citation turn must be an original integer")
        reference = {"session_id": session, "turn": value}
        if reference not in result:
            result.append(reference)
    return result


def _array(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _member(value: object, key: str) -> Any:
    """``value -> key`` as a JSON value: an absent key reads as ``null``."""
    found = get(value, key)
    return None if found is _ABSENT else found


def _ordinal(entries: Sequence[Mapping[str, Any]], clause_id: object) -> int | None:
    for index, entry in enumerate(entries, start=1):
        if entry["id"] == clause_id:
            return index
    return None


# ---------------------------------------------------------------------------
# Resolving a diagnostic and a global audit
# ---------------------------------------------------------------------------

_DIAGNOSTIC_KEYS = frozenset(
    {"session_id", "user_id", "observations", "rule_assessment", "missing_information"}
)
_ASSESSMENT_KEYS = frozenset(
    {
        "attribution",
        "expected_if_followed",
        "reason",
        "evidence_refs",
        "clause_start",
        "clause_end",
        "target_scope",
        "counterevidence_refs",
    }
)
_OBSERVATION_KEYS = frozenset(
    {"dimension", "kind", "observation", "consequence", "evidence_refs"}
)
_DISPOSITIONS = {
    "instruction_conflicts_requirement": "rule_gap",
    "observed_instruction_failure": "rule_gap",
    "missing_required_step": "rule_gap",
    "adequate_guidance_ignored": "adherence_failure",
    "no_supported_target_defect": "no_issue",
    "trigger_not_established": "not_applicable",
    "deciding_evidence_missing": "unknown",
}


def _blank(value: object) -> bool:
    """``coalesce(rule_evidence_trim(value), '') = ''`` for a ``->>`` value."""
    return trim(value) == "" if isinstance(value, str) else True


def resolve(diagnostic: object, content: str) -> dict[str, Any]:
    """``rule_evidence_resolve``: the resolved view of one raw diagnostic."""
    if not isinstance(diagnostic, dict):
        integrity("session diagnostic fields changed")
    assessment = get(diagnostic, "rule_assessment")
    missing = get(diagnostic, "missing_information")
    if (
        set(diagnostic) - _DIAGNOSTIC_KEYS
        or json_type(get(diagnostic, "session_id")) != "string"
        or json_type(get(diagnostic, "user_id")) != "string"
        or json_type(missing) != "array"
        or len(missing) > 4
        or any(json_type(item) != "string" or _blank(item) for item in missing)
        or not isinstance(assessment, dict)
        or set(assessment) != _ASSESSMENT_KEYS
        or json_type(assessment["target_scope"]) != "string"
        or json_type(assessment["clause_start"]) not in ("null", "string")
        or json_type(assessment["clause_end"]) not in ("null", "string")
        or json_type(assessment["evidence_refs"]) != "object"
        or json_type(assessment["counterevidence_refs"]) != "object"
        or json_type(assessment["expected_if_followed"]) != "string"
        or _blank(assessment["expected_if_followed"])
        or json_type(assessment["reason"]) != "string"
        or _blank(assessment["reason"])
    ):
        integrity("session diagnostic fields changed")
    observations = get(diagnostic, "observations")
    if json_type(observations) != "array" or len(observations) > 8:
        integrity("invalid session diagnostic")
    entries = catalog(content)
    quote = ""
    if assessment["clause_start"] is not None or assessment["clause_end"] is not None:
        first = _ordinal(entries, assessment["clause_start"])
        last = _ordinal(entries, assessment["clause_end"])
        if first is None or last is None or first > last:
            integrity("invalid indexed rule clause")
        quote = trim("".join(entry["text"] for entry in entries[first - 1 : last]))
        if quote == "" or len(quote) > 1200:
            integrity("invalid bounded rule quote")
    disposition = _DISPOSITIONS.get(text(assessment["attribution"]) or "")
    if disposition is None or _blank(assessment["target_scope"]):
        integrity("invalid session attribution or scope")
    session_id = text(get(diagnostic, "session_id"))
    resolved_observations = []
    for observation in observations:
        if (
            not isinstance(observation, dict)
            or set(observation) - _OBSERVATION_KEYS
            or text(get(observation, "dimension"))
            not in (
                "task_completion",
                "constraint_compliance",
                "evidence_grounding",
                "execution_effectiveness",
            )
            or text(get(observation, "kind"))
            not in ("strength", "problem", "uncertainty")
            or json_type(get(observation, "evidence_refs")) != "object"
            or json_type(get(observation, "observation")) != "string"
            or _blank(observation["observation"])
            or json_type(get(observation, "consequence")) != "string"
            or _blank(observation["consequence"])
        ):
            integrity("invalid structured session observation")
        resolved_observations.append(
            {
                **observation,
                "evidence_refs": slots(
                    observation["evidence_refs"], session_id, 6, True
                ),
            }
        )
    resolved_assessment = {
        **without(
            assessment,
            "clause_start",
            "clause_end",
            "attribution",
            "target_scope",
            "counterevidence_refs",
        ),
        "incumbent_quote": quote,
        "disposition": disposition,
        "evidence_refs": slots(assessment["evidence_refs"], session_id, 6, True),
    }
    return {
        **without(diagnostic, "rule_assessment", "observations"),
        "observations": resolved_observations,
        "rule_assessment": resolved_assessment,
    }


_AUDIT_KEYS = frozenset({"hypothesis", "edits", "abstention_reason"})
_EDIT_KEYS = frozenset(
    {"clause_start", "clause_end", "replacement", "defect", "evidence_refs"}
)


def resolve_audit(source: object, observed: object) -> dict[str, Any]:
    """``rule_evidence_resolve_audit``: the revised content a proposal yields."""
    original = text_at(source, "incumbent", "content")
    edits = get(observed, "edits")
    if (
        not isinstance(observed, dict)
        or set(observed) - _AUDIT_KEYS
        or json_type(get(observed, "hypothesis")) != "string"
        or json_type(get(observed, "abstention_reason")) != "string"
        or json_type(edits) != "array"
        or len(edits) > 4
        or original is None
    ):
        integrity("invalid global rule audit")
    hypothesis = trim(observed["hypothesis"])
    abstention = trim(observed["abstention_reason"])
    if (edits and (hypothesis == "" or abstention != "")) or (
        not edits and (hypothesis != "" or abstention == "")
    ):
        integrity("global proposal/abstention fields disagree")
    entries = catalog(original)
    refs = references(source)
    resolved: list[dict[str, Any]] = []
    spans: list[tuple[int, int, str]] = []
    for edit in edits:
        evidence = get(edit, "evidence_refs")
        if (
            not isinstance(edit, dict)
            or set(edit) - _EDIT_KEYS
            or json_type(get(edit, "replacement")) != "string"
            or json_type(get(edit, "clause_start")) != "string"
            or json_type(get(edit, "clause_end")) != "string"
            or json_type(get(edit, "defect")) != "string"
            or _blank(edit["defect"])
            or json_type(evidence) != "object"
            or set(evidence) - {f"r{index}" for index in range(1, 7)}
            or json_type(get(evidence, "r1")) != "string"
        ):
            integrity("invalid global edit shape or evidence")
        selected: list[dict[str, Any]] = []
        for index in range(1, 7):
            reference = get(evidence, f"r{index}")
            if null_or_absent(reference):
                continue
            if json_type(reference) != "string" or reference not in refs:
                integrity("global edit cites foreign source")
            if refs[reference] not in selected:
                selected.append(refs[reference])
        first = _ordinal(entries, edit["clause_start"])
        last = _ordinal(entries, edit["clause_end"])
        if first is None or last is None or first > last:
            integrity("global edit clause range invalid")
        start = sum(len(entry["text"]) for entry in entries[: first - 1])
        end = sum(len(entry["text"]) for entry in entries[:last])
        span = original[start:end]
        trimmed = trim(span)
        if trimmed == "" or len(trimmed) > 1200:
            integrity("global edit quote exceeds bound or is blank")
        at = span.find(trimmed)
        replacement = span[:at] + trim(edit["replacement"]) + span[at + len(trimmed) :]
        if span == replacement:
            integrity("global edit is a no-op")
        resolved.append(
            {
                **without(edit, "evidence_refs", "replacement"),
                "replacement": replacement,
                "evidence_refs": selected,
            }
        )
        spans.append((start, end, replacement))
    revised, position = "", 0
    for start, end, replacement in sorted(spans, key=lambda item: item[0]):
        if start < position:
            integrity("global edit ranges overlap or repeat")
        revised += original[position:start] + replacement
        position = end
    revised += original[position:]
    if resolved and trim(revised) == "":
        integrity("global repair removed all guidance")
    return {
        "rule_audit": {**without(observed, "edits"), "edits": resolved},
        "revised_content": revised,
    }


def assert_resolved(diagnostics: Sequence[Any], source: object) -> None:
    """``assert_rule_evidence_resolved``: resolved diagnostics admit natively."""
    content = text_at(source, "incumbent", "content") or ""
    sessions = _array(get(source, "sessions"))
    if len(diagnostics) != len(sessions):
        integrity("resolved diagnostic population changed")
    for diagnostic, session in zip(diagnostics, sessions, strict=True):
        assessment = get(diagnostic, "rule_assessment")
        quote = text(get(assessment, "incumbent_quote"))
        disposition = text(get(assessment, "disposition"))
        observations = _array(get(diagnostic, "observations"))
        if (
            text(get(diagnostic, "user_id")) != text(get(session, "user_id"))
            or text(get(diagnostic, "session_id")) != text(get(session, "session_id"))
            or disposition
            not in (
                "rule_gap",
                "adherence_failure",
                "no_issue",
                "not_applicable",
                "unknown",
            )
            or quote is None
            or len(quote) > 1200
            or (quote != "" and quote not in content)
            or (disposition in ("rule_gap", "adherence_failure") and trim(quote) == "")
            or (
                disposition == "rule_gap"
                and (
                    quote in content[content.find(quote) + 1 :]
                    or not any(
                        text(get(item, "kind")) == "problem" for item in observations
                    )
                )
            )
            or (
                disposition == "unknown"
                and not _array(get(diagnostic, "missing_information"))
            )
        ):
            integrity("corrected diagnostic violates native admission")
        turns = [get(message, "turn") for message in _array(get(session, "messages"))]
        for item in [*observations, assessment]:
            evidence = _array(get(item, "evidence_refs"))
            if not 1 <= len(evidence) <= 6:
                integrity("corrected diagnostic citations invalid")
            for reference in evidence:
                if (
                    text(get(reference, "session_id"))
                    != text(get(session, "session_id"))
                    or get(reference, "turn") not in turns
                ):
                    integrity("corrected diagnostic cites foreign source")


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


def _set_path(value: Any, keys: Sequence[str | int], replacement: Any) -> Any:
    """``jsonb_set(value, keys, replacement, false)``: never creates a path."""
    if not keys:
        return replacement
    head, *rest = keys
    if isinstance(value, dict) and isinstance(head, str) and head in value:
        return {**value, head: _set_path(value[head], rest, replacement)}
    if isinstance(value, list) and isinstance(head, int) and 0 <= head < len(value):
        items = list(value)
        items[head] = _set_path(items[head], rest, replacement)
        return items
    return value


def _counter_slots(schema: object) -> int:
    keys = sorted(_dict(path(schema, "$defs", "CounterTurns", "properties")))
    if keys == ["r1", "r2", "r3", "r4"]:
        return 4
    if keys == ["r1", "r2", "r3", "r4", "r5", "r6"]:
        return 6
    integrity("unsupported counterevidence slot schema")


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def schema_template(schema: object) -> Any:
    """``rule_evidence_schema_template``: the input-independent schema shape."""
    result = copy.deepcopy(schema)
    title = text(get(schema, "title"))
    if title == "BoundedTargetDiagnostic":
        counter_slots = _counter_slots(schema)
        for name in ("SupportTurns", "CounterTurns"):
            for index in range(1, (counter_slots if name == "CounterTurns" else 6) + 1):
                keys: list[str | int] = ["$defs", name, "properties", f"r{index}"]
                if name == "SupportTurns" and index == 1:
                    node: Any = {"title": "R1", "type": "integer"}
                else:
                    current = path(result, *keys)
                    if not isinstance(current, dict):
                        integrity("missing bounded turn schema")
                    node = {
                        **_set_path(current, ["anyOf", 0], {"type": "integer"}),
                        "title": f"R{index}",
                    }
                result = _set_path(result, keys, node)
        definitions = get(result, "$defs")
        if isinstance(result, dict) and isinstance(definitions, dict):
            result = {**result, "$defs": without(definitions, "OriginalTurn")}
        for field in ("clause_start", "clause_end"):
            keys = ["$defs", "BoundedSlotAssessment", "properties", field, "anyOf", 0]
            current = path(result, *keys)
            if not isinstance(current, dict):
                integrity("missing bounded clause schema")
            result = _set_path(result, keys, without(current, "enum", "const"))
    elif title == "GlobalRuleAudit":
        for field in ("clause_start", "clause_end"):
            keys = ["$defs", "GlobalAuditEdit", "properties", field]
            current = path(result, *keys)
            if not isinstance(current, dict):
                integrity("missing global clause schema")
            result = _set_path(result, keys, without(current, "enum", "const"))
        for index in range(1, 7):
            keys = ["$defs", "GlobalSourceSlots", "properties", f"r{index}"]
            if index > 1:
                keys += ["anyOf", 0]
            current = path(result, *keys)
            if not isinstance(current, dict):
                integrity("missing global reference schema")
            result = _set_path(result, keys, without(current, "enum", "const"))
    else:
        integrity("unsupported V7 wire schema")
    return result


def assert_schema_domains(schema: object, payload: object, role: str) -> None:
    """``assert_rule_evidence_schema_domains``: the input-bound enums."""
    if role == "discovery":
        counter_slots = _counter_slots(schema)
        clauses = [
            get(item, "id")
            for item in _array(path(payload, "incumbent", "content_clauses"))
        ]
        turns = [get(item, "turn") for item in _array(get(payload, "messages"))]
        if path(schema, "$defs", "OriginalTurn") != {
            "title": "OriginalTurn",
            "type": "integer",
            "enum": turns or None,
        }:
            integrity("original turn enum changed")
        for name in ("SupportTurns", "CounterTurns"):
            for index in range(1, (counter_slots if name == "CounterTurns" else 6) + 1):
                node = path(schema, "$defs", name, "properties", f"r{index}")
                if name == "SupportTurns" and index == 1:
                    expected: Any = {"$ref": "#/$defs/OriginalTurn"}
                else:
                    expected = {
                        "anyOf": [{"$ref": "#/$defs/OriginalTurn"}, {"type": "null"}],
                        "default": None,
                    }
                if node != expected:
                    integrity("fixed turn-slot schema changed")
        for field in ("clause_start", "clause_end"):
            node = path(
                schema,
                "$defs",
                "BoundedSlotAssessment",
                "properties",
                field,
                "anyOf",
                0,
            )
            if node != choice_schema(clauses):
                integrity("original clause enum changed")
    elif role == "candidate":
        clauses = [get(item, "id") for item in _array(get(payload, "original_clauses"))]
        choices = sorted(
            _dict(get(payload, "source_references")), key=lambda key: int(key[4:])
        )
        for field in ("clause_start", "clause_end"):
            node = path(schema, "$defs", "GlobalAuditEdit", "properties", field)
            if not isinstance(node, dict) or without(node, "title") != choice_schema(
                clauses
            ):
                integrity("global clause enum changed")
        for index in range(1, 7):
            keys: list[str | int] = [
                "$defs",
                "GlobalSourceSlots",
                "properties",
                f"r{index}",
            ]
            if index > 1:
                keys += ["anyOf", 0]
            node = path(schema, *keys)
            if not isinstance(node, dict) or without(node, "title") != choice_schema(
                choices
            ):
                integrity("global reference enum changed")
    else:
        integrity("unsupported wire role")


def counter_slot_count(request_json: object) -> int:
    """6 when the request's output schema declares an ``r5`` counter slot, else 4."""
    request = decode(request_json, "rule-evidence request is not JSON")
    properties = path(request, "output_schema", "$defs", "CounterTurns", "properties")
    return 6 if isinstance(properties, dict) and "r5" in properties else 4


def assert_proposal_context(
    request: object, collection: object, bundle: object
) -> None:
    """``assert_rule_evidence_proposal_context``: the proposal's request is
    exactly the original source plus the immutable resolved diagnostics."""
    sessions, diagnostics, audits = [], [], []
    incumbent: Any = None
    content = ""
    for session in _array(get(collection, "sessions")):
        prepared_request = decode(
            text_at(session, "prepared", "request_json"),
            "rule-evidence request is not JSON",
        )
        source = get(prepared_request, "payload")
        clauses = _array(path(source, "incumbent", "content_clauses"))
        content = "".join(text(get(item, "text")) or "" for item in clauses)
        incumbent = {
            **without(_dict(get(source, "incumbent")), "content_clauses"),
            "content": content,
        }
        sessions.append(
            {
                "session_id": _member(session, "session_id"),
                "user_id": _member(session, "user_id"),
                "exposed_version": _member(
                    _dict(get(bundle, "target")), "incumbent_full_version_fingerprint"
                ),
                "source_limitations": _member(source, "source_limitations"),
                "messages": _member(source, "messages"),
            }
        )
        diagnostics.append(resolve(get(session, "observed_output"), content))
        audit = {
            "session_id": _member(session, "session_id"),
            "attribution": _member(
                path(session, "observed_output", "rule_assessment"), "attribution"
            ),
            "target_scope": _member(
                path(session, "observed_output", "rule_assessment"), "target_scope"
            ),
            "counterevidence_refs": slots(
                path(
                    session,
                    "observed_output",
                    "rule_assessment",
                    "counterevidence_refs",
                ),
                text(get(session, "session_id")),
                counter_slot_count(text_at(session, "prepared", "request_json")),
                False,
            ),
        }
        turns = [get(message, "turn") for message in _array(get(source, "messages"))]
        if any(
            reference["turn"] not in turns
            for reference in audit["counterevidence_refs"]
        ):
            integrity("target audit cites foreign counterevidence")
        audits.append(audit)
    task_context = _dict(get(bundle, "task_context"))
    original = {
        "incumbent": incumbent,
        "sessions": sessions,
        "agent_context": _member(task_context, "agent_context"),
        "success_definition": _member(task_context, "success_definition"),
        "available_tools": _member(task_context, "available_tools"),
        "tool_context_status": _member(task_context, "tool_context_status"),
    }
    assert_resolved(diagnostics, original)
    if request != {
        "original_source": original,
        "diagnostics": diagnostics,
        "target_audits": audits,
        "original_clauses": catalog(content),
        "source_references": references(original),
    }:
        integrity(
            "global audit request changed original evidence or immutable diagnostics"
        )


# ---------------------------------------------------------------------------
# Source spans
# ---------------------------------------------------------------------------


def _speaker(role: object) -> str:
    normalized = (text(role) or "").strip(" ").lower()
    if normalized in ("assistant", "agent"):
        return "agent"
    if normalized == "user":
        return "user"
    return "other"


def assert_source_span(span: object, trajectory: object) -> None:
    """``assert_rule_evidence_source_span``: a span projects original evidence."""
    match = _SPAN_PATH.fullmatch(text(get(span, "field_path")) or "")
    if match is None:
        integrity("source span coordinate is not an original trajectory field")
    request_index, interaction_index, field, start, end = match.groups()
    if any(
        int(value) > _INT32_MAX
        for value in (request_index, interaction_index, start or "0", end or "0")
    ):
        integrity("source span coordinate exceeds original bounds")
    request = path(trajectory, "requests", int(request_index))
    interaction = path(request, "interactions", int(interaction_index))
    value: str | None
    if field == "content":
        value = text(get(interaction, "content"))
        speaker = _speaker(get(interaction, "role"))
    else:
        tools = get(interaction, "tools_used")
        if null_or_absent(tools) or tools == []:
            integrity("source tool span has no original tools")
        value = canonical_trajectory_bytes(tools).decode()
        speaker = "other"
    if start is not None and value is not None:
        if int(start) >= int(end) or int(end) > len(value):
            integrity("source span substring exceeds original bounds")
        value = value[int(start) : int(end)]
    if (
        null_or_absent(request)
        or null_or_absent(interaction)
        or not value
        or text(get(span, "request_id")) != text_at(request, "request", "request_id")
        or get(span, "interaction_id") != get(interaction, "interaction_id")
        or text(get(span, "speaker")) != speaker
        or text(get(span, "immutable_text")) != value
    ):
        integrity("source span does not project original trajectory evidence")


__all__ = [
    "COMPONENT_IDENTITY_KEYS",
    "assert_proposal_context",
    "assert_resolved",
    "assert_schema_domains",
    "assert_source_span",
    "canonical_text",
    "catalog",
    "choice_schema",
    "conflict",
    "counter_slot_count",
    "decode",
    "get",
    "integrity",
    "is_hex64",
    "json_digest",
    "json_type",
    "null_or_absent",
    "path",
    "references",
    "refuse",
    "resolve",
    "resolve_audit",
    "schema_template",
    "slots",
    "text",
    "text_at",
    "text_digest",
    "trim",
    "without",
]
