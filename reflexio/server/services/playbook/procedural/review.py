"""Reduce exactly one semantic verdict per full-result unit into atomic groups."""

from copy import deepcopy
from typing import Any

from .models import ContractError, ResultVerdict, Review, Verdict

CHECKS = (
    "instruction_support",
    "scope_conditions",
    "useful_nonduplicate",
    "preservation_and_correction",
    "resource_consistency",
)


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = list(payload["proposed_records"].values())
    records += [
        unit["resulting_record"]
        for unit in payload["result_review_units"]
        if unit["resulting_record"] is not None
    ]
    return list({id(record): record for record in records}.values())


def _expand_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    """Invert only local inherited-span references; never load external data."""
    restored = deepcopy(payload)
    existing = restored["source"]["existing"]
    for record in _records(restored):
        value = record.get("source_span")
        if not isinstance(value, dict):
            continue
        if set(value) != {"verbatim_concatenation"}:
            raise ContractError("Invalid review provenance representation")
        pieces = []
        for piece in value["verbatim_concatenation"]:
            if set(piece) == {"literal"} and isinstance(piece["literal"], str):
                pieces.append(piece["literal"])
            elif set(piece) == {"incumbent_source_span"}:
                row = existing.get(piece["incumbent_source_span"])
                if (
                    row is None
                    or row["user_playbook_id"] not in record.get("replaces", [])
                    or not isinstance(row.get("source_span"), str)
                ):
                    raise ContractError("Review provenance needs an exact incumbent")
                pieces.append(row["source_span"])
            else:
                raise ContractError("Invalid review provenance part")
        record["source_span"] = "".join(pieces)
    return restored


def input_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Avoid repeating inherited evidence, without altering the logical review."""
    view = deepcopy(payload)
    existing = view["source"]["existing"]
    by_id = {row["user_playbook_id"]: (key, row) for key, row in existing.items()}
    if len(by_id) != len(existing):
        return view  # Ambiguous provenance remains raw and subject to admission.
    for record in _records(view):
        span = record.get("source_span")
        replacements = record.get("replaces", [])
        if not isinstance(span, str) or not replacements:
            continue
        if len(set(replacements)) != len(replacements) or any(
            key not in by_id for key in replacements
        ):
            continue
        parts: list[dict[str, str]] = []
        old_spans = []
        for identity in replacements:
            key, row = by_id[identity]
            old_span = row.get("source_span")
            if isinstance(old_span, str) and old_span and old_span not in old_spans:
                if old_spans:
                    parts.append({"literal": "\n\n"})
                old_spans.append(old_span)
                parts.append({"incumbent_source_span": key})
        prefix = "\n\n".join(old_spans)
        if not prefix or not (span == prefix or span.startswith(prefix + "\n\n")):
            continue
        suffix = span[len(prefix) :]
        if suffix:
            parts.append({"literal": suffix})
        record["source_span"] = {"verbatim_concatenation": parts}
    if _expand_provenance(view) != payload:
        raise ContractError("Review provenance projection changed original evidence")
    return view


def prepare(payload: dict[str, Any]) -> dict[str, Any]:
    if "result_review_units" in payload:
        raise ContractError("Reserved result-review field already exists")
    units = []
    groups = {}
    for index, change in enumerate(payload["changes"]):
        identity = f"U{index}"
        groups.setdefault(change["group"], []).append(identity)
        key = f"W{index}"
        if change["operation"] == "upsert" and key not in payload["proposed_records"]:
            raise ContractError("Upsert review requires its complete compiled result")
        units.append(
            {
                "unit": identity,
                "change": change,
                "resulting_record": payload["proposed_records"].get(key),
            }
        )
    for unit in units:
        unit["coupled_units"] = groups[unit["change"]["group"]]
    return deepcopy({**payload, "result_review_units": units})


def reduce(payload: dict[str, Any], verdicts: list[ResultVerdict]) -> Review:
    identities = [f"U{i}" for i in range(len(payload["changes"]))]
    if len(verdicts) != len(identities) or {v.unit for v in verdicts} != set(
        identities
    ):
        raise ContractError("Review every full result exactly once")
    by_id = {v.unit: v for v in verdicts}
    grouped = {}
    for index, change in enumerate(payload["changes"]):
        raw = by_id[f"U{index}"]
        value = raw.model_dump(mode="json")
        value["group"] = value.pop("unit")
        value["instruction_support"] = value.pop("result_support")
        grouped.setdefault(change["group"], []).append(Verdict.model_validate(value))
    results = []
    for group, members in grouped.items():
        checks = {}
        for key in CHECKS:
            states = [getattr(value, key) for value in members]
            checks[key] = (
                "fail"
                if "fail" in states
                else "unknown"
                if "unknown" in states
                else "pass"
            )
        decisions = [value.decision for value in members]
        decision = (
            "reject"
            if "reject" in decisions or "fail" in checks.values()
            else "unknown"
            if "unknown" in decisions or "unknown" in checks.values()
            else "accept"
        )
        results.append(
            Verdict(
                group=group,
                decision=decision,
                **checks,
                reason="\n".join(f"{value.group}: {value.reason}" for value in members),
            )
        )
    return Review(groups=results)


def accepted_groups(review: Review) -> set[str]:
    return {
        verdict.group
        for verdict in review.groups
        if verdict.decision == "accept"
        and all(getattr(verdict, key) == "pass" for key in CHECKS)
    }
