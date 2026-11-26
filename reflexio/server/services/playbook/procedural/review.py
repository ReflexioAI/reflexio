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
