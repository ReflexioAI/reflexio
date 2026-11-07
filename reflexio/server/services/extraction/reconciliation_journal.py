"""Validation for source selections inside the existing learning attempt journal.

No storage, model dispatch or publication lives here. The existing parent CAS
atomically reserves the physical attempt and its reconciliation attribution.
"""

from __future__ import annotations

from typing import Any

KEY = "reconciliation_context"

# Everything persisted after a compression plan is saved that carries that
# plan's semantics. A saved plan may be replaced by a re-planned witness only
# while none of these exists. Once one does, a changed witness fails closed, so
# old selections never mix with a newer settlement contract.
#
# - ``nodes``: selection receipts settled under the saved plan's model and
#   reference contract (the cached selections).
# - ``audited_conversation``: the consumer conversation over the view those
#   selections rendered. It is saved before every consumer request, so it
#   precedes every consumer step receipt and the approved-batch receipt, and no
#   writer removes it: its absence proves that no consumer work exists.
#
# Kept, not derived: ``attempt_ids``, the matching
# ``audited_consolidation_budget.calls`` and the parent's attempt usage. They
# are append-only physical-request accounting with no model output, and a
# refund is forbidden -- except an attempt the provider never answered during
# an outage, which the parent's journal releases together with its entry here
# (``_release_unanswered_attempt``). A re-planned selection claims only attempts reserved
# after it starts, so spent attempts stay charged without being attributed to
# new work. ``version``, ``operation_key`` and ``limit`` bind the operation's
# payload, policy and model rather than the plan, and never change.
PLAN_DERIVED_ARTIFACTS = ("nodes", "audited_conversation")


def plan_derived_artifacts(checkpoint: dict[str, Any]) -> tuple[str, ...]:
    """Name each persisted artifact built from the saved compression plan."""
    present = {
        "nodes": bool(checkpoint[KEY]["nodes"]),
        "audited_conversation": checkpoint.get("audited_conversation") is not None,
    }
    return tuple(name for name in PLAN_DERIVED_ARTIFACTS if present[name])


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_context(manifest: dict[str, Any]) -> None:
    checkpoint = manifest["checkpoint"]
    context = checkpoint.get(KEY)
    if context is None and KEY not in checkpoint:
        return  # Previously admitted journals retain their original protocol.
    if (
        not isinstance(context, dict)
        or set(context)
        != {"version", "operation_key", "limit", "plan", "attempt_ids", "nodes"}
        or type(context["version"]) is not int
        or context["version"] != 1
        or not _digest(context["operation_key"])
        or type(context["limit"]) is not int
        or not 2 <= context["limit"] <= 24
        or not isinstance(context["attempt_ids"], list)
        or not all(isinstance(value, str) for value in context["attempt_ids"])
        or len(set(context["attempt_ids"])) != len(context["attempt_ids"])
        or len(context["attempt_ids"]) > context["limit"]
        or not isinstance(context["nodes"], dict)
        or len(context["nodes"]) > context["limit"]
    ):
        raise ValueError("Invalid reconciliation source journal")
    budget = checkpoint.get("audited_consolidation_budget")
    if (
        not isinstance(budget, dict)
        or budget.get("key") != context["operation_key"]
        or type(budget.get("calls")) is not int
        or budget["calls"] != len(context["attempt_ids"])
    ):
        raise ValueError("Reconciliation attempts and allowance differ")
    attempts = {attempt["id"]: attempt for attempt in manifest["attempts"]}
    if not set(context["attempt_ids"]) <= attempts.keys():
        raise ValueError("Reconciliation source work lacks reserved attempts")
    plan = context["plan"]
    if plan is not None and (
        not isinstance(plan, dict)
        or set(plan) != {"witness", "source_digest", "request_contract_digest"}
        or not all(_digest(value) for value in plan.values())
    ):
        raise ValueError("Invalid reconciliation compression plan")
    for key, node in context["nodes"].items():
        if (
            plan is None
            or not _digest(key)
            or not isinstance(node, dict)
            or set(node)
            != {"input_refs_digest", "request_contract_digest", "attempt_ids", "output"}
            or not _digest(node["input_refs_digest"])
            or node["request_contract_digest"] != plan["request_contract_digest"]
            or not isinstance(node["output"], dict)
            or not isinstance(node["attempt_ids"], list)
            or not node["attempt_ids"]
            or not all(isinstance(value, str) for value in node["attempt_ids"])
            or len(set(node["attempt_ids"])) != len(node["attempt_ids"])
            or not set(node["attempt_ids"]) <= set(context["attempt_ids"])
            or not any(
                attempts[value]["model_provenance"] is not None
                for value in node["attempt_ids"]
            )
        ):
            raise ValueError("Invalid reconciliation selection receipt")


def validate_advance(previous: dict[str, Any], current: dict[str, Any]) -> None:
    old = previous.get(KEY)
    if old is None:
        return
    new = current.get(KEY)
    if (
        not isinstance(new, dict)
        or any(
            new.get(field) != old[field]
            for field in ("version", "operation_key", "limit")
        )
        or (
            old["plan"] is not None
            and new.get("plan") != old["plan"]
            and (new.get("plan") is None or plan_derived_artifacts(previous))
        )
        or new.get("attempt_ids", [])[: len(old["attempt_ids"])] != old["attempt_ids"]
        or any(
            new.get("nodes", {}).get(key) != node for key, node in old["nodes"].items()
        )
    ):
        raise ValueError("Reconciliation journal cannot be changed or refunded")
