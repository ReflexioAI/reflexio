"""Apply complete audited batches to a benchmark book without choosing survivors."""

from __future__ import annotations

from reflexio.server.services.playbook.components.reconciliation import (
    Decisions,
    project,
    validate_decisions,
)
from tests.eval.scenarios.case import BookRule


def _next_id(book: list[BookRule]) -> int:
    return max((row.id for row in book), default=0) + 1


def apply_consolidation(
    book: list[BookRule],
    candidate: BookRule,
    decision: Decisions,
    *,
    existing_order: list[BookRule],
) -> list[BookRule]:
    if {row.id for row in book} != {row.id for row in existing_order}:
        raise ValueError("Book and supplied existing order differ")
    existing = {
        f"EXISTING-{index}": row.model_dump(mode="json")
        for index, row in enumerate(existing_order)
    }
    references = {
        (ref.session_id, ref.turn): None
        for row in [candidate, *existing_order]
        for ref in row.evidence_refs
    }
    validate_decisions(
        decision, {"NEW-0": candidate.model_dump(mode="json")}, existing, references
    )
    consumed = {
        key
        for action in decision.decisions
        for key in getattr(action, "existing_ids", [])
    }
    # Project uses the same Create/Replace/Remove semantics as the method's
    # independent audit; it retains untouched inputs and every authored output.
    produced = project(decision, existing)
    retained = {
        f"EXISTING-{index}": row
        for index, row in enumerate(existing_order)
        if f"EXISTING-{index}" not in consumed
    }
    fresh_id = max(_next_id(book), candidate.id)
    result = []
    for row in produced:
        source_ids = row["source_ids"]
        if len(source_ids) == 1 and source_ids[0] in retained:
            result.append(retained[source_ids[0]].model_copy(deep=True))
        else:
            result.append(
                BookRule(
                    id=fresh_id,
                    content=row["content"],
                    trigger=row.get("trigger"),
                    rationale=row.get("rationale", ""),
                    evidence_refs=row.get("evidence_refs", []),
                )
            )
            fresh_id += 1
    return result
