"""Plan-level invariants for the agentic-v2 extraction pipeline.

Invariants are pure functions over ``ExtractionCtx``. Hard violations drop
offending ops from the commit; soft violations are logged and applied.
See spec §6 for the full catalog and severity policy.
"""

from __future__ import annotations

from reflexio.server.services.extraction.plan import (
    CreateUserPlaybookOp,
    CreateUserProfileOp,
    DeleteUserPlaybookOp,
    DeleteUserProfileOp,
    ExtractionCtx,
    Violation,
)

PLAN_SIZE_CAP = 30


# --- Hard invariants ---


def inv_A_search_before_create(ctx: ExtractionCtx) -> list[Violation]:  # noqa: N802
    """Every CreateOp must be preceded by ≥1 search_* call this run."""
    create_indices = [
        i
        for i, op in enumerate(ctx.plan)
        if isinstance(op, (CreateUserProfileOp, CreateUserPlaybookOp))
    ]
    if create_indices and ctx.search_count == 0:
        return [
            Violation(
                code="A",
                severity="hard",
                affected_op_indices=create_indices,
                msg="Plan has create ops but no search was performed this run",
            )
        ]
    return []


def inv_B_delete_known_id(ctx: ExtractionCtx) -> list[Violation]:  # noqa: N802
    """Every DeleteOp(id) must reference an id in ctx.known_ids.

    known_ids is populated by search/get/create tool handlers — so deletes
    targeting hallucinated ids (agent never saw them) are rejected.
    """
    violations: list[Violation] = []
    for i, op in enumerate(ctx.plan):
        if (
            isinstance(op, (DeleteUserProfileOp, DeleteUserPlaybookOp))
            and op.id not in ctx.known_ids
        ):
            violations.append(
                Violation(
                    code="B",
                    severity="hard",
                    affected_op_indices=[i],
                    msg=f"Delete of unknown id {op.id!r}",
                )
            )
    return violations


def inv_D_plan_size_cap(ctx: ExtractionCtx) -> list[Violation]:  # noqa: N802
    """Plan cannot exceed PLAN_SIZE_CAP ops — guards runaway loops."""
    if len(ctx.plan) > PLAN_SIZE_CAP:
        overflow = list(range(PLAN_SIZE_CAP, len(ctx.plan)))
        return [
            Violation(
                code="D",
                severity="hard",
                affected_op_indices=overflow,
                msg=f"Plan size {len(ctx.plan)} exceeds cap {PLAN_SIZE_CAP}",
            )
        ]
    return []


def inv_F_no_duplicate_deletes(ctx: ExtractionCtx) -> list[Violation]:  # noqa: N802
    """Same id cannot be deleted twice in one plan."""
    seen: set[str] = set()
    violations: list[Violation] = []
    for i, op in enumerate(ctx.plan):
        if isinstance(op, (DeleteUserProfileOp, DeleteUserPlaybookOp)):
            if op.id in seen:
                violations.append(
                    Violation(
                        code="F",
                        severity="hard",
                        affected_op_indices=[i],
                        msg=f"Duplicate delete of id {op.id!r}",
                    )
                )
            else:
                seen.add(op.id)
    return violations


def inv_J_scope_match(_ctx: ExtractionCtx) -> list[Violation]:  # noqa: N802
    """User_id scope is primarily enforced at the storage layer (handlers inject
    ctx.user_id). This invariant is a placeholder for future cross-user checks;
    for v1 it is a no-op."""
    return []


HARD_INVARIANTS = (
    inv_A_search_before_create,
    inv_B_delete_known_id,
    inv_D_plan_size_cap,
    inv_F_no_duplicate_deletes,
    inv_J_scope_match,
)
