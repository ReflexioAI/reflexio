"""Plan-level invariants for the agentic-v2 extraction pipeline.

Invariants are pure functions over ``ExtractionCtx``. Hard violations drop
offending ops from the commit; soft violations are logged and applied.
See spec §6 for the full catalog and severity policy.
"""

from __future__ import annotations

import logging

from reflexio.server.services.extraction.plan import (
    CommitResult,
    CreateUserPlaybookOp,
    CreateUserProfileOp,
    DeleteUserPlaybookOp,
    DeleteUserProfileOp,
    ExtractionCtx,
    Violation,
)

logger = logging.getLogger(__name__)

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


# --- Soft invariants ---


def inv_E_no_duplicate_creates(ctx: ExtractionCtx) -> list[Violation]:  # noqa: N802
    """Two CreateOps with identical content in one plan = oscillation smell."""
    seen: dict[str, int] = {}
    violations: list[Violation] = []
    for i, op in enumerate(ctx.plan):
        key = None
        if isinstance(op, CreateUserProfileOp):
            key = f"profile::{op.content}"
        elif isinstance(op, CreateUserPlaybookOp):
            key = f"playbook::{op.trigger}::{op.content}"
        if key is None:
            continue
        if key in seen:
            violations.append(
                Violation(
                    code="E",
                    severity="soft",
                    affected_op_indices=[i],
                    msg=f"Duplicate create content at op {i}",
                )
            )
        else:
            seen[key] = i
    return violations


def inv_H_source_span_present(ctx: ExtractionCtx) -> list[Violation]:  # noqa: N802
    """CreateOps must have non-whitespace source_span.

    Schema enforces min_length=1, but whitespace-only slips through —
    this is the secondary guard.
    """
    violations: list[Violation] = []
    for i, op in enumerate(ctx.plan):
        if (
            isinstance(op, (CreateUserProfileOp, CreateUserPlaybookOp))
            and not op.source_span.strip()
        ):
            violations.append(
                Violation(
                    code="H",
                    severity="soft",
                    affected_op_indices=[i],
                    msg=f"Empty/whitespace source_span on create op {i}",
                )
            )
    return violations


def inv_K_deletes_without_creates(ctx: ExtractionCtx) -> list[Violation]:  # noqa: N802
    """Plan with deletes but no creates is unusual — worth logging."""
    has_delete = any(
        isinstance(op, (DeleteUserProfileOp, DeleteUserPlaybookOp)) for op in ctx.plan
    )
    has_create = any(
        isinstance(op, (CreateUserProfileOp, CreateUserPlaybookOp)) for op in ctx.plan
    )
    if has_delete and not has_create:
        indices = [
            i
            for i, op in enumerate(ctx.plan)
            if isinstance(op, (DeleteUserProfileOp, DeleteUserPlaybookOp))
        ]
        return [
            Violation(
                code="K",
                severity="soft",
                affected_op_indices=indices,
                msg="Plan contains deletes without any matching creates",
            )
        ]
    return []


SOFT_INVARIANTS = (
    inv_E_no_duplicate_creates,
    inv_H_source_span_present,
    inv_K_deletes_without_creates,
)


# --- Oscillation resolver ---


def resolve_tentative_oscillations(plan: list) -> set[int]:
    """Return plan indices to drop: create+delete-tentative pairs cancel.

    When the agent creates an entity (issuing a tentative_id) and later
    deletes that same tentative_id within the same plan, both ops are
    dropped before invariants fire. This is the "oscillated self-correction"
    pattern — the agent changed its mind mid-run.

    The tentative_id format is ``tentative::<kind>::<plan_index_at_issue_time>``,
    matching ``_next_tentative_id`` in tools.py which uses ``len(ctx.plan)``
    (the plan length BEFORE the op is appended, i.e. the future index of the op).

    Args:
        plan: The accumulated list of PlanOp instances from ctx.plan.

    Returns:
        Set of plan indices to exclude from apply. Both the create and the
        delete are dropped when a matching pair is found.
    """
    drop: set[int] = set()
    pending_creates: dict[str, int] = {}
    for i, op in enumerate(plan):
        if isinstance(op, CreateUserProfileOp):
            tentative_id = f"tentative::profile::{i}"
            pending_creates[tentative_id] = i
        elif isinstance(op, CreateUserPlaybookOp):
            tentative_id = f"tentative::user_playbook::{i}"
            pending_creates[tentative_id] = i
        elif isinstance(op, (DeleteUserProfileOp, DeleteUserPlaybookOp)):
            if op.id.startswith("tentative::") and op.id in pending_creates:
                drop.add(pending_creates.pop(op.id))
                drop.add(i)
    return drop


# --- commit_plan ---


def commit_plan(
    ctx: ExtractionCtx,
    storage: object,
    *,
    outcome: str,  # Literal["finish_tool","max_steps","error"]
) -> CommitResult:
    """Run all invariants, then apply surviving ops atomically.

    Args:
        ctx: Populated ExtractionCtx from the agent loop.
        storage: BaseStorage handle for apply.
        outcome: How the loop terminated.

    Returns:
        CommitResult containing applied ops + all violations (hard + soft).
    """
    # Error outcome — discard everything, do not apply
    if outcome == "error":
        return CommitResult(applied=[], violations=[], outcome="error")

    violations: list[Violation] = []
    for check in HARD_INVARIANTS:
        violations.extend(check(ctx))
    for check in SOFT_INVARIANTS:
        violations.extend(check(ctx))

    dropped: set[int] = set()
    # Oscillation resolver runs first: matching create+delete-tentative pairs
    # cancel before invariants decide what to keep.
    dropped.update(resolve_tentative_oscillations(ctx.plan))
    for v in violations:
        if v.severity == "hard":
            dropped.update(v.affected_op_indices)

    ops_to_apply = [op for i, op in enumerate(ctx.plan) if i not in dropped]

    for v in violations:
        logger.info(
            "invariant_violation user_id=%s code=%s severity=%s op_indices=%s msg=%s",
            ctx.user_id,
            v.code,
            v.severity,
            v.affected_op_indices,
            v.msg,
        )

    # Delegate actual storage writes to the tool-handler module (Task 5 wires this in).
    # Lazy import so Task 3 can land before tools.py exists.
    from reflexio.server.services.extraction.tools import (
        apply_plan_op,  # noqa: PLC0415  # type: ignore[import-not-found]
    )

    for op in ops_to_apply:
        apply_plan_op(op, storage, ctx)

    return CommitResult(applied=ops_to_apply, violations=violations, outcome=outcome)  # type: ignore[arg-type]
