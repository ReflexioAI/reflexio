"""Unit tests for plan-level invariants. Pure-function — no LLM, no storage."""

from reflexio.server.services.extraction.invariants import (
    inv_A_search_before_create,
    inv_B_delete_known_id,
    inv_D_plan_size_cap,
    inv_F_no_duplicate_deletes,
    inv_J_scope_match,
)
from reflexio.server.services.extraction.plan import (
    CreateUserPlaybookOp,
    CreateUserProfileOp,
    DeleteUserPlaybookOp,
    DeleteUserProfileOp,
    ExtractionCtx,
)


def _mk_ctx(**kw):
    return ExtractionCtx(user_id="u_1", agent_version="v1", **kw)


# --- Invariant A: search-before-create ---


def test_inv_A_empty_plan_no_violations():  # noqa: N802
    assert inv_A_search_before_create(_mk_ctx()) == []


def test_inv_A_create_with_no_search_violates():  # noqa: N802
    ctx = _mk_ctx(search_count=0)
    ctx.plan.append(CreateUserProfileOp(content="x", ttl="infinity", source_span="y"))
    v = inv_A_search_before_create(ctx)
    assert len(v) == 1
    assert v[0].code == "A"
    assert v[0].affected_op_indices == [0]


def test_inv_A_create_after_search_ok():  # noqa: N802
    ctx = _mk_ctx(search_count=1)
    ctx.plan.append(CreateUserProfileOp(content="x", ttl="infinity", source_span="y"))
    assert inv_A_search_before_create(ctx) == []


def test_inv_A_multiple_creates_all_flagged_when_no_search():  # noqa: N802
    ctx = _mk_ctx(search_count=0)
    ctx.plan.append(CreateUserProfileOp(content="a", ttl="infinity", source_span="s"))
    ctx.plan.append(CreateUserPlaybookOp(trigger="t", content="c", source_span="s"))
    v = inv_A_search_before_create(ctx)
    assert len(v) == 1
    assert v[0].affected_op_indices == [0, 1]


# --- Invariant B: delete-references-known-id ---


def test_inv_B_delete_of_unknown_id_violates():  # noqa: N802
    ctx = _mk_ctx()
    ctx.plan.append(DeleteUserProfileOp(id="p_999"))
    v = inv_B_delete_known_id(ctx)
    assert len(v) == 1
    assert v[0].code == "B"
    assert v[0].affected_op_indices == [0]


def test_inv_B_delete_of_searched_id_ok():  # noqa: N802
    ctx = _mk_ctx()
    ctx.known_ids.add("p_123")
    ctx.plan.append(DeleteUserProfileOp(id="p_123"))
    assert inv_B_delete_known_id(ctx) == []


def test_inv_B_delete_of_in_plan_tentative_id_ok():  # noqa: N802
    """Self-correction: delete an id issued earlier in the same plan."""
    ctx = _mk_ctx()
    ctx.known_ids.add("tentative_0")  # the handler adds this when create_* runs
    ctx.plan.append(CreateUserProfileOp(content="x", ttl="infinity", source_span="s"))
    ctx.plan.append(DeleteUserProfileOp(id="tentative_0"))
    assert inv_B_delete_known_id(ctx) == []


def test_inv_B_playbook_delete_of_unknown_id_violates():  # noqa: N802
    ctx = _mk_ctx()
    ctx.plan.append(DeleteUserPlaybookOp(id="pb_999"))
    v = inv_B_delete_known_id(ctx)
    assert v[0].affected_op_indices == [0]


# --- Invariant D: plan-size cap ---


def test_inv_D_under_cap_ok():  # noqa: N802
    ctx = _mk_ctx()
    ctx.known_ids.add("tentative_0")
    for _ in range(30):
        ctx.plan.append(
            CreateUserProfileOp(content="x", ttl="infinity", source_span="y")
        )
    assert inv_D_plan_size_cap(ctx) == []


def test_inv_D_over_cap_flags_overflow():  # noqa: N802
    ctx = _mk_ctx()
    for _ in range(35):
        ctx.plan.append(
            CreateUserProfileOp(content="x", ttl="infinity", source_span="y")
        )
    v = inv_D_plan_size_cap(ctx)
    assert len(v) == 1
    assert v[0].affected_op_indices == list(range(30, 35))


# --- Invariant F: no-duplicate-deletes ---


def test_inv_F_duplicate_delete_flagged():  # noqa: N802
    ctx = _mk_ctx()
    ctx.known_ids.add("p_1")
    ctx.plan.append(DeleteUserProfileOp(id="p_1"))
    ctx.plan.append(DeleteUserProfileOp(id="p_1"))
    v = inv_F_no_duplicate_deletes(ctx)
    assert len(v) == 1
    # second (later) occurrence is the one we drop
    assert v[0].affected_op_indices == [1]


def test_inv_F_distinct_deletes_ok():  # noqa: N802
    ctx = _mk_ctx()
    ctx.known_ids.update({"p_1", "p_2"})
    ctx.plan.append(DeleteUserProfileOp(id="p_1"))
    ctx.plan.append(DeleteUserProfileOp(id="p_2"))
    assert inv_F_no_duplicate_deletes(ctx) == []


# --- Invariant J: scope-match (placeholder for storage-layer guard) ---


def test_inv_J_returns_empty_for_v1():  # noqa: N802
    """J is enforced primarily at storage layer (user_id injection).
    v1 invariant returns empty — future cross-user-check scaffolding."""
    ctx = _mk_ctx()
    assert inv_J_scope_match(ctx) == []


from unittest.mock import MagicMock

from reflexio.server.services.extraction.invariants import (
    commit_plan,
    inv_E_no_duplicate_creates,
    inv_H_source_span_present,
    inv_K_deletes_without_creates,
    resolve_tentative_oscillations,
)

# --- Soft invariants ---


def test_inv_E_identical_creates_flagged():  # noqa: N802
    ctx = _mk_ctx(search_count=1)
    ctx.plan.append(
        CreateUserProfileOp(content="user is a PM", ttl="infinity", source_span="s")
    )
    ctx.plan.append(
        CreateUserProfileOp(content="user is a PM", ttl="infinity", source_span="s")
    )
    v = inv_E_no_duplicate_creates(ctx)
    assert len(v) == 1
    assert v[0].severity == "soft"
    assert v[0].code == "E"


def test_inv_H_empty_source_span_is_caught_at_schema_level():  # noqa: N802
    """source_span is schema-required non-empty; this invariant is a
    secondary log guard if future schema changes relax that."""
    ctx = _mk_ctx(search_count=1)
    # construct op with non-empty source_span — schema enforces min_length=1
    ctx.plan.append(CreateUserProfileOp(content="x", ttl="infinity", source_span=" "))
    v = inv_H_source_span_present(ctx)
    assert len(v) == 1
    assert v[0].code == "H"
    assert v[0].severity == "soft"


def test_inv_K_deletes_only_flagged():  # noqa: N802
    ctx = _mk_ctx()
    ctx.known_ids.add("p_1")
    ctx.plan.append(DeleteUserProfileOp(id="p_1"))
    v = inv_K_deletes_without_creates(ctx)
    assert len(v) == 1
    assert v[0].severity == "soft"


def test_inv_K_delete_plus_create_ok():  # noqa: N802
    ctx = _mk_ctx(search_count=1)
    ctx.known_ids.add("p_1")
    ctx.plan.append(DeleteUserProfileOp(id="p_1"))
    ctx.plan.append(CreateUserProfileOp(content="x", ttl="infinity", source_span="y"))
    assert inv_K_deletes_without_creates(ctx) == []


# --- commit_plan orchestrator ---


def test_commit_plan_applies_valid_ops():  # noqa: N802
    """With no violations, every op reaches storage."""
    ctx = _mk_ctx(search_count=1)
    ctx.known_ids.add("p_exists")
    ctx.plan.append(DeleteUserProfileOp(id="p_exists"))
    ctx.plan.append(
        CreateUserProfileOp(content="new", ttl="infinity", source_span="evidence")
    )

    storage = MagicMock()
    result = commit_plan(ctx, storage, outcome="finish_tool")

    assert len(result.applied) == 2
    assert result.outcome == "finish_tool"
    assert result.violations == []


def test_commit_plan_drops_hard_violation_ops():  # noqa: N802
    """Hard-invariant-violating ops are excluded from apply."""
    ctx = _mk_ctx(search_count=0)
    # create without prior search → invariant A
    ctx.plan.append(CreateUserProfileOp(content="x", ttl="infinity", source_span="y"))
    # delete of unknown id → invariant B
    ctx.plan.append(DeleteUserProfileOp(id="never_retrieved"))

    storage = MagicMock()
    result = commit_plan(ctx, storage, outcome="finish_tool")

    assert result.applied == []
    codes = {v.code for v in result.violations}
    assert {"A", "B"}.issubset(codes)


def test_commit_plan_keeps_soft_violation_ops():  # noqa: N802
    """Soft violations are logged but ops commit."""
    ctx = _mk_ctx(search_count=1)
    ctx.plan.append(DeleteUserProfileOp(id="p_1"))
    ctx.known_ids.add("p_1")

    storage = MagicMock()
    result = commit_plan(ctx, storage, outcome="finish_tool")

    assert len(result.applied) == 1  # the delete got applied
    assert any(v.code == "K" for v in result.violations)  # but K flagged it


# --- resolve_tentative_oscillations ---


def test_resolve_oscillation_cancels_matching_pair():  # noqa: N802
    """Create at index 0 + delete targeting tentative::profile::0 cancel each other."""
    plan = [
        CreateUserProfileOp(content="x", ttl="infinity", source_span="y"),
        DeleteUserProfileOp(id="tentative::profile::0"),
        CreateUserProfileOp(content="real", ttl="infinity", source_span="z"),
    ]
    assert resolve_tentative_oscillations(plan) == {0, 1}


def test_resolve_oscillation_ignores_real_id_delete():  # noqa: N802
    """Delete of a non-tentative id is not touched by the resolver."""
    plan = [
        CreateUserProfileOp(content="x", ttl="infinity", source_span="y"),
        DeleteUserProfileOp(id="p_real_uuid_123"),
    ]
    assert resolve_tentative_oscillations(plan) == set()


def test_resolve_oscillation_unmatched_tentative_delete_passes_through():  # noqa: N802
    """Delete of a tentative id that doesn't match any create — resolver ignores it.
    Invariant B will catch it separately if it's truly unknown."""
    plan = [
        DeleteUserProfileOp(id="tentative::profile::99"),
    ]
    assert resolve_tentative_oscillations(plan) == set()


def test_resolve_oscillation_user_playbook_pair():  # noqa: N802
    """Same oscillation-cancel logic applies to user_playbook creates/deletes."""
    plan = [
        CreateUserPlaybookOp(trigger="t", content="c", source_span="s"),
        DeleteUserPlaybookOp(id="tentative::user_playbook::0"),
    ]
    assert resolve_tentative_oscillations(plan) == {0, 1}
