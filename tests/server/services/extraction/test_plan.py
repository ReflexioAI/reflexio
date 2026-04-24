"""Unit tests for PlanOp types + ExtractionCtx."""

import pytest
from pydantic import ValidationError

from reflexio.server.services.extraction.plan import (
    CommitResult,
    CreateUserPlaybookOp,
    CreateUserProfileOp,
    DeleteUserPlaybookOp,
    DeleteUserProfileOp,
    ExtractionCtx,
    Violation,
)


def test_create_user_profile_op_requires_content_ttl_source_span():
    op = CreateUserProfileOp(
        content="user likes pasta",
        ttl="infinity",
        source_span="I love pasta",
    )
    assert op.content == "user likes pasta"
    assert op.ttl == "infinity"
    assert op.source_span == "I love pasta"


def test_create_user_profile_op_rejects_empty_content():
    with pytest.raises(ValidationError):
        CreateUserProfileOp(content="", ttl="infinity", source_span="evidence")


def test_create_user_profile_op_rejects_invalid_ttl():
    with pytest.raises(ValidationError):
        CreateUserProfileOp(
            content="x",
            ttl="two_days",  # type: ignore[arg-type]
            source_span="y",  # not in ProfileTimeToLive
        )


def test_delete_user_profile_op_requires_id():
    op = DeleteUserProfileOp(id="p_42")
    assert op.id == "p_42"
    with pytest.raises(ValidationError):
        DeleteUserProfileOp(id="")


def test_create_user_playbook_op_fields():
    op = CreateUserPlaybookOp(
        trigger="code help",
        content="show examples",
        rationale="user prefers examples",
        strength="soft",
        source_span="…",
    )
    assert op.strength == "soft"


def test_create_user_playbook_op_rejects_bad_strength():
    with pytest.raises(ValidationError):
        CreateUserPlaybookOp(
            trigger="t",
            content="c",
            rationale="r",
            strength="weak",  # type: ignore[arg-type]
            source_span="s",
        )


def test_delete_user_playbook_op_requires_id():
    op = DeleteUserPlaybookOp(id="pb_7")
    assert op.id == "pb_7"


def test_extraction_ctx_defaults():
    ctx = ExtractionCtx(user_id="u_1", agent_version="v1")
    assert ctx.user_id == "u_1"
    assert ctx.agent_version == "v1"
    assert ctx.plan == []
    assert ctx.known_ids == set()
    assert ctx.search_count == 0
    assert ctx.finished is False


def test_violation_and_commit_result_shapes():
    v = Violation(code="A", severity="hard", affected_op_indices=[0, 2], msg="x")
    assert v.severity == "hard"
    r = CommitResult(applied=[], violations=[v], outcome="finish_tool")
    assert r.outcome == "finish_tool"
    assert len(r.violations) == 1
