"""Tests for the UnifyAgent — cross-axis dedup pass over deferred extractions."""

from __future__ import annotations

from unittest.mock import MagicMock

from reflexio.server.services.extraction.extraction_agent import (
    DeferredExtractionRun,
)
from reflexio.server.services.extraction.plan import (
    CreateUserPlaybookOp,
    CreateUserProfileOp,
    DeleteUserProfileOp,
    ExtractionCtx,
)
from reflexio.server.services.extraction.unify import UnifyAgent


def _ctx(plan: list) -> ExtractionCtx:
    ctx = ExtractionCtx(
        user_id="u_test",
        agent_version="v1",
        extractor_name="ext",
    )
    ctx.plan = plan
    return ctx


def _profile_op(content: str) -> CreateUserProfileOp:
    return CreateUserProfileOp(
        content=content,
        ttl="one_year",
        source_span="s1",
    )


def _playbook_op(trigger: str, content: str) -> CreateUserPlaybookOp:
    return CreateUserPlaybookOp(
        trigger=trigger,
        content=content,
        source_span="s1",
    )


def _runner(response_text: str) -> UnifyAgent:
    """Build a UnifyAgent whose LLM returns a fixed response."""
    pm = MagicMock()
    pm.render_prompt.return_value = "stub prompt"
    client = MagicMock()
    client.generate_response.return_value = response_text
    return UnifyAgent(client=client, prompt_manager=pm)


def test_unify_drop_one_op_from_pass_b():
    """A single 'DROP B.0' line removes the corresponding op."""
    runs = [
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("user is vegan")]),
            outcome="finish_tool",
            kind="UserProfile",
        ),
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("agent told user about vegan options")]),
            outcome="finish_tool",
            kind="UserProfileAgentRec",
        ),
    ]
    dropped = _runner("DROP B.0").run(runs)
    assert dropped == 1
    assert len(runs[0].ctx.plan) == 1
    assert runs[1].ctx.plan == []


def test_unify_keep_all_when_response_says_so():
    """No DROP lines (or 'KEEP ALL') leaves every plan intact."""
    runs = [
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("a"), _profile_op("b")]),
            outcome="finish_tool",
            kind="UserProfile",
        ),
        DeferredExtractionRun(
            ctx=_ctx([_playbook_op("t", "c")]),
            outcome="finish_tool",
            kind="UserPlaybook",
        ),
    ]
    dropped = _runner("KEEP ALL").run(runs)
    assert dropped == 0
    assert len(runs[0].ctx.plan) == 2
    assert len(runs[1].ctx.plan) == 1


def test_unify_llm_failure_keeps_all():
    """LLM exception → graceful fallback, plans unchanged."""
    pm = MagicMock()
    pm.render_prompt.return_value = "stub"
    client = MagicMock()
    client.generate_response.side_effect = RuntimeError("network")
    agent = UnifyAgent(client=client, prompt_manager=pm)

    runs = [
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("x")]),
            outcome="finish_tool",
            kind="UserProfile",
        ),
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("y")]),
            outcome="finish_tool",
            kind="UserProfileAgentRec",
        ),
    ]
    dropped = agent.run(runs)
    assert dropped == 0
    assert len(runs[0].ctx.plan) == 1
    assert len(runs[1].ctx.plan) == 1


def test_unify_never_drops_delete_ops():
    """DROP A.<idx> targeting a delete op is a no-op (deletes are protected)."""
    runs = [
        DeferredExtractionRun(
            ctx=_ctx(
                [
                    _profile_op("kept"),
                    DeleteUserProfileOp(id="profile-123"),
                ]
            ),
            outcome="finish_tool",
            kind="UserProfile",
        ),
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("other")]),
            outcome="finish_tool",
            kind="UserProfileAgentRec",
        ),
    ]
    # Try to drop the delete op — should be ignored.
    dropped = _runner("DROP A.1").run(runs)
    assert dropped == 0
    assert len(runs[0].ctx.plan) == 2  # both ops survive


def test_unify_skips_when_only_one_run():
    """A single deferred run has no cross-axis to unify."""
    runs = [
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("a")]),
            outcome="finish_tool",
            kind="UserProfile",
        )
    ]
    dropped = _runner("DROP A.0").run(runs)
    assert dropped == 0
    assert len(runs[0].ctx.plan) == 1


def test_unify_parses_multiple_drop_lines():
    """Multiple DROP lines apply across multiple passes."""
    runs = [
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("a"), _profile_op("b"), _profile_op("c")]),
            outcome="finish_tool",
            kind="UserProfile",
        ),
        DeferredExtractionRun(
            ctx=_ctx([_profile_op("d"), _profile_op("e")]),
            outcome="finish_tool",
            kind="UserProfileAgentRec",
        ),
    ]
    dropped = _runner("DROP A.0\nDROP A.2\nDROP B.1").run(runs)
    assert dropped == 3
    # Pass A: dropped indices 0 and 2 → only "b" survives
    assert [op.content for op in runs[0].ctx.plan] == ["b"]
    # Pass B: dropped index 1 → only "d" survives
    assert [op.content for op in runs[1].ctx.plan] == ["d"]
