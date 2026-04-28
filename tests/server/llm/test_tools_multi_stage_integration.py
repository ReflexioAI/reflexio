"""Integration tests for the multi-stage fallback path in ``run_tool_loop``.

These tests target the multi-turn structured-output flow used when the
configured model lacks native tool-calling but should still observe
prior tool results before planning the next call (e.g. the search agent
running on ``minimax/MiniMax-M2.7``).

The mocked LLM client is scripted to return one ``MultiStagePlan``
instance per turn; the test asserts that:

  - The loop emits multiple structured-output calls in sequence.
  - Each tool result is appended to the shared ``messages`` list so the
    next turn's prompt sees it.
  - The loop terminates when ``next_call.tool == finish_tool_name``.
  - The loop terminates at ``max_steps`` when no finish is emitted.
  - Each registry tool dispatches via the discriminator literal.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from reflexio.server.llm import tools as tools_mod
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import Tool, ToolRegistry, run_tool_loop

# ---------------------------------------------------------------------------
# Test schemas (mirror SearchAgentTurnPlan shape: reasoning + next_call union)
# ---------------------------------------------------------------------------


class _CallEmit(BaseModel):
    """Test variant: dispatch ``emit``."""

    tool: str = Field(default="emit", pattern="^emit$")
    value: str


class _CallFinish(BaseModel):
    """Test variant: dispatch ``finish``."""

    tool: str = Field(default="finish", pattern="^finish$")
    answer: str | None = None


class MultiStagePlan(BaseModel):
    """Mirror of ``SearchAgentTurnPlan``: one turn of multi-stage fallback."""

    reasoning: str
    # We use a plain Union (no discriminator field) so the tests can
    # construct either variant directly without pydantic's discriminator
    # validation overhead — the real schema uses a discriminated union.
    next_call: _CallEmit | _CallFinish


# ---------------------------------------------------------------------------
# Test ctx + registry
# ---------------------------------------------------------------------------


class _Ctx:
    """Mutable per-run state for tool-loop tests."""

    def __init__(self) -> None:
        self.emitted: list[str] = []
        self.finished: bool = False
        self.finish_answer: str | None = None


class _EmitArgs(BaseModel):
    """Emit a value (test tool)."""

    value: str


class _FinishArgs(BaseModel):
    """Terminate the test loop."""

    answer: str | None = None


def _make_registry(ctx: _Ctx) -> ToolRegistry:
    def _emit_handler(args: BaseModel, c: _Ctx) -> dict:
        c.emitted.append(args.value)  # type: ignore[attr-defined]
        return {"ok": True, "echo": args.value}  # type: ignore[attr-defined]

    def _finish_handler(args: BaseModel, c: _Ctx) -> dict:
        c.finished = True
        c.finish_answer = args.answer  # type: ignore[attr-defined]
        return {"finished": True}

    reg = ToolRegistry()
    reg.register(Tool(name="emit", args_model=_EmitArgs, handler=_emit_handler))
    reg.register(Tool(name="finish", args_model=_FinishArgs, handler=_finish_handler))
    return reg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_no_tool_calling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the capability fallback path so we always exercise multi-stage."""
    monkeypatch.setattr(tools_mod, "supports_tool_calling", lambda _model: False)


def _scripted_client(
    monkeypatch: pytest.MonkeyPatch, plans: list[MultiStagePlan]
) -> LiteLLMClient:
    """Build a LiteLLMClient whose ``generate_chat_response`` returns plans in order."""
    client = LiteLLMClient(LiteLLMConfig(model="some-non-tool-calling-model"))
    iterator = iter(plans)

    def fake_generate(**_kwargs: object) -> MultiStagePlan:
        return next(iterator)

    monkeypatch.setattr(client, "generate_chat_response", fake_generate)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multi_stage_loop_emits_multiple_turns_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3-turn loop: emit, emit, finish — asserts trace shape and ctx mutations."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _force_no_tool_calling(monkeypatch)

    plans = [
        MultiStagePlan(reasoning="first emit", next_call=_CallEmit(value="alpha")),
        MultiStagePlan(reasoning="second emit", next_call=_CallEmit(value="beta")),
        MultiStagePlan(
            reasoning="done",
            next_call=_CallFinish(answer="all set"),
        ),
    ]
    client = _scripted_client(monkeypatch, plans)
    ctx = _Ctx()
    reg = _make_registry(ctx)

    messages = [{"role": "user", "content": "begin"}]
    result = run_tool_loop(
        client=client,
        messages=messages,
        registry=reg,
        model_role=ModelRole.EXTRACTION_AGENT,
        ctx=ctx,
        finish_tool_name="finish",
        multi_stage_schema=MultiStagePlan,
    )

    assert result.finished_reason == "finish_tool"
    assert result.trace.finished is True
    assert [t.tool_name for t in result.trace.turns] == ["emit", "emit", "finish"]
    assert ctx.emitted == ["alpha", "beta"]
    assert ctx.finished is True
    assert ctx.finish_answer == "all set"


def test_multi_stage_loop_appends_tool_results_to_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each tool result must land in ``messages`` so the next turn observes it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _force_no_tool_calling(monkeypatch)

    plans = [
        MultiStagePlan(reasoning="r1", next_call=_CallEmit(value="x")),
        MultiStagePlan(reasoning="r2", next_call=_CallFinish()),
    ]
    client = _scripted_client(monkeypatch, plans)
    ctx = _Ctx()
    reg = _make_registry(ctx)

    messages: list[dict[str, object]] = [{"role": "user", "content": "go"}]
    run_tool_loop(
        client=client,
        messages=messages,
        registry=reg,
        model_role=ModelRole.EXTRACTION_AGENT,
        ctx=ctx,
        finish_tool_name="finish",
        multi_stage_schema=MultiStagePlan,
    )

    # Seed + (assistant plan + user result) for the first turn + assistant plan for finish
    # The finish branch does not append a user-result message — it returns directly.
    roles = [m["role"] for m in messages]
    contents = [m["content"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    # The user message holding the tool result must mention the tool name
    # AND the handler's payload (so the next turn really sees it).
    tool_result_msg = contents[2]
    assert isinstance(tool_result_msg, str)
    assert "Tool emit returned" in tool_result_msg
    assert '"echo": "x"' in tool_result_msg
    # The assistant plan messages echo the tool name + args JSON.
    plan_msg_1 = contents[1]
    plan_msg_2 = contents[3]
    assert isinstance(plan_msg_1, str)
    assert isinstance(plan_msg_2, str)
    assert "Reasoning: r1" in plan_msg_1
    assert "Next call: emit(" in plan_msg_1
    assert "Reasoning: r2" in plan_msg_2
    assert "Next call: finish(" in plan_msg_2


def test_multi_stage_loop_terminates_at_max_steps_when_no_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loop must stop at ``max_steps`` when the agent never emits ``finish``."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _force_no_tool_calling(monkeypatch)

    plans = [
        MultiStagePlan(reasoning=f"r{i}", next_call=_CallEmit(value=f"v{i}"))
        for i in range(10)
    ]
    client = _scripted_client(monkeypatch, plans)
    ctx = _Ctx()
    reg = _make_registry(ctx)

    result = run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "go"}],
        registry=reg,
        model_role=ModelRole.EXTRACTION_AGENT,
        max_steps=3,
        ctx=ctx,
        finish_tool_name="finish",
        multi_stage_schema=MultiStagePlan,
    )

    assert result.finished_reason == "max_steps"
    assert result.trace.finished is False
    assert len(result.trace.turns) == 3
    assert ctx.emitted == ["v0", "v1", "v2"]
    assert ctx.finished is False


def test_multi_stage_loop_dispatches_each_call_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the registry handler actually runs for each turn (not just stubbed)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _force_no_tool_calling(monkeypatch)

    plans = [
        MultiStagePlan(reasoning="r1", next_call=_CallEmit(value="a")),
        MultiStagePlan(reasoning="r2", next_call=_CallEmit(value="b")),
        MultiStagePlan(reasoning="r3", next_call=_CallFinish()),
    ]
    client = _scripted_client(monkeypatch, plans)
    ctx = _Ctx()
    reg = _make_registry(ctx)

    result = run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "go"}],
        registry=reg,
        model_role=ModelRole.EXTRACTION_AGENT,
        ctx=ctx,
        finish_tool_name="finish",
        multi_stage_schema=MultiStagePlan,
    )

    # Every recorded turn should carry the handler's actual return value.
    emit_turns = [t for t in result.trace.turns if t.tool_name == "emit"]
    assert [t.result for t in emit_turns] == [
        {"ok": True, "echo": "a"},
        {"ok": True, "echo": "b"},
    ]
    finish_turn = next(t for t in result.trace.turns if t.tool_name == "finish")
    assert finish_turn.result == {"finished": True}


def test_multi_stage_loop_takes_priority_over_single_shot_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both ``multi_stage_schema`` and ``fallback_schema`` are passed, multi-stage wins."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _force_no_tool_calling(monkeypatch)

    plans = [MultiStagePlan(reasoning="r", next_call=_CallFinish())]
    client = _scripted_client(monkeypatch, plans)
    ctx = _Ctx()
    reg = _make_registry(ctx)

    class _SingleShotSchema(BaseModel):
        items: list[_EmitArgs] = []

    result = run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "go"}],
        registry=reg,
        model_role=ModelRole.EXTRACTION_AGENT,
        ctx=ctx,
        finish_tool_name="finish",
        fallback_schema=_SingleShotSchema,
        fallback_tool_name="emit",
        multi_stage_schema=MultiStagePlan,
    )

    # Single-shot would have produced 0 emits with empty list and returned
    # finished_reason='finish_tool' too — but we can prove multi-stage ran
    # by checking the recorded tool_name: single-shot would record "emit",
    # multi-stage records "finish".
    assert [t.tool_name for t in result.trace.turns] == ["finish"]
    assert ctx.finished is True


def test_multi_stage_loop_logs_per_turn_when_label_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``log_label='X'`` should produce one prompt+response log per turn with multi-stage suffix."""
    from unittest.mock import patch

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _force_no_tool_calling(monkeypatch)

    plans = [
        MultiStagePlan(reasoning="r1", next_call=_CallEmit(value="x")),
        MultiStagePlan(reasoning="r2", next_call=_CallFinish()),
    ]
    client = _scripted_client(monkeypatch, plans)
    ctx = _Ctx()
    reg = _make_registry(ctx)

    with (
        patch(
            "reflexio.server.services.service_utils.log_llm_messages"
        ) as mock_log_msgs,
        patch(
            "reflexio.server.services.service_utils.log_model_response"
        ) as mock_log_resp,
    ):
        run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "go"}],
            registry=reg,
            model_role=ModelRole.EXTRACTION_AGENT,
            ctx=ctx,
            finish_tool_name="finish",
            multi_stage_schema=MultiStagePlan,
            log_label="search_agent",
        )

    assert mock_log_msgs.call_count == 2
    assert mock_log_resp.call_count == 2
    msg_labels = [c.args[1] for c in mock_log_msgs.call_args_list]
    assert msg_labels == [
        "search_agent (multi-stage turn 1)",
        "search_agent (multi-stage turn 2)",
    ]
