import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import (
    Tool,
    ToolLoopResult,  # noqa: F401
    ToolLoopTrace,  # noqa: F401
    ToolRegistry,
    run_tool_loop,
)


class EmitProfileArgs(BaseModel):
    """Emit a candidate user profile item."""

    content: str
    time_to_live: str


class Ctx:
    def __init__(self):
        self.calls = []
        self.finished = False

    def emit(self, args, ctx):
        self.calls.append(args)
        return {"ok": True}


def test_tool_openai_spec_uses_docstring_and_schema():
    t = Tool(name="emit_profile", args_model=EmitProfileArgs, handler=lambda _a, _c: {})
    spec = t.openai_spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "emit_profile"
    assert "Emit a candidate user profile item." in spec["function"]["description"]
    assert spec["function"]["parameters"]["properties"]["content"]["type"] == "string"


def test_registry_handle_parses_and_dispatches():
    ctx = Ctx()
    t = Tool(name="emit_profile", args_model=EmitProfileArgs, handler=ctx.emit)
    reg = ToolRegistry()
    reg.register(t)
    result = reg.handle(
        "emit_profile", json.dumps({"content": "hi", "time_to_live": "persistent"}), ctx
    )
    assert result == {"ok": True}
    assert ctx.calls[0].content == "hi"


def test_registry_handle_converts_validation_error_to_tool_error():
    ctx = Ctx()
    reg = ToolRegistry()
    reg.register(
        Tool(name="emit_profile", args_model=EmitProfileArgs, handler=ctx.emit)
    )
    # Missing required field.
    result = reg.handle("emit_profile", json.dumps({"content": "hi"}), ctx)
    assert "error" in result
    assert "time_to_live" in result["error"]
    assert ctx.calls == []


def test_registry_rejects_unknown_tool():
    reg = ToolRegistry()
    result = reg.handle("not_a_tool", "{}", None)
    assert "error" in result
    assert "unknown tool" in result["error"].lower()


def test_openai_specs_lists_all_registered_tools():
    reg = ToolRegistry()
    reg.register(Tool(name="a", args_model=EmitProfileArgs, handler=lambda *_: {}))
    reg.register(Tool(name="b", args_model=EmitProfileArgs, handler=lambda *_: {}))
    specs = reg.openai_specs()
    assert {s["function"]["name"] for s in specs} == {"a", "b"}


def test_mock_tool_call_response_shape(tool_call_completion):
    make_tc, make_stop = tool_call_completion
    r = make_tc("emit_profile", {"content": "x"})
    assert r.choices[0].finish_reason == "tool_calls"
    assert r.choices[0].message.tool_calls[0].function.name == "emit_profile"
    s = make_stop()
    assert s.choices[0].finish_reason == "stop"
    assert s.choices[0].message.tool_calls is None


# ---------------------------------------------------------------------------
# run_tool_loop tests
# ---------------------------------------------------------------------------


class EmitArgs(BaseModel):
    """Emit a value."""

    value: str


class LoopCtx:
    """Simple mutable context for tool-loop tests."""

    def __init__(self):
        self.emitted: list[str] = []
        self.finished: bool = False


def _make_registry(ctx: LoopCtx) -> ToolRegistry:
    """Build a registry with 'emit' and 'finish' tools that mutate *ctx*."""

    def _emit_handler(args: BaseModel, c: LoopCtx) -> dict:
        c.emitted.append(args.value)  # type: ignore[attr-defined]
        return {"ok": True}

    def _finish_handler(args: BaseModel, c: LoopCtx) -> dict:
        c.finished = True
        return {"done": True}

    class FinishArgs(BaseModel):
        """Signal that extraction is complete."""

    reg = ToolRegistry()
    reg.register(Tool(name="emit", args_model=EmitArgs, handler=_emit_handler))
    reg.register(Tool(name="finish", args_model=FinishArgs, handler=_finish_handler))
    return reg


def test_run_tool_loop_drives_multiple_turns_until_finish(
    monkeypatch, tool_call_completion
):
    """Three LLM turns (emit, emit, finish) should yield finished_reason='finish_tool'."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    make_tc, _make_stop = tool_call_completion
    responses = [
        make_tc("emit", {"value": "alpha"}),
        make_tc("emit", {"value": "beta"}),
        make_tc("finish", {}),
    ]

    config = LiteLLMConfig(model="claude-sonnet-4-6")
    client = LiteLLMClient(config)
    ctx = LoopCtx()
    registry = _make_registry(ctx)

    with patch("litellm.completion", side_effect=responses):
        result = run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "go"}],
            registry=registry,
            model_role=ModelRole.ANGLE_READER,
            ctx=ctx,
        )

    assert result.finished_reason == "finish_tool"
    assert result.trace.finished is True
    assert len(result.trace.turns) == 3
    assert ctx.emitted == ["alpha", "beta"]
    assert ctx.finished is True


def test_run_tool_loop_honours_max_steps(monkeypatch, tool_call_completion):
    """With max_steps=3 and unlimited emit responses, the loop caps at 3 turns."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    make_tc, _make_stop = tool_call_completion
    # Supply more responses than max_steps so we are cap-limited, not response-limited.
    responses = [make_tc("emit", {"value": f"item-{i}"}) for i in range(10)]

    config = LiteLLMConfig(model="claude-sonnet-4-6")
    client = LiteLLMClient(config)
    ctx = LoopCtx()
    registry = _make_registry(ctx)

    with patch("litellm.completion", side_effect=responses):
        result = run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "go"}],
            registry=registry,
            model_role=ModelRole.ANGLE_READER,
            max_steps=3,
            ctx=ctx,
        )

    assert result.finished_reason == "max_steps"
    assert len(ctx.emitted) == 3


def test_run_tool_loop_capability_fallback_uses_response_format(monkeypatch):
    """When supports_tool_calling is False, generate_chat_response uses response_format."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    from reflexio.server.llm import tools as tools_mod

    monkeypatch.setattr(tools_mod, "supports_tool_calling", lambda _model: False)

    config = LiteLLMConfig(model="some-legacy-model")
    client = LiteLLMClient(config)

    class FallbackSchema(BaseModel):
        emissions: list[EmitArgs]

    fake_parsed = FallbackSchema(emissions=[EmitArgs(value="x"), EmitArgs(value="y")])
    monkeypatch.setattr(client, "generate_chat_response", lambda **_: fake_parsed)

    ctx = LoopCtx()
    registry = _make_registry(ctx)

    result = run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "go"}],
        registry=registry,
        model_role=ModelRole.ANGLE_READER,
        fallback_schema=FallbackSchema,
        fallback_tool_name="emit",
        ctx=ctx,
    )

    assert result.finished_reason == "finish_tool"
    assert result.trace.finished is True
    assert len(result.trace.turns) == 2
    assert ctx.emitted == ["x", "y"]


def test_run_tool_loop_returns_error_on_client_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When generate_chat_response raises, the loop returns finished_reason='error'."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    ctx = LoopCtx()  # reuse the helper class defined earlier in the test file

    def _emit_handler(args: BaseModel, c: LoopCtx) -> dict:
        c.emitted.append(args.value)  # type: ignore[attr-defined]
        return {"ok": True}

    reg = ToolRegistry([Tool(name="emit", args_model=EmitArgs, handler=_emit_handler)])

    config = LiteLLMConfig(model="claude-sonnet-4-6")
    client = LiteLLMClient(config)

    def boom(**_kwargs):
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(client, "generate_chat_response", boom)

    result = run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "go"}],
        registry=reg,
        model_role=ModelRole.ANGLE_READER,
        max_steps=5,
        ctx=ctx,
        finish_tool_name="finish",
    )

    assert result.finished_reason == "error"
    assert result.trace.finished is False
    assert result.trace.turns == []
