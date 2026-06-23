import json
import logging
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    ToolCallingChatResponse,
)
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import (
    AsyncAccepted,
    AsyncInfoTool,
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
    assert spec["function"]["strict"] is True
    assert spec["function"]["parameters"]["additionalProperties"] is False
    assert set(spec["function"]["parameters"]["required"]) == {
        "content",
        "time_to_live",
    }
    assert spec["function"]["parameters"]["properties"]["content"]["type"] == "string"


def test_openai_spec_guard_raises_on_unsafe_tool_arg_union():
    """Tool-arg schemas bypass the response_format path + the registry contract
    test, so ``openai_spec()`` runs the boundary guard. A plain-BaseModel
    discriminated-union args_model emits ``oneOf`` → the guard raises under pytest.
    """
    from typing import Annotated, Literal

    from pydantic import Field

    class _A(BaseModel):
        kind: Literal["a"] = "a"
        a: int

    class _B(BaseModel):
        kind: Literal["b"] = "b"
        b: str

    class _UnsafeArgs(BaseModel):
        choice: Annotated[_A | _B, Field(discriminator="kind")]

    t = Tool(name="unsafe", args_model=_UnsafeArgs, handler=lambda _a, _c: {})
    with pytest.raises(ValueError, match="provider-unsafe"):
        t.openai_spec()


def test_openai_spec_passes_for_strict_structured_output_tool_arg():
    """A tool-arg model inheriting StrictStructuredOutput is provider-safe by
    construction — ``openai_spec()`` does not raise even with a union."""
    from typing import Annotated, Literal

    from pydantic import Field

    from reflexio.models.structured_output import StrictStructuredOutput

    class _A(BaseModel):
        kind: Literal["a"] = "a"
        a: int

    class _B(BaseModel):
        kind: Literal["b"] = "b"
        b: str

    class _SafeArgs(StrictStructuredOutput):
        choice: Annotated[_A | _B, Field(discriminator="kind")]

    t = Tool(name="safe", args_model=_SafeArgs, handler=lambda _a, _c: {})
    spec = t.openai_spec()  # must not raise
    assert spec["function"]["name"] == "safe"


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


def test_registry_handle_unwraps_async_accepted_result():
    """AsyncAccepted remains a normal tool result for existing callers."""

    class AskArgs(BaseModel):
        """Ask for missing information."""

        question: str

    reg = ToolRegistry()
    reg.register(
        AsyncInfoTool(
            name="ask_human",
            args_model=AskArgs,
            handler=lambda _a, _c: AsyncAccepted(
                pending_tool_call_id="ptc_1",
                result={
                    "status": "request_pending",
                    "pending_tool_call_id": "ptc_1",
                },
            ),
        )
    )

    assert reg.handle("ask_human", json.dumps({"question": "Target?"}), None) == {
        "status": "request_pending",
        "pending_tool_call_id": "ptc_1",
    }


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
            model_role=ModelRole.EXTRACTION_AGENT,
            ctx=ctx,
        )

    assert result.finished_reason == "finish_tool"
    assert result.trace.finished is True
    assert len(result.trace.turns) == 3
    assert ctx.emitted == ["alpha", "beta"]
    assert ctx.finished is True


def test_run_tool_loop_records_async_accepted_and_continues(
    monkeypatch,
    tool_call_completion,
):
    """Async info tools return a tool message and do not stop the loop."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    class AskArgs(BaseModel):
        """Ask for missing information."""

        question: str

    class FinishArgs(BaseModel):
        """Signal that extraction is complete."""

    make_tc, _make_stop = tool_call_completion
    responses = [
        make_tc("ask_human", {"question": "What deployment target?"}),
        make_tc("finish", {}),
    ]

    config = LiteLLMConfig(model="claude-sonnet-4-6")
    client = LiteLLMClient(config)
    ctx = LoopCtx()
    registry = ToolRegistry()
    registry.register(
        AsyncInfoTool(
            name="ask_human",
            args_model=AskArgs,
            handler=lambda _a, _c: AsyncAccepted(
                pending_tool_call_id="ptc_1",
                result={
                    "status": "request_pending",
                    "pending_tool_call_id": "ptc_1",
                    "instruction": "Continue with available evidence.",
                },
            ),
        )
    )
    registry.register(
        Tool(
            name="finish",
            args_model=FinishArgs,
            handler=lambda _a, c: setattr(c, "finished", True) or {"done": True},
        )
    )

    with patch("litellm.completion", side_effect=responses):
        result = run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "go"}],
            registry=registry,
            model_role=ModelRole.EXTRACTION_AGENT,
            ctx=ctx,
        )

    assert result.finished_reason == "finish_tool"
    assert result.pending_tool_call_ids == ["ptc_1"]
    assert ctx.finished is True
    assert [turn.tool_name for turn in result.trace.turns] == ["ask_human", "finish"]
    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert "ptc_1" in tool_messages[0]["content"]


def test_run_tool_loop_sends_plain_dict_tool_calls_in_followup_request(monkeypatch):
    """Provider tool-call objects should not be reused in the next request."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    emit_call = MagicMock()
    emit_call.id = "call_emit"
    emit_call.type = "function"
    emit_call.function = MagicMock()
    emit_call.function.name = "emit"
    emit_call.function.arguments = json.dumps({"value": "hello"})

    finish_call = MagicMock()
    finish_call.id = "call_finish"
    finish_call.type = "function"
    finish_call.function = MagicMock()
    finish_call.function.name = "finish"
    finish_call.function.arguments = "{}"

    calls: list[list[dict]] = []

    def fake_generate_chat_response(**kwargs):
        calls.append(deepcopy(kwargs["messages"]))
        tool_calls = [emit_call] if len(calls) == 1 else [finish_call]
        return ToolCallingChatResponse(
            content=None,
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )

    config = LiteLLMConfig(model="claude-sonnet-4-6")
    client = LiteLLMClient(config)
    monkeypatch.setattr(client, "generate_chat_response", fake_generate_chat_response)
    ctx = LoopCtx()

    result = run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "go"}],
        registry=_make_registry(ctx),
        model_role=ModelRole.EXTRACTION_AGENT,
        ctx=ctx,
    )

    assert result.finished_reason == "finish_tool"
    assistant_history = calls[1][-2]
    assert assistant_history["role"] == "assistant"
    assert assistant_history["tool_calls"] == [
        {
            "id": "call_emit",
            "type": "function",
            "function": {
                "name": "emit",
                "arguments": json.dumps({"value": "hello"}),
            },
        }
    ]
    assert assistant_history["tool_calls"][0] is not emit_call


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
            model_role=ModelRole.EXTRACTION_AGENT,
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
        model_role=ModelRole.EXTRACTION_AGENT,
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
        model_role=ModelRole.EXTRACTION_AGENT,
        max_steps=5,
        ctx=ctx,
        finish_tool_name="finish",
    )

    assert result.finished_reason == "error"
    assert result.trace.finished is False
    assert result.trace.turns == []


def test_run_tool_loop_logs_llm_client_error_as_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LiteLLMClientError (timeouts, provider errors after retries) is a known
    failure mode: finished_reason='error' but logged at WARNING, not ERROR."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    ctx = LoopCtx()

    def _emit_handler(args: BaseModel, c: LoopCtx) -> dict:
        c.emitted.append(args.value)  # type: ignore[attr-defined]
        return {"ok": True}

    reg = ToolRegistry([Tool(name="emit", args_model=EmitArgs, handler=_emit_handler)])

    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))

    def boom(**_kwargs):
        raise LiteLLMClientError("API call failed: hard timeout")

    monkeypatch.setattr(client, "generate_chat_response", boom)

    with caplog.at_level(logging.WARNING, logger="reflexio.server.llm.tools"):
        result = run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "go"}],
            registry=reg,
            model_role=ModelRole.EXTRACTION_AGENT,
            max_steps=5,
            ctx=ctx,
            finish_tool_name="finish",
        )

    assert result.finished_reason == "error"
    assert result.trace.finished is False
    tool_loop_records = [
        r for r in caplog.records if r.name == "reflexio.server.llm.tools"
    ]
    assert any("tool_loop_llm_error" in r.getMessage() for r in tool_loop_records)
    assert all(r.levelno < logging.ERROR for r in tool_loop_records)


# ---------------- log_label (llm_io.log) integration ---------------- #


def test_run_tool_loop_log_label_none_does_not_invoke_llm_io_helpers(
    monkeypatch, tool_call_completion
):
    """Default log_label=None → zero calls to log_llm_messages / log_model_response."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    make_tc, _ = tool_call_completion
    responses = [make_tc("finish", {})]
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    ctx = LoopCtx()
    registry = _make_registry(ctx)

    with (
        patch(
            "reflexio.server.services.service_utils.log_llm_messages"
        ) as mock_log_msgs,
        patch(
            "reflexio.server.services.service_utils.log_model_response"
        ) as mock_log_resp,
        patch("litellm.completion", side_effect=responses),
    ):
        run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "go"}],
            registry=registry,
            model_role=ModelRole.EXTRACTION_AGENT,
            ctx=ctx,
        )

    mock_log_msgs.assert_not_called()
    mock_log_resp.assert_not_called()


def test_run_tool_loop_log_label_native_path_logs_each_turn(
    monkeypatch, tool_call_completion
):
    """log_label='X' → one log_llm_messages + one log_model_response per native turn.

    Across 2 turns, we expect:
      - 2 prompt log entries labelled "X (turn 1)" and "X (turn 2)"
      - 2 response log entries with matching labels
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    make_tc, _ = tool_call_completion
    responses = [make_tc("emit", {"value": "a"}), make_tc("finish", {})]
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    ctx = LoopCtx()
    registry = _make_registry(ctx)

    with (
        patch(
            "reflexio.server.services.service_utils.log_llm_messages"
        ) as mock_log_msgs,
        patch(
            "reflexio.server.services.service_utils.log_model_response"
        ) as mock_log_resp,
        patch("litellm.completion", side_effect=responses),
    ):
        run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "go"}],
            registry=registry,
            model_role=ModelRole.EXTRACTION_AGENT,
            ctx=ctx,
            log_label="profile_reader_facts",
        )

    assert mock_log_msgs.call_count == 2
    assert mock_log_resp.call_count == 2
    # Label suffixes increment per turn
    msg_labels = [c.args[1] for c in mock_log_msgs.call_args_list]
    resp_labels = [c.args[1] for c in mock_log_resp.call_args_list]
    assert msg_labels == [
        "profile_reader_facts (turn 1)",
        "profile_reader_facts (turn 2)",
    ]
    assert resp_labels == [
        "profile_reader_facts (turn 1)",
        "profile_reader_facts (turn 2)",
    ]


def test_run_tool_loop_log_label_fallback_path_logs_once(monkeypatch):
    """Capability-fallback path logs exactly one prompt + one response with '(fallback)' suffix."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    # Force capability-fallback path
    monkeypatch.setattr(
        "reflexio.server.llm.tools.supports_tool_calling", lambda _model: False
    )

    class EmitListSchema(BaseModel):
        items: list[EmitArgs] = []

    class FinishArgs(BaseModel):
        """Signal end."""

    reg = ToolRegistry()
    ctx = LoopCtx()

    def _emit(args: BaseModel, c: LoopCtx) -> dict:
        c.emitted.append(args.value)  # type: ignore[attr-defined]
        return {"ok": True}

    reg.register(Tool(name="emit", args_model=EmitArgs, handler=_emit))
    reg.register(
        Tool(
            name="finish",
            args_model=FinishArgs,
            handler=lambda _a, _c: {"done": True},
        )
    )

    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    parsed = EmitListSchema(items=[EmitArgs(value="a"), EmitArgs(value="b")])

    with (
        patch(
            "reflexio.server.services.service_utils.log_llm_messages"
        ) as mock_log_msgs,
        patch(
            "reflexio.server.services.service_utils.log_model_response"
        ) as mock_log_resp,
        patch.object(client, "generate_chat_response", return_value=parsed),
    ):
        run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "go"}],
            registry=reg,
            model_role=ModelRole.EXTRACTION_AGENT,
            ctx=ctx,
            fallback_schema=EmitListSchema,
            fallback_tool_name="emit",
            log_label="profile_reader_facts",
        )

    assert mock_log_msgs.call_count == 1
    assert mock_log_resp.call_count == 1
    assert mock_log_msgs.call_args.args[1] == "profile_reader_facts (fallback)"
    assert mock_log_resp.call_args.args[1] == "profile_reader_facts (fallback)"


# ---------------------------------------------------------------------------
# ToolLoopTurn usage field tests
# ---------------------------------------------------------------------------


def test_run_tool_loop_captures_usage_on_tool_loop_turn(monkeypatch):
    """Each ToolLoopTurn should carry prompt/completion/total tokens, model name,
    and cost_usd when the ToolCallingChatResponse carries a usage object."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)

    # Build a fake usage object.
    fake_usage = MagicMock()
    fake_usage.prompt_tokens = 100
    fake_usage.completion_tokens = 50
    fake_usage.total_tokens = 150

    # Build scripted ToolCallingChatResponse objects (one tool call, then finish).
    tc = MagicMock()
    tc.id = "tc_emit"
    tc.function = MagicMock()
    tc.function.name = "emit"
    tc.function.arguments = json.dumps({"value": "hello"})

    resp_with_usage = ToolCallingChatResponse(
        content=None,
        tool_calls=[tc],
        finish_reason="tool_calls",
        usage=fake_usage,
        cost_usd=0.002,
    )
    resp_finish = ToolCallingChatResponse(
        content=None,
        tool_calls=None,
        finish_reason="stop",
        usage=None,
        cost_usd=None,
    )

    config = LiteLLMConfig(model="claude-sonnet-4-6")
    client = LiteLLMClient(config)
    ctx = LoopCtx()
    registry = _make_registry(ctx)

    monkeypatch.setattr(
        client,
        "generate_chat_response",
        MagicMock(side_effect=[resp_with_usage, resp_finish]),
    )

    result = run_tool_loop(
        client=client,
        messages=[{"role": "user", "content": "go"}],
        registry=registry,
        model_role=ModelRole.EXTRACTION_AGENT,
        ctx=ctx,
    )

    # The terminal turn carried no tool calls, so the loop reports the distinct
    # "no_tool_call" reason (the finish handler never ran).
    assert result.finished_reason == "no_tool_call"
    assert len(result.trace.turns) == 1
    turn = result.trace.turns[0]
    assert turn.prompt_tokens == 100
    assert turn.completion_tokens == 50
    assert turn.total_tokens == 150
    assert turn.cost_usd == pytest.approx(0.002)
    # model field is populated from the resolved model name (non-None)
    assert turn.model is not None


class TestSupportsToolCallingOverrides:
    """Verify ``supports_tool_calling`` overrides litellm's False for models
    we know support function calling per vendor docs.

    Surfaced when litellm 1.80.x's model_cost registry had
    ``minimax/MiniMax-M2`` (with tool support) but not ``MiniMax-M2.7``,
    even though MiniMax's vendor docs explicitly say M2.7 supports tools
    and a live tool call round-trip succeeded.
    """

    def test_litellm_true_returns_true(self, monkeypatch):
        """Happy path: litellm says True, function returns True."""
        from reflexio.server.llm import tools as tools_mod

        monkeypatch.setattr(
            "litellm.supports_function_calling",
            lambda model: True,  # noqa: ARG005
        )
        assert tools_mod.supports_tool_calling("openai/gpt-5.4-mini") is True

    def test_litellm_false_unknown_model_returns_false(self, monkeypatch):
        """litellm says False for a model not in the override list — return False."""
        from reflexio.server.llm import tools as tools_mod

        monkeypatch.setattr(
            "litellm.supports_function_calling",
            lambda model: False,  # noqa: ARG005
        )
        assert (
            tools_mod.supports_tool_calling("some-random/model-without-tools") is False
        )

    def test_litellm_false_minimax_m2_overrides_to_true(self, monkeypatch):
        """litellm says False for minimax/MiniMax-M2.7 (registry gap), but our
        override says True — confirmed by vendor docs + live round-trip."""
        from reflexio.server.llm import tools as tools_mod

        monkeypatch.setattr(
            "litellm.supports_function_calling",
            lambda model: False,  # noqa: ARG005
        )
        # M2.7 is the model name not registered in litellm's model_cost yet
        assert tools_mod.supports_tool_calling("minimax/MiniMax-M2.7") is True
        # Family override applies to all M2.x variants
        assert tools_mod.supports_tool_calling("minimax/MiniMax-M2") is True
        assert tools_mod.supports_tool_calling("minimax/MiniMax-M2-special") is True
        # M3 has its own override entry (same registry gap as the M2 family)
        assert tools_mod.supports_tool_calling("minimax/MiniMax-M3") is True

    def test_override_does_not_apply_to_other_minimax_models(self, monkeypatch):
        """The override is prefix-scoped: 'minimax/MiniMax-M2' applies to M2 family
        only, not e.g. abab6.5 or older models."""
        from reflexio.server.llm import tools as tools_mod

        monkeypatch.setattr(
            "litellm.supports_function_calling",
            lambda model: False,  # noqa: ARG005
        )
        assert tools_mod.supports_tool_calling("minimax/abab6.5-chat") is False

    def test_litellm_raises_returns_true(self, monkeypatch):
        """Existing behavior: any litellm exception → optimistically assume True."""
        from reflexio.server.llm import tools as tools_mod

        def boom(model):  # noqa: ARG001
            raise RuntimeError("litellm internal error")

        monkeypatch.setattr("litellm.supports_function_calling", boom)
        assert tools_mod.supports_tool_calling("any/model") is True
