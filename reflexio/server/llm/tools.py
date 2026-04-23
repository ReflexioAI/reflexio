"""Tool-calling primitives shared by agentic extraction and search pipelines."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)

from pydantic import BaseModel, ConfigDict, ValidationError

from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient


class Tool(BaseModel):
    """A single LLM-callable tool.

    Arguments are defined by a Pydantic model (its schema goes to the LLM,
    its docstring becomes the tool description). The handler takes a
    validated args instance plus a caller-supplied context object and
    returns a JSON-serialisable dict that is fed back as the tool result.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, Any], dict]

    def openai_spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (self.args_model.__doc__ or "").strip(),
                "parameters": self.args_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def openai_specs(self) -> list[dict]:
        return [t.openai_spec() for t in self._tools.values()]

    def handle(self, name: str, args_json: str, ctx: Any) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}
        try:
            raw = json.loads(args_json or "{}")
            args = tool.args_model.model_validate(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            return {"error": f"invalid args for {name}: {e}"}
        try:
            return tool.handler(args, ctx)
        except Exception as e:  # handler errors are recoverable tool-turn errors
            logger.exception("tool handler %s failed", name)
            return {"error": f"handler error: {type(e).__name__}"}


class ToolLoopTurn(BaseModel):
    """A single tool call turn in a tool-loop trace."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]
    latency_ms: int
    tokens: int | None = None


class ToolLoopTrace(BaseModel):
    """Full trace of a tool-loop execution."""

    turns: list[ToolLoopTurn] = []
    finished: bool = False


class ToolLoopResult(BaseModel):
    """Outcome of ``run_tool_loop``: final ``ctx``, trace, and terminator reason."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ctx: Any
    trace: ToolLoopTrace
    finished_reason: Literal["finish_tool", "max_steps", "error"]


def supports_tool_calling(model: str) -> bool:
    """Return True when litellm reports native function-calling support.

    Wrapped so tests can monkeypatch the probe without touching litellm.
    On any internal error we optimistically assume support — cheaper to
    attempt a real call than to wrongly fall back.

    Args:
        model (str): Fully-qualified model name.

    Returns:
        bool: True if litellm advertises function-calling for ``model``.
    """
    try:
        import litellm

        return bool(litellm.supports_function_calling(model=model))
    except Exception as e:
        logger.warning(
            "supports_function_calling probe failed for %s: %s: %s — assuming True",
            model,
            type(e).__name__,
            e,
        )
        return True


def run_tool_loop(
    client: LiteLLMClient,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    model_role: ModelRole,
    *,
    max_steps: int = 8,
    ctx: Any = None,
    finish_tool_name: str = "finish",
    fallback_schema: type[BaseModel] | None = None,
    fallback_tool_name: str | None = None,
) -> ToolLoopResult:
    """Drive an LLM through a tool-calling loop until ``finish_tool_name`` or ``max_steps``.

    For providers that lack native tool-calling, falls back to a single
    structured-output call whose parsed schema is converted into synthetic
    tool calls.

    Args:
        client (LiteLLMClient): Configured client — ``generate_chat_response``
            is invoked with ``tools=`` in native mode and with
            ``response_format=`` in fallback mode.
        messages (list[dict]): Seed message list; extended in place per turn.
        registry (ToolRegistry): Tools exposed to the LLM.
        model_role (ModelRole): Role used to resolve the target model.
        max_steps (int): Cap on tool-calling turns.
        ctx (Any): Caller-supplied context object passed to each tool handler.
        finish_tool_name (str): Name of the sentinel tool that terminates the loop.
        fallback_schema (type[BaseModel] | None): Pydantic schema for the
            capability-fallback path; required when tool-calling is unsupported.
        fallback_tool_name (str | None): Name of the tool each fallback item
            is dispatched against.

    Returns:
        ToolLoopResult: ``ctx``, trace, and the terminator reason.

    Raises:
        RuntimeError: If the model lacks tool-calling AND no fallback schema is provided.
    """
    model = resolve_model_name(
        role=model_role,
        site_var_value=None,
        config_override=None,
        api_key_config=getattr(client.config, "api_key_config", None),
    )
    trace = ToolLoopTrace()

    # ---- Capability fallback ------------------------------------------
    if not supports_tool_calling(model):
        if fallback_schema is None or fallback_tool_name is None:
            raise RuntimeError(
                f"Model {model} lacks tool-calling and no fallback_schema provided"
            )
        parsed = client.generate_chat_response(
            messages=messages,
            response_format=fallback_schema,
            model_role=model_role,
        )
        # The fallback path always passes response_format so the client
        # returns a parsed BaseModel instance. Narrow the type so pyright
        # can see model_fields is available.
        if not isinstance(parsed, BaseModel):
            raise RuntimeError(
                f"Fallback structured call returned unexpected type {type(parsed)}"
            )
        # Expect the schema's first field to be a list of items whose
        # ``model_dump_json()`` matches the fallback tool's args model.
        items = getattr(parsed, next(iter(type(parsed).model_fields)))
        for item in items:
            t0 = time.monotonic()
            res = registry.handle(fallback_tool_name, item.model_dump_json(), ctx)
            trace.turns.append(
                ToolLoopTurn(
                    tool_name=fallback_tool_name,
                    args=item.model_dump(),
                    result=res,
                    latency_ms=int((time.monotonic() - t0) * 1000),
                )
            )
        trace.finished = True
        return ToolLoopResult(ctx=ctx, trace=trace, finished_reason="finish_tool")

    # ---- Native tool loop ---------------------------------------------
    local_msgs = list(messages)
    try:
        for _step in range(max_steps):
            t0 = time.monotonic()
            resp = client.generate_chat_response(
                messages=local_msgs,
                tools=registry.openai_specs(),
                tool_choice="auto",
                model_role=model_role,
            )
            tool_calls = getattr(resp, "tool_calls", None)
            if not tool_calls:
                trace.finished = True
                return ToolLoopResult(
                    ctx=ctx, trace=trace, finished_reason="finish_tool"
                )
            # Emit ONE assistant message carrying ALL tool_calls from this turn.
            # OpenAI/Anthropic strict mode requires this shape.
            local_msgs.append(
                {"role": "assistant", "content": None, "tool_calls": list(tool_calls)}
            )
            # Process every tool call and append per-call tool result messages.
            for tc in tool_calls:
                name = tc.function.name
                args_json = tc.function.arguments
                result = registry.handle(name, args_json, ctx)
                try:
                    args_dict = json.loads(args_json or "{}")
                except json.JSONDecodeError:
                    args_dict = {}
                trace.turns.append(
                    ToolLoopTurn(
                        tool_name=name,
                        args=args_dict,
                        result=result,
                        latency_ms=int((time.monotonic() - t0) * 1000),
                    )
                )
                local_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )
            # After processing ALL tool calls, check whether the finish sentinel
            # appeared in this turn (may be alongside sibling calls).
            if any(tc.function.name == finish_tool_name for tc in tool_calls):
                trace.finished = True
                return ToolLoopResult(
                    ctx=ctx, trace=trace, finished_reason="finish_tool"
                )
    except Exception:
        logger.exception("Tool loop raised an unexpected exception")
        trace.finished = False
        return ToolLoopResult(ctx=ctx, trace=trace, finished_reason="error")

    return ToolLoopResult(ctx=ctx, trace=trace, finished_reason="max_steps")
