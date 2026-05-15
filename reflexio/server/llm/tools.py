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
    # Populated from the LLM response's ``usage`` object when available
    # (native tool-call mode). All None in capability-fallback mode and
    # when the provider doesn't report usage.
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


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


# Models we know support function calling per vendor docs but that litellm's
# model_cost registry hasn't catalogued yet. When litellm returns False
# (without raising) for a model whose name starts with one of these prefixes,
# treat that as a registry gap rather than an actual capability gap.
#
# Each entry must be justified by (a) the vendor docs and (b) a confirmed
# round-trip tool call against the live API. Update this list when litellm
# upstreams the registration so the override becomes redundant.
_TOOL_CALLING_OVERRIDES: tuple[str, ...] = (
    # https://platform.minimax.io/docs/guides/text-m2-function-call says
    # MiniMax-M2.7 supports tool use + interleaved thinking via OpenAI-compatible
    # tools format. Verified by a live `litellm.completion(model='minimax/MiniMax-M2.7',
    # tools=[...])` round-trip that returned a proper tool_call message.
    # litellm 1.80.x has 'minimax/MiniMax-M2' in model_cost but not 'MiniMax-M2.7'.
    "minimax/MiniMax-M2",
    # claude-code/* models route through our local CLI provider
    # (see providers/claude_code_provider.py). litellm has no registry
    # entry for them, so it returns False. The provider handles tool
    # calling explicitly by rendering tool specs into the system prompt
    # and parsing the model's JSON output back into ChatCompletionMessageToolCall
    # blocks. Verified end-to-end against the agentic ExtractionAgent loop.
    "claude-code/",
)


def supports_tool_calling(model: str) -> bool:
    """Return True when litellm reports native function-calling support.

    Wrapped so tests can monkeypatch the probe without touching litellm.
    On any internal error we optimistically assume support — cheaper to
    attempt a real call than to wrongly fall back. When litellm returns
    False (without raising) for a model in :data:`_TOOL_CALLING_OVERRIDES`,
    we override to True — see the constant for the rationale.

    Args:
        model (str): Fully-qualified model name.

    Returns:
        bool: True if litellm advertises function-calling for ``model``,
            or the model name matches a known-good override prefix.
    """
    try:
        import litellm

        if bool(litellm.supports_function_calling(model=model)):
            return True
        if any(model.startswith(prefix) for prefix in _TOOL_CALLING_OVERRIDES):
            logger.debug(
                "litellm.supports_function_calling returned False for %s; "
                "applying override (see _TOOL_CALLING_OVERRIDES)",
                model,
            )
            return True
        return False
    except Exception as e:
        logger.warning(
            "supports_function_calling probe failed for %s: %s: %s — assuming True",
            model,
            type(e).__name__,
            e,
        )
        return True


# Cap on tool-result payload size injected back into the message history
# in multi-stage mode. Without this, a single fat search response could
# blow the model's context window in two or three turns.
_MULTI_STAGE_RESULT_CHAR_CAP = 4000


def _serialize_tool_result_for_history(result: dict[str, Any]) -> str:
    """Render a tool result dict as a JSON string capped at a fixed size.

    Args:
        result (dict[str, Any]): The tool handler's return value.

    Returns:
        str: A JSON string truncated to ``_MULTI_STAGE_RESULT_CHAR_CAP``
            characters with a ``... [truncated]`` marker on overflow.
    """
    payload = json.dumps(result, default=str)
    if len(payload) <= _MULTI_STAGE_RESULT_CHAR_CAP:
        return payload
    return f"{payload[:_MULTI_STAGE_RESULT_CHAR_CAP]}... [truncated]"


def _run_multi_stage_fallback(
    *,
    client: LiteLLMClient,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    model_role: ModelRole,
    max_steps: int,
    ctx: Any,
    finish_tool_name: str,
    multi_stage_schema: type[BaseModel],
    log_label: str | None,
    trace: ToolLoopTrace,
) -> ToolLoopResult:
    """Drive a multi-turn tool loop using one structured-output call per turn.

    Used when the configured model lacks native tool-calling but the
    caller wants observe-decide-act semantics (e.g. the search agent on
    ``minimax/MiniMax-M2.7``). Each turn:

    1. Asks the model for a ``multi_stage_schema`` instance whose
       ``next_call`` field carries a discriminator literal naming the
       desired tool.
    2. Dispatches that call against the registry.
    3. Appends the agent's plan as an assistant message and the tool
       result as a user message, so the next turn's model call sees both.

    Loop terminates when ``next_call.tool == finish_tool_name`` or
    ``max_steps`` is exhausted.

    Args:
        client (LiteLLMClient): Configured client.
        messages (list[dict]): Seed message list; extended in place.
        registry (ToolRegistry): Tools exposed to the LLM.
        model_role (ModelRole): Role used to resolve the target model.
        max_steps (int): Cap on tool-calling turns.
        ctx (Any): Per-run context passed to each tool handler.
        finish_tool_name (str): Sentinel literal that ends the loop.
        multi_stage_schema (type[BaseModel]): Schema with a ``next_call``
            discriminated-union field.
        log_label (str | None): Optional llm_io.log label.
        trace (ToolLoopTrace): Trace to extend with per-turn entries.

    Returns:
        ToolLoopResult: ``ctx``, trace, and the terminator reason.
    """
    if log_label:
        from reflexio.server.services.service_utils import (
            log_llm_messages,
            log_model_response,
        )

    for turn_idx in range(max_steps):
        turn_label = f"(multi-stage turn {turn_idx + 1})"
        if log_label:
            log_llm_messages(logger, f"{log_label} {turn_label}", messages)
        tool_t0 = time.monotonic()
        parsed = client.generate_chat_response(
            messages=messages,
            response_format=multi_stage_schema,
            model_role=model_role,
        )
        if log_label:
            log_model_response(logger, f"{log_label} {turn_label}", parsed)
        if not isinstance(parsed, BaseModel):
            raise RuntimeError(
                f"Multi-stage structured call returned unexpected type {type(parsed)}"
            )

        next_call = getattr(parsed, "next_call", None)
        if next_call is None:
            raise RuntimeError(
                "Multi-stage schema must expose a 'next_call' field; "
                f"got {type(parsed).__name__}"
            )
        tool_name = getattr(next_call, "tool", None)
        if not isinstance(tool_name, str):
            raise RuntimeError(
                "Multi-stage next_call must carry a 'tool' discriminator literal; "
                f"got {type(next_call).__name__}"
            )

        reasoning = getattr(parsed, "reasoning", "") or ""
        args_dict = next_call.model_dump(exclude={"tool"})
        args_json = next_call.model_dump_json(exclude={"tool"})

        # Echo the agent's plan back into history so subsequent turns can
        # reason about what was tried already.
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Reasoning: {reasoning}\nNext call: {tool_name}({args_json})"
                ),
            }
        )

        if tool_name == finish_tool_name:
            # Dispatch finish through the registry so any ctx-side
            # bookkeeping (e.g. stashing the answer) still runs.
            result = registry.handle(tool_name, args_json, ctx)
            trace.turns.append(
                ToolLoopTurn(
                    tool_name=tool_name,
                    args=args_dict,
                    result=result,
                    latency_ms=int((time.monotonic() - tool_t0) * 1000),
                )
            )
            trace.finished = True
            return ToolLoopResult(ctx=ctx, trace=trace, finished_reason="finish_tool")

        result = registry.handle(tool_name, args_json, ctx)
        trace.turns.append(
            ToolLoopTurn(
                tool_name=tool_name,
                args=args_dict,
                result=result,
                latency_ms=int((time.monotonic() - tool_t0) * 1000),
            )
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Tool {tool_name} returned: "
                    f"{_serialize_tool_result_for_history(result)}"
                ),
            }
        )

    trace.finished = False
    return ToolLoopResult(ctx=ctx, trace=trace, finished_reason="max_steps")


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
    multi_stage_schema: type[BaseModel] | None = None,
    log_label: str | None = None,
) -> ToolLoopResult:
    """Drive an LLM through a tool-calling loop until ``finish_tool_name`` or ``max_steps``.

    For providers that lack native tool-calling there are two fallback
    modes (in priority order):

    1. **Multi-stage** (``multi_stage_schema`` set): one structured-output
       call per turn whose parsed schema carries a ``next_call``
       discriminated-union. The server dispatches ``next_call`` against
       the registry, appends the result to the message history, and asks
       for the next turn — preserving observe-decide-act semantics.
    2. **Single-shot** (``fallback_schema`` + ``fallback_tool_name``):
       one structured-output call whose parsed list is converted into
       synthetic tool calls dispatched against ``fallback_tool_name``.
       All calls are planned upfront so the agent never observes any
       tool result.

    Args:
        client (LiteLLMClient): Configured client — ``generate_chat_response``
            is invoked with ``tools=`` in native mode and with
            ``response_format=`` in either fallback mode.
        messages (list[dict]): Seed message list; extended in place per turn.
        registry (ToolRegistry): Tools exposed to the LLM.
        model_role (ModelRole): Role used to resolve the target model.
        max_steps (int): Cap on tool-calling turns.
        ctx (Any): Caller-supplied context object passed to each tool handler.
        finish_tool_name (str): Name of the sentinel tool that terminates the loop.
        fallback_schema (type[BaseModel] | None): Pydantic schema for the
            single-shot fallback path. Used only if ``multi_stage_schema``
            is None.
        fallback_tool_name (str | None): Name of the tool each single-shot
            fallback item is dispatched against.
        multi_stage_schema (type[BaseModel] | None): Pydantic schema for
            the multi-stage fallback path. The schema must expose a
            ``next_call`` field whose value is a Pydantic model carrying a
            ``tool`` discriminator literal — that literal names the tool
            to dispatch, all other fields become its args. Takes priority
            over ``fallback_schema``.
        log_label (str | None): When set, each LLM call in the loop is
            mirrored into ``~/.reflexio/logs/llm_io.log`` using this label
            (suffixed with ``(turn N)``, ``(fallback)``, or
            ``(multi-stage turn N)``). Matches classic per-call logging
            parity. Leave unset (default) to suppress file-level logging
            for tool-loop callers like unit tests.

    Returns:
        ToolLoopResult: ``ctx``, trace, and the terminator reason.

    Raises:
        RuntimeError: If the model lacks tool-calling AND no fallback
            (multi-stage or single-shot) is provided.
    """
    model = resolve_model_name(
        role=model_role,
        site_var_value=None,
        config_override=None,
        api_key_config=getattr(client.config, "api_key_config", None),
    )
    trace = ToolLoopTrace()

    # Lazily import the llm_io helpers only when logging is requested —
    # matches classic's per-call lazy-import pattern in profile_deduplicator.py.
    if log_label:
        from reflexio.server.services.service_utils import (
            log_llm_messages,
            log_model_response,
        )

    # ---- Capability fallback ------------------------------------------
    if not supports_tool_calling(model):
        if multi_stage_schema is not None:
            return _run_multi_stage_fallback(
                client=client,
                messages=messages,
                registry=registry,
                model_role=model_role,
                max_steps=max_steps,
                ctx=ctx,
                finish_tool_name=finish_tool_name,
                multi_stage_schema=multi_stage_schema,
                log_label=log_label,
                trace=trace,
            )
        if fallback_schema is None or fallback_tool_name is None:
            raise RuntimeError(
                f"Model {model} lacks tool-calling and no fallback_schema provided"
            )
        if log_label:
            log_llm_messages(logger, f"{log_label} (fallback)", messages)
        parsed = client.generate_chat_response(
            messages=messages,
            response_format=fallback_schema,
            model_role=model_role,
        )
        if log_label:
            log_model_response(logger, f"{log_label} (fallback)", parsed)
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
        # Respect the configured max_steps budget even on the fallback path
        # — otherwise a non-tool-calling provider could blow past the loop
        # cap when the structured response includes more items than expected.
        bounded_items = items[:max_steps]
        for item in bounded_items:
            tool_t0 = time.monotonic()
            res = registry.handle(fallback_tool_name, item.model_dump_json(), ctx)
            trace.turns.append(
                ToolLoopTurn(
                    tool_name=fallback_tool_name,
                    args=item.model_dump(),
                    result=res,
                    latency_ms=int((time.monotonic() - tool_t0) * 1000),
                )
            )
        exceeded = len(items) > max_steps
        trace.finished = not exceeded
        return ToolLoopResult(
            ctx=ctx,
            trace=trace,
            finished_reason="max_steps" if exceeded else "finish_tool",
        )

    # ---- Native tool loop ---------------------------------------------
    local_msgs = list(messages)
    try:
        for _step in range(max_steps):
            if log_label:
                log_llm_messages(logger, f"{log_label} (turn {_step + 1})", local_msgs)
            resp = client.generate_chat_response(
                messages=local_msgs,
                tools=registry.openai_specs(),
                tool_choice="auto",
                model_role=model_role,
            )
            if log_label:
                log_model_response(logger, f"{log_label} (turn {_step + 1})", resp)

            # Extract per-turn usage from the response (populated by LiteLLMClient
            # when the provider reports it; None otherwise).
            turn_usage = getattr(resp, "usage", None)
            turn_prompt_tokens = (
                getattr(turn_usage, "prompt_tokens", None) if turn_usage else None
            )
            turn_completion_tokens = (
                getattr(turn_usage, "completion_tokens", None) if turn_usage else None
            )
            turn_total_tokens = (
                getattr(turn_usage, "total_tokens", None) if turn_usage else None
            )
            turn_cost_usd = getattr(resp, "cost_usd", None)

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
            # A single response's usage is attached to every turn it produced —
            # the summary helpers dedup by (model, prompt_tokens, completion_tokens).
            for tc in tool_calls:
                # Time each tool individually — using the turn-start clock
                # would inflate later tools' latencies with model time and
                # earlier tools' work, masking the actual per-tool cost.
                tool_t0 = time.monotonic()
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
                        latency_ms=int((time.monotonic() - tool_t0) * 1000),
                        model=model,
                        prompt_tokens=turn_prompt_tokens,
                        completion_tokens=turn_completion_tokens,
                        total_tokens=turn_total_tokens,
                        cost_usd=turn_cost_usd,
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
