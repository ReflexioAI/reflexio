"""LLM-client adapters for the auto-publishing wrapper.

Each adapter maps one provider call shape into Reflexio ``InteractionData``:

- :class:`OpenAIChatAdapter` — OpenAI Chat Completions, litellm, OpenRouter
- :class:`OpenAIResponsesAdapter` — OpenAI Responses API
- :class:`AnthropicAdapter` — Anthropic Messages

Adapters are stateless and use duck typing (no hard import of ``openai`` /
``anthropic``) so the same adapter handles both the SDK object shape and the
dict shape that ``litellm`` sometimes returns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from reflexio.models.api_schema.common import ToolUsed
from reflexio.models.api_schema.domain.entities import InteractionData

logger = logging.getLogger(__name__)

# Exact casing is load-bearing: server-side collectors compare ``role == "User"``
# (base_generation_service) and ``role == "Assistant"`` (reflection/service).
ROLE_USER = "User"
ROLE_ASSISTANT = "Assistant"
# Tool-only assistant turns must carry non-empty content, otherwise the history
# renderer (``format_interactions_to_history_string``) drops the line and the
# tool usage is invisible to extraction.
TOOL_CALL_PLACEHOLDER = "(tool call)"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an object (attr) or a mapping (item), EAFP-friendly."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _maybe_json(raw: Any) -> Any:
    """Parse a JSON string tool-argument blob, falling back to the raw value."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw
    return raw


def _last_user_message_text(messages: Any) -> str | None:
    """Best-effort: text of the last ``role == "user"`` message in a list.

    Handles both plain-string content and content-block lists (Anthropic /
    OpenAI multimodal). Returns ``None`` when nothing usable is found.
    """
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if _get(msg, "role") != "user":
            continue
        content = _get(msg, "content")
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            return "".join(parts) or None
        return None
    return None


@dataclass
class LLMCallContext:
    """Everything an adapter (or hook) needs about a single intercepted call."""

    path: tuple[str, ...]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    adapter: LLMAdapter | None = None
    user_content: str | None = None
    source: str = ""
    user_id: str = ""
    session_id: str = ""
    agent_version: str = ""


class LLMAdapter(Protocol):
    """Maps a provider call shape into Reflexio interactions."""

    # Namespace path prefixes the proxy must keep walking even though they are
    # not the terminal callable, e.g. ``{("chat",), ("chat", "completions")}``.
    namespace_prefixes: frozenset[tuple[str, ...]]

    def is_completion_call(self, path: tuple[str, ...], attr: Any) -> bool: ...

    def build_interactions(
        self, ctx: LLMCallContext, response: Any
    ) -> list[InteractionData]: ...

    def assemble_stream(
        self, ctx: LLMCallContext, chunks: list[Any]
    ) -> list[InteractionData]: ...

    def default_user_content(self, ctx: LLMCallContext) -> str | None: ...


class BaseAdapter:
    """Shared turn-assembly so concrete adapters only describe their wire shape.

    Subclasses provide ``namespace_prefixes``, ``is_completion_call``,
    ``default_user_content``, and the two parsing hooks
    ``_assistant_from_response`` / ``_assistant_from_chunks`` which each return
    ``(text, tools)``.
    """

    namespace_prefixes: frozenset[tuple[str, ...]] = frozenset()

    def is_completion_call(self, path: tuple[str, ...], attr: Any) -> bool:
        raise NotImplementedError

    def default_user_content(self, ctx: LLMCallContext) -> str | None:
        return None

    def _assistant_from_response(
        self, response: Any
    ) -> tuple[str | None, list[ToolUsed]]:
        raise NotImplementedError

    def _assistant_from_chunks(
        self, chunks: list[Any]
    ) -> tuple[str | None, list[ToolUsed]]:
        raise NotImplementedError

    def build_interactions(
        self, ctx: LLMCallContext, response: Any
    ) -> list[InteractionData]:
        text, tools = self._assistant_from_response(response)
        return self._assemble(ctx, text, tools)

    def assemble_stream(
        self, ctx: LLMCallContext, chunks: list[Any]
    ) -> list[InteractionData]:
        text, tools = self._assistant_from_chunks(chunks)
        return self._assemble(ctx, text, tools)

    def _assemble(
        self, ctx: LLMCallContext, text: str | None, tools: list[ToolUsed]
    ) -> list[InteractionData]:
        interactions: list[InteractionData] = []
        if ctx.user_content and ctx.user_content.strip():
            interactions.append(
                InteractionData(role=ROLE_USER, content=ctx.user_content)
            )
        content = text or ""
        if not content and tools:
            content = TOOL_CALL_PLACEHOLDER
        # Drop a turn only when it carries neither content nor tools.
        if content or tools:
            interactions.append(
                InteractionData(role=ROLE_ASSISTANT, content=content, tools_used=tools)
            )
        return interactions


class OpenAIChatAdapter(BaseAdapter):
    """OpenAI Chat Completions, litellm ``completion``/``acompletion``, OpenRouter."""

    namespace_prefixes = frozenset({("chat",), ("chat", "completions")})

    def is_completion_call(self, path: tuple[str, ...], attr: Any) -> bool:
        if len(path) >= 3 and path[-3:] == ("chat", "completions", "create"):
            return True
        # litellm module-level callables: litellm.completion / litellm.acompletion
        return bool(path) and path[-1] in ("completion", "acompletion")

    def default_user_content(self, ctx: LLMCallContext) -> str | None:
        return _last_user_message_text(ctx.kwargs.get("messages"))

    def _message(self, response: Any) -> Any:
        choices = _get(response, "choices") or []
        if not choices:
            return None
        return _get(choices[0], "message")

    def _tools_from_message(self, msg: Any) -> list[ToolUsed]:
        tools: list[ToolUsed] = []
        for tc in _get(msg, "tool_calls") or []:
            fn = _get(tc, "function")
            tools.append(
                ToolUsed(
                    tool_name=_get(fn, "name") or "",
                    tool_data={"input": _maybe_json(_get(fn, "arguments"))},
                )
            )
        return tools

    def _assistant_from_response(
        self, response: Any
    ) -> tuple[str | None, list[ToolUsed]]:
        msg = self._message(response)
        return _get(msg, "content"), self._tools_from_message(msg)

    def _assistant_from_chunks(
        self, chunks: list[Any]
    ) -> tuple[str | None, list[ToolUsed]]:
        text_parts: list[str] = []
        tool_acc: dict[Any, dict[str, str]] = {}
        for chunk in chunks:
            choices = _get(chunk, "choices") or []
            if not choices:
                continue
            delta = _get(choices[0], "delta")
            piece = _get(delta, "content")
            if piece:
                text_parts.append(piece)
            for tc in _get(delta, "tool_calls") or []:
                idx = _get(tc, "index", 0)
                slot = tool_acc.setdefault(idx, {"name": "", "args": ""})
                fn = _get(tc, "function")
                name = _get(fn, "name")
                if name:
                    slot["name"] = name
                args = _get(fn, "arguments")
                if args:
                    slot["args"] += args
        tools = [
            ToolUsed(tool_name=v["name"], tool_data={"input": _maybe_json(v["args"])})
            for v in tool_acc.values()
            if v["name"] or v["args"]
        ]
        return ("".join(text_parts) or None), tools


class OpenAIResponsesAdapter(BaseAdapter):
    """OpenAI Responses API (``client.responses.create``)."""

    namespace_prefixes = frozenset({("responses",)})

    def is_completion_call(self, path: tuple[str, ...], attr: Any) -> bool:
        return len(path) >= 2 and path[-2:] in (
            ("responses", "create"),
            ("responses", "acreate"),
        )

    def default_user_content(self, ctx: LLMCallContext) -> str | None:
        inp = ctx.kwargs.get("input")
        if isinstance(inp, str):
            return inp or None
        return _last_user_message_text(inp)

    def _assistant_from_response(
        self, response: Any
    ) -> tuple[str | None, list[ToolUsed]]:
        text = _get(response, "output_text")
        tools: list[ToolUsed] = [
            ToolUsed(
                tool_name=_get(item, "name") or "",
                tool_data={"input": _maybe_json(_get(item, "arguments"))},
            )
            for item in _get(response, "output") or []
            if _get(item, "type") == "function_call"
        ]
        return text, tools

    def _assistant_from_chunks(
        self, chunks: list[Any]
    ) -> tuple[str | None, list[ToolUsed]]:
        text_parts: list[str] = []
        tool_acc: dict[Any, dict[str, str]] = {}
        for ev in chunks:
            etype = _get(ev, "type") or ""
            if etype == "response.output_text.delta":
                piece = _get(ev, "delta")
                if piece:
                    text_parts.append(piece)
            elif etype == "response.output_item.added":
                item = _get(ev, "item")
                if _get(item, "type") == "function_call":
                    key = _get(item, "id") or _get(ev, "output_index")
                    tool_acc.setdefault(
                        key, {"name": _get(item, "name") or "", "args": ""}
                    )
            elif etype == "response.function_call_arguments.delta":
                key = _get(ev, "item_id") or _get(ev, "output_index")
                slot = tool_acc.setdefault(key, {"name": "", "args": ""})
                piece = _get(ev, "delta")
                if piece:
                    slot["args"] += piece
        tools = [
            ToolUsed(tool_name=v["name"], tool_data={"input": _maybe_json(v["args"])})
            for v in tool_acc.values()
            if v["name"] or v["args"]
        ]
        return ("".join(text_parts) or None), tools


class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages API (``client.messages.create``)."""

    namespace_prefixes = frozenset({("messages",)})

    def is_completion_call(self, path: tuple[str, ...], attr: Any) -> bool:
        return len(path) >= 2 and path[-2:] in (
            ("messages", "create"),
            ("messages", "acreate"),
        )

    def default_user_content(self, ctx: LLMCallContext) -> str | None:
        return _last_user_message_text(ctx.kwargs.get("messages"))

    def _assistant_from_response(
        self, response: Any
    ) -> tuple[str | None, list[ToolUsed]]:
        text_parts: list[str] = []
        tools: list[ToolUsed] = []
        for block in _get(response, "content") or []:
            btype = _get(block, "type")
            if btype == "text":
                text_parts.append(_get(block, "text") or "")
            elif btype == "tool_use":
                tools.append(
                    ToolUsed(
                        tool_name=_get(block, "name") or "",
                        tool_data={"input": _get(block, "input")},
                    )
                )
        return ("".join(text_parts) or None), tools

    def _assistant_from_chunks(
        self, chunks: list[Any]
    ) -> tuple[str | None, list[ToolUsed]]:
        text_parts: list[str] = []
        tool_acc: dict[Any, dict[str, str]] = {}
        for ev in chunks:
            etype = _get(ev, "type")
            if etype == "content_block_start":
                block = _get(ev, "content_block")
                if _get(block, "type") == "tool_use":
                    tool_acc[_get(ev, "index")] = {
                        "name": _get(block, "name") or "",
                        "args": "",
                    }
            elif etype == "content_block_delta":
                delta = _get(ev, "delta")
                dtype = _get(delta, "type")
                if dtype == "text_delta":
                    piece = _get(delta, "text")
                    if piece:
                        text_parts.append(piece)
                elif dtype == "input_json_delta":
                    slot = tool_acc.get(_get(ev, "index"))
                    if slot is not None:
                        piece = _get(delta, "partial_json")
                        if piece:
                            slot["args"] += piece
        tools = [
            ToolUsed(tool_name=v["name"], tool_data={"input": _maybe_json(v["args"])})
            for v in tool_acc.values()
            if v["name"] or v["args"]
        ]
        return ("".join(text_parts) or None), tools


def default_adapters() -> list[LLMAdapter]:
    """The built-in adapters, all active and matched by call path."""
    return [OpenAIChatAdapter(), OpenAIResponsesAdapter(), AnthropicAdapter()]
