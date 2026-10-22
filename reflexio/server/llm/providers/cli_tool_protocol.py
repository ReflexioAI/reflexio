"""Prompt-level tool calling for CLI-backed LiteLLM providers.

A local CLI (Claude Code, Codex, OpenClaw) takes one prompt and returns text;
it has no function-calling API. Reflexio's extraction agent is a tool loop, so
these providers render the tool specs into the prompt, ask for exactly one
``{"tool": ..., "args": ...}`` JSON object back, and parse it into a LiteLLM
tool call. The same goes for ``response_format``: the schema is written into
the prompt. Moved verbatim from the claude-code provider (#65) so the OpenClaw
provider shares one implementation instead of growing a second one.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from pydantic import BaseModel


def schema_instruction(response_format: Any) -> str | None:
    """Build a schema instruction to append to the system prompt.

    Accepts either a Pydantic model class, a LiteLLM ``json_schema``
    response_format dict, or a plain JSON-schema dict. Returns None
    when nothing usable is found — callers fall through to unstructured
    completion.

    Args:
        response_format: The response_format value from LiteLLM kwargs
            or ``optional_params`` — a Pydantic class or a dict.

    Returns:
        str | None: Instruction text to append to the system prompt,
            or None if no schema could be extracted.
    """
    schema = extract_json_schema(response_format)
    if not schema:
        return None
    return (
        "You MUST respond with a single JSON object that strictly matches "
        "the schema below. Output JSON only — no markdown fences, no prose, "
        "no explanation.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )


def extract_json_schema(response_format: Any) -> dict[str, Any] | None:
    """Extract a JSON schema from LiteLLM's response_format values.

    Args:
        response_format: Pydantic class, LiteLLM dict
            (``{"type": "json_schema", "json_schema": {"schema": ...}}``),
            or a raw JSON-schema dict.

    Returns:
        dict | None: The JSON schema, or None if one cannot be extracted.
    """
    if response_format is None:
        return None
    if inspect.isclass(response_format) and issubclass(response_format, BaseModel):
        return response_format.model_json_schema()
    if isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            inner = response_format.get("json_schema") or {}
            if isinstance(inner, dict):
                schema = inner.get("schema") or inner
                if isinstance(schema, dict):
                    return schema
        if "properties" in response_format or "$ref" in response_format:
            return response_format
    return None


def tool_call_attr(tc: Any, attr: str) -> str | None:
    """Read a tool-call field from either a LiteLLM object or a plain dict.

    ``tool_calls`` entries may be ``ChatCompletionMessageToolCall`` objects
    (each carrying a ``.function`` with ``.name`` / ``.arguments``) or
    raw dicts (``{"function": {"name": ..., "arguments": ...}}``). Walk
    both shapes.

    Args:
        tc: A single ``tool_calls`` entry (object or dict).
        attr: The attribute to read (``"name"`` or ``"arguments"``).

    Returns:
        str | None: The attribute's string value, or ``None`` if absent.
    """
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        value = fn.get(attr) if isinstance(fn, dict) else getattr(fn, attr, None)
    else:
        fn = getattr(tc, "function", None)
        value = getattr(fn, attr, None) if fn is not None else None
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value)


TOOL_USE_INSTRUCTION_TEMPLATE = (
    "## EXTERNAL TOOL-CALLING MODE\n"
    "\n"
    "You are running as a non-interactive subprocess driven by an external "
    "orchestrator. The orchestrator will execute tools on your behalf. You "
    "MUST NOT use Read, Edit, Write, Bash, Glob, Grep, TodoWrite, Task, or "
    "ANY of your built-in Claude Code tools. The tools listed below are "
    "NOT real callable functions in this session — they are the external "
    "tools the ORCHESTRATOR exposes, which YOU describe by emitting a "
    "structured JSON request as your final text response.\n"
    "\n"
    "Your response MUST be EXACTLY one JSON object on a single line, with "
    "this shape:\n"
    '{{"tool": "<tool_name>", "args": {{...}} }}\n'
    "\n"
    "Hard rules — failing any of these will break the orchestrator:\n"
    "1. Output ONLY the JSON object as plain text. No prose before or after.\n"
    "2. No markdown. No ```json``` code fences.\n"
    "3. Do NOT invoke any built-in tool. Do NOT search the codebase. Just "
    "   emit the JSON.\n"
    "4. ``tool`` must be exactly one of the names listed below.\n"
    "5. ``args`` must be a JSON object matching that tool's parameters.\n"
    "6. To terminate the orchestrator's loop, call the tool named "
    "``{finish}`` with the appropriate args.\n"
    "\n"
    "Available orchestrator tools (each described by name + params schema):\n"
    "{tool_specs}\n"
    "\n"
    "Now decide which tool to call and output ONLY the JSON object.\n"
)


def render_tools_instruction(tools: list[Any], finish_tool: str = "finish") -> str:
    """Render a LiteLLM tools spec into a system-prompt addendum.

    Args:
        tools: LiteLLM-style tools list. Each entry is a dict with
            ``{"type": "function", "function": {"name", "description", "parameters"}}``.
        finish_tool: The conventional finish tool name (echoed in the
            instructions so the model has a recognised termination signal).

    Returns:
        str: The system-prompt block to append.
    """
    lines: list[str] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        raw_fn = tool.get("function")
        fn: dict[str, Any] = raw_fn if isinstance(raw_fn, dict) else {}
        name = fn.get("name") or "?"
        desc = fn.get("description") or ""
        params = fn.get("parameters") or {}
        lines.append(
            f"- {name}: {desc}\n  parameters: {json.dumps(params, separators=(',', ':'))}"
        )
    tool_specs = "\n".join(lines) if lines else "(no tools)"
    return TOOL_USE_INSTRUCTION_TEMPLATE.format(
        finish=finish_tool, tool_specs=tool_specs
    )


def append_tools_instruction(system_prompt: str, tools: list[Any] | None) -> str:
    """Append the tool-use instructions when tools are present.

    Args:
        system_prompt: Existing system prompt (possibly empty).
        tools: LiteLLM-style tools list, or None/empty.

    Returns:
        str: The (possibly-augmented) system prompt.
    """
    if not tools:
        return system_prompt
    instruction = render_tools_instruction(tools)
    if system_prompt:
        return f"{system_prompt}\n\n{instruction}"
    return instruction


def parse_tool_use(text: str, tool_names: set[str]) -> dict[str, Any] | None:
    """Try to parse a ``{"tool": ..., "args": ...}`` block from claude's output.

    Walks candidate JSON substrings (plain, code-fenced, first balanced
    ``{...}``) in order, returning the first that parses to a dict with a
    recognised ``tool`` name and a dict ``args``.

    Args:
        text: Raw text from claude's ``-p --output-format json`` ``result``.
        tool_names: Set of tool names registered by the caller. The returned
            ``tool`` must be in this set.

    Returns:
        dict | None: ``{"name": str, "args": dict}`` on success; ``None`` if
            no valid tool-use JSON could be located.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    candidates: list[str] = [stripped]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    # Find the first balanced ``{...}`` object via JSONDecoder.raw_decode
    # instead of the greedy ``r"\{.*\}"`` regex. The greedy form swallows
    # any trailing ``{...}`` on the same line and silently misses a valid
    # tool-call when claude appends explanatory text after the JSON.
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(stripped[idx : idx + end])
            break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("tool")
        args = parsed.get("args")
        if name in tool_names and isinstance(args, dict):
            return {"name": name, "args": args}
    return None


def append_schema_instruction(system_prompt: str, response_format: Any) -> str:
    """Return *system_prompt* extended with a JSON-schema instruction when applicable.

    Args:
        system_prompt: Existing system prompt (possibly empty).
        response_format: The LiteLLM response_format value, if any.

    Returns:
        str: The (possibly-augmented) system prompt.
    """
    instruction = schema_instruction(response_format)
    if not instruction:
        return system_prompt
    if system_prompt:
        return f"{system_prompt}\n\n{instruction}"
    return instruction


def tool_names(tools: list[Any] | None) -> set[str]:
    """Return the function names declared in a LiteLLM tools list."""
    return {
        name
        for tool in tools or []
        if isinstance(tool, dict)
        and isinstance(name := (tool.get("function") or {}).get("name"), str)
    }
