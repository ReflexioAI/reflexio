"""Claude Code CLI as a LiteLLM custom provider.

Routes ``litellm.completion(model="claude-code/...", ...)`` through the
user's locally-installed ``claude`` CLI (the Claude Code binary), so
reflexio's extractors can run with no external LLM API key — they reuse
whatever auth the user already has for Claude Code.

Activation is opt-in via ``CLAUDE_SMART_USE_LOCAL_CLI=1``. Without it,
the provider does not register and reflexio falls back to its normal
OpenAI/Anthropic/etc. provider priority.

Structured output: when callers pass a Pydantic ``response_format``,
the JSON schema is appended to the system prompt instructing the CLI
to reply with matching JSON. The CLI's text reply is returned as
``message.content``; ``LiteLLMClient._maybe_parse_structured_output``
then parses it into the Pydantic instance via the existing pipeline.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import shutil
import subprocess  # noqa: S404 — subprocess is the integration point; inputs are sanitised.
import tempfile
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)
from pydantic import BaseModel

from reflexio.server.llm.providers.claude_code_stream_parser import (
    ParseResult,
    classify_stall,
    parse_reset_estimate,
    parse_stream_json,
)

_LOGGER = logging.getLogger(__name__)

PROVIDER_KEY = "claude-code"
ENV_ENABLE = "CLAUDE_SMART_USE_LOCAL_CLI"
_ENV_CLI_PATH = "CLAUDE_SMART_CLI_PATH"
_ENV_HOST = "CLAUDE_SMART_HOST"
_ENV_CODEX_PATH = "CLAUDE_SMART_CODEX_PATH"
_ENV_TIMEOUT = "CLAUDE_SMART_CLI_TIMEOUT"
_ENV_MODEL = "CLAUDE_SMART_CLI_MODEL"
_HOST_CODEX = "codex"
_HOST_CLAUDE_CODE = "claude-code"
_CODEX_COMPAT_SCRIPT_NAMES = (
    ("codex-claude-compat.cmd", "codex-claude-compat")
    if os.name == "nt"
    else ("codex-claude-compat", "codex-claude-compat.cmd")
)
_CODEX_COMPAT_SCRIPT_NAME_SET = set(_CODEX_COMPAT_SCRIPT_NAMES)
_DEFAULT_TIMEOUT_SECONDS = 120
_DEFAULT_CLI_MODEL = "claude-sonnet-4-6"

_TRUTHY_ENV_VALUES = {"1", "true", "yes"}
_UNSUPPORTED_PARAMS_WARNED: set[str] = set()
_IMAGE_WARNED = False
_MULTITURN_WARNED = False


class ClaudeCodeCLIError(RuntimeError):
    """Raised when the claude CLI subprocess fails in a way we cannot recover from."""


def _env_enabled() -> bool:
    """Return True when ``CLAUDE_SMART_USE_LOCAL_CLI`` is set to a truthy value.

    Returns:
        bool: True if the opt-in env var is set, False otherwise.
    """
    raw = os.environ.get(ENV_ENABLE)
    return bool(raw) and raw.lower() in _TRUTHY_ENV_VALUES


def _host() -> str:
    """Return the host that owns this backend process."""
    return _HOST_CODEX if os.environ.get(_ENV_HOST) == _HOST_CODEX else _HOST_CLAUDE_CODE


def _cli_name() -> str:
    """Return the expected local CLI binary for the active host."""
    return "codex" if _host() == _HOST_CODEX else "claude"


def _candidate_codex_compat_path() -> Path | None:
    """Return the Codex compatibility wrapper from plugin roots, if present."""
    for env_var in ("PLUGIN_ROOT", "CLAUDE_PLUGIN_ROOT"):
        root = os.environ.get(env_var)
        if not root:
            continue
        for name in _CODEX_COMPAT_SCRIPT_NAMES:
            candidate = Path(root) / "scripts" / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def _resolve_cli_path() -> str | None:
    """Return the path to the active host CLI, or None if unavailable.

    Honours the ``CLAUDE_SMART_CLI_PATH`` override before falling back to
    host-specific defaults. Claude Code uses ``claude``. Codex prefers the
    compatibility wrapper shipped with the plugin, then falls back to
    ``codex`` directly.

    Returns:
        str | None: Absolute path to the CLI, or None if not found.
    """
    override = os.environ.get(_ENV_CLI_PATH)
    if override:
        candidate = Path(override)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        _LOGGER.warning(
            "%s=%s is not an executable file; falling back to PATH",
            _ENV_CLI_PATH,
            override,
        )
    if _host() == _HOST_CODEX:
        compat = _candidate_codex_compat_path()
        if compat is not None:
            return str(compat)
        return os.environ.get(_ENV_CODEX_PATH) or shutil.which("codex")
    return shutil.which("claude")


def is_claude_code_available() -> bool:
    """Return True when the local CLI provider is usable right now.

    Both the opt-in env var *and* a resolvable CLI path are required, so
    an unrelated env var can't silently redirect extraction traffic.

    Returns:
        bool: True iff ``CLAUDE_SMART_USE_LOCAL_CLI`` is truthy AND a
            host CLI is resolvable.
    """
    return _env_enabled() and _resolve_cli_path() is not None


def _flatten_content(content: Any) -> str:
    """Collapse LiteLLM content (string or content-block list) to plain text.

    Image blocks are silently skipped with a one-time WARN log (see
    ``_warn_image_dropped_once``). cache_control markers are ignored
    since the CLI does not accept them.

    Args:
        content: LiteLLM content — string, list of content blocks, or None.

    Returns:
        str: Plain-text content; empty string if no text survives.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in {"image", "image_url"}:
                    _warn_image_dropped_once()
                    continue
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _warn_image_dropped_once() -> None:
    """Emit a single WARN when image content is dropped by the CLI bridge.

    Returns:
        None
    """
    global _IMAGE_WARNED
    if not _IMAGE_WARNED:
        _LOGGER.warning(
            "claude-code provider: image content blocks are dropped — "
            "the CLI bridge accepts text only. Install an image-capable "
            "LLM provider for screenshot analysis."
        )
        _IMAGE_WARNED = True


def _warn_multiturn_once() -> None:
    """Emit a single WARN when multi-turn context gets flattened to text.

    Returns:
        None
    """
    global _MULTITURN_WARNED
    if not _MULTITURN_WARNED:
        _LOGGER.warning(
            "claude-code provider: multi-turn messages are flattened into a "
            "single 'User:/Assistant:' transcript. Quality may differ from "
            "the Anthropic messages API."
        )
        _MULTITURN_WARNED = True


def _warn_unsupported_param_once(name: str) -> None:
    """Emit a single WARN when a LiteLLM param has no CLI equivalent.

    Args:
        name: Parameter name that was ignored (e.g. ``"temperature"``).

    Returns:
        None
    """
    if name in _UNSUPPORTED_PARAMS_WARNED:
        return
    _UNSUPPORTED_PARAMS_WARNED.add(name)
    _LOGGER.warning(
        "claude-code provider: ignoring unsupported parameter %r — "
        "the CLI does not expose this control.",
        name,
    )


def _schema_instruction(response_format: Any) -> str | None:
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
    schema = _extract_json_schema(response_format)
    if not schema:
        return None
    return (
        "You MUST respond with a single JSON object that strictly matches "
        "the schema below. Output JSON only — no markdown fences, no prose, "
        "no explanation.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )


def _extract_json_schema(response_format: Any) -> dict[str, Any] | None:
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


def _split_system_and_dialogue(
    messages: list[dict[str, Any]],
) -> tuple[str, str]:
    """Split chat messages into (system_prompt, dialogue) for the CLI.

    The ``claude -p`` CLI takes one stdin prompt and an optional
    ``--append-system-prompt``. Multi-turn context is flattened into a
    single textual dialogue prefixed with role labels, since the CLI
    does not accept a messages array.

    System messages are merged (joined with blank lines) and returned
    separately for the ``--append-system-prompt`` flag. ``tool`` role
    messages are folded in as ``Tool:`` lines.

    Args:
        messages: LiteLLM-style chat messages.

    Returns:
        tuple[str, str]: ``(system_prompt, dialogue)``. Either may be empty.
    """
    systems: list[str] = []
    turns: list[str] = []
    non_system_roles = 0
    for msg in messages:
        role = msg.get("role", "user")
        content = _flatten_content(msg.get("content"))
        tool_calls = msg.get("tool_calls") if role == "assistant" else None
        if not content and not tool_calls:
            continue
        if role == "system":
            systems.append(content)
            continue
        non_system_roles += 1
        if role == "assistant":
            # When the assistant message carries tool_calls (content is
            # typically None), serialise them as breadcrumbs so the CLI's
            # next-turn context shows which tools were invoked. This is
            # required for multi-turn tool loops to converge.
            if tool_calls:
                breadcrumbs = []
                for tc in tool_calls:
                    name = _tool_call_attr(tc, "name") or "?"
                    args = _tool_call_attr(tc, "arguments") or "{}"
                    breadcrumbs.append(f"called {name} with {args}")
                prefix = "; ".join(breadcrumbs)
                if content:
                    turns.append(f"Assistant: {content}\n[tools: {prefix}]")
                else:
                    turns.append(f"Assistant: [tools: {prefix}]")
            else:
                turns.append(f"Assistant: {content}")
        elif role == "tool":
            tcid = msg.get("tool_call_id") or "?"
            turns.append(f"Tool[{tcid}]: {content}")
        else:
            turns.append(f"User: {content}")
    if non_system_roles > 1:
        _warn_multiturn_once()
    return "\n\n".join(systems), "\n\n".join(turns)


def _tool_call_attr(tc: Any, attr: str) -> str | None:
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


def _run_cli_stream(
    cli_path: str,
    system_prompt: str,
    dialogue: str,
    timeout_seconds: int,
) -> ParseResult:
    """Invoke the active host CLI and return a ParseResult.

    Args:
        cli_path (str): Path to the host executable or compatibility wrapper.
        system_prompt (str): Combined system prompt to append (may be empty).
        dialogue (str): Flattened user/assistant dialogue sent on stdin.
        timeout_seconds (int): Subprocess timeout.

    Returns:
        ParseResult: Aggregated state of the stream — success flag, terminal
            text, retry errors observed, and stderr.

    Raises:
        ClaudeCodeCLIError: On timeout or missing binary.
    """
    if _host() == _HOST_CODEX and Path(cli_path).name not in _CODEX_COMPAT_SCRIPT_NAME_SET:
        return _run_codex_stream(
            codex_path=cli_path,
            system_prompt=system_prompt,
            dialogue=dialogue,
            timeout_seconds=timeout_seconds,
        )
    return _run_claude_stream(
        cli_path=cli_path,
        system_prompt=system_prompt,
        dialogue=dialogue,
        timeout_seconds=timeout_seconds,
    )


def _run_claude_stream(
    *,
    cli_path: str,
    system_prompt: str,
    dialogue: str,
    timeout_seconds: int,
) -> ParseResult:
    """Invoke ``claude -p --output-format stream-json`` and return a ParseResult."""
    model = os.environ.get(_ENV_MODEL) or _DEFAULT_CLI_MODEL
    cmd = [
        cli_path,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--model",
        model,
    ]
    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    # Tag the child process so any hooks it fires (e.g. claude-smart's
    # Stop hook) can detect that this is a reflexio-internal invocation
    # and skip publishing — otherwise extractor system prompts get
    # re-published as user interactions and contaminate the corpus.
    #
    # CLAUDE_CODE_MAX_RETRIES=3 keeps short infrastructure blips tolerated
    # while bounding the worst-case stall to a few seconds before we
    # surface the failure to reflexio's stall_state table.
    env = os.environ.copy()
    env["CLAUDE_SMART_INTERNAL"] = "1"
    env["CLAUDE_CODE_MAX_RETRIES"] = "3"

    try:
        proc = subprocess.run(  # noqa: S603 — cmd is constructed from validated parts.
            cmd,
            input=dialogue,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCodeCLIError(
            f"claude CLI timed out after {timeout_seconds}s"
        ) from exc
    except FileNotFoundError as exc:
        raise ClaudeCodeCLIError(f"claude CLI not found at {cli_path}") from exc

    return parse_stream_json(
        proc.stdout, exit_code=proc.returncode, stderr_text=proc.stderr
    )


def _run_codex_stream(
    *,
    codex_path: str,
    system_prompt: str,
    dialogue: str,
    timeout_seconds: int,
) -> ParseResult:
    """Invoke ``codex exec`` and shape its output like a terminal stream result."""
    output_path = _temporary_output_path()
    cmd = [
        codex_path,
        "exec",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-rules",
        "--output-last-message",
        str(output_path),
        "-",
    ]

    env = os.environ.copy()
    env[_ENV_HOST] = _HOST_CODEX
    env["CLAUDE_SMART_INTERNAL"] = "1"

    try:
        proc = subprocess.run(  # noqa: S603 — cmd is constructed from validated parts.
            cmd,
            input=_codex_prompt(prompt=dialogue, system_prompt=system_prompt),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        try:
            terminal_text = output_path.read_text(encoding="utf-8").strip()
        except OSError:
            terminal_text = ""
    except subprocess.TimeoutExpired as exc:
        raise ClaudeCodeCLIError(
            f"codex CLI timed out after {timeout_seconds}s"
        ) from exc
    except FileNotFoundError as exc:
        raise ClaudeCodeCLIError(f"codex CLI not found at {codex_path}") from exc
    finally:
        with suppress(OSError):
            output_path.unlink()

    return ParseResult(
        success=proc.returncode == 0 and bool(terminal_text),
        terminal_text=terminal_text,
        stderr_text=proc.stderr,
        raw_lines_parsed=1 if terminal_text else 0,
    )


def _temporary_output_path() -> Path:
    with tempfile.NamedTemporaryFile(
        prefix="claude-smart-codex-", delete=False
    ) as handle:
        return Path(handle.name)


def _codex_prompt(*, prompt: str, system_prompt: str) -> str:
    if not system_prompt:
        return prompt
    return f"{system_prompt}\n\n## Task\n{prompt}"


def _build_model_response(
    model: str,
    terminal_text: str,
    elapsed_seconds: float,
) -> ModelResponse:
    """Wrap the CLI's terminal text in a LiteLLM ``ModelResponse``.

    The stream-json transport does not surface usage tokens at the terminal
    event, so prompt/completion counts are reported as zero. Downstream
    LiteLLM callers tolerate this (usage is informational, not load-bearing).

    Args:
        model (str): The model string originally requested
            (e.g. ``claude-code/default``).
        terminal_text (str): The terminal ``result`` text from the CLI.
        elapsed_seconds (float): Wall time the subprocess took — for logging only.

    Returns:
        ModelResponse: Shaped to match what callers of ``litellm.completion`` expect.
    """
    message = Message(role="assistant", content=terminal_text)
    choice = Choices(index=0, message=message, finish_reason="stop")
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    response = ModelResponse(
        id=f"claude-code-{int(time.time())}",
        choices=[choice],
        created=int(time.time()),
        model=model,
        object="chat.completion",
        usage=usage,
    )
    _LOGGER.debug(
        "claude-code provider: model=%s elapsed=%.2fs",
        model,
        elapsed_seconds,
    )
    return response


_TOOL_USE_INSTRUCTION_TEMPLATE = (
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


def _render_tools_instruction(tools: list[Any], finish_tool: str = "finish") -> str:
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
    return _TOOL_USE_INSTRUCTION_TEMPLATE.format(
        finish=finish_tool, tool_specs=tool_specs
    )


def _maybe_append_tools_instruction(system_prompt: str, tools: list[Any] | None) -> str:
    """Append the tool-use instructions when tools are present.

    Args:
        system_prompt: Existing system prompt (possibly empty).
        tools: LiteLLM-style tools list, or None/empty.

    Returns:
        str: The (possibly-augmented) system prompt.
    """
    if not tools:
        return system_prompt
    instruction = _render_tools_instruction(tools)
    if system_prompt:
        return f"{system_prompt}\n\n{instruction}"
    return instruction


def _parse_tool_use(text: str, tool_names: set[str]) -> dict[str, Any] | None:
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


def _build_model_response_with_tool_call(
    *,
    model: str,
    terminal_text: str,
    elapsed_seconds: float,
    tool_use: dict[str, Any],
) -> ModelResponse:
    """Wrap the CLI terminal text as a ``ModelResponse`` carrying one ``tool_calls`` entry.

    The stream-json transport does not surface usage tokens at the terminal
    event, so prompt/completion counts are reported as zero (same convention
    as :func:`_build_model_response`). Downstream LiteLLM callers tolerate
    this — usage is informational, not load-bearing.

    Args:
        model: Model string passed in by LiteLLM.
        terminal_text: The terminal ``result`` text from the CLI (retained
            for signature parity with the plain-text branch; surfaced via
            logging only).
        elapsed_seconds: Subprocess wall time, for logging only.
        tool_use: Parsed ``{"name": str, "args": dict}`` from the model output.

    Returns:
        ModelResponse: A LiteLLM response with ``choices[0].message.tool_calls`` set
            and ``content`` set to ``None`` — matches OpenAI/Anthropic tool-call shape.
    """
    del terminal_text  # retained for signature parity only
    call_id = f"call_{int(time.time() * 1000)}"
    tool_call = ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(
            name=tool_use["name"],
            arguments=json.dumps(tool_use["args"]),
        ),
    )
    message = Message(role="assistant", content=None, tool_calls=[tool_call])
    choice = Choices(index=0, message=message, finish_reason="tool_calls")
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    response = ModelResponse(
        id=f"claude-code-{int(time.time())}",
        choices=[choice],
        created=int(time.time()),
        model=model,
        object="chat.completion",
        usage=usage,
    )
    _LOGGER.debug(
        "claude-code provider: tool_call name=%s elapsed=%.2fs",
        tool_use["name"],
        elapsed_seconds,
    )
    return response


def _maybe_append_schema(system_prompt: str, response_format: Any) -> str:
    """Return *system_prompt* extended with a JSON-schema instruction when applicable.

    Args:
        system_prompt: Existing system prompt (possibly empty).
        response_format: The LiteLLM response_format value, if any.

    Returns:
        str: The (possibly-augmented) system prompt.
    """
    instruction = _schema_instruction(response_format)
    if not instruction:
        return system_prompt
    if system_prompt:
        return f"{system_prompt}\n\n{instruction}"
    return instruction


_IGNORED_PARAMS: tuple[str, ...] = (
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "seed",
    "frequency_penalty",
    "presence_penalty",
)


def _warn_on_ignored_params(*sources: Any) -> None:
    """Emit one-time warnings for LiteLLM params the CLI cannot honour.

    Args:
        *sources: Any number of dict-like sources (kwargs, ``optional_params``).

    Returns:
        None
    """
    for source in sources:
        if not isinstance(source, dict):
            continue
        for name in _IGNORED_PARAMS:
            if source.get(name) is not None:
                _warn_unsupported_param_once(name)


class ClaudeCodeLLM(CustomLLM):
    """LiteLLM custom handler routing completions through the ``claude`` CLI."""

    def __init__(
        self,
        cli_path: str | None = None,
        timeout_seconds: int | None = None,
        storage: Any | None = None,
    ):
        """Initialise the handler.

        Args:
            cli_path (str | None): Override for the ``claude`` binary path.
            timeout_seconds (int | None): Override for subprocess timeout.
            storage (Any | None): A BaseStorage-shaped object used to persist
                stall_state on credit/auth failures. Typed as ``Any`` to avoid
                a circular import with ``server.services.storage``. When None,
                stall state is not recorded (back-compat).
        """
        super().__init__()
        self._explicit_cli_path = cli_path
        self._explicit_timeout = timeout_seconds
        self._storage = storage

    def _cli_path(self) -> str:
        """Resolve the CLI path, raising when unavailable.

        Returns:
            str: Absolute path to the ``claude`` executable.

        Raises:
            ClaudeCodeCLIError: If the CLI cannot be located.
        """
        path = self._explicit_cli_path or _resolve_cli_path()
        if not path:
            raise ClaudeCodeCLIError(
                f"{_cli_name()} CLI not found for {_host()}. Install the host "
                f"CLI or set {_ENV_CLI_PATH} to an executable path."
            )
        return path

    def _timeout(self) -> int:
        """Resolve the subprocess timeout.

        Returns:
            int: Timeout in seconds.
        """
        if self._explicit_timeout is not None:
            return self._explicit_timeout
        raw = os.environ.get(_ENV_TIMEOUT)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                _LOGGER.warning("Ignoring non-integer %s=%r", _ENV_TIMEOUT, raw)
        return _DEFAULT_TIMEOUT_SECONDS

    def completion(  # type: ignore[override]
        self,
        *args: Any,
        model: str = "claude-code/default",
        messages: list[dict[str, Any]] | None = None,
        optional_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Execute a completion via the ``claude`` CLI.

        Args:
            *args: Ignored; LiteLLM may pass ``model_response`` positionally.
            model: The requested model slug (e.g. ``claude-code/default``).
            messages: LiteLLM-style chat messages.
            optional_params: LiteLLM's bag of generation params — checked for
                ``response_format`` and logged when unsupported knobs are set.
            **kwargs: Other LiteLLM arguments (api_key, logging_obj, etc.);
                ignored by this handler.

        Returns:
            ModelResponse: Shaped to match what ``litellm.completion`` callers expect.

        Raises:
            ClaudeCodeCLIError: On CLI failure or missing binary.
        """
        del args
        messages = messages or []
        optional_params = optional_params or {}

        # Tools may arrive in either ``optional_params["tools"]`` (LiteLLM's
        # standard plumbing for custom providers) or as a top-level ``tools``
        # kwarg. Read both before discarding kwargs.
        tools = optional_params.get("tools") or kwargs.get("tools")
        del kwargs

        _warn_on_ignored_params(optional_params)

        response_format = optional_params.get("response_format")
        system_prompt, dialogue = _split_system_and_dialogue(messages)
        # When ``tools`` is present, render the tools spec into the system
        # prompt and ignore response_format (mutually exclusive with our
        # tool_use JSON output contract).
        if tools:
            system_prompt = _maybe_append_tools_instruction(system_prompt, tools)
        else:
            system_prompt = _maybe_append_schema(system_prompt, response_format)

        started = time.perf_counter()
        result = _run_cli_stream(
            cli_path=self._cli_path(),
            system_prompt=system_prompt,
            dialogue=dialogue,
            timeout_seconds=self._timeout(),
        )

        if result.success:
            self._clear_stall_safely()
            elapsed = time.perf_counter() - started

            # When ``tools`` are provided, attempt to parse the model's
            # terminal text as a ``{"tool": ..., "args": ...}`` JSON object.
            # If parsing succeeds, return a tool-call ModelResponse; otherwise
            # warn (so the silent fall-through is observable) and return a
            # plain-text response, which the caller treats as "no tool_calls"
            # and uses to terminate the tool loop.
            if tools:
                tool_names: set[str] = {
                    name
                    for tool in tools
                    if isinstance(tool, dict)
                    and isinstance(
                        name := (tool.get("function") or {}).get("name"), str
                    )
                }
                tool_use = _parse_tool_use(result.terminal_text, tool_names)
                if tool_use is not None:
                    return _build_model_response_with_tool_call(
                        model=model,
                        terminal_text=result.terminal_text,
                        elapsed_seconds=elapsed,
                        tool_use=tool_use,
                    )
                # Log a metadata-only warning (no raw payload) — the model
                # output can carry user content / source code; deferring the
                # body to a DEBUG fingerprint avoids turning a recoverable
                # parse miss into a log-retention concern.
                _LOGGER.warning(
                    "claude-code provider: tools=%s were provided but no valid "
                    "tool_use JSON was parsed from model output; the tool loop "
                    "will terminate without a finish_tool call.",
                    sorted(tool_names),
                )
                _LOGGER.debug(
                    "claude-code provider: unparsable tool-use payload length=%d",
                    len(result.terminal_text),
                )

            return _build_model_response(
                model=model,
                terminal_text=result.terminal_text,
                elapsed_seconds=elapsed,
            )

        self._record_stall_safely(result)
        raise ClaudeCodeCLIError(
            f"claude -p stream failed; retry_errors={result.retry_errors}; "
            f"stderr={result.stderr_text[:200]!r}"
        )

    def _record_stall_safely(self, result: ParseResult) -> None:
        """Persist a stall_state row on credit/auth failure. Never raises.

        Args:
            result (ParseResult): Output of :func:`_run_cli_stream`.

        Returns:
            None
        """
        reason = classify_stall(result)
        if reason is None or self._storage is None:
            return
        try:
            self._storage.upsert_stall_state(
                reason=reason,
                stalled_at=datetime.now(UTC),
                reset_estimate=parse_reset_estimate(
                    f"{result.stderr_text} {result.terminal_text}"
                ),
                error_message=(result.stderr_text or result.terminal_text)[:1000],
            )
        except Exception as exc:  # noqa: BLE001 — never crash the provider over telemetry.
            _LOGGER.warning("Failed to record stall_state: %s", exc)

    def _clear_stall_safely(self) -> None:
        """Clear any prior stall_state row after a successful run. Never raises.

        Returns:
            None
        """
        if self._storage is None:
            return
        try:
            self._storage.clear_stall_state()
        except Exception as exc:  # noqa: BLE001 — never crash the provider over telemetry.
            _LOGGER.warning("Failed to clear stall_state: %s", exc)

    async def acompletion(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> ModelResponse:
        """Async entry point — delegates to the sync CLI call via ``to_thread``.

        Args:
            *args: Forwarded to :meth:`completion`.
            **kwargs: Forwarded to :meth:`completion`.

        Returns:
            ModelResponse: The CLI-backed completion result.
        """
        import asyncio

        return await asyncio.to_thread(self.completion, *args, **kwargs)


_REGISTERED = False
_HANDLER: ClaudeCodeLLM | None = None


def register_if_enabled(storage: Any | None = None) -> bool:
    """Register the ``claude-code`` provider with LiteLLM if enabled and available.

    Idempotent — safe to call more than once per process. Opt-in via
    ``CLAUDE_SMART_USE_LOCAL_CLI=1``. Skips registration (with a warning)
    when the env var is set but the CLI is not on PATH.

    Args:
        storage (Any | None): Optional BaseStorage-shaped handle used by the
            provider to persist stall_state on credit/auth failures. The
            caller (``LiteLLMClient`` import-time wiring) typically has no
            storage available, so use :func:`set_storage` to late-bind it
            once a request context exists.

    Returns:
        bool: True if the provider is registered after this call.
    """
    global _REGISTERED, _HANDLER
    if _REGISTERED:
        if storage is not None and _HANDLER is not None:
            _HANDLER._storage = storage
        return True
    if not _env_enabled():
        return False
    cli_path = _resolve_cli_path()
    if not cli_path:
        _LOGGER.warning(
            "%s=1 is set but the %s CLI is not available for %s. "
            "Install the host CLI or set %s; skipping provider registration.",
            ENV_ENABLE,
            _cli_name(),
            _host(),
            _ENV_CLI_PATH,
        )
        return False

    existing = list(getattr(litellm, "custom_provider_map", None) or [])
    if any(entry.get("provider") == PROVIDER_KEY for entry in existing):
        _REGISTERED = True
        return True
    _HANDLER = ClaudeCodeLLM(storage=storage)
    existing.append({"provider": PROVIDER_KEY, "custom_handler": _HANDLER})
    litellm.custom_provider_map = existing
    _REGISTERED = True
    _LOGGER.info("Registered %s LiteLLM provider (cli=%s)", PROVIDER_KEY, cli_path)
    return True


def set_storage(storage: Any) -> None:
    """Bind storage onto the registered handler after registration.

    The provider is registered at LiteLLM-import time, before any
    request-scoped storage exists. Once a storage instance is available,
    call this to enable stall_state persistence on the live handler.

    Args:
        storage (Any): BaseStorage-shaped instance.

    Returns:
        None
    """
    if _HANDLER is not None:
        _HANDLER._storage = storage


__all__ = [
    "ENV_ENABLE",
    "PROVIDER_KEY",
    "ClaudeCodeCLIError",
    "ClaudeCodeLLM",
    "is_claude_code_available",
    "register_if_enabled",
    "set_storage",
]
