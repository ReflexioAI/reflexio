"""openClaw CLI as a LiteLLM custom provider.

Routes ``litellm.completion(model="openclaw/...", ...)`` through the
user's locally-installed ``openclaw`` CLI by spawning
``openclaw infer model run --prompt <p> --json [--model <m>]``.

Activation is opt-in via ``OPENCLAW_SMART_USE_LOCAL_CLI=1``. Without it,
the provider does not register and reflexio falls back to its normal
provider priority. Every spawn sets ``OPENCLAW_SMART_INTERNAL=1`` so the
openclaw-smart plugin's hooks short-circuit when this provider invokes
the CLI (recursion guard).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess  # noqa: S404 — subprocess is the integration point.
import time
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

from reflexio.server.env_utils import env_truthy
from reflexio.server.llm.providers.cli_tool_protocol import (
    append_schema_instruction,
    append_tools_instruction,
    parse_tool_use,
    tool_call_attr,
    tool_names,
)

_LOGGER = logging.getLogger(__name__)

PROVIDER_KEY = "openclaw"
ENV_ENABLE = "OPENCLAW_SMART_USE_LOCAL_CLI"
ENV_CLI_PATH = "OPENCLAW_BIN"
ENV_DEFAULT_MODEL = "OPENCLAW_DEFAULT_MODEL"
ENV_TIMEOUT = "OPENCLAW_CLI_TIMEOUT"
_DEFAULT_TIMEOUT_SECONDS = 180
_BUSY_MARKER = "retry when it finishes"
_BUSY_ATTEMPTS = 4
_BUSY_BACKOFF_SECONDS = 2.0
# Headroom under Linux MAX_ARG_STRLEN (131072) for the argv string.
_MAX_PROMPT_BYTES = 120_000


# Module-level state reset by tests via the _reset_module_state fixture.
_REGISTERED: bool = False
_HANDLER: OpenClawLLM | None = None


class OpenClawCLIError(RuntimeError):
    """Raised when the openclaw CLI subprocess fails."""


def _env_enabled() -> bool:
    """Return True when OPENCLAW_SMART_USE_LOCAL_CLI is truthy.

    Returns:
        bool: True if the opt-in env var is set to a truthy value, else False.
    """
    return env_truthy(os.environ.get(ENV_ENABLE, ""))


def _resolve_cli_path() -> str | None:
    """Resolve the openclaw CLI absolute path.

    Honors ``OPENCLAW_BIN`` env override, otherwise falls back to
    ``shutil.which("openclaw")``.

    Returns:
        str | None: Absolute path to the openclaw binary, or None if not found.
    """
    override = os.environ.get(ENV_CLI_PATH)
    if override:
        path = Path(override)
        if path.is_file() and os.access(override, os.X_OK):
            return override
    return shutil.which("openclaw")


def is_openclaw_available() -> bool:
    """Return True when the openclaw provider is usable right now.

    Both the opt-in env var *and* a resolvable CLI path are required.

    Returns:
        bool: True iff ``OPENCLAW_SMART_USE_LOCAL_CLI`` is truthy AND a
            CLI binary is resolvable.
    """
    return _env_enabled() and _resolve_cli_path() is not None


def _extract_text(payload: Any) -> str:
    """Extract the model's text reply from ``openclaw infer model run --json``.

    Observed against openclaw 2026.10.6, the envelope is::

        {"ok": true, "capability": "model.run", "outputs": [{"text": "..."}]}

    The original parser guessed top-level ``text``/``content``/``output`` keys
    before any provider could be probed; none of them exist, so every
    completion through this provider raised "no completion text".

    Args:
        payload (Any): Parsed JSON response from openclaw.

    Returns:
        str: Concatenated ``outputs[].text``, or ``""`` when there is none.
    """
    if not isinstance(payload, dict):
        return ""
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        return ""
    texts = [
        item["text"]
        for item in outputs
        if isinstance(item, dict) and isinstance(item.get("text"), str)
    ]
    return "".join(texts)


def _json_or_none(text: str) -> Any:
    """Parse the CLI's JSON stdout, or return None when it is not JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _call_cli(*, prompt: str, model: str | None, timeout_s: int) -> str:
    """Invoke openclaw CLI for a one-shot completion.

    Args:
        prompt (str): User prompt text passed via ``--prompt``.
        model (str | None): Optional model id passed via ``--model``.
        timeout_s (int): Subprocess timeout in seconds.

    Returns:
        str: Completion text extracted from the CLI's JSON stdout.

    Raises:
        OpenClawCLIError: If the CLI is not found, exits non-zero, returns
            non-JSON stdout, or returns JSON with no recognized text field.
    """
    cli = _resolve_cli_path()
    if not cli:
        raise OpenClawCLIError("openclaw CLI not found; set OPENCLAW_BIN")

    # `infer model run` takes the prompt only as an argument, and Linux caps a
    # single argv string at MAX_ARG_STRLEN (128 KiB). Refuse clearly instead
    # of surfacing an opaque E2BIG from exec.
    if len(prompt.encode()) > _MAX_PROMPT_BYTES:
        raise OpenClawCLIError(
            f"prompt is {len(prompt.encode())} bytes; openclaw accepts at most "
            f"{_MAX_PROMPT_BYTES} via --prompt"
        )
    argv = [cli, "infer", "model", "run", "--prompt", prompt, "--json"]
    if model:
        argv += ["--model", model]

    env = os.environ.copy()
    env["OPENCLAW_SMART_INTERNAL"] = "1"

    for attempt in range(1, _BUSY_ATTEMPTS + 1):
        try:
            proc = subprocess.run(  # noqa: S603 — argv is fully constructed.
                argv,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OpenClawCLIError(f"openclaw timed out after {timeout_s}s") from exc
        payload = _json_or_none(proc.stdout)
        error = payload.get("error") if isinstance(payload, dict) else None
        # Two embedded OpenClaw processes on one state directory (an agent run
        # with --local and this one) contend for its startup maintenance; the
        # loser is told to retry. A running Gateway avoids it.
        if _BUSY_MARKER in str(error) and attempt < _BUSY_ATTEMPTS:
            time.sleep(_BUSY_BACKOFF_SECONDS * attempt)
            continue
        break

    if proc.returncode != 0 or (
        isinstance(payload, dict) and payload.get("ok") is False
    ):
        # On failure the CLI prints its JSON error on stdout and only a generic
        # "The CLI command failed." on stderr; report the specific one.
        detail = (error or {}).get("message") if isinstance(error, dict) else error
        raise OpenClawCLIError(
            f"openclaw reported failure: {detail}"
            if detail
            else proc.stderr.strip() or f"openclaw exit {proc.returncode}"
        )
    if payload is None:
        raise OpenClawCLIError(f"openclaw returned non-JSON: {proc.stdout[:200]}")

    text = _extract_text(payload)
    if not text:
        raise OpenClawCLIError(
            f"openclaw JSON had no completion text: {list(payload)[:5]}"
        )
    return text


class OpenClawLLM(CustomLLM):
    """LiteLLM CustomLLM that routes completions through the openclaw CLI."""

    def completion(self, *args: Any, **kwargs: Any) -> ModelResponse:  # noqa: ARG002 — LiteLLM CustomLLM signature
        """Synchronous completion via ``openclaw infer model run``.

        The ``openclaw/`` prefix is stripped from the requested model id.
        If no model is provided, ``OPENCLAW_DEFAULT_MODEL`` is used; if
        that's also unset, the CLI's own default is used.

        Args:
            *args: Unused (LiteLLM signature).
            **kwargs: LiteLLM completion kwargs (``messages``, ``model``).

        Returns:
            ModelResponse: Reply text in ``choices[0].message.content``.
        """
        # ``kwargs.get("model")`` can be ``None`` — coercing via ``str(...)``
        # would yield the literal "None" and skip the default-model fallback.
        raw_model = kwargs.get("model")
        model_kwarg = "" if raw_model is None else str(raw_model)
        # LiteLLM hands a custom handler the model with its provider prefix
        # already removed ("openclaw/openai/gpt-x" arrives as "openai/gpt-x"),
        # and OpenClaw needs that provider/model pair intact. Splitting off the
        # first segment unconditionally left a bare model OpenClaw rejects.
        model = model_kwarg.removeprefix(f"{PROVIDER_KEY}/") or None
        # `openclaw/default` (the auto-detected model) means "whatever model
        # OpenClaw is configured to use", like `claude-code/default`.
        if model == "default":
            model = None
        if not model:
            model = os.environ.get(ENV_DEFAULT_MODEL) or None

        messages = kwargs.get("messages", [])
        # Tools and response_format arrive in optional_params (LiteLLM's
        # custom-provider plumbing) or as top-level kwargs; read both.
        optional_params = kwargs.get("optional_params") or {}
        tools = optional_params.get("tools") or kwargs.get("tools")
        response_format = optional_params.get("response_format") or kwargs.get(
            "response_format"
        )
        prompt = _messages_to_prompt(
            messages, tools=tools, response_format=response_format
        )
        # An invalid OPENCLAW_CLI_TIMEOUT must not crash the completion path —
        # fall back to the default with a warning so misconfiguration degrades
        # to working-but-slow rather than working-then-erroring.
        raw_timeout = os.environ.get(ENV_TIMEOUT)
        try:
            timeout_s = (
                int(raw_timeout)
                if raw_timeout is not None
                else _DEFAULT_TIMEOUT_SECONDS
            )
        except ValueError:
            _LOGGER.warning(
                "Invalid %s=%r; using default %d",
                ENV_TIMEOUT,
                raw_timeout,
                _DEFAULT_TIMEOUT_SECONDS,
            )
            timeout_s = _DEFAULT_TIMEOUT_SECONDS

        text = _call_cli(prompt=prompt, model=model, timeout_s=timeout_s)

        if tools:
            names = tool_names(tools)
            tool_use = parse_tool_use(text, names)
            if tool_use is not None:
                return _tool_call_response(model_kwarg or PROVIDER_KEY, tool_use)
            # Metadata only: the reply can carry user content.
            _LOGGER.warning(
                "openclaw provider: tools=%s were provided but the reply held no "
                "tool-call JSON; the tool loop will end without its finish tool.",
                sorted(names),
            )
        return ModelResponse(
            choices=[Choices(message=Message(content=text, role="assistant"))],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model=model_kwarg or PROVIDER_KEY,
        )


def _tool_call_response(model: str, tool_use: dict[str, Any]) -> ModelResponse:
    """Shape a parsed ``{"name", "args"}`` tool use as an OpenAI-style tool call."""
    call = ChatCompletionMessageToolCall(
        id=f"call_{time.time_ns()}",
        type="function",
        function=Function(
            name=tool_use["name"], arguments=json.dumps(tool_use["args"])
        ),
    )
    message = Message(role="assistant", content=None, tool_calls=[call])
    return ModelResponse(
        choices=[Choices(index=0, message=message, finish_reason="tool_calls")],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        model=model,
    )


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return "" if content is None else str(content)


def _messages_to_prompt(
    messages: list[dict[str, Any]],
    *,
    tools: list[Any] | None = None,
    response_format: Any = None,
) -> str:
    """Flatten chat messages into the single ``--prompt`` the CLI accepts.

    System messages come first, followed by the tool-calling or schema
    instructions (``cli_tool_protocol``), then the dialogue as ``role:
    content`` blocks. Earlier tool calls and their results are kept as
    breadcrumbs: without them a multi-step tool loop never sees what it
    already did and cannot converge.

    Args:
        messages (list[dict[str, Any]]): LiteLLM-style chat messages.
        tools (list[Any] | None): LiteLLM tools spec, if the caller is a tool loop.
        response_format (Any): Pydantic class or JSON-schema dict to ask for.

    Returns:
        str: The prompt.
    """
    system = "\n\n".join(
        _text(m.get("content")) for m in messages if m.get("role") == "system"
    )
    system = (
        append_tools_instruction(system, tools)
        if tools
        else append_schema_instruction(system, response_format)
    )
    parts = [f"system: {system}"] if system else []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue
        content = _text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            calls = "; ".join(
                f"called {tool_call_attr(tc, 'name') or '?'} with "
                f"{tool_call_attr(tc, 'arguments') or '{}'}"
                for tc in m["tool_calls"]
            )
            content = f"{content}\n[tools: {calls}]" if content else f"[tools: {calls}]"
        elif role == "tool":
            role = f"tool[{m.get('tool_call_id') or '?'}]"
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def register_if_enabled() -> bool:
    """Register the openclaw provider with LiteLLM if the opt-in env is set.

    Idempotent: safe to call multiple times.

    Returns:
        bool: True if the provider is now registered, False if env is off or
            the CLI is missing.
    """
    global _REGISTERED, _HANDLER
    if _REGISTERED:
        return True
    if not _env_enabled():
        return False
    if not _resolve_cli_path():
        _LOGGER.warning(
            "%s=1 but openclaw CLI not found; set %s",
            ENV_ENABLE,
            ENV_CLI_PATH,
        )
        return False
    _HANDLER = OpenClawLLM()
    litellm.custom_provider_map = [
        *getattr(litellm, "custom_provider_map", []),
        {"provider": PROVIDER_KEY, "custom_handler": _HANDLER},
    ]
    _REGISTERED = True
    _LOGGER.info("Registered openclaw LiteLLM provider")
    return True
