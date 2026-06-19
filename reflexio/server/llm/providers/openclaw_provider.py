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
from pathlib import Path
from typing import Any

import litellm
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import Choices, Message, ModelResponse, Usage

_LOGGER = logging.getLogger(__name__)

PROVIDER_KEY = "openclaw"
ENV_ENABLE = "OPENCLAW_SMART_USE_LOCAL_CLI"
ENV_CLI_PATH = "OPENCLAW_BIN"
ENV_DEFAULT_MODEL = "OPENCLAW_DEFAULT_MODEL"
ENV_TIMEOUT = "OPENCLAW_CLI_TIMEOUT"
_DEFAULT_TIMEOUT_SECONDS = 180

_TRUTHY = {"1", "true", "yes"}

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
    return os.environ.get(ENV_ENABLE, "").lower() in _TRUTHY


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
    """Extract the model's text reply from openclaw's JSON output.

    The exact field name in ``openclaw infer model run --json`` output
    couldn't be probed at design time (no model providers were
    auth-configured). Scan common keys in priority order: ``text`` →
    ``content`` → ``output`` → ``message.content``. The first end-to-end
    test will surface the real key; we can pin it then if needed.

    Args:
        payload (Any): Parsed JSON response from openclaw.

    Returns:
        str: Completion text, or empty string if no known key matches.
    """
    if not isinstance(payload, dict):
        return ""
    for key in ("text", "content", "output"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    msg = payload.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    return ""


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

    argv = [cli, "infer", "model", "run", "--prompt", prompt, "--json"]
    if model:
        argv += ["--model", model]

    env = os.environ.copy()
    env["OPENCLAW_SMART_INTERNAL"] = "1"

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

    if proc.returncode != 0:
        raise OpenClawCLIError(
            proc.stderr.strip() or f"openclaw exit {proc.returncode}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise OpenClawCLIError(
            f"openclaw returned non-JSON: {proc.stdout[:200]}"
        ) from exc

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
        if "/" in model_kwarg:
            model = model_kwarg.split("/", 1)[1] or None
        else:
            model = model_kwarg or None
        if not model:
            model = os.environ.get(ENV_DEFAULT_MODEL) or None

        messages = kwargs.get("messages", [])
        prompt = _messages_to_prompt(messages)
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

        return ModelResponse(
            choices=[Choices(message=Message(content=text, role="assistant"))],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model=model_kwarg or "openclaw",
        )


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten chat messages into a single prompt string.

    ``openclaw infer model run`` takes a single ``--prompt``; we serialize
    the chat as ``role: content`` lines separated by blank lines. Content
    blocks (lists) are flattened to text.

    Args:
        messages (list[dict[str, Any]]): LiteLLM-style chat messages.

    Returns:
        str: Newline-joined role:content lines.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
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
