"""Local-script assistant backend for Reflexio playbook optimization.

Reflexio's ``LocalScriptAssistant`` sends one JSON payload on stdin and expects
one JSON object on stdout. This module bridges that protocol to a guarded
``openclaw infer model run`` subprocess so candidate playbooks can be evaluated
against the local model without re-entering openclaw-smart hooks.

Compared to claude-smart's variant, openClaw's ``infer model run`` is a one-shot
completion (not a full agent with tool access), so we cannot grant read-only
tool calls during evaluation. The optimizer therefore evaluates rules purely
based on the prompt context. ``OPENCLAW_SMART_INTERNAL=1`` is set on the
subprocess env to short-circuit our own hooks (recursion guard).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # noqa: S404 — subprocess is the integration point.
import sys
from typing import Any

_CLI_TIMEOUT_SECONDS = 300
ENV_CLI_PATH = "OPENCLAW_BIN"
ENV_DEFAULT_MODEL = "OPENCLAW_DEFAULT_MODEL"
ENV_TIMEOUT = "OPENCLAW_OPTIMIZER_TIMEOUT"


class OptimizerAssistantError(Exception):
    """Raised for any local assistant protocol or openClaw CLI failure."""


def main() -> int:
    """Console-script entrypoint for ``openclaw-smart-optimizer-assistant``.

    Returns:
        int: Exit code (0 on success, 1 on any error).
    """
    try:
        payload = _read_payload()
        messages = _validated_list(payload, "messages")
        playbooks = _validated_list(payload, "playbooks")
        prompt, system_prompt = _build_prompt(messages, playbooks)
        content = _run_openclaw_cli(prompt=prompt, system_prompt=system_prompt)
    except Exception as exc:  # noqa: BLE001 — script errors become LocalScript failures.
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 1

    json.dump({"content": content}, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def _read_payload() -> dict[str, Any]:
    """Read and parse a JSON payload from stdin.

    Returns:
        dict[str, Any]: Parsed payload.

    Raises:
        OptimizerAssistantError: If stdin is not valid JSON or not an object.
    """
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OptimizerAssistantError("stdin must be a JSON object") from exc
    if not isinstance(payload, dict):
        raise OptimizerAssistantError("stdin must be a JSON object")
    return payload


def _validated_list(payload: dict[str, Any], field: str) -> list[Any]:
    """Return ``payload[field]`` as a list, raising if missing or wrong type."""
    value = payload.get(field)
    if not isinstance(value, list):
        raise OptimizerAssistantError(f"payload.{field} must be a list")
    return value


def _build_prompt(messages: list[Any], playbooks: list[Any]) -> tuple[str, str]:
    """Build ``(prompt, system_prompt)`` from the optimizer payload.

    Args:
        messages (list[Any]): Conversation messages from the optimizer.
        playbooks (list[Any]): Candidate playbook rules to evaluate.

    Returns:
        tuple[str, str]: ``(prompt, system_prompt)`` for the CLI call.

    Raises:
        OptimizerAssistantError: If messages contain no content.
    """
    normalized = [_normalize_message(message) for message in messages]
    normalized = [message for message in normalized if message["content"]]
    if not normalized:
        raise OptimizerAssistantError("payload.messages must contain content")

    final_message = normalized[-1]
    prior_messages = normalized[:-1]

    system_sections = [_render_playbooks(playbooks)]
    existing_system = [
        message["content"] for message in normalized if message["role"] == "system"
    ]
    if existing_system:
        system_sections.append(
            "## Existing system context\n" + "\n\n".join(existing_system)
        )

    prior_dialogue = [
        message
        for message in prior_messages
        if message["role"] in {"user", "assistant"}
    ]
    if prior_dialogue:
        system_sections.append(
            "## Conversation so far\n" + _render_transcript(prior_dialogue)
        )

    prompt = final_message["content"]
    if final_message["role"] != "user":
        prompt = _render_transcript([final_message])
    system_prompt = "\n\n".join(section for section in system_sections if section)
    return prompt, system_prompt


def _normalize_message(message: Any) -> dict[str, str]:
    """Coerce a raw message dict to ``{role, content}`` with safe defaults."""
    if not isinstance(message, dict):
        raise OptimizerAssistantError("each message must be an object")
    role = str(message.get("role") or "user").strip().lower()
    if role not in {"user", "assistant", "system"}:
        role = "user"
    content = message.get("content")
    if not isinstance(content, str):
        raise OptimizerAssistantError("each message.content must be a string")
    return {"role": role, "content": content.strip()}


def _render_playbooks(playbooks: list[Any]) -> str:
    """Render the candidate playbook rules as a markdown section."""
    if not playbooks:
        return ""
    lines = ["## Candidate playbook rules"]
    for index, playbook in enumerate(playbooks, start=1):
        if not isinstance(playbook, dict):
            raise OptimizerAssistantError("each playbook must be an object")
        content = playbook.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OptimizerAssistantError("each playbook.content must be a string")
        trigger = playbook.get("trigger")
        suffix = ""
        if isinstance(trigger, str) and trigger.strip():
            suffix = f" (when: {trigger.strip()})"
        lines.append(f"{index}. {content.strip()}{suffix}")
    return "\n".join(lines)


def _render_transcript(messages: list[dict[str, str]]) -> str:
    """Render a list of normalized messages as a ``Role: content`` transcript."""
    labels = {"user": "User", "assistant": "Assistant", "system": "System"}
    return "\n\n".join(
        f"{labels.get(message['role'], 'User')}: {message['content']}"
        for message in messages
    )


def _resolve_cli_path() -> str | None:
    """Return the openclaw binary path, honoring ``OPENCLAW_BIN`` then PATH."""
    override = os.environ.get(ENV_CLI_PATH)
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    return shutil.which("openclaw")


def _run_openclaw_cli(*, prompt: str, system_prompt: str) -> str:
    """Invoke ``openclaw infer model run`` and return the completion text.

    Args:
        prompt (str): User-facing portion of the prompt.
        system_prompt (str): System context (playbook rules + transcript)
            concatenated into the same prompt — openclaw does not have a
            separate ``--system-prompt`` flag.

    Returns:
        str: Completion text extracted from the CLI's JSON stdout.

    Raises:
        OptimizerAssistantError: On missing CLI, timeout, non-zero exit, or
            unparseable JSON output.
    """
    cli = _resolve_cli_path()
    if not cli:
        raise OptimizerAssistantError("openclaw CLI not found; set OPENCLAW_BIN")

    full_prompt = _join_prompt(prompt=prompt, system_prompt=system_prompt)
    argv = [cli, "infer", "model", "run", "--prompt", full_prompt, "--json"]
    model = os.environ.get(ENV_DEFAULT_MODEL)
    if model:
        argv += ["--model", model]

    env = os.environ.copy()
    env["OPENCLAW_SMART_INTERNAL"] = "1"
    timeout_s = int(os.environ.get(ENV_TIMEOUT, str(_CLI_TIMEOUT_SECONDS)))

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
        raise OptimizerAssistantError(
            f"openclaw CLI timed out after {timeout_s}s"
        ) from exc
    except FileNotFoundError as exc:
        raise OptimizerAssistantError("openclaw CLI not found on PATH") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise OptimizerAssistantError(
            f"openclaw CLI exited {proc.returncode}: {stderr[:500]}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise OptimizerAssistantError(
            f"openclaw CLI returned non-JSON: {proc.stdout[:200]}"
        ) from exc
    if not isinstance(data, dict):
        raise OptimizerAssistantError("openclaw CLI JSON output must be an object")

    content = _extract_content(data)
    if not content:
        raise OptimizerAssistantError(
            f"openclaw CLI JSON had no completion text: {list(data)[:5]}"
        )
    return content


def _join_prompt(*, prompt: str, system_prompt: str) -> str:
    """Concatenate system context above the user prompt."""
    if not system_prompt:
        return prompt
    return f"{system_prompt}\n\n## Task\n{prompt}"


def _extract_content(payload: dict[str, Any]) -> str:
    """Pull the completion text from openclaw's JSON output.

    The exact field name in ``openclaw infer model run --json`` output is not
    pinned at design time; scan common keys in priority order.
    """
    for key in ("text", "content", "output", "result", "response"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    msg = payload.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
