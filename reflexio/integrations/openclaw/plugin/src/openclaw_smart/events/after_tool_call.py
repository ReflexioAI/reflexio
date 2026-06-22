"""after_tool_call hook — record the tool invocation and its outcome.

openClaw delivers ``after_tool_call`` payloads with camelCase fields
(``toolName``, ``params``, ``result``). Per Phase 0 finding B1 the handler
translates these to snake_case before persisting so downstream consumers
(state buffer, reflexio API) see a stable shape.
"""

from __future__ import annotations

import re
import time
from typing import Any

from openclaw_smart import state

# Tool inputs are persisted locally and later published to reflexio, so we
# apply a conservative redaction pass at ingestion time. Chosen to avoid
# false positives over maximal coverage — the dashboard shows these
# verbatim, and users noticing a masked command is far less surprising
# than a masked ``LOG_LEVEL=INFO``.
_MAX_STR_LEN = 4096
_SECRET_ASSIGNMENT = re.compile(
    r"(?P<key>[A-Z][A-Z0-9_]{2,})=(?P<quote>['\"]?)"
    r"(?P<value>[A-Za-z0-9+/=_\-]{20,})(?P=quote)"
)


def _looks_like_secret(value: str) -> bool:
    """Heuristic: mixed-case letters plus digits suggest a high-entropy token."""
    has_lower = any(c.islower() for c in value)
    has_upper = any(c.isupper() for c in value)
    has_digit = any(c.isdigit() for c in value)
    return has_lower and has_upper and has_digit


def _mask_secrets(text: str) -> str:
    """Mask values that look like high-entropy secrets in ``KEY=value`` form."""

    def sub(match: re.Match[str]) -> str:
        value = match.group("value")
        if not _looks_like_secret(value):
            return match.group(0)
        key = match.group("key")
        quote = match.group("quote")
        return f"{key}={quote}<redacted:{len(value)}>{quote}"

    return _SECRET_ASSIGNMENT.sub(sub, text)


def _redact_string(value: str) -> str:
    """Mask secrets in ``value`` then truncate to ``_MAX_STR_LEN`` chars."""
    masked = _mask_secrets(value)
    if len(masked) > _MAX_STR_LEN:
        return masked[:_MAX_STR_LEN] + "…(truncated)"
    return masked


def _redact(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Redact obvious secrets and truncate oversized string values."""
    return {
        k: _redact_string(v) if isinstance(v, str) else v for k, v in tool_input.items()
    }


def _derive_status(tool_response: Any) -> str:
    """Classify the tool outcome as 'success' or 'error'.

    openClaw's ``after_tool_call`` payload places the tool result under
    ``result``; a structured failure may use ``is_error``/``error`` keys,
    while plain string results default to success.
    """
    if isinstance(tool_response, dict) and (
        tool_response.get("is_error") or tool_response.get("error")
    ):
        return "error"
    return "success"


_OUTPUT_TEXT_KEYS = ("stdout", "stderr", "output", "content", "text", "error")


def _flatten_tool_response_text(tool_response: Any) -> str:
    """Flatten ``tool_response`` into a single string for buffering.

    Tool responses arrive in heterogeneous shapes — Bash sends a dict with
    ``stdout``/``stderr``, Edit/Read send a string or a dict with
    ``content``/``output``, and failures populate ``error``. Joining the
    well-known string-valued keys preserves the parts most useful for
    downstream learning (failure messages, command output) without
    serializing entire structured payloads.
    """
    if tool_response is None:
        return ""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        parts = [
            tool_response[key]
            for key in _OUTPUT_TEXT_KEYS
            if isinstance(tool_response.get(key), str) and tool_response[key]
        ]
        if parts:
            return "\n".join(parts)
        for key in ("text", "content"):
            value = tool_response.get(key)
            if isinstance(value, str):
                return value
        return ""
    for attr in ("text", "content"):
        value = getattr(tool_response, attr, None)
        if isinstance(value, str):
            return value
    return ""


def _extract_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Map openClaw's camelCase keys onto our snake_case schema.

    Accepts either style so the handler is resilient to dispatcher changes —
    if a future TS shim emits already-translated keys, the snake_case
    branch wins.
    """
    return {
        "session_id": payload.get("sessionKey") or payload.get("sessionId"),
        "tool_name": payload.get("tool_name") or payload.get("toolName") or "",
        "tool_input": payload.get("tool_input") or payload.get("params") or {},
        "tool_response": payload.get("tool_response")
        if "tool_response" in payload
        else payload.get("result"),
    }


def handle(payload: dict[str, Any]) -> None:
    """Persist one tool invocation to the session JSONL buffer."""
    fields = _extract_fields(payload)
    session_id = fields["session_id"]
    tool_name = fields["tool_name"]
    if not session_id or not tool_name:
        return

    tool_response = fields["tool_response"]
    output_text = _flatten_tool_response_text(tool_response)
    # ``_redact`` expects a dict; non-dict payloads (raw string, list, or None)
    # are wrapped under ``_raw`` so the redaction path stays uniform without
    # crashing on the unusual shape.
    raw_tool_input = fields["tool_input"]
    if isinstance(raw_tool_input, dict):
        safe_tool_input: dict[str, Any] = _redact(raw_tool_input)
    else:
        safe_tool_input = {"_raw": _redact_string(str(raw_tool_input))}
    record = {
        "ts": int(time.time()),
        "role": "Assistant_tool",
        "tool_name": tool_name,
        "tool_input": safe_tool_input,
        "tool_output": _redact_string(output_text) if output_text else "",
        "status": _derive_status(tool_response),
    }
    state.append(session_id, record)
