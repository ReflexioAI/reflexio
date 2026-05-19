"""Dispatch table for openclaw-smart hook events.

The TS shim spawns ``python -m openclaw_smart.hook <event>`` (or the
``openclaw-smart-hook`` console script) once per hook invocation, piping
the openClaw event payload on stdin. This module reads the JSON, routes
to the matching handler, and makes sure no unhandled exception ever
propagates back to openClaw.

Handlers emit their own ``{"prependContext": …}`` JSON to stdout when
they have context to inject; otherwise they emit nothing. The dispatcher
emits nothing on no-op paths (unknown event, recursion guard, handler
exception) so the TS shim can treat empty stdout as "no return value".
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

from openclaw_smart import runtime
from openclaw_smart.internal_call import is_internal_invocation

_LOGGER = logging.getLogger(__name__)


def _load_handlers() -> dict[str, Callable[[dict[str, Any]], None]]:
    """Import handler modules lazily to keep cold-start cost low."""
    from openclaw_smart.events import (
        after_tool_call,
        agent_end,
        before_prompt_build,
        before_tool_call,
        session_end,
        session_start,
    )

    return {
        "session-start": session_start.handle,
        "before-prompt-build": before_prompt_build.handle,
        "before-tool-call": before_tool_call.handle,
        "after-tool-call": after_tool_call.handle,
        "agent-end": agent_end.handle,
        "session-end": session_end.handle,
    }


_HANDLERS: dict[str, Callable[[dict[str, Any]], None]] | None = None


def _handlers() -> dict[str, Callable[[dict[str, Any]], None]]:
    global _HANDLERS
    if _HANDLERS is None:
        _HANDLERS = _load_handlers()
    return _HANDLERS


def _read_stdin_json() -> dict[str, Any]:
    """Parse stdin as JSON. Returns ``{}`` on empty or malformed input."""
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError) as exc:
        _LOGGER.debug("stdin read failed: %s", exc)
        return {}
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        _LOGGER.debug("stdin JSON decode failed: %s", exc)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_args(argv: list[str]) -> str:
    """Return the event name from ``argv``.

    Accepts either ``[event]`` or ``[host, event]`` for symmetry with
    claude-smart, but the host arg (if present) must be ``"openclaw"`` and
    is otherwise ignored — runtime.set_host is called either way.
    """
    if not argv:
        return ""
    if len(argv) >= 2 and argv[0] in runtime.VALID_HOSTS:
        return argv[1]
    return argv[0]


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``python -m openclaw_smart.hook`` and the console script.

    Args:
        argv (list[str] | None): Command-line args (sans program name).
            Defaults to ``sys.argv[1:]``.

    Returns:
        int: Always 0 — failures are absorbed so the host never sees a
            non-zero exit and decides to surface a plugin error.
    """
    argv = argv if argv is not None else sys.argv[1:]
    event = _parse_args(argv)
    runtime.set_host(runtime.HOST_OPENCLAW)
    if not event:
        _LOGGER.warning("hook dispatcher called with no event name")
        return 0

    payload = _read_stdin_json()

    # Self-feedback guard: when this hook fires inside reflexio's own
    # openclaw subprocess (the openclaw LLM provider), skip all handlers
    # so we don't publish the extractor's system prompt back into reflexio.
    # See openclaw_smart.internal_call for detection logic.
    if is_internal_invocation(payload):
        return 0

    handler = _handlers().get(event)
    if handler is None:
        _LOGGER.warning("unknown hook event: %s", event)
        return 0

    try:
        handler(payload)
    except Exception as exc:  # noqa: BLE001 — hooks must never crash the session.
        _LOGGER.exception("hook handler %s raised: %s", event, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
