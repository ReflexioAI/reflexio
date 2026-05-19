"""before_tool_call hook — observe-only stub.

Phase 0 probe finding Q1: openClaw's ``before_tool_call`` hook does NOT
honor ``prependContext`` returned by plugins, so injecting context here
would have no effect on the upcoming tool call. (In contrast, Claude
Code's PreToolUse hook is the primary just-in-time injection point.)

This handler is therefore a silent no-op in v1. The TS shim may choose
not to register it at all — having a stub here keeps the dispatcher
symmetric and leaves room for future observe-only telemetry.
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)


def handle(payload: dict[str, Any]) -> None:
    """before_tool_call dispatcher — silent no-op.

    Args:
        payload (dict[str, Any]): Unused. openClaw delivers
            ``{toolName, params, sessionKey, agentId}`` here but we have no
            actionable side effect to apply per Phase 0 finding Q1.
    """
    del payload
    _LOGGER.debug("before_tool_call invoked (observe-only stub)")
