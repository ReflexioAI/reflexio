"""before_prompt_build hook — buffer the user turn and inject matching context.

Two responsibilities, in order:

1. Buffer the prompt into the session JSONL (this is the sole source of
   ``"User"`` role turns downstream — openClaw replays the rest of the
   transcript via tool events, not before_prompt_build).
2. Use the prompt text as a search query against reflexio's preferences +
   skills and emit the top hits as ``prependContext`` so the model sees
   relevant rules before planning the response.

The shared pipeline lives in ``context_inject.emit_context``. Retrieval is
best-effort: any failure from search (reflexio unreachable, HTTP timeout,
unexpected shape) is caught so the buffered-prompt behaviour is always
preserved.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openclaw_smart import context_inject, ids, state

_LOGGER = logging.getLogger(__name__)
_TOP_K = 3


def handle(payload: dict[str, Any]) -> None:
    """before_prompt_build dispatcher — buffers the prompt, then injects context.

    Args:
        payload (dict[str, Any]): The openClaw event/ctx blob (merged). Expected
            keys ``sessionKey`` (or ``sessionId``), ``prompt``, optionally
            ``agentId`` and ``workspaceDir``.

    Returns:
        None: Side effects only — appends to the session buffer and may
            write a ``{"prependContext": ...}`` JSON document to stdout.
    """
    session_id = payload.get("sessionKey") or payload.get("sessionId")
    prompt = payload.get("prompt") or ""
    if not session_id or not prompt:
        return

    project_id = ids.resolve_project_id_with_fallback(
        cwd=payload.get("workspaceDir"),
        agent_id=payload.get("agentId"),
    )
    state.append(
        session_id,
        {
            "ts": int(time.time()),
            "role": "User",
            "content": prompt,
            "user_id": project_id,
        },
    )

    try:
        context_inject.emit_context(
            session_id=session_id,
            project_id=project_id,
            query=prompt,
            top_k=_TOP_K,
        )
    except Exception as exc:  # noqa: BLE001 — never break the user's turn
        _LOGGER.debug("before_prompt_build context inject failed: %s", exc)
