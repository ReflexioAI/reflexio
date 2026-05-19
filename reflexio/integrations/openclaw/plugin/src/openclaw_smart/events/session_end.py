"""session_end hook — flush any remaining interactions with forced extraction.

Fires when an openClaw session terminates. We force extraction here (unlike
``agent_end``, which lets reflexio queue extraction for its next sweep) so
the user's distilled lessons from this session are usable in the very next
session, not after the next scheduled sweep.
"""

from __future__ import annotations

from typing import Any

from openclaw_smart import ids, publish


def handle(payload: dict[str, Any]) -> None:
    """Drain the buffer and trigger synchronous extraction.

    Args:
        payload (dict[str, Any]): openClaw event/ctx blob, expected to
            contain ``sessionKey`` (or ``sessionId``) and optionally
            ``agentId`` / ``workspaceDir``.
    """
    session_id = payload.get("sessionKey") or payload.get("sessionId")
    if not session_id:
        return
    project_id = ids.resolve_project_id_with_fallback(
        cwd=payload.get("workspaceDir"),
        agent_id=payload.get("agentId"),
    )
    publish.publish_unpublished(
        session_id=session_id,
        project_id=project_id,
        force_extraction=True,
        skip_aggregation=False,
    )
