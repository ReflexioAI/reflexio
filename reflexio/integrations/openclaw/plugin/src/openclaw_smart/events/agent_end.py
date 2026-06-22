"""agent_end hook — finalize the assistant turn and publish to reflexio.

This handler is the openClaw analogue of claude-smart's ``stop.py``, but
substantially simpler. Per Phase 0 finding Q2, openClaw delivers the final
messages inline as ``event.messages: unknown[]``, so we read assistant text
directly from the payload instead of polling a transcript JSONL on disk.

Pipeline:
1. Extract final assistant text from ``payload.messages``.
2. Parse ``[oc:…]`` citation markers from that text and resolve them
   against the per-session injected registry.
3. Append an ``Assistant`` record (always, even when the text is empty —
   ``state.unpublished_slice`` folds any buffered ``Assistant_tool``
   records into the next ``Assistant`` turn's ``tools_used``).
4. Drain the buffer to reflexio via ``publish.publish_unpublished``.

claude-smart's plan-mode decision scan and ``cs-cite`` Bash tool scan are
intentionally NOT ported — both rely on Claude-Code-specific transcript
shapes that openClaw does not emit.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openclaw_smart import ids, oc_cite, publish, state

_LOGGER = logging.getLogger(__name__)


def _extract_assistant_text(messages: Any) -> str:
    """Return the final assistant turn's text from a messages list.

    Walks ``messages`` from the end, collecting every contiguous assistant
    entry up to the most recent non-assistant boundary, then joins their
    text content. Mirrors claude-smart's transcript walker but works on
    inline message dicts.

    Args:
        messages (Any): openClaw's ``event.messages`` blob, expected to be
            a list of dicts with ``role`` and ``content`` fields.

    Returns:
        str: Concatenated assistant text, or ``""`` when none is found.
    """
    if not isinstance(messages, list):
        return ""
    collected: list[str] = []
    for entry in reversed(messages):
        if not isinstance(entry, dict):
            continue
        role = _entry_role(entry)
        if role != "assistant":
            # Boundary — stop collecting; everything before this belongs to
            # earlier turns and has already been (or will be) published.
            break
        content = _entry_content(entry)
        text_parts = _extract_text_blocks(content)
        if text_parts:
            # Prepend because we walked in reverse order.
            collected = text_parts + collected
    return "\n\n".join(part for part in collected if part)


def _entry_role(entry: dict[str, Any]) -> str:
    """Read the role from an openClaw or Claude-Code-shaped message dict."""
    role = entry.get("role")
    if isinstance(role, str):
        return role.lower()
    message = entry.get("message")
    if isinstance(message, dict):
        nested = message.get("role")
        if isinstance(nested, str):
            return nested.lower()
    return ""


def _entry_content(entry: dict[str, Any]) -> Any:
    """Return the content payload from a message entry, accepting both shapes."""
    if "content" in entry:
        return entry["content"]
    message = entry.get("message")
    if isinstance(message, dict):
        return message.get("content")
    return None


def _extract_text_blocks(content: Any) -> list[str]:
    """Return assistant-visible text from a message content payload.

    Accepts:
    - bare strings (``content: "hi"``)
    - block lists (``content: [{type: "text", text: "hi"}, ...]``)
    """
    if isinstance(content, str):
        return [content] if content else []
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            out.append(text)
            continue
        inner = block.get("content")
        if isinstance(inner, str) and inner:
            out.append(inner)
    return out


def _resolve_cited_items(session_id: str, cited_ids: list[str]) -> list[dict[str, Any]]:
    """Map citation ids to ``{id, kind, title}`` entries via the session registry.

    Unknown ids (model hallucinations, or items injected in a newer session
    than this hook can see) are dropped. Duplicate ids within one turn
    collapse to a single entry — the user-facing badge row doesn't need
    the multiplicity.
    """
    if not cited_ids:
        return []
    registry = state.read_injected(session_id)
    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for cid in cited_ids:
        if cid in seen:
            continue
        entry = registry.get(cid)
        if not entry:
            continue
        seen.add(cid)
        item: dict[str, Any] = {
            "id": entry.get("id", cid),
            "kind": entry.get("kind", ""),
            "title": entry.get("title", ""),
        }
        real_id = entry.get("real_id")
        if real_id:
            item["real_id"] = real_id
        source_kind = entry.get("source_kind")
        if isinstance(source_kind, str) and source_kind:
            item["source_kind"] = source_kind
        resolved.append(item)
    return resolved


def handle(payload: dict[str, Any]) -> None:
    """Finalize the current assistant turn and publish to reflexio.

    Args:
        payload (dict[str, Any]): openClaw event/ctx blob, expected to contain
            ``sessionKey`` (or ``sessionId``), ``messages``, and optionally
            ``agentId`` / ``workspaceDir``.
    """
    session_id = payload.get("sessionKey") or payload.get("sessionId")
    if not session_id:
        return

    project_id = ids.resolve_project_id_with_fallback(
        cwd=payload.get("workspaceDir"),
        agent_id=payload.get("agentId"),
    )

    assistant_text = _extract_assistant_text(payload.get("messages"))
    cited_ids = oc_cite.parse_text_citations(assistant_text)
    cited_items = _resolve_cited_items(session_id, cited_ids)

    now = int(time.time())
    record: dict[str, Any] = {
        "ts": now,
        "role": "Assistant",
        "content": assistant_text,
        "user_id": project_id,
    }
    if cited_items:
        record["cited_items"] = cited_items
    state.append(session_id, record)

    publish.publish_unpublished(
        session_id=session_id,
        project_id=project_id,
        force_extraction=False,
        skip_aggregation=False,
    )
