"""Render reflexio preferences + skills as markdown for display or injection."""

from __future__ import annotations

from typing import Any, Iterable

from openclaw_smart import oc_cite


def _first_nonempty(*values: Any) -> str:
    """Return the first truthy string value, or an empty string."""
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def render(
    *,
    project_id: str,
    user_playbooks: Iterable[Any],
    agent_playbooks: Iterable[Any],
    profiles: Iterable[Any],
) -> str:
    """Render skills + preferences as full audit markdown.

    Empty sections are omitted. When all sections are empty, returns "".
    """
    markdown, _ = render_with_registry(
        project_id=project_id,
        user_playbooks=user_playbooks,
        agent_playbooks=agent_playbooks,
        profiles=profiles,
    )
    return markdown


def render_with_registry(
    *,
    project_id: str,
    user_playbooks: Iterable[Any],
    agent_playbooks: Iterable[Any],
    profiles: Iterable[Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Variant of ``render`` that also returns the citation registry.

    Every skill and preference bullet is tagged with a short ``[oc:ID]``
    prefix. The registry maps those ids back to ``{id, kind, title,
    content}`` entries so ``events.agent_end`` can resolve citations into
    human-readable titles for the dashboard.

    Agent playbooks (cross-project, distilled) are listed before user
    playbooks (this project's lessons) under one ``### Project-specific
    skills`` heading. The model doesn't need to reason about the split.
    """
    playbook_lines, playbook_entries = _format_combined_playbooks(
        agent_playbooks=agent_playbooks, user_playbooks=user_playbooks
    )
    profile_lines, profile_entries = _format_profiles(profiles)
    if not playbook_lines and not profile_lines:
        return "", []

    sections: list[str] = [f"## openclaw-smart — project `{project_id}`"]
    if playbook_lines:
        sections.append("### Project-specific skills")
        sections.extend(playbook_lines)
    if profile_lines:
        sections.append("### Project preferences")
        sections.extend(profile_lines)
    sections.append(oc_cite.CITATION_INSTRUCTION)
    return "\n".join(sections) + "\n", playbook_entries + profile_entries


def render_inline(
    *,
    project_id: str,
    user_playbooks: Iterable[Any],
    agent_playbooks: Iterable[Any],
    profiles: Iterable[Any],
) -> str:
    """Render skills + preferences for mid-session injection.

    Same bullet format as ``render`` but with no top-level project header.
    This block is injected just-in-time alongside an in-flight user prompt or
    tool call, so the caller already has project context.
    """
    markdown, _ = render_inline_with_registry(
        project_id=project_id,
        user_playbooks=user_playbooks,
        agent_playbooks=agent_playbooks,
        profiles=profiles,
    )
    return markdown


def render_inline_with_registry(
    *,
    project_id: str,
    user_playbooks: Iterable[Any],
    agent_playbooks: Iterable[Any],
    profiles: Iterable[Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Variant of ``render_inline`` that also returns the citation registry."""
    del project_id  # kept for symmetry with ``render_with_registry``.
    playbook_lines, playbook_entries = _format_combined_playbooks(
        agent_playbooks=agent_playbooks, user_playbooks=user_playbooks
    )
    profile_lines, profile_entries = _format_profiles(profiles)
    if not playbook_lines and not profile_lines:
        return "", []
    sections: list[str] = []
    if playbook_lines:
        sections.append("### Relevant project-specific skills")
        sections.extend(playbook_lines)
    if profile_lines:
        sections.append("### Relevant project preferences")
        sections.extend(profile_lines)
    sections.append(oc_cite.CITATION_INSTRUCTION)
    return "\n".join(sections) + "\n", playbook_entries + profile_entries


def _format_combined_playbooks(
    *,
    agent_playbooks: Iterable[Any],
    user_playbooks: Iterable[Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Render agent playbooks first, then user playbooks, with one shared rank counter."""
    lines: list[str] = []
    entries: list[dict[str, Any]] = []
    rank = 0
    for pb in agent_playbooks:
        rank = _append_playbook_bullet(
            pb, "agent_playbook_id", "agent_playbook", rank, lines, entries
        )
    for pb in user_playbooks:
        rank = _append_playbook_bullet(
            pb, "user_playbook_id", "user_playbook", rank, lines, entries
        )
    return lines, entries


def _append_playbook_bullet(
    pb: Any,
    id_field: str,
    source_kind: str,
    rank: int,
    lines: list[str],
    entries: list[dict[str, Any]],
) -> int:
    content = _first_nonempty(_field(pb, "content"))
    if not content:
        return rank
    rank += 1
    trigger = _first_nonempty(_field(pb, "trigger"))
    rationale = _first_nonempty(_field(pb, "rationale"))
    real_id = _field(pb, id_field)
    item_id = oc_cite.rank_id("playbook", rank, real_id)
    title = _title_from_content(content)
    bullet = f"- [oc:{item_id}] {content}"
    if trigger:
        bullet += f" _(when: {trigger})_"
    if rationale:
        bullet += f" — *why:* {rationale}"
    lines.append(bullet)
    entries.append(
        {
            "id": item_id,
            "kind": "playbook",
            "title": title,
            "content": content,
            "real_id": str(real_id) if real_id is not None else None,
            "source_kind": source_kind,
        }
    )
    return rank


def _format_profiles(
    profiles: Iterable[Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    lines: list[str] = []
    entries: list[dict[str, Any]] = []
    rank = 0
    for p in profiles:
        content = _first_nonempty(_field(p, "content"))
        if not content:
            continue
        rank += 1
        real_id = _field(p, "profile_id")
        item_id = oc_cite.rank_id("profile", rank, real_id)
        title = _title_from_content(content)
        lines.append(f"- [oc:{item_id}] {content}")
        entries.append(
            {
                "id": item_id,
                "kind": "profile",
                "title": title,
                "content": content,
                "real_id": str(real_id) if real_id is not None else None,
            }
        )
    return lines, entries


def _title_from_content(content: str, limit: int = 80) -> str:
    """Derive a compact human-readable title from a bullet's content."""
    text = content.strip()
    if not text:
        return ""
    for terminator in (". ", "\n"):
        idx = text.find(terminator)
        if 0 < idx <= limit:
            return text[:idx].rstrip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from either an attribute or a dict key."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
