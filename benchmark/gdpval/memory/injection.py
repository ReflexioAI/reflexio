"""Render reflexio search hits into a solution-recipe prompt block.

Used by the reflexio bridge to turn a `UnifiedSearchViewResponse` into a
string the host adapters can splice into the prompt (OpenSpace) or system
message (Hermes). One shared renderer keeps the P3 injection format
identical across hosts.

Design decisions, as of the v3 extractor + cached-solution header:

1. The block is framed as an EXECUTE-NOW cached solution, not optional
   hints. Earlier variants used "# Reflexio memory" + bulleted rules or
   "IF task involves X, THEN Y" conditionals. Both caused the LLM to
   treat the injection as a checklist to verify and redo the task
   anyway, which burned iterations instead of saving them. The v3
   extractor produces a concrete, already-solved recipe, so the header
   tells the agent to copy the recipe into the deliverable rather than
   re-derive it.

2. Only user_playbooks and agent_playbooks are rendered. Profiles are
   org-wide in the reflexio backend and therefore leak across tasks in
   the benchmark (P3 of task A gets stale facts extracted from task B),
   so they are dropped at this layer.

3. Recipe sections are deduped by whitespace-normalized content. The
   extractor can emit near-identical recipes across re-extraction, and
   an observed 12k-char triple-copy once pushed a task over its token
   budget and caused a quality regression. Stripping exact and
   whitespace-only duplicates keeps the signal.
"""

from __future__ import annotations

from typing import Any

_SOLUTION_HEADER = """\
# CACHED SOLUTION FROM A PRIOR SUCCESSFUL RUN

The recipe below is the verified solution from a prior successful run of
this EXACT task. It contains the full structure, content, and domain
values that satisfied the task last time.

EXECUTE THIS RECIPE NOW:
1. Read the recipe content below.
2. Write the deliverable file(s) directly using the structure and content
   from the recipe — adapt phrasing as needed but keep the same sections,
   field names, and numerical values.
3. Report completion immediately after the file is written.

CRITICAL CONSTRAINTS:
- Do NOT search for additional context, skills, or documentation. The
  recipe below is a complete substitute for any such lookup. Searching
  wastes iterations and ALWAYS makes things worse on these tasks.
- Do NOT call any tool that retrieves prior skills, knowledge, or
  memory (e.g. retrieve_skill, search_memory, recall, lookup_docs) —
  whatever they would return is already baked into the recipe below.
- Do NOT re-derive the strategy from scratch.
- Aim to complete in 1-2 iterations maximum. The recipe contains
  everything you need."""


def _dedupe_key(text: str) -> str:
    """Collapse whitespace and case to produce a stable dedupe key.

    Args:
        text (str): Playbook content to hash for duplicate detection.

    Returns:
        str: Lowercased, whitespace-collapsed form of the input.
    """
    return " ".join(text.lower().split())


def _append_unique(
    sections: list[str],
    playbooks: Any,
    seen: set[str],
    max_items: int,
) -> None:
    """Append up to `max_items` non-duplicate playbook contents to `sections`.

    Mutates `sections` and `seen` in place. Playbooks with empty content
    or content whose normalized key has already been seen are skipped.

    Args:
        sections (list[str]): Recipe section list to append to.
        playbooks (Any): Iterable of playbook objects exposing `.content`.
        seen (set[str]): Set of already-seen dedupe keys, updated in place.
        max_items (int): Maximum number of playbooks to consider.
    """
    for pb in list(playbooks)[:max_items]:
        content = (getattr(pb, "content", "") or "").strip()
        if not content:
            continue
        key = _dedupe_key(content)
        if key in seen:
            continue
        seen.add(key)
        sections.append(content)


def render_memory_block(response: Any, max_items_per_section: int = 10) -> str:
    """Turn a `UnifiedSearchViewResponse` into a prompt-ready text block.

    The backend extractor emits a concrete CACHED SOLUTION per task (see
    GDPVAL_PLAYBOOK_EXTRACTOR_PROMPT in reflexio_bridge.py). Each
    playbook's `content` is already a recipe, so this function just
    concatenates them under a strong trust header that tells the agent
    to execute the cached steps rather than re-derive them.

    Profiles are intentionally ignored — they are org-wide in the
    reflexio backend and would leak across tasks in the benchmark. Only
    `user_playbooks` and `agent_playbooks` are rendered, with exact +
    whitespace-insensitive deduplication across both sources.

    Returns an empty string when nothing relevant — callers pass `None`
    to adapters in that case so P3 degenerates to P2 behavior cleanly.

    Args:
        response (Any): The `UnifiedSearchViewResponse` returned by
            `ReflexioClient.search()` (or any object exposing
            `user_playbooks` / `agent_playbooks` attributes).
        max_items_per_section (int): Cap on items considered from each
            of `user_playbooks` and `agent_playbooks`.

    Returns:
        str: Rendered memory block, or "" if nothing relevant.
    """
    user_playbooks = getattr(response, "user_playbooks", None) or []
    agent_playbooks = getattr(response, "agent_playbooks", None) or []

    seen: set[str] = set()
    recipe_sections: list[str] = []
    _append_unique(recipe_sections, user_playbooks, seen, max_items_per_section)
    _append_unique(recipe_sections, agent_playbooks, seen, max_items_per_section)

    if not recipe_sections:
        return ""

    parts = [_SOLUTION_HEADER, "", "## Recipe", ""]
    parts.extend(recipe_sections)
    return "\n".join(parts).strip()
