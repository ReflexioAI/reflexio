"""Compose a reflexio search query from a before_tool_call payload.

Deterministic — no LLM call — so the before_tool_call hook can stay inside
its latency budget. The output is fed to ``ReflexioClient.search(query=...)``
(the unified ``/api/search`` endpoint, which fans out to user playbooks,
agent playbooks, and preferences server-side), which tokenizes via reflexio's
FTS5 sanitizer (OR-joined, stemmed) plus a vector-similarity leg. Short,
meaning-dense strings give the most selective hybrid ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

_MAX_SNIPPET_LEN = 400


def from_tool_call(tool_name: str, tool_input: Mapping[str, Any]) -> str:
    """Compose a search query from an openClaw before_tool_call payload.

    Args:
        tool_name (str): Tool name as reported by openClaw (after
            camelCase→snake_case translation, e.g. ``"Edit"``, ``"Bash"``).
        tool_input (Mapping[str, Any]): The tool's input dict as delivered
            by the hook payload (after camelCase translation).

    Returns:
        str: A short query suitable for reflexio hybrid search, or ``""``
            when the tool is not one we compose for (caller should then
            skip the search entirely).
    """
    match tool_name:
        case "Edit" | "Write" | "NotebookEdit":
            return _from_file_edit(tool_input)
        case "Bash":
            return _from_bash(tool_input)
        case _:
            return ""


def _as_str(value: Any) -> str:
    """Coerce a tool-payload field to a string, treating non-strings as empty.

    Tool inputs come from external openClaw payloads; a malformed event with
    a non-string ``new_string`` or ``command`` would otherwise crash the
    hook with ``AttributeError`` / ``TypeError``. We prefer a clean empty
    query over a partial failure.
    """
    return value if isinstance(value, str) else ""


def _from_file_edit(tool_input: Mapping[str, Any]) -> str:
    path = _as_str(tool_input.get("file_path"))
    snippet = _as_str(tool_input.get("new_string")) or _as_str(
        tool_input.get("content")
    )
    basename = Path(path).name if path else ""
    return f"{basename} {snippet[:_MAX_SNIPPET_LEN]}".strip()


def _from_bash(tool_input: Mapping[str, Any]) -> str:
    command = _as_str(tool_input.get("command"))
    first_line = command.splitlines()[0] if command else ""
    return first_line[:_MAX_SNIPPET_LEN].strip()
