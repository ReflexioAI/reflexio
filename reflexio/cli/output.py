"""Structured output formatting for Reflexio CLI.

All commands produce output through this module, ensuring a consistent
JSON envelope for agent consumption and Rich formatting for humans.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC
from typing import Any

# ---------------------------------------------------------------------------
# JSON envelope
# ---------------------------------------------------------------------------


def render(
    data: Any,
    *,
    json_mode: bool = False,
    meta: dict[str, Any] | None = None,
    exclude_none: bool = True,
) -> None:
    """Render command output to stdout.

    In JSON mode, wraps data in a standard envelope:
    ``{"ok": true, "data": ..., "meta": {...}}``

    In human mode, prints the data as-is (string) or pretty-prints dicts/lists.

    Args:
        data: The data to render (dict, list, string, or Pydantic model)
        json_mode: If True, output JSON envelope; otherwise human-readable
        meta: Optional metadata (count, limit, has_more, etc.)
        exclude_none: If True (default), omit None fields from Pydantic models
    """
    if json_mode:
        envelope: dict[str, Any] = {
            "ok": True,
            "data": _serialize(data, exclude_none=exclude_none),
        }
        if meta:
            envelope["meta"] = meta
        print(json.dumps(envelope, indent=2, default=str))
    elif isinstance(data, str):
        print(data)
    elif isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, default=str))
    elif hasattr(data, "model_dump"):
        # Pydantic model safety net — serialize to dict instead of __repr__
        print(
            json.dumps(
                data.model_dump(mode="json", exclude_none=exclude_none),
                indent=2,
                default=str,
            )
        )
    else:
        print(data)


def _serialize(data: Any, *, exclude_none: bool = True) -> Any:
    """Serialize data for JSON output, handling Pydantic models.

    Args:
        data: Data to serialize
        exclude_none: If True, omit None fields from Pydantic models

    Returns:
        Any: JSON-serializable data
    """
    if hasattr(data, "model_dump"):
        return data.model_dump(mode="json", exclude_none=exclude_none)
    if isinstance(data, list):
        return [_serialize(item, exclude_none=exclude_none) for item in data]
    return data


# ---------------------------------------------------------------------------
# Human-readable formatters (kept from _output.py for backward compat)
# ---------------------------------------------------------------------------


def print_info(msg: str) -> None:
    """Print an informational message to stderr.

    Args:
        msg: Message to print
    """
    print(msg, file=sys.stderr)


def print_error(msg: str) -> None:
    """Print an error message to stderr.

    Args:
        msg: Error message to print
    """
    print(f"Error: {msg}", file=sys.stderr)


def _lifecycle_tag(obj: Any) -> str:
    """Return a lifecycle status tag for objects with a ``status`` attribute.

    Only emits a tag for non-current statuses (PENDING, ARCHIVED).  Objects
    whose status is CURRENT or None are considered active and get no tag.

    Args:
        obj: Any object that may have a ``status`` attribute

    Returns:
        str: Tag string like " [PENDING]" or "", including leading space
    """
    status = getattr(obj, "status", None)
    if (
        status is not None
        and isinstance(status, str)
        and status.upper() not in ("CURRENT", "")
    ):
        return f" [{status.upper()}]"
    # Handle enum-style status where .value is not None
    if status is not None and hasattr(status, "value") and status.value is not None:
        return f" [{str(status.value).upper()}]"
    return ""


def format_profiles(profiles: list[Any]) -> str:
    """Format profile objects as markdown bullet list.

    Shows a lifecycle tag ([PENDING]/[ARCHIVED]) when the profile status is
    not CURRENT or None.

    Args:
        profiles: Profile objects with a profile_content attribute

    Returns:
        str: Formatted profiles, one per line
    """
    if not profiles:
        return ""
    return "\n".join(f"- {p.content}{_lifecycle_tag(p)}" for p in profiles)


def _structured_lines(sd: Any) -> list[str]:
    """Build indented structured-data lines from a StructuredData-like object.

    Uses the canonical labels: Trigger, Instruction, Pitfall, Rationale.

    Args:
        sd: Object with optional trigger, instruction, pitfall, rationale attrs

    Returns:
        list[str]: Indented lines (may be empty)
    """
    lines: list[str] = []
    if sd is None:
        return lines
    if getattr(sd, "trigger", None):
        lines.append(f"  Trigger: {sd.trigger}")
    if getattr(sd, "instruction", None):
        lines.append(f"  Instruction: {sd.instruction}")
    if getattr(sd, "pitfall", None):
        lines.append(f"  Pitfall: {sd.pitfall}")
    if getattr(sd, "rationale", None):
        lines.append(f"  Rationale: {sd.rationale}")
    return lines


def format_agent_playbooks(agent_playbooks: list[Any]) -> str:
    """Format agent playbook objects with structured data.

    Shows an approval status tag ([APPROVED]/[PENDING]/[REJECTED]) from
    ``playbook_status`` and a lifecycle tag ([PENDING]/[ARCHIVED]) from
    ``status`` when not CURRENT/None.

    Args:
        agent_playbooks: Agent playbook objects with content and structured_data

    Returns:
        str: Formatted agent playbooks with structured action lines
    """
    if not agent_playbooks:
        return ""
    blocks: list[str] = []
    for playbook in agent_playbooks:
        # Approval status tag from playbook_status
        approval_tag = ""
        ps = getattr(playbook, "playbook_status", None)
        if ps is not None:
            ps_str = str(ps).upper() if not isinstance(ps, str) else ps.upper()
            if ps_str:
                approval_tag = f" [{ps_str}]"

        lifecycle = _lifecycle_tag(playbook)
        parts = [f"- {playbook.content}{approval_tag}{lifecycle}"]
        parts.extend(_structured_lines(playbook.structured_data))
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def format_user_playbooks(user_playbooks: list[Any]) -> str:
    """Format user playbook objects with structured data and source metadata.

    User playbooks show source/request_id metadata instead of approval status.
    A lifecycle tag ([PENDING]/[ARCHIVED]) is shown when status is not
    CURRENT/None.

    Args:
        user_playbooks: User playbook objects with content, structured_data,
            source, and request_id

    Returns:
        str: Formatted user playbooks with structured action lines and source
    """
    if not user_playbooks:
        return ""
    blocks: list[str] = []
    for playbook in user_playbooks:
        lifecycle = _lifecycle_tag(playbook)
        parts = [f"- {playbook.content}{lifecycle}"]
        parts.extend(_structured_lines(playbook.structured_data))
        source = getattr(playbook, "source", None) or "unknown"
        request_id = getattr(playbook, "request_id", None) or "unknown"
        parts.append(f"  Source: {source} (request: {request_id})")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Rich human-readable rendering
# ---------------------------------------------------------------------------
#
# The ``format_*`` functions above return plain text and are used by
# ``format_context`` (which builds the agent bootstrap context block —
# pure text, no rich). The ``print_*`` functions below render directly
# to stdout via a ``rich.console.Console`` so they can wrap long lines,
# colorize labels, and visually separate entries. ``rich.console.Console``
# auto-detects whether stdout is a TTY and degrades to plain text when
# piped, so these are safe to use unconditionally.


def _human_relative_timestamp(ts: int | None) -> str:
    """Render a Unix timestamp as a short relative description.

    Examples: ``"2m ago"``, ``"3h ago"``, ``"5d ago"``, ``"2026-04-10"``.
    Returns ``"unknown"`` for None or invalid input.
    """
    if not ts:
        return "unknown"
    try:
        from datetime import datetime

        delta = datetime.now(UTC) - datetime.fromtimestamp(int(ts), tz=UTC)
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return f"{seconds}s ago"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        if seconds < 86400 * 30:
            return f"{seconds // 86400}d ago"
        return datetime.fromtimestamp(int(ts), tz=UTC).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "unknown"


def print_user_profiles(profiles: list[Any]) -> None:
    """Render user profiles with rich formatting: numbered, wrapped, dimmed metadata.

    Each profile is shown as a numbered, word-wrapped block. The
    profile content is the primary visual element; source / age /
    lifecycle tag render in a dim metadata footer beneath. When stdout
    is piped, rich auto-detects and emits plain text without colors so
    the output stays grep-friendly.

    Args:
        profiles: Profile objects with at least ``content``; optional
            ``source``, ``last_modified_timestamp``, ``status``.
    """
    if not profiles:
        return
    from rich.console import Console
    from rich.text import Text

    console = Console()
    for i, p in enumerate(profiles, 1):
        # Index + content. ``Text.assemble`` keeps the styles separate
        # while wrapping the whole line at the terminal width.
        line = Text.assemble(
            (f"[{i}] ", "bold cyan"),
            getattr(p, "content", "") or "",
        )
        console.print(line)

        # Dim metadata footer: source, age, lifecycle.
        meta_parts: list[str] = []
        source = getattr(p, "source", None)
        if source:
            meta_parts.append(f"source: {source}")
        ts = getattr(p, "last_modified_timestamp", None)
        if ts:
            meta_parts.append(_human_relative_timestamp(ts))
        lifecycle = _lifecycle_tag(p).strip()
        if lifecycle:
            meta_parts.append(lifecycle)
        if meta_parts:
            console.print(Text("    " + " · ".join(meta_parts), style="dim"))
        console.print()  # blank line between entries


def print_user_playbooks(user_playbooks: list[Any]) -> None:
    """Render user playbooks as rich panels with aligned structured fields.

    Each playbook becomes a bordered panel. The panel title is the
    rule's content (the human-readable summary). The body is a
    two-column grid of the structured fields (Trigger / Instruction
    / Pitfall / Rationale) with right-aligned bold labels and
    word-wrapped values. A dim footer shows source + short request ID.

    Args:
        user_playbooks: Playbook objects with ``content``,
            ``structured_data`` (trigger/instruction/pitfall/rationale),
            ``source``, ``request_id``, optional ``status``.
    """
    if not user_playbooks:
        return
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    for i, pb in enumerate(user_playbooks, 1):
        # Build a borderless grid for the structured fields. The label
        # column is right-aligned and bold-dim; the value column wraps
        # at the panel width.
        grid = Table.grid(padding=(0, 2), expand=True)
        grid.add_column(style="bold dim", justify="right", no_wrap=True)
        grid.add_column(overflow="fold")

        sd = getattr(pb, "structured_data", None)
        for label, attr in (
            ("Trigger", "trigger"),
            ("Instruction", "instruction"),
            ("Pitfall", "pitfall"),
            ("Rationale", "rationale"),
        ):
            value = getattr(sd, attr, None) if sd is not None else None
            if value:
                grid.add_row(label, value)

        # Footer: source + truncated request_id.
        source = getattr(pb, "source", None) or "unknown"
        request_id = getattr(pb, "request_id", None) or "unknown"
        short_req = request_id[:8] if request_id != "unknown" else request_id
        footer = Text(f"source: {source} · request: {short_req}", style="dim")

        # Content goes inside the panel body, NOT the title — Rich's
        # Panel title doesn't word-wrap and would truncate long rule
        # summaries with a "─" continuation, leaving the most
        # important part of the playbook unreadable. The title stays
        # short (just the index) so the panel border is always tidy.
        content_text = Text(
            getattr(pb, "content", "") or "", style="bold", overflow="fold"
        )
        title = Text.assemble((f"[{i}]", "bold cyan"))
        lifecycle = _lifecycle_tag(pb).strip()
        if lifecycle:
            title.append(f" {lifecycle}", style="yellow")

        panel = Panel(
            Group(content_text, Text(""), grid, Text(""), footer),
            title=title,
            title_align="left",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(panel)
        console.print()  # spacer between panels


def mask_api_key(api_key: str) -> str:
    """Mask an API key for safe display in command output.

    Keeps the short provider prefix (``rflx-``, ``sk-``, etc.) and the
    last 4 characters so the user can still distinguish which key is
    active without exposing the full secret. Used by ``reflexio status
    whoami`` and ``reflexio auth status``.

    Args:
        api_key (str): The raw API key, possibly empty.

    Returns:
        str: A masked representation safe for printing. Returns
            ``"<unset>"`` for an empty key, ``"*" * len`` for very
            short keys (<=8 chars), and ``"{prefix}****{last4}"``
            for normal keys.
    """
    if not api_key:
        return "<unset>"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    prefix = api_key[:5] if "-" in api_key[:8] else api_key[:4]
    return f"{prefix}****{api_key[-4:]}"


def print_agent_playbooks(agent_playbooks: list[Any]) -> None:
    """Render agent playbooks as rich panels with approval badge.

    Mirrors :func:`print_user_playbooks` but the title row shows an
    approval badge (``[APPROVED]`` / ``[PENDING]`` / ``[REJECTED]``)
    colour-coded green/yellow/red instead of the source metadata
    footer. The body still shows the structured Trigger / Instruction
    / Pitfall / Rationale grid. The footer shows ``playbook_name ·
    agent_version · <relative time>``.

    Args:
        agent_playbooks: Agent playbook objects with ``content``,
            ``structured_data``, ``playbook_status``, optional
            ``playbook_name``, ``agent_version``, ``created_at``,
            ``status``.
    """
    if not agent_playbooks:
        return
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    badge_styles = {
        "APPROVED": "bold green",
        "PENDING": "bold yellow",
        "REJECTED": "bold red",
    }

    console = Console()
    for i, pb in enumerate(agent_playbooks, 1):
        grid = Table.grid(padding=(0, 2), expand=True)
        grid.add_column(style="bold dim", justify="right", no_wrap=True)
        grid.add_column(overflow="fold")

        sd = getattr(pb, "structured_data", None)
        for label, attr in (
            ("Trigger", "trigger"),
            ("Instruction", "instruction"),
            ("Pitfall", "pitfall"),
            ("Rationale", "rationale"),
        ):
            value = getattr(sd, attr, None) if sd is not None else None
            if value:
                grid.add_row(label, value)

        # Title: [N] + approval badge + lifecycle tag
        title = Text.assemble((f"[{i}]", "bold cyan"))
        ps = getattr(pb, "playbook_status", None)
        if ps is not None:
            ps_str = (ps if isinstance(ps, str) else str(ps)).upper()
            if ps_str:
                style = badge_styles.get(ps_str, "bold white")
                title.append(f" [{ps_str}]", style=style)
        lifecycle = _lifecycle_tag(pb).strip()
        if lifecycle:
            title.append(f" {lifecycle}", style="yellow")

        # Footer: playbook_name · agent_version · relative created_at
        footer_parts: list[str] = []
        playbook_name = getattr(pb, "playbook_name", None)
        if playbook_name:
            footer_parts.append(str(playbook_name))
        agent_version = getattr(pb, "agent_version", None)
        if agent_version:
            footer_parts.append(f"v{agent_version}")
        created_at = getattr(pb, "created_at", None)
        if created_at:
            footer_parts.append(_human_relative_timestamp(int(created_at)))

        content_text = Text(
            getattr(pb, "content", "") or "", style="bold", overflow="fold"
        )
        body_parts: list[Any] = [content_text, Text(""), grid]
        if footer_parts:
            body_parts.extend([Text(""), Text(" · ".join(footer_parts), style="dim")])

        panel = Panel(
            Group(*body_parts),
            title=title,
            title_align="left",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(panel)
        console.print()


def print_interactions(interactions: list[Any]) -> None:
    """Render interactions grouped by request_id as rich panels.

    Each request_id becomes one bordered panel. The title shows the
    short request id, relative timestamp, and turn count. The body is
    a borderless two-column grid with role labels in the left column
    (User cyan, Assistant magenta, System dim, others white) and
    folded content on the right. When stdout is piped, rich
    auto-detects and degrades to plain text.

    Args:
        interactions: InteractionView-like objects with
            ``request_id``, ``role``, ``content``, ``created_at``.
    """
    if not interactions:
        return
    from collections import defaultdict

    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    role_styles: dict[str, str] = {
        "user": "bold cyan",
        "assistant": "bold magenta",
        "system": "bold dim",
    }

    groups: dict[str, list[Any]] = defaultdict(list)
    group_mins: dict[str, int] = {}
    for ix in interactions:
        groups[ix.request_id].append(ix)
        existing = group_mins.get(ix.request_id)
        if existing is None or ix.created_at < existing:
            group_mins[ix.request_id] = ix.created_at

    # Stable order: earliest group first (O(N) pre-pass above + O(G log G) sort)
    sorted_groups = sorted(groups.items(), key=lambda g: group_mins[g[0]])

    console = Console()
    for request_id, items in sorted_groups:
        ts = group_mins[request_id]
        try:
            relative = _human_relative_timestamp(int(ts))
        except (TypeError, ValueError):
            relative = "unknown"
        short_id = (request_id or "unknown")[:8]
        turn_count = len(items)
        turn_label = "turn" if turn_count == 1 else "turns"
        title = Text.assemble(
            ("req: ", "dim"),
            (short_id, "bold cyan"),
            ("  ·  ", "dim"),
            (relative, "dim"),
            ("  ·  ", "dim"),
            (f"{turn_count} {turn_label}", "dim"),
        )

        grid = Table.grid(padding=(0, 2), expand=True)
        grid.add_column(justify="right", no_wrap=True)
        grid.add_column(overflow="fold")

        for item in sorted(items, key=lambda i: i.created_at):
            role = (item.role or "").strip() or "unknown"
            style = role_styles.get(role.lower(), "bold white")
            grid.add_row(Text(role.capitalize(), style=style), item.content or "")

        panel = Panel(
            grid,
            title=title,
            title_align="left",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(panel)
        console.print()


def print_storage_credentials(
    storage_type: str | None,
    storage_config: dict[str, Any],
    *,
    revealed: bool,
) -> None:
    """Render the output of ``reflexio config storage`` as a rich table.

    Prints a bold "Storage type: ..." header followed by a two-column
    grid (field name / value). When the credentials are masked, a
    dim-italic hint reminds the user that ``--reveal`` exists.

    Args:
        storage_type: The server's storage_type label (e.g.,
            ``"sqlite"``, ``"supabase"``).
        storage_config: The serialized StorageConfig dict (already
            masked or revealed by the caller).
        revealed: Whether ``storage_config`` contains raw credentials.
            When False, a reveal-hint footer is printed.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    console.print(
        Text.assemble(
            ("Storage type: ", "bold dim"),
            (storage_type or "unknown", "bold"),
        )
    )

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    for key, value in storage_config.items():
        grid.add_row(str(key), "<none>" if value is None else str(value))
    console.print(grid)

    if not revealed:
        console.print(
            Text(
                "(use --reveal to print raw credential values)",
                style="dim italic",
            )
        )


def print_whoami_summary(
    *,
    endpoint: str,
    api_key: str,
    org_id: str | None,
    storage_type: str | None,
    storage_label: str | None,
    storage_configured: bool,
    message: str | None,
) -> None:
    """Render the ``reflexio status whoami`` summary as a rich table.

    Emits a two-column grid (right-aligned bold-dim label column +
    value column) covering endpoint, masked API key, org id, and
    storage info. The ``[configured]`` / ``[unconfigured]`` marker is
    colour-coded green/yellow. A trailing dim-italic line renders any
    message the server attached to the whoami payload.

    Args:
        endpoint: Fully-qualified server URL (``client.base_url``).
        api_key: The raw resolved API key — masked internally via
            :func:`mask_api_key`.
        org_id: Organization id from the whoami response, or None.
        storage_type: Storage backend label from the whoami response.
        storage_label: Human-friendly storage name (e.g., a DSN with
            masked credentials), or None.
        storage_configured: Whether the server reports configured
            storage for this org.
        message: Optional advisory message from the whoami response.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", justify="right", no_wrap=True)
    grid.add_column(overflow="fold")

    grid.add_row("Endpoint", endpoint)
    grid.add_row("API key", mask_api_key(api_key))
    grid.add_row("Org ID", org_id or "<none>")
    if storage_type:
        grid.add_row("Storage type", storage_type)

    if storage_label:
        marker_text = "[configured]" if storage_configured else "[unconfigured]"
        marker_style = "green" if storage_configured else "yellow"
        storage_value = Text.assemble(
            (storage_label, ""),
            ("  ", ""),
            (marker_text, marker_style),
        )
        grid.add_row("Storage", storage_value)
    elif not storage_configured:
        grid.add_row("Storage", Text("<not configured>", style="yellow"))

    console.print(grid)

    if message:
        console.print(Text(message, style="dim italic"))


def print_doctor_checks(checks: list[dict[str, Any]]) -> None:
    """Render ``reflexio doctor check`` results as a rich grid.

    Each check row becomes ``<icon> <name>: <message>`` with the icon
    and name colour-coded by status — green ``✓`` for pass, yellow
    ``⚠`` for warn, red ``✗`` for fail. If a check has a hint, it is
    rendered on an indented second line in dim style.

    Args:
        checks: Doctor check dicts with keys ``status`` (``"pass"``,
            ``"warn"``, or ``"fail"``), ``name``, ``message``, and
            ``hint`` (possibly None).
    """
    if not checks:
        return
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    icon_styles: dict[str, tuple[str, str]] = {
        "pass": ("✓", "bold green"),
        "warn": ("⚠", "bold yellow"),
        "fail": ("✗", "bold red"),
    }

    console = Console()
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(no_wrap=True)  # icon
    grid.add_column(no_wrap=True)  # name
    grid.add_column(overflow="fold")  # message + hint

    for check in checks:
        status = check.get("status", "warn")
        icon, style = icon_styles.get(status, ("?", "bold white"))
        name = check.get("name", "") or ""
        message = check.get("message", "") or ""
        hint = check.get("hint")

        body = Text(message)
        if hint:
            body.append("\n")
            body.append(Text(f"  Hint: {hint}", style="dim"))

        grid.add_row(
            Text(icon, style=style),
            Text(f"{name}:", style="bold"),
            body,
        )
    console.print(grid)


def print_auth_status(url: str, api_key: str, env_path: str) -> None:
    """Render ``reflexio auth status`` as a two-column rich grid.

    Three rows: URL, masked API Key (via :func:`mask_api_key`), and
    env file path.

    Args:
        url: The configured REFLEXIO_URL value, possibly empty.
        api_key: The raw REFLEXIO_API_KEY value, possibly empty.
        env_path: Absolute path to the env file.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold dim", justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("URL", url or "<unset>")
    grid.add_row("API Key", mask_api_key(api_key))
    grid.add_row("Env file", env_path)
    console.print(grid)


def format_interactions(interactions: list[Any]) -> str:
    """Format interaction objects grouped by request_id as conversation turns.

    Args:
        interactions: InteractionView objects with request_id, role, content, created_at

    Returns:
        str: Formatted conversation turns
    """
    if not interactions:
        return ""

    from collections import defaultdict
    from datetime import datetime

    groups: dict[str, list[Any]] = defaultdict(list)
    for ix in interactions:
        groups[ix.request_id].append(ix)

    # Sort groups by earliest created_at
    sorted_groups = sorted(
        groups.items(), key=lambda g: min(i.created_at for i in g[1])
    )

    blocks: list[str] = []
    for request_id, items in sorted_groups:
        ts = min(i.created_at for i in items)
        try:
            dt = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError, OSError):
            dt = "unknown"
        short_id = request_id[:8]
        header = f"── {short_id} ({dt} UTC) ──"
        lines = [header]
        for item in sorted(items, key=lambda i: i.created_at):
            role = item.role.capitalize()
            lines.append(f"  {role}: {item.content}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def format_context(
    profiles: list[Any],
    agent_playbooks: list[Any],
) -> str:
    """Build a combined context block from profiles and agent playbooks.

    Args:
        profiles: Profile objects
        agent_playbooks: Agent playbook objects

    Returns:
        str: Combined markdown context block, or empty string if all empty
    """
    sections: list[str] = []

    if profile_text := format_profiles(profiles):
        sections.append(f"## User Preferences — APPLY THESE\n{profile_text}")

    if playbook_text := format_agent_playbooks(agent_playbooks):
        sections.append(f"## Behavior Corrections — FOLLOW THESE\n{playbook_text}")

    if not sections:
        return ""

    return (
        "---\n# IMPORTANT: Apply These Corrections (from Reflexio)\n\n"
        + "\n\n".join(sections)
        + "\n\nFailure to follow these means repeating mistakes the user already corrected.\n---"
    )


def pagination_meta(
    items: list[Any],
    limit: int,
) -> dict[str, Any]:
    """Build pagination metadata for output envelope.

    Args:
        items: The items returned
        limit: The requested limit

    Returns:
        dict: Metadata with count, limit, and has_more flag
    """
    return {
        "count": len(items),
        "limit": limit,
        "has_more": len(items) == limit,
    }
