"""Shared display utilities for Reflexio notebooks.

Provides consistent, pretty output formatting across all notebooks.
Import with: `from _display_helpers import *`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from IPython.display import HTML, Markdown, display
from rich import print as rprint
from rich.console import Console
from rich.json import JSON as RichJSON  # noqa: N811

if TYPE_CHECKING:
    from reflexio import Config, UserProfile

console = Console(force_jupyter=True)

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------


def load_env() -> tuple[str, str]:
    """Load environment variables and return the Reflexio server URL and API key.

    Walks up from the notebooks directory to find the nearest `.env` file,
    loads it, then reads REFLEXIO_API_URL and REFLEXIO_API_KEY.

    Returns:
        tuple[str, str]: A (url, api_key) pair for constructing a ReflexioClient.
    """
    from dotenv import load_dotenv

    _this_dir = Path(__file__).resolve().parent
    for parent in (_this_dir, *_this_dir.parents):
        if (env_file := parent / ".env").exists():
            load_dotenv(env_file)
            break

    url = os.environ.get(
        "REFLEXIO_API_URL", f"http://localhost:{os.environ.get('BACKEND_PORT', '8081')}"
    )
    api_key = os.environ.get("REFLEXIO_API_KEY", "")

    show_success(f"Environment loaded — server URL: {url}")
    return url, api_key


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def show_profiles(profiles: list[UserProfile], title: str = "User Profiles") -> None:
    """Display user profiles as a formatted pandas DataFrame."""
    display(Markdown(f"### {title}"))
    if not profiles:
        rprint(
            "[dim italic]No profiles found. Publish some interactions first![/dim italic]"
        )
        return

    rows = []
    for i, p in enumerate(profiles, 1):
        content = p.content
        if len(content) > 120:
            content = content[:117] + "..."
        rows.append(
            {
                "#": i,
                "Content": content,
                "Extractor": getattr(p, "extractor_names", ["—"])[0]
                if getattr(p, "extractor_names", None)
                else "—",
                "Status": getattr(p, "status", None) or "current",
            }
        )

    df = pd.DataFrame(rows)
    display(df)


def show_interactions(interactions: list, title: str = "Interactions") -> None:
    """Display interactions as a formatted pandas DataFrame with chat-style layout."""
    display(Markdown(f"### {title}"))
    if not interactions:
        rprint("[dim italic]No interactions found.[/dim italic]")
        return

    rows = []
    for i, interaction in enumerate(interactions, 1):
        content = (
            interaction.content if hasattr(interaction, "content") else str(interaction)
        )
        if len(content) > 150:
            content = content[:147] + "..."
        role = interaction.role if hasattr(interaction, "role") else "—"
        tools = ""
        if hasattr(interaction, "tools_used") and interaction.tools_used:
            tools = ", ".join(t.tool_name for t in interaction.tools_used)
        action = ""
        if (
            hasattr(interaction, "user_action")
            and interaction.user_action
            and str(interaction.user_action) != "none"
        ):
            action = str(interaction.user_action)

        rows.append(
            {
                "#": i,
                "Role": role,
                "Content": content,
                "Tools": tools or "—",
                "Action": action or "—",
            }
        )

    df = pd.DataFrame(rows)
    display(df)


def show_playbooks(playbooks: list, title: str = "Playbooks") -> None:
    """Display user or agent playbooks as a formatted DataFrame.

    Works with both UserPlaybook and AgentPlaybook objects.
    """
    display(Markdown(f"### {title}"))
    if not playbooks:
        rprint(
            "[dim italic]No playbooks found. The agent needs more interactions to learn from![/dim italic]"
        )
        return

    rows = []
    for i, fb in enumerate(playbooks, 1):
        # Handle both UserPlaybook and AgentPlaybook objects
        content = getattr(fb, "content", "") or getattr(fb, "feedback_content", "")
        if len(content) > 120:
            content = content[:117] + "..."

        name = getattr(fb, "playbook_name", "") or getattr(fb, "feedback_name", "—")
        status = (
            getattr(fb, "playbook_status", None)
            or getattr(fb, "feedback_status", None)
            or getattr(fb, "status", None)
            or "current"
        )

        # Trigger field (user playbooks)
        trigger = ""
        if hasattr(fb, "trigger") and fb.trigger:
            trigger = fb.trigger or ""
        if len(trigger) > 80:
            trigger = trigger[:77] + "..."

        row = {
            "#": i,
            "Content": content,
            "Name": name,
            "Status": str(status),
        }
        if trigger:
            row["Trigger/When"] = trigger
        rows.append(row)

    df = pd.DataFrame(rows)
    display(df)


def show_config(config: Config) -> None:
    """Display a structured summary of the Reflexio configuration."""
    display(Markdown("### Configuration Summary"))

    extractors = config.profile_extractor_configs or []
    playbook_configs = config.user_playbook_extractor_configs or []
    tools = config.tool_can_use or []
    success_configs = config.agent_success_configs or []

    summary = f"""| Setting | Value |
|---------|-------|
| **Profile Extractors** | {len(extractors)} configured |
| **Playbook Configs** | {len(playbook_configs)} configured |
| **Tools Registered** | {len(tools)} tools |
| **Success Evaluators** | {len(success_configs)} configured |
| **Extraction Window** | size={config.batch_size}, stride={config.batch_interval} |"""

    if extractors:
        names = ", ".join(f"`{e.extractor_name}`" for e in extractors)
        summary += f"\n| **Extractor Names** | {names} |"
    if playbook_configs:
        names = ", ".join(f"`{f.extractor_name}`" for f in playbook_configs)
        summary += f"\n| **Playbook Names** | {names} |"
    if tools:
        names = ", ".join(f"`{t.tool_name}`" for t in tools)
        summary += f"\n| **Tool Names** | {names} |"

    display(Markdown(summary))


def show_json(data, title: str | None = None) -> None:
    """Display any dict or Pydantic model as syntax-highlighted JSON."""
    if title:
        display(Markdown(f"### {title}"))

    if hasattr(data, "model_dump"):
        data = data.model_dump()

    json_str = json.dumps(data, indent=2, default=str)
    console.print(RichJSON(json_str))


def show_response(response, title: str | None = None) -> None:
    """Display any API response as pretty JSON."""
    if title:
        display(Markdown(f"### {title}"))

    if hasattr(response, "model_dump"):
        data = response.model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        rprint(response)
        return

    json_str = json.dumps(data, indent=2, default=str)
    console.print(RichJSON(json_str))


def collapsible(content: str, summary: str = "Show details") -> None:
    """Display content in a collapsible HTML details block."""
    html = f"<details><summary>{summary}</summary>\n\n```\n{content}\n```\n</details>"
    display(HTML(html))


def show_success(msg: str) -> None:
    """Display a green success message."""
    rprint(f"[bold green]OK[/bold green] {msg}")


def show_error(msg: str) -> None:
    """Display a red error message."""
    rprint(f"[bold red]ERROR[/bold red] {msg}")


# ---------------------------------------------------------------------------
# Convenience re-exports so notebooks can `from _display_helpers import *`
# ---------------------------------------------------------------------------

__all__ = [
    "load_env",
    "show_profiles",
    "show_interactions",
    "show_playbooks",
    "show_config",
    "show_json",
    "show_response",
    "collapsible",
    "show_success",
    "show_error",
    "console",
    "display",
    "Markdown",
    "HTML",
    "pd",
    "rprint",
]
