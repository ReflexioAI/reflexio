"""Top-level shortcut commands for common operations.

These are registered at the root level of the CLI for quick access:
    reflexio publish    (alias for interactions publish)
    reflexio search     (unified search across all types)
    reflexio context    (fetch formatted context for agent injection)
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from reflexio.cli.errors import handle_errors
from reflexio.cli.output import (
    print_info,
    render,
)
from reflexio.cli.state import (
    get_client,
    require_agent_version,
    require_user_id,
)


def _build_publish_args(
    *,
    user_id: str | None,
    session_id: str | None,
    source: str,
    agent_version: str | None,
    wait: bool,
    user_message: str | None,
    agent_response: str | None,
    data: str | None,
    file: Path | None,
    stdin: bool,
    skip_aggregation: bool,
    force_extraction: bool,
) -> list[str]:
    """Assemble the argv the ``interactions publish`` command expects.

    The shortcut ``reflexio publish`` re-invokes the full interactions
    Typer app rather than duplicating its logic; this helper keeps the
    shortcut's cyclomatic complexity under ruff's C901 limit by hiding
    the long chain of ``if`` guards behind a single call.

    Args:
        user_id: Optional user identifier.
        session_id: Optional session identifier.
        source: Source tag recorded on the interaction.
        agent_version: Optional agent version tag.
        wait: When True, wait for processing to complete.
        user_message: Single-turn user message shortcut.
        agent_response: Single-turn agent response shortcut.
        data: Inline JSON or ``@filepath`` reference.
        file: Path to a JSON/JSONL payload file.
        stdin: When True, read JSON payload from stdin.
        skip_aggregation: When True, skip post-extract aggregation.
        force_extraction: When True, bypass batch interval checks.

    Returns:
        list[str]: argv list ready to hand to ``interactions_app(...)``.
    """
    args: list[str] = ["publish"]
    if user_id:
        args.extend(["--user-id", user_id])
    if session_id:
        args.extend(["--session-id", session_id])
    args.extend(["--source", source])
    if agent_version:
        args.extend(["--agent-version", agent_version])
    if wait:
        args.append("--wait")
    if user_message:
        args.extend(["--user-message", user_message])
    if agent_response:
        args.extend(["--agent-response", agent_response])
    if data:
        args.extend(["--data", data])
    if file:
        args.extend(["--file", str(file)])
    if stdin:
        args.append("--stdin")
    if skip_aggregation:
        args.append("--skip-aggregation")
    if force_extraction:
        args.append("--force-extraction")
    return args


def register_shortcuts(app: typer.Typer) -> None:
    """Register shortcut commands on the root Typer app.

    Args:
        app: The root Typer application
    """

    @app.command()
    @handle_errors
    def publish(
        ctx: typer.Context,
        user_id: Annotated[
            str | None,
            typer.Option("--user-id", help="User identifier (optional if in payload)"),
        ] = None,
        session_id: Annotated[
            str | None, typer.Option("--session-id", help="Session ID")
        ] = None,
        source: Annotated[str, typer.Option(help="Source tag")] = "cli",
        agent_version: Annotated[
            str | None, typer.Option("--agent-version", help="Agent version")
        ] = None,
        wait: Annotated[
            bool, typer.Option("--wait", help="Wait for processing to complete")
        ] = False,
        user_message: Annotated[
            str | None, typer.Option("--user-message", help="Single-turn user message")
        ] = None,
        agent_response: Annotated[
            str | None,
            typer.Option("--agent-response", help="Single-turn agent response"),
        ] = None,
        data: Annotated[
            str | None, typer.Option("--data", help="JSON data or @filepath")
        ] = None,
        file: Annotated[
            Path | None, typer.Option("--file", help="Path to JSON/JSONL file")
        ] = None,
        stdin: Annotated[
            bool, typer.Option("--stdin", help="Read JSON from stdin")
        ] = False,
        skip_aggregation: Annotated[
            bool,
            typer.Option(
                "--skip-aggregation",
                help="Extract profiles/playbooks but skip aggregation",
            ),
        ] = False,
        force_extraction: Annotated[
            bool,
            typer.Option(
                "--force-extraction",
                help="Bypass batch_interval checks and always run extractors",
            ),
        ] = False,
    ) -> None:
        """Publish interactions (shortcut for: interactions publish)."""
        from reflexio.cli.commands.interactions import app as interactions_app

        args = _build_publish_args(
            user_id=user_id,
            session_id=session_id,
            source=source,
            agent_version=agent_version,
            wait=wait,
            user_message=user_message,
            agent_response=agent_response,
            data=data,
            file=file,
            stdin=stdin,
            skip_aggregation=skip_aggregation,
            force_extraction=force_extraction,
        )

        # Propagate context
        interactions_app(args, standalone_mode=False, obj=ctx.obj)

    @app.command()
    @handle_errors
    def search(
        ctx: typer.Context,
        query: Annotated[str, typer.Argument(help="Search query text")],
        top_k: Annotated[int, typer.Option("--top-k", help="Max results per type")] = 5,
        threshold: Annotated[float, typer.Option(help="Similarity threshold")] = 0.4,
        user_id: Annotated[
            str | None, typer.Option("--user-id", help="Filter by user ID")
        ] = None,
        agent_version: Annotated[
            str | None, typer.Option("--agent-version", help="Filter by agent version")
        ] = None,
    ) -> None:
        """Unified semantic search across profiles and playbooks."""
        client = get_client(ctx)
        json_mode = ctx.obj.json_mode

        response = client.search(
            query=query,
            top_k=top_k,
            threshold=threshold,
            user_id=user_id,
            agent_version=agent_version,
        )

        if json_mode:
            render(response, json_mode=True)
        else:
            from reflexio.cli.output import format_context

            context = format_context(
                profiles=response.profiles,
                agent_playbooks=response.user_playbooks,
            )
            if context:
                print(context)

            total = len(response.profiles) + len(response.user_playbooks)
            print_info(
                f"Found {len(response.profiles)} profiles, "
                f"{len(response.user_playbooks)} playbooks ({total} total)"
            )

    @app.command()
    @handle_errors
    def context(
        ctx: typer.Context,
        user_id: Annotated[
            str | None, typer.Option("--user-id", help="User ID for profile lookup")
        ] = None,
        agent_version: Annotated[
            str | None,
            typer.Option(
                "--agent-version", help="Agent version for playbook filtering"
            ),
        ] = None,
        query: Annotated[
            str, typer.Option(help="Search query")
        ] = "general preferences",
    ) -> None:
        """Fetch formatted context for agent bootstrap injection."""
        from reflexio.cli.output import format_context

        client = get_client(ctx)
        json_mode = ctx.obj.json_mode
        # The context shortcut scopes profiles to a specific user and
        # playbooks to a specific agent_version. Silently defaulting
        # either would produce misleading bootstrap context — fail
        # loudly instead.
        resolved_user_id = require_user_id(user_id, command_hint="context bootstrap")
        resolved_agent_version = require_agent_version(
            agent_version, command_hint="context bootstrap"
        )

        profiles = []
        resp = client.search_profiles(user_id=resolved_user_id, query=query, top_k=5)
        if resp.success:
            profiles = resp.user_profiles

        agent_playbooks = []
        resp_pb = client.search_user_playbooks(
            query=query, agent_version=resolved_agent_version, top_k=5
        )
        if resp_pb.success:
            agent_playbooks = resp_pb.user_playbooks

        if json_mode:
            render(
                {
                    "profiles": [p.model_dump(mode="json") for p in profiles],
                    "playbooks": [p.model_dump(mode="json") for p in agent_playbooks],
                },
                json_mode=True,
            )
        else:
            context_text = format_context(
                profiles=profiles,
                agent_playbooks=agent_playbooks,
            )
            if context_text:
                print(context_text)

            print_info(
                f"Found {len(profiles)} profiles, {len(agent_playbooks)} playbooks"
            )
