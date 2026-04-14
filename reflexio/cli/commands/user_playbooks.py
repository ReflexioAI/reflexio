"""User playbook management commands (list, search, add, delete, regenerate)."""

from __future__ import annotations

from typing import Annotated

import typer

from reflexio.cli.errors import (
    EXIT_VALIDATION,
    CliError,
    handle_errors,
    raise_if_failed,
)
from reflexio.cli.output import (
    pagination_meta,
    print_info,
    print_user_playbooks,
    render,
)
from reflexio.cli.state import get_client, require_agent_version

app = typer.Typer(help="Manage user playbooks.")

_STATUS_MAP = {
    "current": "current",
    "pending": "pending",
    "archived": "archived",
}


@app.command(name="list")
@handle_errors
def list_user_playbooks(
    ctx: typer.Context,
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="Filter by user ID"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum number of playbooks to return"),
    ] = 20,
    playbook_name: Annotated[
        str | None,
        typer.Option("--playbook-name", help="Filter by playbook name"),
    ] = None,
    agent_version: Annotated[
        str | None,
        typer.Option("--agent-version", help="Filter by agent version"),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="Status filter (current/pending/archived)"),
    ] = None,
) -> None:
    """List user playbooks.

    Args:
        ctx: Typer context with CliState in ctx.obj
        user_id: Optional user ID filter
        limit: Maximum number of playbooks to return
        playbook_name: Optional playbook name filter
        agent_version: Optional agent version filter
        status: Optional status filter (current, pending, archived)
    """
    from reflexio.models.api_schema.service_schemas import Status

    if status and status not in _STATUS_MAP:
        raise CliError(
            error_type="validation",
            message=f"Invalid status: {status}. Must be current, pending, or archived.",
            exit_code=EXIT_VALIDATION,
        )

    status_filter: list[Status | None] | None = None
    if status == "current":
        status_filter = [Status.CURRENT]
    elif status == "pending":
        status_filter = [Status.PENDING]
    elif status == "archived":
        status_filter = [Status.ARCHIVED]

    client = get_client(ctx)
    resp = client.get_user_playbooks(
        limit=limit,
        user_id=user_id,
        playbook_name=playbook_name,
        agent_version=agent_version,
        status_filter=status_filter,
    )
    playbooks = resp.user_playbooks or []

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(playbooks, json_mode=True, meta=pagination_meta(playbooks, limit))
    else:
        print_info(f"Found {len(playbooks)} user playbook(s)")
        if playbooks:
            print_user_playbooks(playbooks)


@app.command()
@handle_errors
def search(
    ctx: typer.Context,
    query: Annotated[
        str,
        typer.Argument(help="Semantic search query for user playbooks"),
    ],
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum number of results"),
    ] = 10,
    agent_version: Annotated[
        str | None,
        typer.Option("--agent-version", help="Agent version to filter by"),
    ] = None,
) -> None:
    """Search user playbooks by semantic query.

    Args:
        ctx: Typer context with CliState in ctx.obj
        query: Search query string
        limit: Maximum number of results
        agent_version: Optional agent version filter
    """
    client = get_client(ctx)

    resp = client.search_user_playbooks(
        query=query,
        agent_version=agent_version,
        top_k=limit,
    )
    playbooks = resp.user_playbooks or []

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(playbooks, json_mode=True, meta=pagination_meta(playbooks, limit))
    else:
        print_info(f"Found {len(playbooks)} user playbook(s)")
        if playbooks:
            print_user_playbooks(playbooks)


@app.command()
@handle_errors
def add(
    ctx: typer.Context,
    content: Annotated[
        str,
        typer.Option("--content", help="Playbook content text"),
    ],
    playbook_name: Annotated[
        str,
        typer.Option("--playbook-name", help="Playbook category name"),
    ] = "agent_corrections",
    trigger: Annotated[
        str | None,
        typer.Option("--trigger", help="When this playbook applies"),
    ] = None,
    instruction: Annotated[
        str | None,
        typer.Option("--instruction", help="What to do (DO)"),
    ] = None,
    pitfall: Annotated[
        str | None,
        typer.Option("--pitfall", help="What not to do (DON'T)"),
    ] = None,
    rationale: Annotated[
        str | None,
        typer.Option("--rationale", help="Why this playbook matters"),
    ] = None,
) -> None:
    """Add a manual user playbook entry.

    Args:
        ctx: Typer context with CliState in ctx.obj
        content: Playbook content text
        playbook_name: Playbook category name
        trigger: When this playbook applies
        instruction: What to do
        pitfall: What not to do
        rationale: Why this playbook matters
    """
    from reflexio.models.api_schema.service_schemas import StructuredData, UserPlaybook

    structured_data = StructuredData(
        trigger=trigger,
        instruction=instruction,
        pitfall=pitfall,
        rationale=rationale,
    )
    playbook = UserPlaybook(
        agent_version="cli",
        request_id="cli-manual",
        playbook_name=playbook_name,
        content=content,
        structured_data=structured_data,
    )

    client = get_client(ctx)
    resp = client.add_user_playbook([playbook])

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        raise_if_failed(resp, default="Failed to add user playbook")
        print_info(f"User playbook added: {content[:80]}")


@app.command()
@handle_errors
def update(
    ctx: typer.Context,
    playbook_id: Annotated[
        int,
        typer.Option("--playbook-id", help="User playbook ID to update"),
    ],
    content: Annotated[
        str | None,
        typer.Option("--content", help="New content text"),
    ] = None,
    playbook_name: Annotated[
        str | None,
        typer.Option("--playbook-name", help="New playbook category name"),
    ] = None,
) -> None:
    """Update editable fields of a user playbook.

    Pass only the fields you want to change. Currently supports
    ``--content`` and ``--playbook-name``. To change the structured
    data block (trigger / instruction / pitfall / rationale), use
    the ``client.update_user_playbook`` Python API directly — the
    storage layer replaces structured_data wholesale, so a CLI flag
    for individual fields would risk wiping the others.

    Args:
        ctx: Typer context with CliState in ctx.obj
        playbook_id: User playbook ID to update
        content: New content text
        playbook_name: New playbook category name
    """
    if content is None and playbook_name is None:
        raise CliError(
            error_type="validation",
            message=("Pass at least one of --content/--playbook-name to update."),
            exit_code=EXIT_VALIDATION,
        )

    client = get_client(ctx)
    resp = client.update_user_playbook(
        user_playbook_id=playbook_id,
        content=content,
        playbook_name=playbook_name,
    )

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        raise_if_failed(resp, default="Failed to update user playbook")
        print_info(f"User playbook {playbook_id} updated")


@app.command()
@handle_errors
def delete(
    ctx: typer.Context,
    playbook_id: Annotated[
        str,
        typer.Option("--playbook-id", help="User playbook ID"),
    ],
) -> None:
    """Delete a user playbook by ID.

    Args:
        ctx: Typer context with CliState in ctx.obj
        playbook_id: User playbook ID to delete
    """
    client = get_client(ctx)
    resp = client.delete_user_playbook(
        user_playbook_id=int(playbook_id),
        wait_for_response=True,
    )

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        print_info(f"Deleted user playbook {playbook_id}")


@app.command(name="delete-all")
@handle_errors
def delete_all(
    ctx: typer.Context,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Delete all user playbooks (user only, not agent playbooks).

    Uses the dedicated ``DELETE /api/delete_all_user_playbooks`` endpoint,
    which is scoped strictly to user playbooks. To also wipe agent
    playbooks, run ``reflexio agent-playbooks delete-all`` separately.

    Args:
        ctx: Typer context with CliState in ctx.obj
        yes: If True, skip confirmation prompt
    """
    if not yes:
        confirmed = typer.confirm("Are you sure you want to delete all user playbooks?")
        if not confirmed:
            raise typer.Abort()

    client = get_client(ctx)
    resp = client.delete_all_user_playbooks()

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        print_info("All user playbooks deleted")


@app.command()
@handle_errors
def regenerate(
    ctx: typer.Context,
    wait: Annotated[
        bool,
        typer.Option("--wait", help="Wait for regeneration to complete"),
    ] = False,
    agent_version: Annotated[
        str | None,
        typer.Option(
            "--agent-version", help="Agent version to regenerate playbooks for"
        ),
    ] = None,
) -> None:
    """Re-extract user playbooks from published interactions.

    Re-runs the playbook extraction pipeline over all interactions for the
    given agent version, producing fresh user playbooks.

    Args:
        ctx: Typer context with CliState in ctx.obj
        wait: If True, wait for regeneration to complete
        agent_version: Agent version to regenerate (required)
    """
    client = get_client(ctx)
    resolved_version = require_agent_version(
        agent_version, command_hint="user-playbook regeneration"
    )

    resp = client.rerun_playbook_generation(
        agent_version=resolved_version,
        wait_for_response=wait,
    )

    # When waiting, auto-promote PENDING user playbooks to CURRENT so
    # they're immediately visible in `list` output.
    promoted = 0
    if wait:
        try:
            upgrade_resp = client.upgrade_user_playbooks(
                agent_version=resolved_version,
            )
            promoted = upgrade_resp.user_playbooks_promoted
        except Exception:
            print_info("Warning: promotion failed, but regeneration succeeded")

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    elif wait:
        print_info(f"Playbook regeneration complete ({promoted} playbook(s) promoted)")
    else:
        print_info("Playbook regeneration started")
