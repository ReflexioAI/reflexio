"""Agent playbook management commands (list, search, delete, aggregate, regenerate)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer

from reflexio.cli.errors import (
    EXIT_VALIDATION,
    CliError,
    handle_errors,
    raise_if_failed,
)

if TYPE_CHECKING:
    from reflexio.models.api_schema.service_schemas import PlaybookStatus
from reflexio.cli.output import (
    pagination_meta,
    print_agent_playbooks,
    print_info,
    render,
)
from reflexio.cli.state import get_client, require_agent_version

app = typer.Typer(help="Manage agent playbooks.")

_PLAYBOOK_STATUS_MAP = {
    "pending": "pending",
    "approved": "approved",
    "rejected": "rejected",
}


def _validate_playbook_status(value: str | None) -> PlaybookStatus | None:
    """Validate and convert a playbook status string to PlaybookStatus enum.

    Args:
        value (str | None): Raw status string from CLI input

    Returns:
        PlaybookStatus | None: Parsed enum value, or None if input is None

    Raises:
        CliError: If the value is not a valid playbook status
    """
    from reflexio.models.api_schema.service_schemas import PlaybookStatus

    if not value:
        return None
    if value not in _PLAYBOOK_STATUS_MAP:
        raise CliError(
            error_type="validation",
            message=f"Invalid playbook status: {value}. Must be pending, approved, or rejected.",
            exit_code=EXIT_VALIDATION,
        )
    return PlaybookStatus(value)


@app.command(name="list")
@handle_errors
def list_agent_playbooks(
    ctx: typer.Context,
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
    playbook_status: Annotated[
        str | None,
        typer.Option(
            "--playbook-status",
            help="Filter by approval status (pending/approved/rejected)",
        ),
    ] = None,
) -> None:
    """List agent playbooks.

    Args:
        ctx: Typer context with CliState in ctx.obj
        limit: Maximum number of playbooks to return
        playbook_name: Optional playbook name filter
        agent_version: Optional agent version filter
        playbook_status: Optional approval status filter
    """
    pb_status_filter = _validate_playbook_status(playbook_status)

    client = get_client(ctx)
    resp = client.get_agent_playbooks(
        limit=limit,
        playbook_name=playbook_name,
        agent_version=agent_version,
        playbook_status_filter=pb_status_filter,
    )
    playbooks = resp.agent_playbooks or []

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(playbooks, json_mode=True, meta=pagination_meta(playbooks, limit))
    else:
        print_info(f"Found {len(playbooks)} agent playbook(s)")
        if playbooks:
            print_agent_playbooks(playbooks)


@app.command()
@handle_errors
def search(
    ctx: typer.Context,
    query: Annotated[
        str,
        typer.Argument(help="Semantic search query for agent playbooks"),
    ],
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum number of results"),
    ] = 10,
    agent_version: Annotated[
        str | None,
        typer.Option("--agent-version", help="Agent version to filter by"),
    ] = None,
    playbook_status: Annotated[
        str | None,
        typer.Option(
            "--playbook-status",
            help="Filter by approval status (pending/approved/rejected)",
        ),
    ] = None,
) -> None:
    """Search agent playbooks by semantic query.

    Args:
        ctx: Typer context with CliState in ctx.obj
        query: Search query string
        limit: Maximum number of results
        agent_version: Optional agent version filter
        playbook_status: Optional approval status filter
    """
    pb_status_filter = _validate_playbook_status(playbook_status)

    client = get_client(ctx)

    resp = client.search_agent_playbooks(
        query=query,
        agent_version=agent_version,
        playbook_status_filter=pb_status_filter,
        top_k=limit,
    )
    playbooks = resp.agent_playbooks or []

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(playbooks, json_mode=True, meta=pagination_meta(playbooks, limit))
    else:
        print_info(f"Found {len(playbooks)} agent playbook(s)")
        if playbooks:
            print_agent_playbooks(playbooks)


@app.command()
@handle_errors
def add(
    ctx: typer.Context,
    content: Annotated[
        str,
        typer.Option("--content", help="Playbook content text"),
    ],
    agent_version: Annotated[
        str,
        typer.Option(
            "--agent-version",
            help="Agent version this playbook applies to",
        ),
    ],
    playbook_name: Annotated[
        str,
        typer.Option("--playbook-name", help="Playbook category name"),
    ] = "agent_corrections",
    playbook_status: Annotated[
        str,
        typer.Option(
            "--playbook-status",
            help="Approval status (pending/approved/rejected)",
        ),
    ] = "pending",
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
    """Add an agent playbook directly, bypassing the aggregation pipeline.

    Mirrors ``reflexio user-playbooks add`` but for the agent
    (post-aggregation) playbook layer. Useful for seeding rules
    that you want the agent to follow without first producing
    sample user playbooks for the aggregator to cluster.

    Args:
        ctx: Typer context with CliState in ctx.obj
        content: Playbook content text
        agent_version: Agent version this playbook applies to
        playbook_name: Playbook category name
        playbook_status: Initial approval status
        trigger: When this playbook applies
        instruction: What to do
        pitfall: What not to do
        rationale: Why this playbook matters
    """
    # --playbook-status has a non-empty default ("pending"), so
    # _validate_playbook_status always returns a PlaybookStatus here.
    # The cast narrows the Optional return type without the runtime
    # overhead (or lint noise) of an ``assert``.
    from typing import cast

    from reflexio.models.api_schema.service_schemas import (
        AgentPlaybook,
        PlaybookStatus,
        StructuredData,
    )

    pb_status = cast(PlaybookStatus, _validate_playbook_status(playbook_status))

    structured_data = StructuredData(
        trigger=trigger,
        instruction=instruction,
        pitfall=pitfall,
        rationale=rationale,
    )
    playbook = AgentPlaybook(
        agent_version=agent_version,
        playbook_name=playbook_name,
        content=content,
        structured_data=structured_data,
        playbook_status=pb_status,
        playbook_metadata='{"source": "cli-manual"}',
    )

    client = get_client(ctx)
    resp = client.add_agent_playbooks([playbook])

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        raise_if_failed(resp, default="Failed to add agent playbook")
        print_info(f"Agent playbook added: {content[:80]}")


@app.command()
@handle_errors
def update(
    ctx: typer.Context,
    id: Annotated[  # noqa: A002
        int,
        typer.Option("--id", help="Agent playbook ID to update"),
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
    """Update editable fields of an agent playbook.

    Pass only the fields you want to change. Currently supports
    ``--content`` and ``--playbook-name``. To change the structured
    data block (trigger / instruction / pitfall / rationale), use
    the ``client.update_agent_playbook`` Python API directly — the
    storage layer replaces structured_data wholesale, so a CLI flag
    for individual fields would risk wiping the others.

    To change the approval status, use ``update-status`` instead.

    Args:
        ctx: Typer context with CliState in ctx.obj
        id: Agent playbook ID to update
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
    resp = client.update_agent_playbook(
        agent_playbook_id=id,
        content=content,
        playbook_name=playbook_name,
    )

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        raise_if_failed(resp, default="Failed to update agent playbook")
        print_info(f"Agent playbook {id} updated")


@app.command(name="update-status")
@handle_errors
def update_status(
    ctx: typer.Context,
    id: Annotated[  # noqa: A002
        int,
        typer.Option("--id", help="Agent playbook ID"),
    ],
    status: Annotated[
        str,
        typer.Option(
            "--status",
            help="New approval status (pending/approved/rejected)",
        ),
    ],
) -> None:
    """Approve, reject, or mark pending an agent playbook.

    The dedicated approval-workflow command. Use this rather than
    ``update --status`` because the server has a single-purpose
    endpoint for status changes that writes a smaller change log
    and enforces tighter validation.

    Args:
        ctx: Typer context with CliState in ctx.obj
        id: Agent playbook ID
        status: New approval status (pending/approved/rejected)
    """
    pb_status = _validate_playbook_status(status)
    if pb_status is None:
        raise CliError(
            error_type="validation",
            message=(
                f"Invalid status: {status}. Must be pending, approved, or rejected."
            ),
            exit_code=EXIT_VALIDATION,
        )

    client = get_client(ctx)
    resp = client.update_agent_playbook_status(
        agent_playbook_id=id,
        playbook_status=pb_status,
    )

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        raise_if_failed(resp, default="Failed to update agent playbook status")
        print_info(f"Agent playbook {id} status set to {pb_status.value}")


@app.command()
@handle_errors
def delete(
    ctx: typer.Context,
    id: Annotated[  # noqa: A002
        str,
        typer.Option("--id", help="Agent playbook ID"),
    ],
) -> None:
    """Delete an agent playbook by ID.

    Args:
        ctx: Typer context with CliState in ctx.obj
        id: Agent playbook ID to delete
    """
    client = get_client(ctx)
    resp = client.delete_agent_playbook(
        agent_playbook_id=int(id),
        wait_for_response=True,
    )

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        print_info(f"Deleted agent playbook {id}")


@app.command(name="delete-all")
@handle_errors
def delete_all(
    ctx: typer.Context,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Delete all agent playbooks (agent only, not user playbooks).

    Uses the dedicated ``DELETE /api/delete_all_agent_playbooks`` endpoint,
    which is scoped strictly to agent playbooks. To also wipe user
    playbooks, run ``reflexio user-playbooks delete-all`` separately.

    Args:
        ctx: Typer context with CliState in ctx.obj
        yes: If True, skip confirmation prompt
    """
    if not yes:
        confirmed = typer.confirm(
            "Are you sure you want to delete all agent playbooks?"
        )
        if not confirmed:
            raise typer.Abort()

    client = get_client(ctx)
    resp = client.delete_all_agent_playbooks()

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        print_info("All agent playbooks deleted")


@app.command()
@handle_errors
def aggregate(
    ctx: typer.Context,
    wait: Annotated[
        bool,
        typer.Option("--wait", help="Wait for aggregation to complete"),
    ] = False,
    agent_version: Annotated[
        str | None,
        typer.Option(
            "--agent-version",
            help="Agent version to aggregate. If omitted, aggregates all versions.",
        ),
    ] = None,
    playbook_name: Annotated[
        str | None,
        typer.Option("--playbook-name", help="Playbook name to aggregate"),
    ] = None,
) -> None:
    """Run playbook aggregation to cluster similar user playbooks.

    Without --agent-version, discovers all agent versions that have user
    playbooks and aggregates each one separately.

    Args:
        ctx: Typer context with CliState in ctx.obj
        wait: If True, wait for aggregation to complete
        agent_version: Agent version to aggregate (omit to aggregate all)
        playbook_name: Playbook name to aggregate (defaults to first configured playbook)
    """
    client = get_client(ctx)

    # Default to first configured playbook name if not specified
    if not playbook_name:
        config = client.get_config()
        if config and config.user_playbook_extractor_configs:
            playbook_name = config.user_playbook_extractor_configs[0].extractor_name
        else:
            raise CliError(
                error_type="validation",
                message="No playbook_name provided and no playbooks configured on the server",
                exit_code=EXIT_VALIDATION,
            )

    json_mode: bool = ctx.obj.json_mode

    # Determine which agent versions to aggregate
    if agent_version:
        versions = [agent_version]
    else:
        # Discover all agent versions from existing user playbooks
        resp = client.get_user_playbooks(playbook_name=playbook_name, limit=10000)
        versions = sorted(
            {pb.agent_version for pb in (resp.user_playbooks or []) if pb.agent_version}
        )
        if not versions:
            if json_mode:
                render(
                    {"message": "No user playbooks found to aggregate"}, json_mode=True
                )
            else:
                print_info("No user playbooks found to aggregate")
            return
        print_info(
            f"Found {len(versions)} agent version(s) to aggregate: {', '.join(versions)}"
        )

    # Aggregate each version
    for version in versions:
        print_info(
            f"Aggregating user playbooks for '{playbook_name}' (agent_version={version})..."
        )
        resp = client.run_playbook_aggregation(
            agent_version=version,
            playbook_name=playbook_name,
            wait_for_response=wait,
        )

        if json_mode:
            render(resp, json_mode=True)
        else:
            success = getattr(resp, "success", None)
            message = getattr(resp, "message", "")
            if success is False:
                print_info(f"  Failed: {message}")
            elif message:
                print_info(f"  {message}")
            elif wait:
                print_info("  Done")
            else:
                print_info("  Started")


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
        typer.Option("--agent-version", help="Agent version to regenerate"),
    ] = None,
) -> None:
    """Rerun playbook generation for an agent version.

    Args:
        ctx: Typer context with CliState in ctx.obj
        wait: If True, wait for regeneration to complete
        agent_version: Optional agent version filter
    """
    client = get_client(ctx)
    # Regenerate targets a specific agent_version — silently falling
    # back to DEFAULT_AGENT_VERSION would re-run the wrong pool of
    # playbooks. Require an explicit value (flag, env, or config).
    resolved_version = require_agent_version(
        agent_version, command_hint="playbook regeneration"
    )

    resp = client.rerun_playbook_generation(
        agent_version=resolved_version,
        wait_for_response=wait,
    )

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    elif wait:
        print_info("Playbook regeneration complete")
    else:
        print_info("Playbook regeneration started")
