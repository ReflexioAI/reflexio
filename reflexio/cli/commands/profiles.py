"""Profile management commands (list, search, delete, generate)."""

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
    print_user_profiles,
    render,
)
from reflexio.cli.state import get_client, require_user_id

app = typer.Typer(
    help=(
        "Manage user profiles. Profiles are normally derived "
        "automatically from published interactions (via 'reflexio "
        "publish' + 'reflexio user-profiles regenerate'), but you "
        "can also seed them manually with 'reflexio user-profiles add'."
    ),
)


@app.command(name="list")
@handle_errors
def list_profiles(
    ctx: typer.Context,
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="User ID to filter profiles"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum number of profiles to return"),
    ] = 20,
    status: Annotated[
        str | None,
        typer.Option("--status", help="Status filter (current/pending/archived)"),
    ] = None,
) -> None:
    """List user profiles.

    Args:
        ctx: Typer context with CliState in ctx.obj
        user_id: Optional user ID filter
        limit: Maximum number of profiles to return
        status: Optional status filter (current, pending, archived)
    """
    if status and status not in ("current", "pending", "archived"):
        raise CliError(
            error_type="validation",
            message=f"Invalid status: {status}. Must be current, pending, or archived.",
            exit_code=EXIT_VALIDATION,
        )

    client = get_client(ctx)
    if user_id:
        # Filtered: query a single user via /api/get_profiles
        resp = client.get_profiles(
            user_id=user_id,
            top_k=limit,
            status_filter=[status] if status else None,
        )
    else:
        # Unfiltered: query every user via /api/get_all_profiles.
        # We intentionally do NOT fall back to resolve_user_id() here —
        # "no --user-id" means "all users", not "the default user".
        resp = client.get_all_profiles(
            limit=limit,
            status_filter=status,
        )
    profiles = resp.user_profiles or []

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(profiles, json_mode=True, meta=pagination_meta(profiles, limit))
    else:
        print_info(f"Found {len(profiles)} profile(s)")
        if profiles:
            print_user_profiles(profiles)


@app.command()
@handle_errors
def search(
    ctx: typer.Context,
    query: Annotated[
        str,
        typer.Argument(help="Semantic search query for profiles"),
    ],
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="User ID to filter profiles"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum number of results"),
    ] = 10,
    threshold: Annotated[
        float | None,
        typer.Option("--threshold", help="Similarity threshold"),
    ] = None,
) -> None:
    """Search profiles by semantic query.

    Args:
        ctx: Typer context with CliState in ctx.obj
        query: Search query string
        user_id: Optional user ID filter
        limit: Maximum number of results
        threshold: Optional similarity threshold
    """
    client = get_client(ctx)
    # Profile search requires a specific user_id — the server's
    # SearchUserProfileRequest.user_id is NonEmptyStr (cross-user
    # profile search is not currently supported). require_user_id
    # errors out cleanly instead of silently falling back to
    # DEFAULT_USER_ID.
    resolved_user_id = require_user_id(user_id, command_hint="profile search")

    kwargs: dict = {
        "query": query,
        "user_id": resolved_user_id,
        "top_k": limit,
    }
    if threshold is not None:
        kwargs["threshold"] = threshold

    resp = client.search_profiles(**kwargs)
    profiles = resp.user_profiles or []

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(profiles, json_mode=True, meta=pagination_meta(profiles, limit))
    else:
        print_info(f"Found {len(profiles)} profile(s)")
        if profiles:
            print_user_profiles(profiles)


@app.command()
@handle_errors
def delete(
    ctx: typer.Context,
    user_id: Annotated[
        str,
        typer.Option("--user-id", help="User ID (required)"),
    ],
    profile_id: Annotated[
        str,
        typer.Option("--profile-id", help="Specific profile ID to delete"),
    ] = "",
) -> None:
    """Delete a user profile.

    Args:
        ctx: Typer context with CliState in ctx.obj
        user_id: User ID owning the profile
        profile_id: Specific profile ID to delete (empty string for all user profiles)
    """
    client = get_client(ctx)
    resp = client.delete_profile(
        user_id=user_id,
        profile_id=profile_id,
        wait_for_response=True,
    )

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        label = (
            f"profile {profile_id}" if profile_id else f"profiles for user {user_id}"
        )
        print_info(f"Deleted {label}")


@app.command(name="delete-all")
@handle_errors
def delete_all(
    ctx: typer.Context,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Delete all profiles.

    Args:
        ctx: Typer context with CliState in ctx.obj
        yes: If True, skip confirmation prompt
    """
    if not yes:
        confirmed = typer.confirm("Are you sure you want to delete all profiles?")
        if not confirmed:
            raise typer.Abort()

    client = get_client(ctx)
    resp = client.delete_all_profiles()

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        print_info("All profiles deleted")


@app.command()
@handle_errors
def add(
    ctx: typer.Context,
    user_id: Annotated[
        str,
        typer.Option("--user-id", help="User ID owning the profile"),
    ],
    content: Annotated[
        str,
        typer.Option("--content", help="Profile content text"),
    ],
) -> None:
    """Manually add a user profile, bypassing interaction-based inference.

    Useful for seeding a known fact about the user (testing, migration,
    manual fact injection). Most users should rely on automatic profile
    generation via ``reflexio publish`` + ``reflexio user-profiles
    regenerate``.

    Args:
        ctx: Typer context with CliState in ctx.obj
        user_id: User the profile belongs to
        content: Profile content text (used for embedding)
    """
    client = get_client(ctx)
    resp = client.add_user_profile(
        [{"user_id": user_id, "content": content, "source": "cli-manual"}]
    )

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        raise_if_failed(resp, default="Failed to add user profile")
        print_info(f"User profile added: {content[:80]}")


@app.command()
@handle_errors
def regenerate(
    ctx: typer.Context,
    user_id: Annotated[
        str,
        typer.Option("--user-id", help="User ID (required)"),
    ],
    wait: Annotated[
        bool,
        typer.Option("--wait", help="Wait for profile generation to complete"),
    ] = False,
) -> None:
    """Re-run profile generation for a user.

    Re-runs the inference pipeline over the user's published
    interactions to refresh their profiles.  Renamed from
    ``generate`` for consistency with ``agent-playbooks regenerate``.

    Args:
        ctx: Typer context with CliState in ctx.obj
        user_id: User ID to generate profiles for
        wait: If True, wait for generation to complete
    """
    client = get_client(ctx)
    resp = client.rerun_profile_generation(
        user_id=user_id,
        wait_for_response=wait,
    )

    # When waiting, auto-promote PENDING profiles to CURRENT so they're
    # immediately visible in `list` output.  Without this, regenerated
    # profiles stay in PENDING status and the default list filter (CURRENT
    # only) returns 0 results.
    if wait:
        upgrade_resp = client.upgrade_profiles(user_id=user_id)
        promoted = upgrade_resp.profiles_promoted if upgrade_resp else 0
    else:
        promoted = 0

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    elif wait:
        print_info(
            f"Profile generation complete ({promoted} profile(s) promoted)"
        )
    else:
        print_info("Profile generation started")
