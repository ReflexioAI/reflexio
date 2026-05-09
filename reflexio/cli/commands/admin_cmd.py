"""Administrative server-side controls: cache invalidation and friends.

Subcommands live under nested Typer groups (``admin cache invalidate``)
so the surface stays organised as more admin operations are added.
"""

from __future__ import annotations

from typing import Annotated

import typer

from reflexio.cli.errors import handle_errors
from reflexio.cli.output import print_info, render
from reflexio.cli.state import get_client

# Top-level "admin" group.
app = typer.Typer(help="Administrative server-side controls.")

# Nested "admin cache" group — invalidate, plus future stats/clear hooks.
cache_app = typer.Typer(help="Manage the per-org Reflexio cache.")
app.add_typer(cache_app, name="cache")


@cache_app.command(name="invalidate")
@handle_errors
def cache_invalidate(
    ctx: typer.Context,
    org_id: Annotated[
        str | None,
        typer.Option(
            "--org-id",
            help=(
                "Optional org_id verification token. When supplied, the "
                "server will reject the request unless it matches the "
                "caller's authenticated org. Cross-org invalidation is "
                "not supported."
            ),
        ),
    ] = None,
) -> None:
    """Evict the caller's per-org Reflexio cache entry.

    Necessary when the running config was mutated through a channel
    the server can't observe (sibling-replica DB writes, hand-edited
    config files on backends that don't auto-detect mtime changes).

    Args:
        ctx: Typer context with CliState in ctx.obj
        org_id: Optional verification token; must match the caller's
            authenticated org if supplied.
    """
    client = get_client(ctx)
    resp = client.invalidate_cache(org_id=org_id)

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        invalidated = bool(resp.get("invalidated"))
        target_org = resp.get("org_id", "(unknown)")
        if invalidated:
            print_info(f"Cache evicted for org {target_org}")
        else:
            print_info(f"No cache entry to evict for org {target_org}")
