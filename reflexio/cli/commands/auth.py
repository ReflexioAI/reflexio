"""Authentication and configuration setup commands."""

from __future__ import annotations

import os
from typing import Annotated

import typer

from reflexio.cli.env_loader import get_env_path, set_env_var
from reflexio.cli.errors import handle_errors
from reflexio.cli.output import mask_api_key, print_auth_status, render

app = typer.Typer(help="Authentication and configuration setup.")


@app.command()
@handle_errors
def login(
    ctx: typer.Context,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Reflexio API key"),
    ] = None,
    server_url: Annotated[
        str | None,
        typer.Option("--server-url", help="Backend API server URL"),
    ] = None,
) -> None:
    """Save authentication credentials to ~/.reflexio/.env.

    Args:
        ctx: Typer context with CliState in ctx.obj
        api_key: API key to store
        server_url: Backend API server URL to store
    """
    if not server_url and not api_key:
        typer.echo("Nothing saved — provide --api-key and/or --server-url.")
        return

    env_path = get_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)

    if server_url:
        set_env_var(env_path, "REFLEXIO_URL", server_url)
    if api_key:
        set_env_var(env_path, "REFLEXIO_API_KEY", api_key)

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render({"env_path": str(env_path)}, json_mode=True)
    else:
        print(f"Credentials saved to {env_path}")


@app.command()
@handle_errors
def status(
    ctx: typer.Context,
) -> None:
    """Show current authentication status from environment.

    Args:
        ctx: Typer context with CliState in ctx.obj
    """
    env_path = get_env_path()
    raw_key = os.environ.get("REFLEXIO_API_KEY", "")
    raw_url = os.environ.get("REFLEXIO_URL", "")

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(
            {
                "url": raw_url,
                "api_key": mask_api_key(raw_key),
                "env_path": str(env_path),
            },
            json_mode=True,
        )
    else:
        print_auth_status(url=raw_url, api_key=raw_key, env_path=str(env_path))


@app.command()
@handle_errors
def logout(
    ctx: typer.Context,
) -> None:
    """Remove stored credentials by deleting the .env file.

    Args:
        ctx: Typer context with CliState in ctx.obj
    """
    env_path = get_env_path()
    if env_path.exists():
        env_path.unlink()

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render({"message": "Logged out"}, json_mode=True)
    else:
        print("Logged out. Env file removed.")
