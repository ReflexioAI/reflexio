"""Shared root CLI options without command or service imports."""

from __future__ import annotations

from typing import Annotated

import typer

from reflexio import __version__
from reflexio.cli.state import CliState


def _version_callback(ctx: typer.Context, value: bool) -> None:
    """Print version and exit.

    Args:
        value: Whether the --version flag was passed
    """
    if value and not ctx.resilient_parsing:
        typer.echo(f"reflexio {__version__}")
        raise typer.Exit


def create_root_app() -> typer.Typer:
    """Create shared global options without importing service commands."""
    app = typer.Typer(
        name="reflexio",
        help="Reflexio CLI — AI agent memory and feedback system",
        no_args_is_help=True,
        rich_markup_mode="rich",
    )

    @app.callback()
    def main_callback(
        ctx: typer.Context,
        json_mode: Annotated[
            bool,
            typer.Option("--json", help="Output structured JSON envelopes"),
        ] = False,
        server_url: Annotated[
            str | None,
            typer.Option(
                "--server-url", help="Backend API server URL", envvar="REFLEXIO_URL"
            ),
        ] = None,
        api_key: Annotated[
            str | None,
            typer.Option(
                "--api-key", help="Reflexio API key", envvar="REFLEXIO_API_KEY"
            ),
        ] = None,
        _version: Annotated[  # noqa: ARG001 — Typer requires this param for the callback
            bool,
            typer.Option(
                "--version",
                callback=_version_callback,
                is_eager=True,
                help="Show version",
            ),
        ] = False,
    ) -> None:
        """Reflexio CLI — AI agent memory and feedback system."""
        ctx.ensure_object(CliState)
        ctx.obj.json_mode = json_mode
        if server_url:
            ctx.obj.server_url = server_url
        if api_key:
            ctx.obj.api_key = api_key

    return app
