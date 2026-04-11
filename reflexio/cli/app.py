"""Typer app factory for Reflexio CLI.

Creates the root app with global flags, registers all command groups
as sub-apps, and provides the entry point for both open-source and
enterprise CLI variants.
"""

from __future__ import annotations

from typing import Annotated

import typer

from reflexio import __version__
from reflexio.cli.state import CliState


def _version_callback(value: bool) -> None:
    """Print version and exit.

    Args:
        value: Whether the --version flag was passed
    """
    if value:
        typer.echo(f"reflexio {__version__}")
        raise typer.Exit


def create_app(exclude: set[str] | None = None) -> typer.Typer:
    """Create and configure the root Typer application.

    Registers global flags (--json, --url, --api-key, --version) and
    all command group sub-apps.

    Args:
        exclude: Optional set of sub-app names to skip registration for.
            Allows consumers to replace specific sub-apps (e.g., ``exclude={"services"}``
            to provide a custom services implementation).

    Returns:
        typer.Typer: The configured root application
    """
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

    # Register command groups
    from reflexio.cli.commands.agent_playbooks import app as agent_playbooks_app
    from reflexio.cli.commands.api import app as api_app
    from reflexio.cli.commands.auth import app as auth_app
    from reflexio.cli.commands.config_cmd import app as config_app
    from reflexio.cli.commands.doctor import app as doctor_app
    from reflexio.cli.commands.interactions import app as interactions_app
    from reflexio.cli.commands.profiles import app as profiles_app
    from reflexio.cli.commands.services import app as services_app
    from reflexio.cli.commands.setup_cmd import app as setup_app
    from reflexio.cli.commands.shortcuts import register_shortcuts
    from reflexio.cli.commands.status_cmd import app as status_app
    from reflexio.cli.commands.user_playbooks import app as user_playbooks_app

    _skip = exclude or set()

    _sub_apps: list[tuple[typer.Typer, str]] = [
        (services_app, "services"),
        (interactions_app, "interactions"),
        (profiles_app, "user-profiles"),
        (agent_playbooks_app, "agent-playbooks"),
        (user_playbooks_app, "user-playbooks"),
        (config_app, "config"),
        (auth_app, "auth"),
        (status_app, "status"),
        (api_app, "api"),
        (doctor_app, "doctor"),
        (setup_app, "setup"),
    ]
    for sub_app, name in _sub_apps:
        if name not in _skip:
            app.add_typer(sub_app, name=name)

    # Top-level shortcuts (publish, search, context)
    register_shortcuts(app)

    return app
