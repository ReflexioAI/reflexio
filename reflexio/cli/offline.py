"""Import-light dispatch for explicit file operations before operator bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import click
import typer
from typer.main import get_command

from reflexio.cli.errors import EXIT_VALIDATION, CliError, render_error
from reflexio.cli.output import print_info, render
from reflexio.cli.root_app import create_root_app


def upgrade_file(
    ctx: typer.Context,
    source: Annotated[
        Path, typer.Argument(help="Input organization configuration JSON")
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", help="Canonical copy path; existing output is replaced"
        ),
    ],
) -> None:
    """Convert an explicit JSON file offline, preserving its original bytes."""
    from reflexio.cli.config_upgrade import convert_config_file

    try:
        convert_config_file(source, output)
    except (OSError, ValueError) as exc:
        error = (
            CliError(
                message="Could not read the input or publish the output; check paths and permissions."
            )
            if isinstance(exc, OSError)
            else CliError(
                error_type="validation", message=str(exc), exit_code=EXIT_VALIDATION
            )
        )
        render_error(error, json_mode=ctx.obj.json_mode)
        raise typer.Exit(error.exit_code) from None
    if ctx.obj.json_mode:
        render({"output": str(output)}, json_mode=True)
    else:
        print_info(f"Canonical configuration written to {output}")


def dispatch_offline(argv: list[str]) -> bool:
    """Run the explicit converter without loading dotenv, services, or clients.

    Click parses the shared global option definitions before selecting a route;
    option values that happen to equal command names never become commands.
    Other routes retain their usual operator environment bootstrap.
    """
    app = create_root_app()
    config_app = typer.Typer()
    config_app.command(name="upgrade-file")(upgrade_file)
    app.add_typer(config_app, name="config")
    root = get_command(app)
    parser = click.Command(
        "reflexio",
        params=root.params,
        context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
    )
    with parser.make_context("reflexio", list(argv), resilient_parsing=True) as ctx:
        if ctx.args[:2] != ["config", "upgrade-file"]:
            return False
    app(argv, standalone_mode=True)
    return True
