"""Import-light dispatch for explicit file operations before operator bootstrap."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from typer.core import TyperCommand
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


def upgrade_local(
    ctx: typer.Context,
    database: Annotated[
        Path, typer.Argument(help="Absolute path to an existing local SQLite dataset")
    ],
    org_id: Annotated[
        str | None,
        typer.Option(
            help="Explicit namespace for a qualified unclaimed dataset; never overrides an existing claim"
        ),
    ] = None,
) -> None:
    """Establish future learning policy after stopping affected services/hosts.

    Supports one qualified historical schema. Unsupported schemas, retained work
    and financial debt stay held. Never migrates business data or loads .env.
    """
    from reflexio.cli.env_loader import block_implicit_dotenv_walkup

    block_implicit_dotenv_walkup()
    from reflexio.server.services.storage.sqlite_storage._learning_policy_authority import (
        upgrade_existing_sqlite_dataset,
    )
    from reflexio.server.services.storage.storage_base.learning_transition import (
        LearningTransitionBlockedError,
    )

    try:
        result = upgrade_existing_sqlite_dataset(database, org_id=org_id)
    except (OSError, sqlite3.Error, ValueError, LearningTransitionBlockedError) as exc:
        message = (
            str(exc)
            if isinstance(exc, (ValueError, LearningTransitionBlockedError))
            else "Could not inspect or commit the existing local dataset; check its path, permissions and stopped-host state."
        )
        error = CliError(
            error_type="validation", message=message, exit_code=EXIT_VALIDATION
        )
        render_error(error, json_mode=ctx.obj.json_mode)
        raise typer.Exit(error.exit_code) from None
    render(
        {
            "authority": result.authority.to_dict(),
            "established": result.established,
            "provenance": result.provenance,
            "original_provenance_available": result.provenance is not None,
        },
        json_mode=ctx.obj.json_mode,
    )


def create_data_app() -> typer.Typer:
    """Register the same import-light handler for early dispatch and help."""
    app = typer.Typer(help="Explicit offline local-dataset maintenance.")
    app.command(name="upgrade-local")(upgrade_local)
    return app


def cli_command_path(argv: list[str]) -> tuple[str, ...]:
    """Identify the command using the shared globals, before loading dotenv."""
    root = get_command(create_root_app())
    # Typer vendors Click in newer versions. The params must be parsed by
    # Typer's own command class.
    parser = TyperCommand(
        "reflexio",
        params=root.params,
        context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
    )
    with parser.make_context("reflexio", list(argv), resilient_parsing=True) as ctx:
        return tuple(ctx.args[:2])


def dispatch_offline(
    argv: list[str], *, command_path: tuple[str, ...] | None = None
) -> bool:
    """Run the explicit converter without loading dotenv, services, or clients."""
    route = cli_command_path(argv) if command_path is None else command_path
    if route not in (("config", "upgrade-file"), ("data", "upgrade-local")):
        return False
    app = create_root_app()
    config_app = typer.Typer()
    config_app.command(name="upgrade-file")(upgrade_file)
    app.add_typer(config_app, name="config")
    app.add_typer(create_data_app(), name="data")
    app(argv, standalone_mode=True)
    return True
