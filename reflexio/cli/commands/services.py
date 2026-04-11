"""Service management commands (Typer wrapper around existing run/stop logic)."""

from __future__ import annotations

import argparse
import os
from typing import Annotated

import typer

from reflexio.cli import run_services as run_mod
from reflexio.cli import stop_services as stop_mod

app = typer.Typer(help="Start and stop Reflexio services.")

_VALID_STORAGE_BACKENDS = {"sqlite", "supabase", "disk"}


def validate_storage_backend(storage: str | None) -> None:
    """Validate and apply a storage backend selection.

    If *storage* is not None, validates it against known backends and sets
    the ``REFLEXIO_STORAGE`` environment variable.

    Args:
        storage: Storage backend name (e.g. ``"sqlite"``, ``"supabase"``),
            or None to skip validation.

    Raises:
        typer.BadParameter: If *storage* is not a recognised backend.
    """
    if storage is None:
        return
    storage_lower = storage.lower()
    if storage_lower not in _VALID_STORAGE_BACKENDS:
        raise typer.BadParameter(
            f"Invalid storage backend '{storage}'. "
            f"Must be one of: {', '.join(sorted(_VALID_STORAGE_BACKENDS))}"
        )
    os.environ["REFLEXIO_STORAGE"] = storage_lower


@app.command()
def start(
    backend_port: Annotated[int, typer.Option(help="Backend server port")] = 8081,
    docs_port: Annotated[int, typer.Option(help="Docs server port")] = 8082,
    only: Annotated[
        str | None, typer.Option(help="Comma-separated services: backend,docs")
    ] = None,
    no_reload: Annotated[
        bool, typer.Option("--no-reload", help="Disable uvicorn auto-reload")
    ] = False,
    storage: Annotated[
        str | None,
        typer.Option(help="Data storage backend: sqlite (default) or supabase"),
    ] = None,
) -> None:
    """Start Reflexio services (backend, docs)."""
    validate_storage_backend(storage)
    # Bridge to existing argparse-based implementation
    args = argparse.Namespace(
        backend_port=backend_port,
        docs_port=docs_port,
        only=only,
        no_reload=no_reload,
    )
    run_mod.execute(args)


@app.command()
def stop(
    backend_port: Annotated[int, typer.Option(help="Backend server port")] = 8081,
    docs_port: Annotated[int, typer.Option(help="Docs server port")] = 8082,
    only: Annotated[
        str | None, typer.Option(help="Comma-separated services: backend,docs")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="SIGKILL immediately")] = False,
) -> None:
    """Stop Reflexio services."""
    args = argparse.Namespace(
        backend_port=backend_port,
        docs_port=docs_port,
        only=only,
        force=force,
    )
    stop_mod.execute(args)
