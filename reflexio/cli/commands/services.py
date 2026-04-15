"""Service management commands (Typer wrapper around existing run/stop logic)."""

from __future__ import annotations

import argparse
import os
from typing import Annotated

import typer

from reflexio.cli import run_services as run_mod
from reflexio.cli import stop_services as stop_mod
from reflexio.cli.bootstrap_config import _VALID_STORAGE_BACKENDS

app = typer.Typer(help="Start and stop Reflexio services.")


def validate_storage_backend(storage: str | None) -> None:
    """Validate and apply a storage backend selection.

    If *storage* is not None, validates it against known backends and sets
    the ``REFLEXIO_STORAGE`` environment variable.

    .. deprecated::
        Prefer :func:`reflexio.cli.bootstrap_config.resolve_storage` which
        implements the full priority chain (CLI flag > env var > config > default)
        and config file persistence.

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
    backend_port: Annotated[
        int | None, typer.Option(help="Backend server port (default: 8081)")
    ] = None,
    docs_port: Annotated[
        int | None, typer.Option(help="Docs server port (default: 8082)")
    ] = None,
    only: Annotated[
        str | None, typer.Option(help="Comma-separated services: backend,docs")
    ] = None,
    no_reload: Annotated[
        bool, typer.Option("--no-reload", help="Disable uvicorn auto-reload")
    ] = False,
    storage: Annotated[
        str | None,
        typer.Option(help="Data storage backend: sqlite (default), supabase, or disk"),
    ] = None,
) -> None:
    """Start Reflexio services (backend, docs)."""
    from reflexio.cli.bootstrap_config import resolve_storage, save_storage_to_config
    from reflexio.cli.env_loader import load_reflexio_env

    # Load .env BEFORE resolve_storage so env vars from ~/.reflexio/.env
    # (e.g. REFLEXIO_STORAGE=supabase) are visible to the resolution chain.
    load_reflexio_env()

    resolved = resolve_storage(storage)
    os.environ["REFLEXIO_STORAGE"] = resolved

    # If user explicitly passed --storage, also persist to config and .env
    if storage is not None:
        save_storage_to_config(resolved)

        from reflexio.cli.env_loader import get_env_path, set_env_var

        env_path = get_env_path()
        if env_path.exists():
            set_env_var(env_path, "REFLEXIO_STORAGE", resolved)

    args = argparse.Namespace(
        backend_port=backend_port,
        docs_port=docs_port,
        only=only,
        no_reload=no_reload,
    )
    run_mod.execute(args)


@app.command()
def stop(
    backend_port: Annotated[
        int | None, typer.Option(help="Backend server port (default: 8081)")
    ] = None,
    docs_port: Annotated[
        int | None, typer.Option(help="Docs server port (default: 8082)")
    ] = None,
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
