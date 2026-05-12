"""Service management commands (Typer wrapper around existing run/stop logic)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from reflexio.cli import run_services as run_mod
from reflexio.cli import stop_services as stop_mod
from reflexio.cli.bootstrap_config import _VALID_STORAGE_BACKENDS

_logger = logging.getLogger(__name__)

app = typer.Typer(help="Start and stop Reflexio services.")


def _ensure_llm_configured(env_path: Path) -> None:
    """Run the first-run LLM + embedding wizard when no key is configured.

    Called by ``services start`` before uvicorn boots. Replaces the ugly
    ``RuntimeError: No LLM API keys found`` traceback that used to crash the
    FastAPI lifespan handler on a fresh install.

    Behaviour matrix:
        - At least one LLM key AND an embedding-capable provider in env →
          return silently, startup continues.
        - At least one LLM key but no cloud embedder, and ``chromadb`` is
          importable → log "Using local embedder as fallback" and return;
          runtime auto-detection (Layer A) will pick ``local/minilm-l6-v2``
          for the EMBEDDING role. No prompt, no blocking.
        - No LLM key, interactive TTY → prompt for a provider + key, then
          (conditionally) prompt for an embedding key, re-load the .env so
          the new values land in ``os.environ``, and return.
        - Missing only an embedding key (e.g. user set ANTHROPIC_API_KEY
          manually) AND chromadb is unavailable → prompt only for an
          embedding provider.
        - No LLM key, non-interactive stdin (CI, nohup, container) → print a
          clean pointer to the .env file and raise ``typer.Exit(1)`` so the
          user never sees the uvicorn/starlette traceback.

    Args:
        env_path (Path): Path to the user's ``.env`` file (``~/.reflexio/.env``
            in the default installation). Keys selected in the wizard are
            written here.

    Raises:
        typer.Exit: When stdin is not a TTY and no LLM key is configured.
    """
    from dotenv import load_dotenv

    from reflexio.cli.commands.setup_cmd import (
        _prompt_embedding_provider,
        _prompt_llm_provider,
    )
    from reflexio.server.llm.model_defaults import (
        EMBEDDING_CAPABLE_PROVIDERS,
        GENERATION_CAPABLE_PROVIDERS,
        detect_available_providers,
    )
    from reflexio.server.llm.providers.local_embedding_provider import (
        is_chromadb_importable,
    )

    providers = detect_available_providers()
    has_embedding = any(p in EMBEDDING_CAPABLE_PROVIDERS for p in providers)
    has_generation = any(p in GENERATION_CAPABLE_PROVIDERS for p in providers)
    if providers and has_generation and has_embedding:
        return

    # Path 3 of the embedding auto-detection: if a generation provider is
    # configured but no cloud embedder, and chromadb is importable, the
    # runtime will silently fall back to the local MiniLM embedder (see
    # ``_auto_detect_model`` in model_defaults.py). No need to prompt or
    # block startup. We require ``has_generation`` here because providers
    # like ``["local"]`` (embedder-only) leave the GENERATION role
    # unresolvable and must still trip the wizard.
    if has_generation and is_chromadb_importable():
        _logger.info("Using local embedder as fallback (no cloud embedder configured)")
        return

    if not sys.stdin.isatty():
        typer.echo(
            "\nReflexio is not fully configured yet — "
            + (
                "no generation-capable LLM API key"
                if not has_generation
                else "no embedding-capable provider"
            )
            + f" was found in {env_path}.\n"
            "Set one of OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, ... "
            "in that file, or run `reflexio setup init` interactively, "
            "then retry `reflexio services start`."
        )
        raise typer.Exit(1)

    if not has_generation:
        typer.echo(
            "\nWelcome to Reflexio! Let's pick an LLM provider before "
            "starting the local services."
        )
        typer.echo(
            "(If you wanted Managed or Self-hosted Reflexio instead, press "
            "Ctrl+C and run `reflexio setup init`.)"
        )
        _, _, provider_key = _prompt_llm_provider(env_path)
        _prompt_embedding_provider(env_path, provider_key)
    else:
        typer.echo(
            "\nYour LLM provider doesn't support text embeddings — Reflexio "
            "needs an embedding model for semantic search."
        )
        non_embedding_provider = next(
            p for p in providers if p not in EMBEDDING_CAPABLE_PROVIDERS
        )
        _prompt_embedding_provider(env_path, non_embedding_provider)

    load_dotenv(dotenv_path=env_path, override=True)
    typer.echo()


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
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help=(
                "Number of backend worker processes (daemon mode only). "
                "Default 2 enables zero-downtime worker recycling."
            ),
        ),
    ] = 2,
    max_requests: Annotated[
        int,
        typer.Option(
            "--max-requests",
            help="Worker recycles after this many requests (0 disables). Default 10000.",
        ),
    ] = 10000,
    max_requests_jitter: Annotated[
        int,
        typer.Option("--max-requests-jitter", help="Random 0..jitter per worker."),
    ] = 1000,
    graceful_shutdown_sec: Annotated[
        int,
        typer.Option(
            "--graceful-shutdown-sec",
            help="Seconds to drain in-flight requests on shutdown.",
        ),
    ] = 30,
) -> None:
    """Start Reflexio services (backend, docs)."""
    from reflexio.cli.bootstrap_config import resolve_storage, save_storage_to_config
    from reflexio.cli.env_loader import get_env_path, load_reflexio_env

    # Load .env BEFORE resolve_storage so env vars from ~/.reflexio/.env
    # (e.g. REFLEXIO_STORAGE=supabase) are visible to the resolution chain.
    load_reflexio_env()

    # First-run guard: if the backend's lifespan validator would reject the
    # current env (no LLM key, or no embedding-capable provider), prompt now
    # so users see a friendly wizard instead of a lifespan traceback.
    _ensure_llm_configured(get_env_path())

    resolved = resolve_storage(storage)
    os.environ["REFLEXIO_STORAGE"] = resolved

    # If user explicitly passed --storage, also persist to config and .env
    if storage is not None:
        save_storage_to_config(resolved)

        from reflexio.cli.env_loader import set_env_var

        env_path = get_env_path()
        if env_path.exists():
            set_env_var(env_path, "REFLEXIO_STORAGE", resolved)

    args = argparse.Namespace(
        backend_port=backend_port,
        docs_port=docs_port,
        only=only,
        no_reload=no_reload,
        workers=workers,
        max_requests=max_requests,
        max_requests_jitter=max_requests_jitter,
        graceful_shutdown_sec=graceful_shutdown_sec,
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
