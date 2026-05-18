"""Embedding service commands."""

from __future__ import annotations

import os
from typing import Annotated

import typer
import uvicorn

app = typer.Typer(help="Run Reflexio embedding services.")

_DEFAULT_EMBEDDING_PORT = 8072


def _resolve_port(port: int | None) -> int:
    if port is not None:
        return port

    raw_port = os.environ.get("EMBEDDING_PORT")
    if raw_port is None:
        return _DEFAULT_EMBEDDING_PORT

    try:
        return int(raw_port)
    except ValueError:
        typer.echo(
            f"Warning: invalid EMBEDDING_PORT={raw_port!r}; "
            f"using default {_DEFAULT_EMBEDDING_PORT}",
            err=True,
        )
        return _DEFAULT_EMBEDDING_PORT


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option(help="Host interface for the embedding daemon."),
    ] = "127.0.0.1",
    port: Annotated[
        int | None,
        typer.Option(help="Embedding service port (default: EMBEDDING_PORT or 8072)."),
    ] = None,
) -> None:
    """Serve an OpenAI-compatible local embedding endpoint."""
    resolved_port = _resolve_port(port)
    uvicorn.run(
        "reflexio.server.llm.embedding_service:app",
        host=host,
        port=resolved_port,
        log_level="info",
    )
