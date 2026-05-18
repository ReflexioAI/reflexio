"""Embedding service commands."""

from __future__ import annotations

import os
from typing import Annotated

import typer
import uvicorn

app = typer.Typer(help="Run Reflexio embedding services.")


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
    resolved_port = port or int(os.environ.get("EMBEDDING_PORT", "8072"))
    uvicorn.run(
        "reflexio.server.llm.embedding_service:app",
        host=host,
        port=resolved_port,
        log_level="info",
    )
