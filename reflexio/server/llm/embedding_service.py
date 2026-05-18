"""OpenAI-compatible local embedding service."""

from __future__ import annotations

import math
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reflexio.server.llm.providers.local_embedding_provider import LocalEmbedder
from reflexio.server.llm.providers.nomic_embedding_provider import (
    NomicEmbedder,
    is_nomic_model,
)

_MINILM_MODEL = "local/minilm-l6-v2"
_SUPPORTED_MODELS = {
    _MINILM_MODEL,
    "local/nomic-embed-v1.5",
    "local/nomic-embed-text-v1.5",
}
_ACTIVE_MODEL: str | None = None
_ACTIVE_MODEL_LOCK = threading.Lock()

app = FastAPI(title="Reflexio Embedding Service")


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    dimensions: int | None = Field(default=None, gt=0)


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "active_model": _ACTIVE_MODEL}


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings using the daemon's single active local model."""
    _activate_model(request.model)
    texts = [request.input] if isinstance(request.input, str) else list(request.input)
    if is_nomic_model(request.model):
        embeddings = NomicEmbedder.get().embed(texts)
    elif request.model == _MINILM_MODEL:
        embeddings = LocalEmbedder.get().embed(texts)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported model: {request.model}"
        )

    if request.dimensions:
        embeddings = [_resize_embedding(vec, request.dimensions) for vec in embeddings]

    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=embedding, index=index)
            for index, embedding in enumerate(embeddings)
        ],
        model=request.model,
    )


def _activate_model(model: str) -> None:
    if model not in _SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
    global _ACTIVE_MODEL
    with _ACTIVE_MODEL_LOCK:
        if _ACTIVE_MODEL is None:
            _ACTIVE_MODEL = model
            return
        if model != _ACTIVE_MODEL:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Embedding daemon already owns model "
                    f"{_ACTIVE_MODEL}; start a separate daemon for {model}"
                ),
            )


def _resize_embedding(vec: list[float], dimensions: int) -> list[float]:
    if dimensions == len(vec):
        return vec
    if dimensions > len(vec):
        raise HTTPException(
            status_code=400,
            detail=f"dimensions={dimensions} exceeds model output length {len(vec)}",
        )
    sliced = vec[:dimensions]
    norm = math.sqrt(sum(value * value for value in sliced))
    if norm <= 0:
        return sliced
    return [value / norm for value in sliced]
