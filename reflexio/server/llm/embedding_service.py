"""OpenAI-compatible local embedding service."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from reflexio.server.llm.llm_utils import positive_int_env
from reflexio.server.llm.providers.local_embedding_provider import LocalEmbedder
from reflexio.server.llm.providers.nomic_embedding_provider import (
    NomicEmbedder,
    is_nomic_model,
)

logger = logging.getLogger(__name__)

MINILM_MODEL = "local/minilm-l6-v2"
NOMIC_TEXT_MODEL = "local/nomic-embed-text-v1.5"
_SUPPORTED_MODELS = {
    MINILM_MODEL,
    "local/nomic-embed-v1.5",
    NOMIC_TEXT_MODEL,
}
_ACTIVE_MODEL: str | None = None
_ACTIVE_MODEL_LOCK = threading.Lock()

DEFAULT_OSS_EMBEDDING_MODEL = MINILM_MODEL

# Bound how many embed/encode calls run at once. The endpoint is a sync ``def``
# served from Starlette's threadpool, so without a guard a burst of requests
# would run that many model.encode() calls in parallel, stacking their
# activation memory and OOM-killing the daemon. The semaphore caps simultaneous
# encodes; excess requests block on acquire() and are picked up when a slot
# frees — they queue, they are never rejected. Pair with a small encode
# batch_size so each in-flight encode stays cheap.
_DEFAULT_MAX_CONCURRENCY = 4
_ENV_MAX_CONCURRENCY = "REFLEXIO_EMBED_MAX_CONCURRENCY"
_ENCODE_SEMAPHORE: threading.BoundedSemaphore | None = None
_ENCODE_SEMAPHORE_LOCK = threading.Lock()


def _max_concurrency() -> int:
    """Resolve the max simultaneous encodes from env, defaulting to 4."""
    return positive_int_env(_ENV_MAX_CONCURRENCY, _DEFAULT_MAX_CONCURRENCY, logger)


def _encode_semaphore() -> threading.BoundedSemaphore:
    """Return the process-wide encode semaphore, building it on first use."""
    global _ENCODE_SEMAPHORE
    if _ENCODE_SEMAPHORE is None:
        with _ENCODE_SEMAPHORE_LOCK:
            if _ENCODE_SEMAPHORE is None:
                _ENCODE_SEMAPHORE = threading.BoundedSemaphore(_max_concurrency())
    return _ENCODE_SEMAPHORE


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


def create_embedding_app(default_model: str | None = None) -> FastAPI:
    """Create the embedding daemon app and optionally warm a default model."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if not default_model:
            yield
            return
        try:
            _embed_texts(default_model, ["reflexio embedding daemon warmup"])
        except Exception:
            logger.exception("Failed to warm embedding model %s", default_model)
            raise
        yield

    embedding_app = FastAPI(title="Reflexio Embedding Service", lifespan=lifespan)

    @embedding_app.get("/health")
    def health() -> dict[str, Any]:
        """Health check endpoint."""
        return {"status": "ok", "active_model": _ACTIVE_MODEL}

    @embedding_app.post("/v1/embeddings")
    def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        """Create embeddings using the daemon's single active local model."""
        texts = (
            [request.input] if isinstance(request.input, str) else list(request.input)
        )
        embeddings = _embed_texts(request.model, texts)

        if request.dimensions:
            embeddings = [
                _resize_embedding(vec, request.dimensions) for vec in embeddings
            ]

        return EmbeddingResponse(
            data=[
                EmbeddingData(embedding=embedding, index=index)
                for index, embedding in enumerate(embeddings)
            ],
            model=request.model,
        )

    return embedding_app


def _embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    _activate_model(model)
    semaphore = _encode_semaphore()
    # Non-blocking probe purely for observability: if no slot is free, this
    # request will have to wait. The real acquire below blocks until a slot
    # opens, so the request queues and is picked up later — never rejected.
    if not semaphore.acquire(blocking=False):
        logger.info(
            "Embedding request queued; all %d encode slots busy",
            _max_concurrency(),
        )
        semaphore.acquire()
    try:
        if is_nomic_model(model):
            return NomicEmbedder.get().embed(texts)
        if model == MINILM_MODEL:
            return LocalEmbedder.get().embed(texts)
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
    finally:
        semaphore.release()


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


app = create_embedding_app(default_model=DEFAULT_OSS_EMBEDDING_MODEL)
