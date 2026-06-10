"""OpenAI-compatible local embedding service."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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

# Opportunistically coalesce concurrent small requests before calling
# model.encode(). This keeps request concurrency thread-based while giving the
# GPU larger batches during bursts.
_DEFAULT_MICRO_BATCH_DELAY_MS = 5
_DEFAULT_MICRO_BATCH_MAX_TEXTS = 64
_ENV_MICRO_BATCH_DELAY_MS = "REFLEXIO_EMBED_MICRO_BATCH_DELAY_MS"
_ENV_MICRO_BATCH_MAX_TEXTS = "REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS"
_MICRO_BATCH_CONDITION = threading.Condition()
_MICRO_BATCH_QUEUE: list[_EmbeddingJob] = []
_ACTIVE_BATCH_PROCESSORS = 0
# Failsafe bound for a submitter waiting on its job (see _embed_texts).
_JOB_WAIT_TIMEOUT_SECONDS = 600.0


@dataclass
class _EmbeddingJob:
    model: str
    texts: list[str]
    done: threading.Event = field(default_factory=threading.Event)
    result: list[list[float]] | None = None
    error: BaseException | None = None


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to default %d", name, raw, default)
        return default
    return value if value >= 0 else default


def _max_concurrency() -> int:
    """Resolve the max simultaneous encodes from env, defaulting to 4."""
    return positive_int_env(_ENV_MAX_CONCURRENCY, _DEFAULT_MAX_CONCURRENCY, logger)


def _micro_batch_delay_seconds() -> float:
    return (
        _nonnegative_int_env(_ENV_MICRO_BATCH_DELAY_MS, _DEFAULT_MICRO_BATCH_DELAY_MS)
        / 1000
    )


def _micro_batch_max_texts() -> int:
    return positive_int_env(
        _ENV_MICRO_BATCH_MAX_TEXTS, _DEFAULT_MICRO_BATCH_MAX_TEXTS, logger
    )


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
    if not texts:
        return []

    job = _EmbeddingJob(model=model, texts=list(texts))
    should_process = False

    global _ACTIVE_BATCH_PROCESSORS
    with _MICRO_BATCH_CONDITION:
        _MICRO_BATCH_QUEUE.append(job)
        if _max_concurrency() > _ACTIVE_BATCH_PROCESSORS:
            _ACTIVE_BATCH_PROCESSORS += 1
            should_process = True
        _MICRO_BATCH_CONDITION.notify()

    if should_process:
        threading.Thread(
            target=_run_micro_batch_processor,
            daemon=True,
            name="embedding-micro-batch",
        ).start()

    # Last-resort failsafe: a processor thread that dies between taking and
    # completing jobs would otherwise leave this request hanging forever. The
    # bound is far above any legitimate CPU bulk encode.
    if not job.done.wait(timeout=_JOB_WAIT_TIMEOUT_SECONDS):
        raise RuntimeError(
            f"Embedding micro-batch did not complete within "
            f"{_JOB_WAIT_TIMEOUT_SECONDS:.0f}s"
        )
    if job.error is not None:
        raise job.error
    if job.result is None:
        raise RuntimeError("Embedding micro-batch completed without a result")
    return job.result


def _run_micro_batch_processor() -> None:
    global _ACTIVE_BATCH_PROCESSORS
    try:
        while True:
            jobs = _take_micro_batch()
            if not jobs:
                with _MICRO_BATCH_CONDITION:
                    if _MICRO_BATCH_QUEUE:
                        continue
                    _ACTIVE_BATCH_PROCESSORS -= 1
                    _MICRO_BATCH_CONDITION.notify_all()
                    return
            _process_micro_batch(jobs)
    except BaseException:
        with _MICRO_BATCH_CONDITION:
            _ACTIVE_BATCH_PROCESSORS -= 1
            _MICRO_BATCH_CONDITION.notify_all()
        raise


def _take_micro_batch() -> list[_EmbeddingJob]:
    max_texts = _micro_batch_max_texts()
    deadline = time.monotonic() + _micro_batch_delay_seconds()

    with _MICRO_BATCH_CONDITION:
        if not _MICRO_BATCH_QUEUE:
            return []

        first = _MICRO_BATCH_QUEUE.pop(0)
        jobs = [first]
        total_texts = len(first.texts)

        while total_texts < max_texts:
            compatible_index = next(
                (
                    index
                    for index, candidate in enumerate(_MICRO_BATCH_QUEUE)
                    if candidate.model == first.model
                    and total_texts + len(candidate.texts) <= max_texts
                ),
                None,
            )
            if compatible_index is not None:
                candidate = _MICRO_BATCH_QUEUE.pop(compatible_index)
                jobs.append(candidate)
                total_texts += len(candidate.texts)
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _MICRO_BATCH_CONDITION.wait(timeout=remaining)

        return jobs


def _process_micro_batch(jobs: list[_EmbeddingJob]) -> None:
    texts: list[str] = []
    slices: list[tuple[_EmbeddingJob, int, int]] = []
    for job in jobs:
        start = len(texts)
        texts.extend(job.texts)
        slices.append((job, start, len(texts)))

    try:
        embeddings = _encode_texts_now(jobs[0].model, texts)
    except BaseException as exc:
        for job in jobs:
            job.error = exc
            job.done.set()
        return

    if len(embeddings) != len(texts):
        # Slicing a short result would silently hand jobs truncated/empty
        # vectors; fail every job loudly at the source instead.
        mismatch = RuntimeError(
            f"Embedding count mismatch: encoded {len(texts)} texts but got "
            f"{len(embeddings)} embeddings"
        )
        for job in jobs:
            job.error = mismatch
            job.done.set()
        return

    for job, start, end in slices:
        job.result = embeddings[start:end]
        job.done.set()


def _encode_texts_now(model: str, texts: list[str]) -> list[list[float]]:
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
