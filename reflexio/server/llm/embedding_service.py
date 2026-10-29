"""OpenAI-compatible local embedding service."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from _thread import LockType
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import anyio
from anyio.to_thread import run_sync
from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from reflexio.server.background_work import (
    report_background_failure,
    report_background_success,
)
from reflexio.server.llm.providers.local_embedding_provider import LocalEmbedder
from reflexio.server.llm.providers.nomic_embedding_provider import (
    NomicEmbedder,
    is_nomic_model,
)
from reflexio.server.llm.rerank.common import (
    RERANK_MODEL,
    CrossEncoderUnavailableError,
    reranker_enabled,
)
from reflexio.server.llm.rerank.cross_encoder_model import (
    CrossEncoderRunner,
    RerankWork,
)
from reflexio.server.llm.sentence_encoder import embedding_device

logger = logging.getLogger(__name__)
# The standalone inference service is operator-facing infrastructure. Keep its
# lifecycle contract visible even when the hosting uvicorn process leaves the
# root logger at WARNING.
logger.setLevel(logging.INFO)

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

# One queue owner dispatches the daemon's single active model. The model
# runners also serialize direct callers; competing queue processors only wait
# on those model locks and fragment batches. Coalesce work already waiting,
# without delaying an idle request to fill a batch.
_MICRO_BATCH_LOCK = threading.Lock()
_MICRO_BATCH_QUEUE: list[_EmbeddingJob] = []
_BATCH_WORKER_RUNNING = False
# Public request default; the queue uses this same budget rather than a
# separate submitter failsafe. HTTP requests pass their selected timeout.
_DEFAULT_TIMEOUT_MS = 30_000

EmbeddingEncoder = Callable[[list[str]], list[list[float]]]
InferenceSubmit = Callable[
    [
        Callable[[list[str]], list[Any]],
        list[str],
        Literal["search", "interactive", "bulk"],
        int,
        int,
    ],
    Awaitable[list[Any]],
]


@dataclass
class _EmbeddingJob:
    model: str
    texts: list[str]
    encoder: EmbeddingEncoder | None = None
    done: threading.Event = field(default_factory=threading.Event)
    result: list[list[float]] = field(default_factory=list)
    pending: deque[int] = field(init=False)
    turn: list[int] = field(default_factory=list)
    rows: dict[int, list[float]] = field(default_factory=dict)
    capacity: int | None = field(init=False)
    deadline: float = float("inf")
    error: BaseException | None = None

    def __post_init__(self) -> None:
        encoder = (
            self.encoder if self.encoder is not None else _model_encoder(self.model)
        )
        batched = isinstance(encoder, _BatchedEncoder)
        self.capacity = encoder.batch_size if batched else None
        if self.capacity is not None and self.capacity <= 0:
            raise ValueError("Embedding encoder batch capacity must be positive")
        indices = range(len(self.texts))
        # Native sentence encoders already sort by text length to avoid padding.
        # Preserve that packing across requests, then restore original row order.
        self.pending = deque(
            sorted(indices, key=lambda i: len(self.texts[i]), reverse=True)
            if batched
            else indices
        )


@runtime_checkable
class _BatchedEncoder(Protocol):
    @property
    def batch_size(self) -> int: ...

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


def _model_encoder(model: str) -> _BatchedEncoder:
    if is_nomic_model(model):
        return NomicEmbedder.get()
    if model == MINILM_MODEL:
        return LocalEmbedder.get()
    raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    dimensions: int | None = Field(default=None, gt=0)
    priority: Literal["interactive", "bulk"] = "bulk"
    purpose: Literal["document", "query"] = "document"
    timeout_ms: int = Field(default=_DEFAULT_TIMEOUT_MS, gt=0, le=600_000)


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    priority: Literal["search", "interactive"] = "interactive"
    timeout_ms: int = Field(default=5_000, gt=0, le=600_000)


class RerankData(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    object: str = "list"
    data: list[RerankData]
    model: str


class RerankHealthResponse(BaseModel):
    status: Literal["ready", "disabled", "unavailable"]
    configured_model: str
    enabled: bool
    ready: bool


def _validate_embedding_width(
    model: str, embeddings: list[list[float]], width: int | None
) -> None:
    if width is not None:
        for vector in embeddings:
            if len(vector) != width:
                raise HTTPException(
                    500,
                    f"Encoder for {model} returned {len(vector)} dimensions; expected {width}",
                )


def _warm_reranker(
    runner: CrossEncoderRunner, require_ready: bool, *, enabled: bool = True
) -> None:
    if enabled and reranker_enabled() and not runner.prewarm() and require_ready:
        raise RuntimeError("CPU inference reranker failed its startup readiness check")


async def _until_disconnected(
    request: Request, submission: Awaitable[list[Any]]
) -> list[Any]:
    """Cancel queued CPU work when its HTTP borrower disappears."""
    result = None
    error = None

    async def work() -> None:
        nonlocal result, error
        try:
            result = await submission
        except Exception as exc:
            error = exc
        finally:
            group.cancel_scope.cancel()

    async def disconnected() -> None:
        nonlocal error
        while not await request.is_disconnected():
            await anyio.sleep(0.1)
        error = HTTPException(499, "Inference client disconnected")
        group.cancel_scope.cancel()

    # Use the same cancellation scopes as Request.is_disconnected. Mixing
    # asyncio Task.cancel with its AnyIO scope can lose cancellation and hang.
    async with anyio.create_task_group() as group:
        group.start_soon(work)
        group.start_soon(disconnected)
    if error is not None:
        raise error
    if result is None:
        raise RuntimeError("Inference submission ended without a result")
    return result


def _model_health(
    model_contract: Callable[[], Mapping[str, str | int]] | None,
    workload: str,
) -> dict[str, str | int]:
    if model_contract is None:
        raise HTTPException(
            status_code=503, detail="Pinned model identity is not configured"
        )
    # Existing clients bind query-only embedding admission through this contract.
    # Search adds a separately advertised rerank capability without changing it.
    embedding_workload = "query" if workload == "search" else workload
    return {**model_contract(), "workload": embedding_workload}


def _workload(query_only: bool, workload: str | None) -> str:
    resolved = workload or ("query" if query_only else "shared")
    if resolved not in {"shared", "query", "search"}:
        raise ValueError("Inference workload must be shared, query or search")
    if query_only and resolved != "query":
        raise ValueError("query_only conflicts with the explicit workload")
    return resolved


async def _recover_search_reranker(
    request: Request,
    runner: CrossEncoderRunner,
    recovery_lock: LockType,
    inference_submit: InferenceSubmit | None,
    *,
    enabled: bool,
) -> None:
    if not enabled or not reranker_enabled() or runner.ready():
        return

    def recover(_texts: list[str]) -> list[bool]:
        # A timed-out prediction still owns this lock until it actually ends.
        # Other health requests must not start another model recovery attempt.
        if not recovery_lock.acquire(blocking=False):
            return [False]
        try:
            return [runner.ready() or runner.prewarm()]
        finally:
            recovery_lock.release()

    try:
        if inference_submit is not None:
            await _until_disconnected(
                request, inference_submit(recover, ["warmup"], "interactive", 1000, 1)
            )
        else:
            with anyio.fail_after(1):
                # Starlette supports AnyIO 3 and 4; this alias works with both.
                await run_sync(recover, ["warmup"], cancellable=True)
    except (CrossEncoderUnavailableError, HTTPException, TimeoutError):
        # Readiness stays false unless the runner completes a real probe.
        logger.warning("Search reranker health recovery did not complete")


def create_embedding_app(
    default_model: str | None = None,
    *,
    allowed_models: set[str] | None = None,
    model_encoders: Mapping[str, EmbeddingEncoder] | None = None,
    fixed_dimensions: Mapping[str, int] | None = None,
    reranker_model: str = RERANK_MODEL,
    reranker_runner: CrossEncoderRunner | None = None,
    inference_submit: InferenceSubmit | None = None,
    inference_shutdown: Callable[[], None] | None = None,
    query_only: bool = False,
    workload: str | None = None,
    model_contract: Callable[[], Mapping[str, str | int]] | None = None,
) -> FastAPI:
    """Create a daemon with optional encoders and their pinned vector identity.

    Isolated query routing requires ``model_contract`` to report the actual
    encoder's model, immutable revision and dimensions. Workload is derived
    from this app's admission policy, never from the supplied metadata.
    """
    selected_workload = _workload(query_only, workload)
    query_only = selected_workload != "shared"
    rerank_allowed = selected_workload != "query"
    effective_encoders = dict(model_encoders or {})
    effective_fixed_dimensions = dict(fixed_dimensions or {})
    effective_allowed_models = (
        set(allowed_models) if allowed_models is not None else set(_SUPPORTED_MODELS)
    )
    runner = reranker_runner or CrossEncoderRunner(reranker_model)
    recovery_lock = threading.Lock()

    def embed_texts(
        model: str, texts: list[str], timeout_ms: int | None = None
    ) -> list[list[float]]:
        return _embed_texts(
            model,
            texts,
            encoder=effective_encoders.get(model),
            allowed_models=effective_allowed_models,
            timeout_ms=timeout_ms,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if default_model:
            try:
                embed_texts(default_model, ["reflexio embedding daemon warmup"])
            except Exception:
                logger.exception("Failed to warm embedding model %s", default_model)
                raise
        _warm_reranker(
            runner,
            require_ready=inference_submit is not None or selected_workload == "search",
            enabled=rerank_allowed,
        )
        logger.info(
            "event=inference_service_ready configured_model=%s "
            "configured_reranker_model=%s embedding_device=%s "
            "reranker_device=%s reranker_ready=%s hf_offline=%s workload=%s",
            default_model,
            reranker_model,
            embedding_device(),
            os.environ.get("REFLEXIO_RERANK_DEVICE", "cpu"),
            rerank_allowed and runner.ready(),
            os.environ.get("HF_HUB_OFFLINE", "0"),
            selected_workload,
        )
        try:
            yield
        finally:
            if inference_shutdown is not None:
                inference_shutdown()

    embedding_app = FastAPI(title="Reflexio Embedding Service", lifespan=lifespan)

    embedding_app.get("/health/model", name="health_model")(
        lambda: _model_health(model_contract, selected_workload)
    )

    @embedding_app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        """Health check endpoint."""
        await _recover_search_reranker(
            request,
            runner,
            recovery_lock,
            inference_submit,
            enabled=selected_workload == "search",
        )
        return {
            "status": "ok",
            "active_model": _ACTIVE_MODEL,
            "configured_model": default_model,
            "configured_reranker_model": reranker_model,
            "workload": selected_workload,
            "reranker_enabled": rerank_allowed and reranker_enabled(),
            "reranker_ready": rerank_allowed and runner.ready(),
        }

    @embedding_app.get("/health/rerank", response_model=RerankHealthResponse)
    async def rerank_health(
        request: Request, response: Response
    ) -> RerankHealthResponse:
        """Report reranker readiness without affecting embedding health."""
        await _recover_search_reranker(
            request,
            runner,
            recovery_lock,
            inference_submit,
            enabled=selected_workload == "search",
        )
        enabled = rerank_allowed and reranker_enabled()
        ready = enabled and runner.ready()
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return RerankHealthResponse(
            status=runner.status() if enabled else "disabled",
            configured_model=reranker_model,
            enabled=enabled,
            ready=ready,
        )

    @embedding_app.post("/v1/embeddings")
    async def create_embeddings(
        request: EmbeddingRequest, http_request: Request
    ) -> EmbeddingResponse:
        """Create embeddings using the daemon's single active local model."""
        if query_only and (
            request.purpose != "query"
            or (isinstance(request.input, list) and len(request.input) != 1)
        ):
            raise HTTPException(
                status_code=422,
                detail="Query worker accepts one explicitly marked query only",
            )
        if request.model not in effective_allowed_models:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model for this service: {request.model}",
            )
        expected_dimensions = effective_fixed_dimensions.get(request.model)
        if (
            expected_dimensions is not None
            and request.dimensions is not None
            and request.dimensions != expected_dimensions
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{request.model} embeddings use the fixed "
                    f"{expected_dimensions}-dimension storage contract"
                ),
            )
        texts = (
            [request.input] if isinstance(request.input, str) else list(request.input)
        )
        if inference_submit is not None:
            _activate_model(request.model, allowed_models=effective_allowed_models)
            encoder = effective_encoders.get(request.model)
            embeddings = await _until_disconnected(
                http_request,
                inference_submit(
                    encoder or (lambda chunk: _encode_texts_now(request.model, chunk)),
                    texts,
                    "search"
                    if query_only or request.purpose == "query"
                    else request.priority,
                    request.timeout_ms,
                    1,
                ),
            )
        else:
            embeddings = await run_in_threadpool(
                embed_texts, request.model, texts, request.timeout_ms
            )

        _validate_embedding_width(request.model, embeddings, expected_dimensions)

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

    @embedding_app.post("/v1/rerank")
    async def create_rerank_scores(
        request: RerankRequest, http_request: Request
    ) -> RerankResponse:
        """Score query/document pairs using this daemon's local cross-encoder."""
        if request.model != reranker_model:
            raise HTTPException(
                status_code=400, detail=f"Unsupported model: {request.model}"
            )
        if not rerank_allowed or not reranker_enabled():
            raise HTTPException(
                status_code=503,
                detail="Reranking is disabled by REFLEXIO_RERANK_ENABLED",
            )
        try:
            if inference_submit is not None:
                scores = await _until_disconnected(
                    http_request,
                    inference_submit(
                        RerankWork(runner, request.query),
                        request.documents,
                        request.priority,
                        request.timeout_ms,
                        4,
                    ),
                )
            else:
                scores = await run_in_threadpool(
                    runner.score_pairs, request.query, request.documents
                )
        except CrossEncoderUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return RerankResponse(
            data=[
                RerankData(index=index, score=score)
                for index, score in enumerate(scores)
            ],
            model=request.model,
        )

    return embedding_app


def _embed_texts(
    model: str,
    texts: list[str],
    *,
    encoder: EmbeddingEncoder | None = None,
    allowed_models: set[str] | None = None,
    timeout_ms: int | None = _DEFAULT_TIMEOUT_MS,
) -> list[list[float]]:
    _activate_model(model, allowed_models=allowed_models)
    if not texts:
        return []

    job = _EmbeddingJob(
        model=model,
        texts=list(texts),
        encoder=encoder,
        deadline=time.monotonic() + timeout_ms / 1000
        if timeout_ms is not None
        else float("inf"),
    )
    global _BATCH_WORKER_RUNNING
    with _MICRO_BATCH_LOCK:
        _MICRO_BATCH_QUEUE.append(job)
        start_worker = not _BATCH_WORKER_RUNNING
        _BATCH_WORKER_RUNNING = True

    if start_worker:
        try:
            threading.Thread(
                target=_run_micro_batch_processor,
                daemon=True,
                name="embedding-micro-batch",
            ).start()
        except BaseException as error:
            _fail_queued_jobs(error)
            raise

    wait_seconds = (
        max(0, job.deadline - time.monotonic()) if timeout_ms is not None else None
    )
    if not job.done.wait(timeout=wait_seconds):
        with _MICRO_BATCH_LOCK:
            if not job.done.is_set():
                job.error = HTTPException(504, "Embedding deadline exceeded")
                job.done.set()
                if job in _MICRO_BATCH_QUEUE:
                    _MICRO_BATCH_QUEUE.remove(job)
    if job.error is not None:
        raise job.error
    return job.result


def _fail_queued_jobs(error: BaseException, *, release_worker: bool = True) -> None:
    global _BATCH_WORKER_RUNNING
    with _MICRO_BATCH_LOCK:
        for job in _MICRO_BATCH_QUEUE:
            job.error = error
            job.done.set()
        _MICRO_BATCH_QUEUE.clear()
        if release_worker:
            _BATCH_WORKER_RUNNING = False


def _run_micro_batch_processor() -> None:
    global _BATCH_WORKER_RUNNING
    jobs: list[_EmbeddingJob] = []
    try:
        while True:
            with _MICRO_BATCH_LOCK:
                if not _MICRO_BATCH_QUEUE:
                    _BATCH_WORKER_RUNNING = False
                    return
            jobs = _take_micro_batch()
            _process_micro_batch(jobs)
            report_background_success("embedding-queue-worker")
    except BaseException as error:
        for job in jobs:
            if not job.done.is_set():
                job.error = error
                job.done.set()
        # Release existing borrowers before logging, but keep ownership until
        # failure reporting finishes so a successor cannot reset the streak first.
        _fail_queued_jobs(error, release_worker=False)
        try:
            report_background_failure(
                logger,
                "embedding_queue_worker_failed",
                error,
                scope="embedding-queue-worker",
            )
        finally:
            # Also fail arrivals during reporting, even if the reporter raises.
            _fail_queued_jobs(error)


def _take_micro_batch() -> list[_EmbeddingJob]:
    with _MICRO_BATCH_LOCK:
        # Expired submitters may be waiting to acquire this lock. They must not
        # dispatch another physical model turn while their deadline is over.
        now = time.monotonic()
        for job in list(_MICRO_BATCH_QUEUE):
            if job.done.is_set() or now >= job.deadline:
                _MICRO_BATCH_QUEUE.remove(job)
                if not job.done.is_set():
                    job.error = HTTPException(504, "Embedding deadline exceeded")
                    job.done.set()
        if not _MICRO_BATCH_QUEUE:
            return []
        first = _MICRO_BATCH_QUEUE[0]
        if first.capacity is None:
            # A custom callback without native capacity is an opaque atomic
            # invocation. Preserve its request contract; do not invent a cap or
            # combine independent calls into an undocumented oversized batch.
            _MICRO_BATCH_QUEUE.pop(0)
            first.turn = list(first.pending)
            first.pending.clear()
            return [first]
        anchor_length = len(first.texts[first.pending[0]])
        candidates = [
            job
            for job in _MICRO_BATCH_QUEUE
            if job.model == first.model and job.encoder is first.encoder
        ]
        jobs: list[_EmbeddingJob] = []
        selected = first
        for _ in range(first.capacity):
            if not selected.turn:
                _MICRO_BATCH_QUEUE.remove(selected)
                jobs.append(selected)
            selected.turn.append(selected.pending.popleft())
            if not selected.pending:
                candidates.remove(selected)
            if not candidates:
                break
            # The oldest request always anchors a turn, so nearest-length
            # packing cannot starve a waiting request. No delay or target fill.
            selected = min(
                candidates,
                key=lambda job: abs(len(job.texts[job.pending[0]]) - anchor_length),
            )
        return jobs


def _process_micro_batch(jobs: list[_EmbeddingJob]) -> None:
    with _MICRO_BATCH_LOCK:
        for job in jobs:
            if not job.done.is_set() and time.monotonic() >= job.deadline:
                job.error = HTTPException(504, "Embedding deadline exceeded")
                job.done.set()
        jobs = [job for job in jobs if not job.done.is_set()]
    if not jobs:
        return
    texts = [text for job in jobs for index in job.turn for text in [job.texts[index]]]
    try:
        encoder = jobs[0].encoder
        embeddings = (
            encoder(texts)
            if encoder is not None
            else _encode_texts_now(jobs[0].model, texts)
        )
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: encoded {len(texts)} texts but got "
                f"{len(embeddings)} embeddings"
            )
    except BaseException as exc:
        with _MICRO_BATCH_LOCK:
            for job in jobs:
                if not job.done.is_set():
                    job.error = exc
                    job.done.set()
        return

    with _MICRO_BATCH_LOCK:
        start = 0
        for job in jobs:
            end = start + len(job.turn)
            if not job.done.is_set():
                if time.monotonic() >= job.deadline:
                    job.error = HTTPException(504, "Embedding deadline exceeded")
                    job.done.set()
                else:
                    job.rows.update(zip(job.turn, embeddings[start:end], strict=True))
                    if not job.pending:
                        job.result = [
                            job.rows[index] for index in range(len(job.texts))
                        ]
                        job.done.set()
                    else:
                        # Arrivals already waiting get a turn before this large
                        # request continues. Never return its partial result.
                        job.turn.clear()
                        _MICRO_BATCH_QUEUE.append(job)
            start = end


def _encode_texts_now(model: str, texts: list[str]) -> list[list[float]]:
    # The embedder singletons serialize their own model.encode() internally
    # (the shared sentence-transformers / ONNX models are not thread-safe — see
    # NomicEmbedder / LocalEmbedder). Concurrency is already bounded by the
    # queue owner, so no extra encode lock/semaphore is needed here.
    if is_nomic_model(model):
        return NomicEmbedder.get().embed(texts)
    if model == MINILM_MODEL:
        return LocalEmbedder.get().embed(texts)
    raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")


def _activate_model(model: str, *, allowed_models: set[str] | None = None) -> None:
    effective_allowed_models = (
        allowed_models if allowed_models is not None else _SUPPORTED_MODELS
    )
    if model not in effective_allowed_models:
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
