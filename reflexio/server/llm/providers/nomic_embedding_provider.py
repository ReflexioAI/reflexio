"""Local in-process embedder using ``nomic-ai/nomic-embed-text-v1.5``.

A higher-quality alternative to the chromadb-bundled MiniLM-L6-v2: 137M
parameters, 768-dim native, supports Matryoshka representation (64–768
dimensions without retraining), 8192-token context, Apache-2.0 licensed.
Performs comparably to OpenAI's ``text-embedding-3-small`` on MTEB
retrieval at a fraction of the latency cost when run locally on CPU or
Apple Silicon.

Activation
----------

- Set ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1`` in the process environment.
- Pass model name ``local/nomic-embed-v1.5`` (or ``local/nomic-embed-text-v1.5``)
  to :func:`LiteLLMClient.get_embedding`/``get_embeddings``.
- Requires the ``sentence-transformers`` pip dependency.

Storage compatibility
---------------------

Reflexio's vec0 tables expect 512-dim vectors (``EMBEDDING_DIMENSIONS``).
Nomic's native 768 dim is reduced via Matryoshka — slice the first 512
floats, then re-normalize to unit length so cosine similarity remains
comparable. Quality on retrieval tasks at 512 dim is ~95% of the full
768 (per Nomic's own evaluation).
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import threading
from typing import Any

from reflexio.server.llm.llm_utils import positive_int_env

_LOGGER = logging.getLogger(__name__)

_ENV_ENABLE = "CLAUDE_SMART_USE_LOCAL_EMBEDDING"
_ENV_PROVIDER = "REFLEXIO_EMBEDDING_PROVIDER"
_ENV_DAEMON = "REFLEXIO_EMBEDDING_DAEMON"
_MODEL_KEYS = {"local/nomic-embed-v1.5", "local/nomic-embed-text-v1.5"}
_HF_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Reflexio's vec0 schema dim. Nomic v1.5 outputs 768 natively; we slice
# to 512 (Matryoshka) and re-normalize.
_TARGET_DIM = 512
# Nomic v1.5 was trained with task-prefixed inputs; "search_document"
# vs "search_query" prefixes give better asymmetric retrieval. Reflexio's
# storage layer already passes a "search_document: " / "search_query: "
# prefix when calling _get_embedding(purpose=...), so we don't add another
# prefix here — the input arrives correctly tagged.
# The model has a 8192 token context window; we still cap chars
# defensively to avoid pathological multi-MB inputs.
_MAX_CHARS = 32_000

# Encode in small mini-batches so a single large request can't spike memory:
# peak activation memory scales with batch_size, not with the total number of
# texts in the request. A small batch is what makes the daemon's bounded
# concurrency (see ``embedding_service``) safe to raise above 1.
_DEFAULT_ENCODE_BATCH_SIZE = 4
_ENV_BATCH_SIZE = "REFLEXIO_EMBED_BATCH_SIZE"


def _encode_batch_size() -> int:
    """Resolve the encode mini-batch size from env, defaulting to 4."""
    return positive_int_env(_ENV_BATCH_SIZE, _DEFAULT_ENCODE_BATCH_SIZE, _LOGGER)


class NomicEmbedderError(RuntimeError):
    """Raised when the Nomic embedder is requested but its deps are missing."""


def _huggingface_cache_path() -> str:
    return os.environ.get("HF_HOME") or "~/.cache/huggingface"


class NomicEmbedder:
    """Lazily-loaded singleton wrapping a sentence-transformers model.

    Loading the underlying ``nomic-embed-text-v1.5`` model takes ~5–10 s on
    first call (downloads ~550 MB on cold start, then cached under
    ``~/.cache/huggingface/``). After that, embedding latency on CPU is
    ~30–60 ms per single text and ~200 ms per batch of 32 (Apple M-series).
    """

    _instance: NomicEmbedder | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    @classmethod
    def get(cls) -> NomicEmbedder:
        """Return the process-wide singleton, constructing it on first use."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> Any:
        """Lazy-import sentence-transformers and load the Nomic model."""
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import (
                    SentenceTransformer,  # type: ignore[import-not-found]
                )
            except ImportError as exc:
                raise NomicEmbedderError(
                    "sentence-transformers is required for the Nomic local "
                    "embedder. Install with `uv add sentence-transformers`."
                ) from exc
            _LOGGER.info(
                "Loading Nomic embedding model %s — first call may download "
                "~550 MB to %s",
                _HF_MODEL_NAME,
                _huggingface_cache_path(),
            )
            # Force CPU device — MPS init has been observed to hang on some
            # Apple Silicon + macOS combos for several minutes during model
            # load. CPU is fast enough for our use case (137M params) and
            # behaves predictably. Set NOMIC_EMBED_DEVICE=mps|cuda|cpu to
            # override.
            device = os.environ.get("NOMIC_EMBED_DEVICE", "cpu")
            self._model = SentenceTransformer(
                _HF_MODEL_NAME,
                trust_remote_code=True,  # Nomic v1.5 ships custom code
                device=device,
            )
            _LOGGER.info(
                "Nomic embedder ready (model=%s, target_dim=%d, native_dim=%d)",
                _HF_MODEL_NAME,
                _TARGET_DIM,
                _embedding_dimension(self._model),
            )
            return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning ``_TARGET_DIM``-sized unit vectors.

        Args:
            texts: Inputs to encode. Each is char-truncated to ``_MAX_CHARS``
                as a defensive cap; Nomic itself supports 8192 tokens.

        Returns:
            list[list[float]]: One vector per input, each exactly
                ``_TARGET_DIM`` (512) floats and L2-normalised so cosine
                similarity equals dot product.
        """
        model = self._load()
        safe = [(t or "")[:_MAX_CHARS] for t in texts]
        # show_progress_bar=False so server logs stay clean during ingest
        # batches. convert_to_numpy=True returns a numpy ndarray; we slice
        # and renormalise per-row before converting to plain Python lists.
        raw = model.encode(
            safe,
            batch_size=_encode_batch_size(),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [_truncate_and_renormalise(vec.tolist()) for vec in raw]


def _truncate_and_renormalise(vec: list[float]) -> list[float]:
    """Slice to ``_TARGET_DIM`` and L2-renormalise for valid Matryoshka use.

    Args:
        vec (list[float]): Native-dim Nomic embedding (typically 768 floats,
            already L2-unit on the full 768).

    Returns:
        list[float]: Exactly ``_TARGET_DIM`` floats, L2-normalised in the
            truncated subspace so cosine similarity remains a valid metric.
            Zero-padded if the input is shorter than ``_TARGET_DIM``.
    """
    if len(vec) >= _TARGET_DIM:
        sliced = vec[:_TARGET_DIM]
    else:
        sliced = vec + [0.0] * (_TARGET_DIM - len(vec))
    norm = math.sqrt(sum(x * x for x in sliced))
    if norm <= 0:
        return sliced
    return [x / norm for x in sliced]


def _embedding_dimension(model: Any) -> int:
    get_embedding_dimension = getattr(model, "get_embedding_dimension", None)
    if callable(get_embedding_dimension):
        dimension = get_embedding_dimension()
    else:
        dimension = model.get_sentence_embedding_dimension()
    return int(dimension)  # type: ignore[arg-type]


_REGISTERED = False


def register_if_enabled() -> bool:
    """Make the Nomic embedder available when env + deps allow it.

    Idempotent. Returns ``True`` when the embedder is usable after this
    call. Routing happens via prefix-match on the model name in
    ``LiteLLMClient.get_embedding(s)``.

    Eagerly pre-warms the model in a daemon thread so the first request
    doesn't pay the ~30 s cold-start cost. The thread is fire-and-forget;
    callers either land mid-load (and block briefly) or after-load (and
    proceed immediately).
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    if os.environ.get(_ENV_ENABLE) != "1":
        return False
    provider = os.environ.get(_ENV_PROVIDER, "").strip().lower()
    if provider in {"local_service", "internal_service", "off"}:
        _LOGGER.info(
            "Nomic in-process prewarm skipped because %s=%s",
            _ENV_PROVIDER,
            provider,
        )
        return False
    if not provider and os.environ.get(_ENV_DAEMON) != "1":
        _LOGGER.info(
            "Nomic in-process prewarm skipped; %s=1 now defaults to the "
            "shared embedding service.",
            _ENV_ENABLE,
        )
        return False
    if importlib.util.find_spec("sentence_transformers") is None:
        _LOGGER.warning(
            "%s=1 set but `sentence-transformers` not installed; the Nomic "
            "local embedder will not be available.",
            _ENV_ENABLE,
        )
        return False
    _REGISTERED = True
    _LOGGER.info(
        "Nomic local embedding provider enabled (models=%s)", sorted(_MODEL_KEYS)
    )

    def _prewarm() -> None:
        """Background load + dummy inference so the first real request is fast."""
        try:
            embedder = NomicEmbedder.get()
            embedder.embed(["warmup"])
            _LOGGER.info("Nomic embedder pre-warmed")
        except Exception:  # noqa: BLE001
            _LOGGER.exception(
                "Nomic embedder pre-warm failed; first call will pay the cost"
            )

    threading.Thread(target=_prewarm, daemon=True, name="nomic-prewarm").start()
    return True


def is_enabled() -> bool:
    """Return True after a successful :func:`register_if_enabled`."""
    return _REGISTERED


def is_nomic_model(model: str) -> bool:
    """Predicate used by ``LiteLLMClient`` to route by model name.

    Args:
        model (str): The embedding model name passed by the caller.

    Returns:
        bool: True when the model resolves to the Nomic provider.
    """
    return model in _MODEL_KEYS


__all__ = [
    "NomicEmbedder",
    "NomicEmbedderError",
    "is_enabled",
    "is_nomic_model",
    "register_if_enabled",
]
