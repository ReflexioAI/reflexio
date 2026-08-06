"""GPU-only provider for ``intfloat/multilingual-e5-small``.

The model produces 384-dimensional unit vectors. Reflexio storage uses a fixed
512-dimensional vector contract, so the provider appends zeros. Zero-padding a
unit vector preserves its norm and every cosine similarity.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

from reflexio.server.llm.llm_utils import positive_int_env

_LOGGER = logging.getLogger(__name__)

MULTILINGUAL_E5_MODEL = "local/multilingual-e5-small"
MULTILINGUAL_E5_HF_MODEL = "intfloat/multilingual-e5-small"
_NATIVE_DIM = 384
_TARGET_DIM = 512
_MAX_CHARS = 8_000
_DEFAULT_ENCODE_BATCH_SIZE = 16
_ENV_BATCH_SIZE = "REFLEXIO_EMBED_BATCH_SIZE"


class MultilingualE5EmbedderError(RuntimeError):
    """Raised when E5 cannot run on the required CUDA runtime."""


def _encode_batch_size() -> int:
    return positive_int_env(_ENV_BATCH_SIZE, _DEFAULT_ENCODE_BATCH_SIZE, _LOGGER)


def _pad_unit_embedding(vector: list[float]) -> list[float]:
    """Validate and zero-pad one native E5 vector to Reflexio's storage width."""
    if len(vector) != _NATIVE_DIM:
        raise MultilingualE5EmbedderError(
            f"Expected {_NATIVE_DIM}-dimensional E5 output, got {len(vector)}"
        )
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        raise MultilingualE5EmbedderError("E5 returned a zero-norm embedding")
    normalized = [value / norm for value in vector]
    return [*normalized, *([0.0] * (_TARGET_DIM - _NATIVE_DIM))]


class MultilingualE5Embedder:
    """Lazily loaded, process-wide CUDA E5 embedder."""

    _instance: MultilingualE5Embedder | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    @classmethod
    def get(cls) -> MultilingualE5Embedder:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise MultilingualE5EmbedderError(
                    "torch and sentence-transformers are required for multilingual E5"
                ) from exc
            if not torch.cuda.is_available():
                raise MultilingualE5EmbedderError(
                    "multilingual-e5-small is supported only by the dedicated "
                    "CUDA embedding service"
                )
            _LOGGER.info(
                "Loading multilingual E5 model %s on CUDA", MULTILINGUAL_E5_HF_MODEL
            )
            self._model = SentenceTransformer(
                MULTILINGUAL_E5_HF_MODEL,
                device="cuda",
            )
            return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        safe = [(text or "")[:_MAX_CHARS] for text in texts]
        with self._encode_lock:
            encoded = model.encode(
                safe,
                batch_size=_encode_batch_size(),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [_pad_unit_embedding([float(value) for value in row]) for row in encoded]


def is_multilingual_e5_model(model: str) -> bool:
    return model.strip().casefold() == MULTILINGUAL_E5_MODEL
