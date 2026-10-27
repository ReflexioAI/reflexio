"""Local in-process embedder using ``nomic-ai/nomic-embed-text-v1.5``.

A higher-quality alternative to the ONNX MiniLM-L6-v2: 137M
parameters, 768-dim native, supports Matryoshka representation (64–768
dimensions without retraining), 8192-token context, Apache-2.0 licensed.
Performs comparably to OpenAI's ``text-embedding-3-small`` on MTEB
retrieval at a fraction of the latency cost when run locally on CPU or
Apple Silicon.

Lifecycle
---------

The inference service calls ``NomicEmbedder.get().embed(...)`` directly and
owns startup warmup and readiness. API workers route local Nomic requests over
HTTP; they do not register or prewarm an in-process fallback. Requires the
``sentence-transformers`` dependency in the inference service.

Storage compatibility
---------------------

Reflexio's vec0 tables expect 512-dim vectors (``EMBEDDING_DIMENSIONS``).
Nomic's native 768 dim is reduced via Matryoshka — slice the first 512
floats, then re-normalize to unit length so cosine similarity remains
comparable. Quality on retrieval tasks at 512 dim is ~95% of the full
768 (per Nomic's own evaluation).
"""

from __future__ import annotations

import math
from typing import Any

from reflexio.server.llm.sentence_encoder import SentenceEncoder

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


class NomicEmbedderError(RuntimeError):
    """Raised when the Nomic embedder is requested but its deps are missing."""


class NomicEmbedder(SentenceEncoder):
    """Nomic identity and Matryoshka adapter for the shared sentence encoder."""

    model_name = _HF_MODEL_NAME
    max_chars = _MAX_CHARS
    error = NomicEmbedderError

    def _transform(self, vector: list[float]) -> list[float]:
        return _truncate_and_renormalise(vector)


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
    "is_nomic_model",
]
