"""Local cross-encoder reranker for ``(query, document)`` pairs.

Wraps ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~25M params) from
``sentence-transformers``. The model is lazy-loaded on first call and
held as a process-wide singleton — load takes ~3 s but only happens
once per server start. Scoring K=30 pairs takes ~50 ms on CPU.

Usage
-----

>>> from reflexio.server.llm.rerank import score_pairs
>>> scores = score_pairs("italian food", ["pasta lover", "weather report"])
>>> scores[0] > scores[1]
True

The helper is intentionally side-effect free at import time: building
the singleton happens only when ``score_pairs`` is called, so importing
this module never triggers a model download.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

_LOGGER = logging.getLogger(__name__)

# HuggingFace identifier for the cross-encoder. Chosen for the
# size/quality trade-off: 22M parameters, ~50 ms for K=30 on CPU,
# well-known MS-MARCO benchmark performance.
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Device override for the cross-encoder. Defaults to CPU: without an
# explicit device sentence-transformers auto-selects MPS on Apple
# Silicon, where torch's caching allocator accumulates gigabytes of
# GPU buffers it never returns to the OS — and the 22M-param model
# gains nothing from GPU at K=30 pairs. Mirrors NOMIC_EMBED_DEVICE.
_DEVICE_ENV_VAR = "REFLEXIO_RERANK_DEVICE"

# Singleton state — never accessed directly outside ``_get_model``.
_MODEL: Any | None = None
_MODEL_LOCK = threading.Lock()


class CrossEncoderUnavailableError(RuntimeError):
    """Raised when the cross-encoder model cannot be loaded.

    The most common cause is ``sentence-transformers`` being absent from
    the runtime environment. Callers should treat this as a soft failure
    (log + skip rerank) rather than a 500.
    """


def _import_cross_encoder() -> Any:
    """Robustly import ``sentence_transformers.CrossEncoder``.

    The ``sentence_transformers`` package is loaded both by the Nomic
    local-embedding pre-warm thread (kicked off by ``LiteLLMClient.__init__``
    when ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1``) and by this reranker. When
    those concurrent imports race, one thread can see a half-loaded
    ``sentence_transformers`` module in ``sys.modules`` whose ``CrossEncoder``
    attribute was never bound — Python's import machinery hands back the
    partial module without re-running ``__init__``. This helper detects that
    case, drops the stale entry, and re-imports cleanly.

    Returns:
        Any: The ``CrossEncoder`` class.

    Raises:
        CrossEncoderUnavailableError: When the package genuinely isn't
            installed, or every retry yields a partial module.
    """
    import sys

    for _attempt in range(2):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            # Partial import — drop the stale entry and try once more.
            sys.modules.pop("sentence_transformers", None)
            continue
        return CrossEncoder
    try:
        from sentence_transformers import (
            CrossEncoder,  # noqa: F401 — final attempt for the error path
        )
    except ImportError as e:
        raise CrossEncoderUnavailableError(
            "sentence-transformers is not installed; cannot use the "
            "cross-encoder reranker"
        ) from e
    return CrossEncoder


def _get_model() -> Any:
    """Return the lazy-loaded cross-encoder singleton.

    The first caller pays the load cost (~3 s, weights cached under
    ``~/.cache/huggingface/`` after first download). Subsequent callers
    get the warm instance immediately.

    Returns:
        Any: A ``sentence_transformers.CrossEncoder`` instance.

    Raises:
        CrossEncoderUnavailableError: If ``sentence-transformers`` is not
            importable, or if the underlying model fails to load.
    """
    global _MODEL  # noqa: PLW0603 — singleton-pattern intentional
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        cross_encoder_cls = _import_cross_encoder()
        device = os.environ.get(_DEVICE_ENV_VAR, "cpu")
        try:
            _LOGGER.info("Loading reranker model %s (device=%s)", _MODEL_NAME, device)
            _MODEL = cross_encoder_cls(_MODEL_NAME, device=device)
        except Exception as e:  # noqa: BLE001 — surface as a typed failure
            raise CrossEncoderUnavailableError(
                f"Failed to load cross-encoder model {_MODEL_NAME!r}: {e}"
            ) from e
        _LOGGER.info("Reranker model ready (model=%s)", _MODEL_NAME)
        return _MODEL


def score_pairs(query: str, docs: list[str]) -> list[float]:
    """Score ``(query, doc)`` pairs with the cross-encoder.

    Higher score means more relevant. Scores are not bounded to a fixed
    range — they are raw model logits — so callers should treat them as
    opaque relative-ranking signal, not as probabilities.

    Args:
        query (str): The reranking query.
        docs (list[str]): Documents to score against ``query``.

    Returns:
        list[float]: One score per document, in the same order as
            ``docs``. Empty list when ``docs`` is empty.

    Raises:
        CrossEncoderUnavailableError: If the cross-encoder cannot be
            loaded (re-raised from :func:`_get_model`).
    """
    if not docs:
        return []
    model = _get_model()
    pairs = [(query, doc) for doc in docs]
    raw_scores = model.predict(pairs, show_progress_bar=False)
    # ``predict`` returns a numpy array; convert to plain Python floats so
    # the caller can serialise the result without numpy as a dependency.
    return [float(s) for s in raw_scores]


def prewarm() -> bool:
    """Force the cross-encoder model to load at startup.

    Call once during app startup so the ~3s model load never lands on a user
    query (and never serializes a concurrent burst behind the model lock).
    Never raises — startup must not crash if the cross-encoder is unavailable
    or fails to score the smoke input.

    Returns:
        True if the model loaded and scored a smoke input; False otherwise
        (callers should treat the floor as degraded, not crash).
    """
    try:
        score_pairs("warmup", ["warmup"])
    except CrossEncoderUnavailableError:
        _LOGGER.warning(
            "Cross-encoder unavailable at startup; relevance floor will degrade "
            "to unfiltered results until the model is available."
        )
        return False
    except Exception:  # noqa: BLE001 — pre-warm must never crash startup
        _LOGGER.warning(
            "Cross-encoder pre-warm failed unexpectedly; relevance floor will "
            "degrade to unfiltered results until the model is available.",
            exc_info=True,
        )
        return False
    _LOGGER.info("Cross-encoder pre-warmed.")
    return True
