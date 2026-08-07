"""Service-side runners for the supported cross-encoder rerankers."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Literal

from reflexio.server.llm.rerank.common import (
    RERANK_ENABLED_ENV_VAR,
    CrossEncoderUnavailableError,
    reranker_enabled,
    reranker_revision,
)

_LOGGER = logging.getLogger(__name__)
_DEVICE_ENV_VAR = "REFLEXIO_RERANK_DEVICE"


def _import_cross_encoder() -> Any:
    """Import sentence-transformers without leaving a partial module behind."""
    import sys

    for _attempt in range(2):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            sys.modules.pop("sentence_transformers", None)
            continue
        return CrossEncoder
    raise CrossEncoderUnavailableError(
        "sentence-transformers is not installed; cannot use the cross-encoder reranker"
    )


class CrossEncoderRunner:
    """One serialized cross-encoder instance owned by the inference service."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.revision = reranker_revision(model_name)
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._ready = False
        self._last_error: str | None = None

    def status(self) -> Literal["ready", "disabled", "unavailable"]:
        if not reranker_enabled():
            return "disabled"
        return "ready" if self._ready else "unavailable"

    def ready(self) -> bool:
        return self.status() == "ready"

    def last_error(self) -> str | None:
        return self._last_error

    def _set_readiness(self, *, ready: bool, error: str | None = None) -> None:
        self._ready = ready
        self._last_error = error

    def _get_model(self) -> Any:
        if not reranker_enabled():
            raise CrossEncoderUnavailableError(
                f"Reranker is disabled by {RERANK_ENABLED_ENV_VAR}"
            )
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            cross_encoder_cls = _import_cross_encoder()
            device = os.environ.get(_DEVICE_ENV_VAR, "cpu")
            try:
                _LOGGER.info(
                    "Loading reranker model %s revision=%s device=%s",
                    self.model_name,
                    self.revision,
                    device,
                )
                self._model = cross_encoder_cls(
                    self.model_name,
                    device=device,
                    revision=self.revision,
                )
            except Exception as exc:  # noqa: BLE001
                message = (
                    f"Failed to load cross-encoder model {self.model_name!r}: {exc}"
                )
                self._set_readiness(ready=False, error=message)
                raise CrossEncoderUnavailableError(message) from exc
            _LOGGER.info("Reranker model ready (model=%s)", self.model_name)
            return self._model

    def score_pairs(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        model = self._get_model()
        try:
            from torch import nn
        except ImportError as exc:
            message = "torch is not installed; cannot use the cross-encoder reranker"
            self._set_readiness(ready=False, error=message)
            raise CrossEncoderUnavailableError(message) from exc

        try:
            with self._predict_lock:
                raw_scores = model.predict(
                    [(query, doc) for doc in docs],
                    show_progress_bar=False,
                    activation_fn=nn.Identity(),
                )
            scores = [float(score) for score in raw_scores]
        except Exception as exc:  # noqa: BLE001
            message = (
                f"Failed to score with cross-encoder model {self.model_name!r}: {exc}"
            )
            self._set_readiness(ready=False, error=message)
            raise CrossEncoderUnavailableError(message) from exc
        self._set_readiness(ready=True)
        return scores

    def prewarm(self) -> bool:
        if not reranker_enabled():
            self._set_readiness(ready=False)
            _LOGGER.info(
                "Reranker prewarm skipped because %s=false", RERANK_ENABLED_ENV_VAR
            )
            return False
        try:
            self.score_pairs("warmup", ["warmup"])
        except Exception:  # noqa: BLE001
            _LOGGER.warning(
                "Cross-encoder unavailable during service prewarm; reranking remains unavailable",
                exc_info=True,
            )
            return False
        _LOGGER.info("Cross-encoder pre-warmed in shared inference service")
        return True


__all__ = ["CrossEncoderRunner"]
