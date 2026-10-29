"""One native sentence-model lifecycle and bounded, serialized execution."""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Self, cast

from reflexio.native_compute import configure_native_compute
from reflexio.removed_env import reject_removed_env
from reflexio.server.env_utils import env_str, positive_int_env

_LOGGER = logging.getLogger(__name__)
_DEVICE = "REFLEXIO_EMBEDDING_DEVICE"


def embedding_device() -> str:
    """Resolve explicit process placement; the model never chooses hardware."""
    reject_removed_env(os.environ, canonical=_DEVICE)
    return env_str(_DEVICE, "cpu")


def embedding_batch_size(default: int) -> int:
    """Resolve the sole native embedding capacity; never revive a queue cap."""
    reject_removed_env(os.environ, canonical="REFLEXIO_EMBED_BATCH_SIZE")
    return positive_int_env("REFLEXIO_EMBED_BATCH_SIZE", default, _LOGGER)


class SentenceEncoder(ABC):
    """Shared lazy model owner; adapters retain model identity and vector space.

    Subclasses declare model facts, validate supported devices and transform one
    native row. Execution capacity and hardware placement belong to the process.
    No Torch or model import happens until the first actual load.
    """

    model_name: str
    revision: str | None = None
    max_chars: int
    default_batch_size = 4
    error: type[RuntimeError] = RuntimeError
    _instance: SentenceEncoder | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    @classmethod
    def get(cls) -> Self:
        """Own one model per adapter class, constructed without loading weights."""
        with cls._instance_lock:
            if cls.__dict__.get("_instance") is None:
                cls._instance = cls()
            return cast(Self, cls._instance)

    def _validate_device(self, device: str) -> None:
        """The native loader validates general device strings, including indices."""
        import torch

        torch.device(device)

    def _load(self) -> Any:
        device = embedding_device()
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                try:
                    configure_native_compute(torch=True)
                    self._validate_device(device)
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise self.error(
                        "torch and sentence-transformers are required for native embeddings"
                    ) from exc
                options: dict[str, Any] = {"device": device, "trust_remote_code": False}
                if self.revision is not None:
                    options["revision"] = self.revision
                _LOGGER.info("Loading sentence model %s on %s", self.model_name, device)
                self._model = SentenceTransformer(self.model_name, **options)
        return self._model

    @property
    def batch_size(self) -> int:
        """The native batch capacity, also used by daemon scheduling."""
        return embedding_batch_size(self.default_batch_size)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Serialize the shared model with the process's actual mini-batch budget."""
        model = self._load()
        safe = [(text or "")[: self.max_chars] for text in texts]
        batch_size = self.batch_size
        with self._model_lock:
            raw = model.encode(
                safe,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [self._transform([float(value) for value in row]) for row in raw]

    @abstractmethod
    def _transform(self, vector: list[float]) -> list[float]:
        raise NotImplementedError
