"""HTTP embedding service provider.

This module is the routing boundary between Reflexio's LLM client and an
OpenAI-compatible embedding service. It lets local Claude Smart installs keep
one model process alive across backend workers, Claude Code, and Codex, while
also supporting a horizontally-scaled internal service in cloud deployments.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import httpx

_LOGGER = logging.getLogger(__name__)

EmbeddingProviderMode = Literal[
    "cloud", "local_service", "internal_service", "inprocess", "off"
]

_ENV_PROVIDER = "REFLEXIO_EMBEDDING_PROVIDER"
_ENV_SERVICE_URL = "REFLEXIO_EMBEDDING_SERVICE_URL"
_ENV_TIMEOUT_MS = "REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS"
_ENV_EMBEDDING_PORT = "EMBEDDING_PORT"
_ENV_CLAUDE_SMART_LOCAL = "CLAUDE_SMART_USE_LOCAL_EMBEDDING"
_DEFAULT_LOCAL_PORT = 8072
_DEFAULT_TIMEOUT_MS = 2_000
_SERVICE_MODES = {"local_service", "internal_service"}
_VALID_MODES = {"cloud", *_SERVICE_MODES, "inprocess", "off"}


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the configured embedding provider is unavailable."""


def _truthy(value: str | None) -> bool:
    return value in {"1", "true", "True", "yes", "YES", "on", "ON"}


def _local_service_url() -> str:
    port = os.environ.get(_ENV_EMBEDDING_PORT, str(_DEFAULT_LOCAL_PORT))
    return f"http://127.0.0.1:{port}"


def embedding_service_url(mode: EmbeddingProviderMode | None = None) -> str:
    """Return the configured embedding service URL.

    ``local_service`` defaults to ``127.0.0.1:$EMBEDDING_PORT`` so local
    Claude Smart hosts can share a single machine daemon. ``internal_service``
    requires an explicit URL because it is deployment-specific.
    """
    resolved = mode or embedding_provider_mode()
    configured = os.environ.get(_ENV_SERVICE_URL)
    if configured:
        return configured.rstrip("/")
    if resolved == "local_service":
        return _local_service_url()
    raise EmbeddingUnavailableError(
        f"{_ENV_SERVICE_URL} is required when {_ENV_PROVIDER}={resolved}"
    )


def embedding_service_timeout_seconds() -> float:
    raw = os.environ.get(_ENV_TIMEOUT_MS)
    if raw is None:
        return _DEFAULT_TIMEOUT_MS / 1000
    try:
        timeout_ms = int(raw)
    except ValueError as exc:
        raise EmbeddingUnavailableError(
            f"{_ENV_TIMEOUT_MS} must be an integer number of milliseconds"
        ) from exc
    return max(timeout_ms, 1) / 1000


def embedding_provider_mode(model: str | None = None) -> EmbeddingProviderMode:
    """Resolve the embedding provider mode for a model.

    Backwards compatibility: a ``local/*`` model still uses the in-process
    embedder unless the user explicitly chooses a mode or Claude Smart's legacy
    ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1`` flag is present. That legacy flag
    now means "use the shared local service" by default.
    """
    configured = os.environ.get(_ENV_PROVIDER)
    if configured:
        mode = configured.strip().lower()
        if mode not in _VALID_MODES:
            raise EmbeddingUnavailableError(
                f"Invalid {_ENV_PROVIDER}={configured!r}; expected one of "
                f"{', '.join(sorted(_VALID_MODES))}"
            )
        return mode  # type: ignore[return-value]

    if _truthy(os.environ.get(_ENV_CLAUDE_SMART_LOCAL)):
        return "local_service"

    if os.environ.get(_ENV_SERVICE_URL):
        return "internal_service"

    if model and model.startswith("local/"):
        return "inprocess"
    return "cloud"


def should_use_embedding_service(model: str) -> bool:
    """Return True when embedding requests should call the HTTP service."""
    return embedding_provider_mode(model) in _SERVICE_MODES


def get_service_embeddings(
    texts: list[str],
    *,
    model: str,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Call the OpenAI-compatible embedding service.

    Args:
        texts: Inputs to embed.
        model: Embedding model name.
        dimensions: Optional embedding dimensions.

    Returns:
        Embeddings in input order.

    Raises:
        EmbeddingUnavailableError: If the service cannot be reached or returns
            an invalid response.
    """
    if not texts:
        return []

    mode = embedding_provider_mode(model)
    if mode == "off":
        raise EmbeddingUnavailableError("Embedding provider is disabled")
    if mode not in _SERVICE_MODES:
        raise EmbeddingUnavailableError(
            f"Embedding service requested while {_ENV_PROVIDER}={mode}"
        )

    url = f"{embedding_service_url(mode)}/v1/embeddings"
    payload: dict[str, Any] = {"model": model, "input": texts}
    if dimensions:
        payload["dimensions"] = dimensions

    timeout = embedding_service_timeout_seconds()
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            data = body.get("data")
            if not isinstance(data, list):
                raise ValueError("embedding service response is missing data[]")
            sorted_data = sorted(data, key=lambda item: int(item.get("index", 0)))
            embeddings = [item.get("embedding") for item in sorted_data]
            if not all(isinstance(item, list) for item in embeddings):
                raise ValueError("embedding service response has invalid embeddings")
            return [[float(value) for value in item] for item in embeddings]
        except Exception as exc:  # noqa: BLE001 - normalized to typed degradation signal
            last_error = exc
            if attempt == 0:
                time.sleep(0.1)

    _LOGGER.warning("Embedding service unavailable at %s: %s", url, last_error)
    raise EmbeddingUnavailableError(
        f"Embedding service unavailable at {url}: {last_error}"
    ) from last_error
