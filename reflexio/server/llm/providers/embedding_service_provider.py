"""HTTP embedding service provider.

This module is the routing boundary between Reflexio's LLM client and an
OpenAI-compatible embedding service. It lets local Claude Smart installs keep
one model process alive across backend workers, Claude Code, and Codex, while
also supporting a horizontally-scaled internal service in cloud deployments.
"""

from __future__ import annotations

import json
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
_ENV_DAEMON_HOST = "REFLEXIO_EMBEDDING_DAEMON_HOST"
_ENV_LOCAL_SERVICE_PROBE_TIMEOUT_MS = (
    "REFLEXIO_EMBEDDING_LOCAL_SERVICE_PROBE_TIMEOUT_MS"
)
_ENV_CLAUDE_SMART_LOCAL = "CLAUDE_SMART_USE_LOCAL_EMBEDDING"
_ENV_MAX_TEXTS_PER_REQUEST = "REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST"
_DEFAULT_LOCAL_PORT = 8072
_DEFAULT_INTERNAL_SERVICE_TIMEOUT_MS = 2_000
_DEFAULT_LOCAL_SERVICE_TIMEOUT_MS = 30_000
_DEFAULT_LOCAL_SERVICE_PROBE_TIMEOUT_MS = 200
# Each request embeds its whole ``input`` list in one ``model.encode()`` on the
# service, so an unbounded request can exceed the client read timeout on a CPU
# daemon. Cap texts per request and concatenate; encode batching is a separate
# server-side knob (REFLEXIO_EMBED_BATCH_SIZE).
_DEFAULT_MAX_TEXTS_PER_REQUEST = 32
_LOCAL_SERVICE_PROBE_CACHE_SECONDS = 5.0
_SERVICE_MODES = {"local_service", "internal_service"}
_VALID_MODES = {"cloud", *_SERVICE_MODES, "inprocess", "off"}
_local_service_probe_cache: tuple[float, bool, str | None] | None = None


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the configured embedding provider is unavailable."""


def _local_service_url() -> str:
    host = os.environ.get(_ENV_DAEMON_HOST, "").strip() or "127.0.0.1"
    port = os.environ.get(_ENV_EMBEDDING_PORT, str(_DEFAULT_LOCAL_PORT))
    return f"http://{host}:{port}"


def _local_service_probe_timeout_seconds() -> float:
    raw = os.environ.get(_ENV_LOCAL_SERVICE_PROBE_TIMEOUT_MS)
    if raw is None:
        return _DEFAULT_LOCAL_SERVICE_PROBE_TIMEOUT_MS / 1000
    try:
        timeout_ms = int(raw)
    except ValueError:
        _LOGGER.warning(
            "%s must be an integer number of milliseconds; using %dms",
            _ENV_LOCAL_SERVICE_PROBE_TIMEOUT_MS,
            _DEFAULT_LOCAL_SERVICE_PROBE_TIMEOUT_MS,
        )
        return _DEFAULT_LOCAL_SERVICE_PROBE_TIMEOUT_MS / 1000
    return max(timeout_ms, 1) / 1000


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


def embedding_service_timeout_seconds(
    mode: EmbeddingProviderMode | None = None,
) -> float:
    raw = os.environ.get(_ENV_TIMEOUT_MS)
    if raw is None:
        resolved = mode or embedding_provider_mode()
        if resolved == "local_service":
            return _DEFAULT_LOCAL_SERVICE_TIMEOUT_MS / 1000
        return _DEFAULT_INTERNAL_SERVICE_TIMEOUT_MS / 1000
    try:
        timeout_ms = int(raw)
    except ValueError as exc:
        raise EmbeddingUnavailableError(
            f"{_ENV_TIMEOUT_MS} must be an integer number of milliseconds"
        ) from exc
    return max(timeout_ms, 1) / 1000


def _is_local_model(model: str | None) -> bool:
    return bool(model and model.startswith("local/"))


def _local_service_status() -> tuple[bool, str | None]:
    global _local_service_probe_cache
    now = time.monotonic()
    if (
        _local_service_probe_cache is not None
        and now - _local_service_probe_cache[0] < _LOCAL_SERVICE_PROBE_CACHE_SECONDS
    ):
        return _local_service_probe_cache[1], _local_service_probe_cache[2]

    try:
        response = httpx.get(
            f"{_local_service_url()}/health",
            timeout=_local_service_probe_timeout_seconds(),
        )
        reachable = response.status_code < 500
        active_model = response.json().get("active_model") if reachable else None
        if not isinstance(active_model, str):
            active_model = None
    except httpx.HTTPError:
        reachable = False
        active_model = None
    except ValueError:
        reachable = False
        active_model = None
    _local_service_probe_cache = (now, reachable, active_model)
    return reachable, active_model


def _local_service_supports_model(model: str | None) -> bool:
    reachable, active_model = _local_service_status()
    return reachable and (active_model is None or active_model == model)


def _ordered_embeddings_from_response(
    data: Any, expected_count: int
) -> list[list[float]]:
    if not isinstance(data, list):
        raise ValueError("embedding service response is missing data[]")
    if len(data) != expected_count:
        raise ValueError(
            "embedding service response cardinality mismatch: "
            f"expected {expected_count}, got {len(data)}"
        )

    seen: set[int] = set()
    indexed_embeddings: list[tuple[int, list[Any]]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("embedding service response data[] has invalid item")

        index = item.get("index")
        if type(index) is not int:
            raise ValueError("embedding service response has invalid index")
        if index in seen:
            raise ValueError(f"embedding service response has duplicate index {index}")
        seen.add(index)

        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("embedding service response has invalid embeddings")
        indexed_embeddings.append((index, embedding))

    expected_indices = set(range(expected_count))
    if seen != expected_indices:
        raise ValueError(
            "embedding service response indices mismatch: "
            f"expected {sorted(expected_indices)}, got {sorted(seen)}"
        )

    return [
        [float(value) for value in embedding]
        for _, embedding in sorted(indexed_embeddings)
    ]


def embedding_provider_mode(model: str | None = None) -> EmbeddingProviderMode:
    """Resolve the embedding provider mode for a model.

    Explicit legacy env modes keep their historical behavior. With no explicit
    env mode, routing is model-driven: ``local/*`` uses the local daemon when it
    is reachable and otherwise falls back to the in-process embedder; cloud
    models use their provider directly.
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

    if os.environ.get(_ENV_CLAUDE_SMART_LOCAL) == "1":
        return "local_service"

    if os.environ.get(_ENV_SERVICE_URL):
        return "internal_service"

    if _is_local_model(model):
        return "local_service" if _local_service_supports_model(model) else "inprocess"
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
    timeout = embedding_service_timeout_seconds(mode)

    # Bound each request so a single large publish cannot exceed the client read
    # timeout: split into fixed-size chunks and concatenate the results in order.
    chunk_size = _max_texts_per_request()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        embeddings.extend(
            _post_embedding_batch(
                url, model=model, texts=chunk, dimensions=dimensions, timeout=timeout
            )
        )
    return embeddings


def _post_embedding_batch(
    url: str,
    *,
    model: str,
    texts: list[str],
    dimensions: int | None,
    timeout: float,
) -> list[list[float]]:
    """POST one bounded batch of texts to the embedding service."""
    payload: dict[str, Any] = {"model": model, "input": texts}
    if dimensions:
        payload["dimensions"] = dimensions

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            return _ordered_embeddings_from_response(body.get("data"), len(texts))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Connection never established — the request never reached the
            # server, so a single retry is safe and cannot duplicate work.
            last_error = exc
            if attempt == 0:
                time.sleep(0.1)
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            # The server already received the request: a read timeout means it
            # is still encoding, and retrying would queue a second identical
            # encode behind the first and amplify load on an already-saturated
            # daemon. HTTPStatusError (4xx/5xx) and malformed-payload errors
            # (JSONDecodeError / the ValueError raised by
            # _ordered_embeddings_from_response) will not change on retry
            # either. Fail fast. Anything outside this set (AttributeError,
            # TypeError, ...) is a programming bug and propagates raw.
            last_error = exc
            break

    _LOGGER.warning("Embedding service unavailable at %s: %s", url, last_error)
    raise EmbeddingUnavailableError(
        f"Embedding service unavailable at {url}: {last_error}"
    ) from last_error


def _max_texts_per_request() -> int:
    raw = os.environ.get(_ENV_MAX_TEXTS_PER_REQUEST)
    if raw is None:
        return _DEFAULT_MAX_TEXTS_PER_REQUEST
    try:
        value = int(raw)
    except ValueError:
        _LOGGER.warning(
            "%s must be an integer; using %d",
            _ENV_MAX_TEXTS_PER_REQUEST,
            _DEFAULT_MAX_TEXTS_PER_REQUEST,
        )
        return _DEFAULT_MAX_TEXTS_PER_REQUEST
    if value < 1:
        _LOGGER.warning(
            "%s must be >= 1; using %d",
            _ENV_MAX_TEXTS_PER_REQUEST,
            _DEFAULT_MAX_TEXTS_PER_REQUEST,
        )
        return _DEFAULT_MAX_TEXTS_PER_REQUEST
    return value
