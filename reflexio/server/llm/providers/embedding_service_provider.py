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
import threading
import time
from typing import Any, Literal

import httpx

from reflexio.server.tracing import profile_step

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
# A reachable daemon is stable, so positive probe results are cached long
# enough that steady-state requests skip the sequential /health round trip.
# Failures stay short-lived so a restarted daemon is re-adopted quickly and
# the inprocess-fallback window stays small.
_LOCAL_SERVICE_PROBE_CACHE_SECONDS = 60.0
_LOCAL_SERVICE_PROBE_FAILURE_CACHE_SECONDS = 5.0
# Keep-alive must expire below the ALB's default 60s idle timeout so the pool
# never hands out a connection the load balancer already closed.
_HTTP_KEEPALIVE_EXPIRY_SECONDS = 50.0
_EMBEDDING_RETRY_BACKOFF_SECONDS = 0.1
_SERVICE_MODES = {"local_service", "internal_service"}
_VALID_MODES = {"cloud", *_SERVICE_MODES, "inprocess", "off"}
_local_service_probe_cache: tuple[float, bool, str | None] | None = None
_http_client_lock = threading.Lock()
_http_client_instance: httpx.Client | None = None
_http_client_pid: int | None = None


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


def _http_client() -> httpx.Client:
    """Shared keep-alive client for health probes and embedding POSTs.

    A per-request client pays DNS resolution plus a TCP handshake on every
    embedding call. The client is recreated after fork (PID check) so worker
    children never reuse sockets inherited from the parent process.
    """
    global _http_client_instance, _http_client_pid
    pid = os.getpid()
    with _http_client_lock:
        if _http_client_instance is None or _http_client_pid != pid:
            if _http_client_instance is not None and _http_client_pid != pid:
                try:
                    _http_client_instance.close()
                except Exception:
                    _LOGGER.debug(
                        "Failed to close stale embedding HTTP client", exc_info=True
                    )
            _http_client_instance = httpx.Client(
                limits=httpx.Limits(keepalive_expiry=_HTTP_KEEPALIVE_EXPIRY_SECONDS)
            )
            _http_client_pid = pid
        return _http_client_instance


def _local_service_status() -> tuple[bool, str | None]:
    global _local_service_probe_cache
    now = time.monotonic()
    if _local_service_probe_cache is not None:
        cached_at, cached_reachable, cached_model = _local_service_probe_cache
        ttl = (
            _LOCAL_SERVICE_PROBE_CACHE_SECONDS
            if cached_reachable
            else _LOCAL_SERVICE_PROBE_FAILURE_CACHE_SECONDS
        )
        if now - cached_at < ttl:
            return cached_reachable, cached_model

    try:
        response = _http_client().get(
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
    env mode, routing is model-driven: ``local/*`` uses a configured daemon host
    authoritatively, otherwise it uses the local daemon when reachable and falls
    back to the in-process embedder. Cloud models use their provider directly.
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

    if _is_local_model(model) and os.environ.get(_ENV_DAEMON_HOST, "").strip():
        return "local_service"

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
            response = _http_client().post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            return _ordered_embeddings_from_response(body.get("data"), len(texts))
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
        ) as exc:
            # Connection never established, or the server closed a pooled
            # keep-alive connection before sending a response (stale-reuse
            # race) — the request was not processed, so a single retry is
            # safe and cannot duplicate work.
            last_error = exc
            if attempt == 0:
                with profile_step(
                    "search.embedding.api.retry_backoff",
                    retry_reason=type(exc).__name__,
                    retry_backoff_ms=int(_EMBEDDING_RETRY_BACKOFF_SECONDS * 1000),
                    attempt=attempt + 1,
                    max_attempts=2,
                ):
                    time.sleep(_EMBEDDING_RETRY_BACKOFF_SECONDS)
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
