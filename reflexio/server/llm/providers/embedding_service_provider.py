"""HTTP embedding service provider.

This module is the routing boundary between Reflexio's LLM client and an
OpenAI-compatible embedding service. It lets local Claude Smart installs keep
one model process alive across backend workers, Claude Code, and Codex, while
also supporting a horizontally-scaled internal service in cloud deployments.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, replace
from ipaddress import ip_address
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from reflexio.embedding_requests import embedding_request_max_texts
from reflexio.inference_launch import (
    inference_service_url as resolve_inference_service_url,
)
from reflexio.removed_env import reject_removed_env
from reflexio.server.env_utils import env_str, positive_int_env
from reflexio.server.llm.call_deadline import remaining_call_time
from reflexio.server.search_runtime import (
    checkpoint as _search_checkpoint,
)
from reflexio.server.search_runtime import (
    current,
    http_request_deadline,
    increment,
)
from reflexio.server.search_runtime import (
    remaining as _search_remaining,
)
from reflexio.server.tracing import profile_step

_LOGGER = logging.getLogger(__name__)

EmbeddingProviderMode = Literal["cloud", "local_service", "internal_service", "off"]

_ENV_PROVIDER = "REFLEXIO_EMBEDDING_PROVIDER"
_ENV_SERVICE_URL = "REFLEXIO_EMBEDDING_SERVICE_URL"
_ENV_QUERY_URL = "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL"
_query_contract_cache: dict[tuple[int, str, str], tuple[str, str, int]] = {}
_query_contract_lock = threading.Lock()
_ENV_TIMEOUT_MS = "REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS"
CUSTOM_EMBEDDING_MODEL = "custom"
_DEFAULT_INTERNAL_SERVICE_TIMEOUT_MS = 2_000
_DEFAULT_LOCAL_SERVICE_TIMEOUT_MS = 30_000
# Keep-alive must expire below the ALB's default 60s idle timeout so the pool
# never hands out a connection the load balancer already closed.
_HTTP_KEEPALIVE_EXPIRY_SECONDS = 50.0
_EMBEDDING_RETRY_BACKOFF_SECONDS = 0.1
_SERVICE_MODES = {"local_service", "internal_service"}
_VALID_MODES = {"cloud", *_SERVICE_MODES, "off"}
_http_client_lock = threading.Lock()
_http_client_instance: httpx.Client | None = None
_http_client_pid: int | None = None
_configured_model_cache: dict[tuple[int, str], InferenceServiceCapabilities] = {}
_configured_model_lock = threading.Lock()
_ADVERTISED_CAPABILITY_LOCK_SECONDS = 5


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the configured embedding provider is unavailable."""


def remaining(default: float = 30.0) -> float:
    """Respect both the search request and the enclosing compute attempt."""
    return remaining_call_time(_search_remaining(default))


def checkpoint() -> None:
    _search_checkpoint()
    remaining_call_time()


@dataclass(frozen=True)
class InferenceServiceCapabilities:
    """Process-cached model contract advertised by one inference service."""

    configured_model: str
    configured_reranker_model: str | None
    reranker_enabled: bool
    reranker_ready: bool
    workload: str = "shared"


def inference_service_url() -> str:
    """Return the shared local/internal inference-service base URL.

    This resolver is model-agnostic so non-embedding capabilities such as
    reranking can share the same daemon even when an organization uses cloud
    embeddings.
    """
    return resolve_inference_service_url(os.environ)


def remote_inference_service_configured() -> bool:
    """Return whether the configured inference URL targets a non-loopback host."""
    configured = os.environ.get(_ENV_SERVICE_URL, "").strip()
    if not configured:
        return False
    hostname = urlsplit(configured).hostname
    if hostname is None or hostname.lower() == "localhost":
        return False
    try:
        return not ip_address(hostname).is_loopback
    except ValueError:
        return True


def embedding_service_url(mode: EmbeddingProviderMode | None = None) -> str:
    """Return the configured embedding service URL.

    ``local_service`` defaults to ``127.0.0.1:$EMBEDDING_PORT`` so local
    Claude Smart hosts can share a single machine daemon. ``internal_service``
    requires an explicit URL because it is deployment-specific.
    """
    resolved = mode or embedding_provider_mode()
    configured = os.environ.get(_ENV_SERVICE_URL, "").strip()
    if configured:
        return inference_service_url()
    if resolved == "local_service":
        return inference_service_url()
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


def _uses_embedding_service(model: str | None) -> bool:
    return model == CUSTOM_EMBEDDING_MODEL or _is_local_model(model)


def resolve_service_configured_model(
    model: str | None = CUSTOM_EMBEDDING_MODEL,
) -> str:
    """Resolve ``custom`` to the service's authoritative, process-cached model.

    Exact cloud and local model names pass through unchanged. The service model
    is immutable for a data-plane process lifetime; operators restart the data
    plane after changing the model behind an endpoint.
    """
    if model and model != CUSTOM_EMBEDDING_MODEL:
        return model
    return resolve_inference_service_capabilities().configured_model


def resolve_service_configured_reranker_model() -> str:
    """Return the service-advertised concrete reranker model."""
    model = resolve_inference_service_capabilities().configured_reranker_model
    if model is None:
        raise EmbeddingUnavailableError(
            f"{inference_service_url()}/health did not return configured_reranker_model"
        )
    return model


def cached_inference_service_capabilities() -> InferenceServiceCapabilities | None:
    """Return this process/endpoint contract without network or lock waits."""
    return _configured_model_cache.get((os.getpid(), inference_service_url()))


def resolve_inference_service_capabilities(
    *, advertised: dict[str, Any] | None = None, service_url: str | None = None
) -> InferenceServiceCapabilities:
    """Discover and process-cache both models with one ``/health`` request.

    A host may supply a bounded fresh health response to qualify a cold cache
    without another HTTP request. An existing process contract remains immutable;
    supplied metadata never replaces it. Supplied responses wait at most five
    seconds for another caller currently discovering the initial contract.

    Raises:
        EmbeddingUnavailableError: If the provider is ``off``, or the service
            cannot be reached.
    """
    checkpoint()
    # Honour an explicit `off` before deriving a service mode. Deriving one
    # unconditionally made `off` unreachable here: the caller got a connection
    # error against a service it had just disabled, rather than being told the
    # provider is off.
    if embedding_provider_mode() == "off":
        raise EmbeddingUnavailableError(
            f"Embedding provider is disabled ({_ENV_PROVIDER}=off); "
            "no service model can be discovered"
        )
    mode: EmbeddingProviderMode = (
        "internal_service"
        if os.environ.get(_ENV_SERVICE_URL, "").strip()
        else "local_service"
    )
    service_url = (service_url or embedding_service_url(mode)).rstrip("/")
    cache_key = (os.getpid(), service_url)
    cached = _configured_model_cache.get(cache_key)
    if cached is not None:
        return cached
    admission_seconds = (
        _ADVERTISED_CAPABILITY_LOCK_SECONDS
        if advertised is not None
        else embedding_service_timeout_seconds(mode)
    )
    discovery_deadline = time.monotonic() + remaining(admission_seconds)
    admitted = _configured_model_lock.acquire(timeout=remaining(admission_seconds))
    if not admitted:
        raise EmbeddingUnavailableError("Inference capability discovery is busy")
    try:
        checkpoint()
        cached = _configured_model_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            if advertised is None:
                body = _inference_service_health(
                    service_url,
                    seconds=max(0, discovery_deadline - time.monotonic()),
                )
            else:
                body = advertised
            checkpoint()
            configured_model = body.get("configured_model")
        except (
            httpx.HTTPError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            raise EmbeddingUnavailableError(
                f"Could not discover embedding model from {service_url}/health: {exc}"
            ) from exc
        if not isinstance(configured_model, str) or not configured_model.strip():
            raise EmbeddingUnavailableError(
                f"{service_url}/health did not return configured_model"
            )
        reranker_model = body.get("configured_reranker_model")
        if not isinstance(reranker_model, str) or not reranker_model.strip():
            reranker_model = None
        capabilities = InferenceServiceCapabilities(
            configured_model=configured_model.strip(),
            configured_reranker_model=(
                reranker_model.strip() if reranker_model is not None else None
            ),
            reranker_enabled=body.get("reranker_enabled") is True,
            reranker_ready=body.get("reranker_ready") is True,
            workload=body.get("workload", "shared"),
        )
        checkpoint()
        _configured_model_cache[cache_key] = capabilities
        return capabilities
    finally:
        _configured_model_lock.release()


def _inference_service_health(service_url: str, *, seconds: float) -> dict[str, Any]:
    try:
        response = _http_client().get(
            f"{service_url}/health", timeout=remaining(seconds)
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Inference health must be an object")
        return body
    except (httpx.HTTPError, ValueError) as exc:
        raise EmbeddingUnavailableError(
            f"Could not discover inference health from {service_url}/health: {exc}"
        ) from exc


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
                limits=httpx.Limits(keepalive_expiry=_HTTP_KEEPALIVE_EXPIRY_SECONDS),
                event_hooks={"request": [http_request_deadline]},
            )
            _http_client_pid = pid
        return _http_client_instance


def inference_http_client() -> httpx.Client:
    """Return the process/fork-safe keep-alive client for shared inference APIs."""
    return _http_client()


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

    Local models always use the separate inference-service process. Service
    availability never changes the routing decision and never causes a model to
    load inside an API worker. Cloud models use their provider directly.
    """
    reject_removed_env(os.environ, canonical=_ENV_PROVIDER)
    configured = os.environ.get(_ENV_PROVIDER)
    if configured:
        mode = configured.strip().lower()
        if mode not in _VALID_MODES:
            raise EmbeddingUnavailableError(
                f"Invalid {_ENV_PROVIDER}={configured!r}; expected one of "
                f"{', '.join(sorted(_VALID_MODES))}"
            )
        if mode == "off":
            return "off"
        if model and not _uses_embedding_service(model):
            return "cloud"
        if mode == "internal_service" or os.environ.get(_ENV_SERVICE_URL, "").strip():
            return "internal_service"
        return "local_service"

    if model and not _uses_embedding_service(model):
        return "cloud"

    if os.environ.get(_ENV_SERVICE_URL, "").strip():
        return "internal_service"

    if _uses_embedding_service(model):
        return "local_service"
    return "cloud"


def should_use_embedding_service(model: str) -> bool:
    """Return True when embedding requests should call the HTTP service."""
    return embedding_provider_mode(model) in _SERVICE_MODES


def query_embedding_service_url() -> str:
    """Optional isolated query service; an empty value retains shared routing."""
    return env_str(_ENV_QUERY_URL, "").rstrip("/")


def search_reranker_service_url() -> str:
    """Optional search-only reranker; generic reranking keeps shared routing."""
    return env_str("REFLEXIO_SEARCH_RERANK_SERVICE_URL", "").rstrip("/")


def validate_search_reranker_service(
    *, advertised: dict[str, Any] | None = None
) -> str:
    """Bind reserved reranking to immutable models and current operational health.

    Readiness supplies fresh bounded health. Healthy request-path discovery stays
    cached; an unhealthy first discovery is retried without changing model identity.
    """
    endpoint = search_reranker_service_url()
    if not endpoint:
        return inference_service_url()
    if endpoint != query_embedding_service_url() or endpoint == inference_service_url():
        raise EmbeddingUnavailableError(
            "Search reranking must use the isolated query worker"
        )
    shared = resolve_inference_service_capabilities()
    validate_query_embedding_service(shared.configured_model, None)
    cached = _configured_model_cache.get((os.getpid(), endpoint))
    isolated = resolve_inference_service_capabilities(
        advertised=advertised, service_url=endpoint
    )
    if (
        advertised is None
        and cached is not None
        and (
            isolated.workload != "search"
            or isolated.configured_model != shared.configured_model
            or not isolated.reranker_enabled
            or not isolated.reranker_ready
        )
    ):
        advertised = _inference_service_health(
            endpoint, seconds=embedding_service_timeout_seconds()
        )
    if advertised is not None:
        if (
            advertised.get("configured_model") != isolated.configured_model
            or advertised.get("configured_reranker_model")
            != isolated.configured_reranker_model
        ):
            raise EmbeddingUnavailableError(
                "Search worker model identity changed; restart API required"
            )
        # Only operational state may refresh; the two model identities remain
        # bound to this API process even across temporary reranker failures.
        isolated = replace(
            isolated,
            workload=advertised.get("workload", "shared"),
            reranker_enabled=advertised.get("reranker_enabled") is True,
            reranker_ready=advertised.get("reranker_ready") is True,
        )
        _configured_model_cache[(os.getpid(), endpoint)] = isolated
    if (
        isolated.workload != "search"
        or isolated.configured_model != shared.configured_model
        or not isolated.reranker_enabled
        or not isolated.reranker_ready
        or not isolated.configured_reranker_model
        or isolated.configured_reranker_model != shared.configured_reranker_model
    ):
        raise EmbeddingUnavailableError(
            "Search worker reranker contract is incompatible or unready"
        )
    return endpoint


def _vector_contract(body: dict[str, Any], *, query: bool) -> tuple[str, str, int]:
    model, revision, dimensions = (
        body.get("model"),
        body.get("revision"),
        body.get("dimensions"),
    )
    if (
        not isinstance(model, str)
        or not model
        or not isinstance(revision, str)
        or not revision
        or type(dimensions) is not int
        or dimensions <= 0
    ):
        raise EmbeddingUnavailableError("Incomplete vector contract")
    if not query and body.get("workload", "shared") != "shared":
        raise EmbeddingUnavailableError("Document endpoint is not a shared worker")
    if query and body.get("workload") != "query":
        raise EmbeddingUnavailableError("Query endpoint is not a query-only worker")
    return model, revision, dimensions


def validate_query_embedding_service(
    model: str,
    dimensions: int | None,
    *,
    advertised: tuple[dict[str, Any], dict[str, Any]] | None = None,
    deadline: float | None = None,
) -> str:
    """Verify identical immutable vector contracts; never fail over to shared.

    Readiness supplies fresh, bounded health responses. The request path caches
    contracts by PID and both endpoints; a drift requires restarting the API.
    """
    checkpoint()
    query_url = query_embedding_service_url()
    mode = embedding_provider_mode(model)
    if not query_url:
        return embedding_service_url(mode)
    shared_url = embedding_service_url(mode)
    if query_url == shared_url:
        raise EmbeddingUnavailableError("Query embedding capacity must be separate")
    key = (os.getpid(), shared_url, query_url)
    contract = _query_contract_cache.get(key)
    if contract is None or advertised is not None:
        admission_seconds = (
            _ADVERTISED_CAPABILITY_LOCK_SECONDS
            if advertised is not None
            else embedding_service_timeout_seconds(mode)
        )
        discovery_deadline = time.monotonic() + remaining(admission_seconds)
        if deadline is not None:
            discovery_deadline = min(discovery_deadline, deadline)
        if not _query_contract_lock.acquire(
            timeout=remaining(max(0, discovery_deadline - time.monotonic()))
        ):
            raise EmbeddingUnavailableError("Query model contract discovery is busy")
        try:
            checkpoint()
            cached = _query_contract_cache.get(key)
            if cached is not None and advertised is None:
                contract = cached
            else:
                if advertised is None:
                    bodies = []
                    for endpoint in (shared_url, query_url):
                        try:
                            response = _http_client().get(
                                endpoint + "/health/model",
                                timeout=remaining(
                                    max(0, discovery_deadline - time.monotonic())
                                ),
                            )
                            response.raise_for_status()
                            bodies.append(response.json())
                        except (httpx.HTTPError, ValueError) as exc:
                            raise EmbeddingUnavailableError(
                                "Could not verify isolated query embedding contract"
                            ) from exc
                else:
                    bodies = list(advertised)
                if any(not isinstance(body, dict) for body in bodies):
                    raise EmbeddingUnavailableError("Incomplete vector contract")
                contract = _vector_contract(bodies[0], query=False)
                if contract != _vector_contract(bodies[1], query=True):
                    raise EmbeddingUnavailableError(
                        "Query and document vector contracts differ"
                    )
                if cached is not None and cached != contract:
                    raise EmbeddingUnavailableError(
                        "Vector contract changed; restart API required"
                    )
                checkpoint()
                _query_contract_cache[key] = contract
        finally:
            _query_contract_lock.release()
    if contract[0] != model or (dimensions is not None and contract[2] != dimensions):
        raise EmbeddingUnavailableError(
            "Query vector contract differs from requested model/dimensions"
        )
    checkpoint()
    return query_url


def get_service_embeddings(
    texts: list[str],
    *,
    model: str,
    dimensions: int | None = None,
    purpose: Literal["document", "query"] = "document",
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

    checkpoint()
    mode = embedding_provider_mode(model)
    if mode == "off":
        raise EmbeddingUnavailableError("Embedding provider is disabled")
    if mode not in _SERVICE_MODES:
        raise EmbeddingUnavailableError(
            f"Embedding service requested while {_ENV_PROVIDER}={mode}"
        )

    if purpose not in ("document", "query"):
        raise ValueError("Embedding purpose must be document or query")
    separate = purpose == "query" and bool(query_embedding_service_url())
    base = (
        validate_query_embedding_service(model, dimensions)
        if separate
        else embedding_service_url(mode)
    )
    url = f"{base}/v1/embeddings"
    timeout = embedding_service_timeout_seconds(mode)

    # Bound each request so a single large publish cannot exceed the client read
    # timeout: split into fixed-size chunks and concatenate the results in order.
    chunk_size = 1 if separate else embedding_request_max_texts()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), chunk_size):
        checkpoint()
        chunk = texts[start : start + chunk_size]
        embeddings.extend(
            _post_embedding_batch(
                url,
                model=model,
                texts=chunk,
                dimensions=dimensions,
                timeout=timeout,
                mode=mode,
                purpose=purpose,
            )
        )
    checkpoint()
    return embeddings


def _post_embedding_batch(
    url: str,
    *,
    model: str,
    texts: list[str],
    dimensions: int | None,
    timeout: float,
    mode: EmbeddingProviderMode,
    purpose: Literal["document", "query"] = "document",
) -> list[list[float]]:
    """POST one bounded batch of texts to the embedding service."""
    payload: dict[str, Any] = {"model": model, "input": texts}
    if purpose == "query":
        payload["purpose"] = "query"
    interactive = (
        purpose == "query"
        or current() is not None
        or all(text.startswith(("query: ", "search_query: ")) for text in texts)
    )
    payload["priority"] = "interactive" if interactive else "bulk"
    if not interactive:
        timeout = max(
            timeout,
            positive_int_env("REFLEXIO_EMBEDDING_BULK_TIMEOUT_MS", 2_000, _LOGGER)
            / 1000,
        )
    if dimensions:
        payload["dimensions"] = dimensions

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            checkpoint()
            payload["timeout_ms"] = max(1, min(600_000, int(remaining(timeout) * 1000)))
            response = _http_client().post(
                url, json=payload, timeout=remaining(timeout)
            )
            checkpoint()
            response.raise_for_status()
            body = response.json()
            vectors = _ordered_embeddings_from_response(body.get("data"), len(texts))
            if purpose == "query" and query_embedding_service_url():
                expected = (
                    dimensions
                    or _query_contract_cache[
                        (
                            os.getpid(),
                            embedding_service_url(mode),
                            query_embedding_service_url(),
                        )
                    ][2]
                )
                if any(
                    len(vector) != expected
                    or not all(math.isfinite(value) for value in vector)
                    for vector in vectors
                ):
                    raise ValueError("Query worker returned invalid vectors")
                increment("search.embedding.query_worker")
            checkpoint()
            return vectors
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
                    time.sleep(remaining(_EMBEDDING_RETRY_BACKOFF_SECONDS))
                    checkpoint()
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

    # For HTTP-status failures the response body carries the actionable
    # detail (e.g. the daemon's 409 "already owns model X" message) — without
    # it the log reads as a connectivity problem when it is a config conflict.
    error_detail = str(last_error)
    if isinstance(last_error, httpx.HTTPStatusError):
        body = last_error.response.text.strip()
        if body:
            error_detail = f"{last_error} — response body: {body[:300]}"
    _LOGGER.warning(
        "Embedding service unavailable at %s: %s",
        url,
        error_detail,
        extra={"mode": mode},
    )
    raise EmbeddingUnavailableError(
        f"Embedding service unavailable at {url}: {error_detail}"
    ) from last_error
