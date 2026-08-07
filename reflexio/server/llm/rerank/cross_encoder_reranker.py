"""HTTP client for the shared inference service's reranker endpoint."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx

from reflexio.server.env_utils import env_str
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
    inference_http_client,
    inference_service_url,
    resolve_service_configured_reranker_model,
)
from reflexio.server.llm.rerank.common import (
    RERANK_ENABLED_ENV_VAR,
    RERANK_MODEL,
    CrossEncoderUnavailableError,
    reranker_enabled,
)

_SERVICE_TIMEOUT_MS_ENV_VAR = "REFLEXIO_RERANK_SERVICE_TIMEOUT_MS"
_DEFAULT_SERVICE_TIMEOUT_MS = 5_000


def score_pairs(query: str, docs: list[str]) -> list[float]:
    """Score pairs through the shared inference service.

    The main API process never loads a local cross-encoder. Disabled or
    unavailable service paths raise ``CrossEncoderUnavailableError`` so unified
    search can preserve its existing fail-open behavior.
    """
    return score_pairs_with_model(query, docs)[1]


def score_pairs_with_model(query: str, docs: list[str]) -> tuple[str, list[float]]:
    """Score pairs and return the discovered model used for the logits."""
    if not reranker_enabled():
        raise CrossEncoderUnavailableError(
            f"Reranker is disabled by {RERANK_ENABLED_ENV_VAR}"
        )
    service_url = inference_service_url()
    expected_local_unavailability = _is_colocated_service(service_url)
    try:
        model = resolve_service_configured_reranker_model()
    except EmbeddingUnavailableError as exc:
        raise CrossEncoderUnavailableError(
            f"Could not discover reranker model: {exc}",
            report_failure=not expected_local_unavailability,
        ) from exc
    scores = (
        []
        if not docs
        else _score_pairs_remote(
            service_url,
            model,
            query,
            docs,
            expected_local_unavailability=expected_local_unavailability,
        )
    )
    return model, scores


def reranker_service_url() -> str:
    """Return the shared embedding/reranking service base URL."""
    return inference_service_url()


def _is_colocated_service(service_url: str) -> bool:
    """Return whether failure is expected from the colocated child process."""
    if env_str("REFLEXIO_EMBEDDING_SERVICE_URL"):
        return False
    hostname = urlsplit(service_url).hostname
    return hostname in {"127.0.0.1", "::1", "localhost"}


def _score_pairs_remote(
    service_url: str,
    model: str,
    query: str,
    docs: list[str],
    *,
    expected_local_unavailability: bool,
) -> list[float]:
    url = f"{service_url.rstrip('/')}/v1/rerank"
    payload = {"model": model, "query": query, "documents": docs}
    try:
        response = inference_http_client().post(
            url,
            json=payload,
            timeout=_rerank_service_timeout_seconds(),
        )
        response.raise_for_status()
        body = response.json()
        return _ordered_scores_from_response(body.get("data"), len(docs))
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
    ) as exc:
        raise CrossEncoderUnavailableError(
            f"Rerank service request failed at {url}: {exc}",
            report_failure=not expected_local_unavailability,
        ) from exc
    except httpx.HTTPStatusError as exc:
        expected_local_503 = (
            expected_local_unavailability
            and exc.response.status_code == httpx.codes.SERVICE_UNAVAILABLE
        )
        raise CrossEncoderUnavailableError(
            f"Rerank service request failed at {url}: {exc}",
            report_failure=not expected_local_503,
        ) from exc
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise CrossEncoderUnavailableError(
            f"Rerank service request failed at {url}: {exc}"
        ) from exc


def _rerank_service_timeout_seconds() -> float:
    raw = env_str(_SERVICE_TIMEOUT_MS_ENV_VAR, str(_DEFAULT_SERVICE_TIMEOUT_MS))
    try:
        timeout_ms = int(raw)
    except ValueError as exc:
        raise CrossEncoderUnavailableError(
            f"{_SERVICE_TIMEOUT_MS_ENV_VAR} must be an integer number of milliseconds"
        ) from exc
    return max(timeout_ms, 1) / 1000


def _ordered_scores_from_response(data: Any, expected_count: int) -> list[float]:
    if not isinstance(data, list):
        raise ValueError("rerank service response is missing data[]")
    if len(data) != expected_count:
        raise ValueError(
            "rerank service response cardinality mismatch: "
            f"expected {expected_count}, got {len(data)}"
        )

    seen: set[int] = set()
    indexed_scores: list[tuple[int, float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("rerank service response data[] has invalid item")
        index = item.get("index")
        if type(index) is not int:
            raise ValueError("rerank service response has invalid index")
        if index in seen:
            raise ValueError(f"rerank service response has duplicate index {index}")
        if index < 0 or index >= expected_count:
            raise ValueError(f"rerank service response has out-of-range index {index}")
        seen.add(index)

        score = item.get("score")
        if not isinstance(score, int | float) or isinstance(score, bool):
            raise ValueError("rerank service response has invalid score")
        indexed_scores.append((index, float(score)))

    expected_indices = set(range(expected_count))
    if seen != expected_indices:
        raise ValueError(
            "rerank service response indices mismatch: "
            f"expected {sorted(expected_indices)}, got {sorted(seen)}"
        )
    return [score for _, score in sorted(indexed_scores)]


__all__ = [
    "RERANK_MODEL",
    "CrossEncoderUnavailableError",
    "reranker_service_url",
    "score_pairs",
    "score_pairs_with_model",
]
