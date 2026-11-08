"""C45 activation through the existing search inference service owner."""

from __future__ import annotations

import math

import httpx

from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
    checkpoint,
    inference_http_client,
    inference_service_url,
    remaining,
    search_reranker_service_url,
    validate_search_reranker_service,
)
from reflexio.server.llm.rerank.common import (
    MULTILINGUAL_RERANK_MODEL,
    MULTILINGUAL_RERANK_REVISION,
    CrossEncoderUnavailableError,
    reranker_enabled,
)
from reflexio.server.llm.rerank.cross_encoder_reranker import (
    _rerank_service_timeout_seconds,
)
from reflexio.server.tracing import profile_step


def score_user_playbook_activation(
    query: str, documents: list[str]
) -> list[float | None]:
    """Return aligned raw logits or explicit overlength markers; never fall back."""
    if not documents:
        return []
    if len(documents) > 5 or not reranker_enabled():
        raise CrossEncoderUnavailableError("USER-playbook activation is unavailable")
    checkpoint()
    try:
        service_url = (
            validate_search_reranker_service()
            if search_reranker_service_url()
            else inference_service_url()
        )
    except EmbeddingUnavailableError as exc:
        checkpoint()
        raise CrossEncoderUnavailableError(
            "USER-playbook activation route is unavailable"
        ) from exc
    budget = remaining(_rerank_service_timeout_seconds())
    payload = {
        "model": MULTILINGUAL_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "timeout_ms": max(1, min(600_000, int(remaining(budget) * 1000))),
    }
    with profile_step(
        "search.user_selection.activation", document_count=len(documents)
    ) as span:
        try:
            response = inference_http_client().post(
                f"{service_url.rstrip('/')}/v1/user-playbook-activation",
                json=payload,
                timeout=remaining(budget),
            )
            checkpoint()
            span.set_data("http_status_code", response.status_code)
            response.raise_for_status()
            body = response.json()
            if (
                body.get("model") != MULTILINGUAL_RERANK_MODEL
                or body.get("revision") != MULTILINGUAL_RERANK_REVISION
            ):
                raise ValueError("Activation model identity differs")
            scores = body.get("scores")
            if not isinstance(scores, list) or len(scores) != len(documents):
                raise ValueError("Activation score cardinality differs")
            if any(
                value is not None
                and (type(value) not in (int, float) or not math.isfinite(value))
                for value in scores
            ):
                raise ValueError("Activation requires finite raw logits")
            checkpoint()
            span.set_data("outcome", "success")
            return [float(value) if value is not None else None for value in scores]
        except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
            checkpoint()
            span.set_data("outcome", "unavailable")
            raise CrossEncoderUnavailableError(
                "USER-playbook activation service failed"
            ) from exc
