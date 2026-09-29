"""Embedding model policy, independent of application entity schemas."""

from dataclasses import dataclass
from typing import Literal

SEARCH_DOCUMENT_PREFIX = "search_document: "
SEARCH_QUERY_PREFIX = "search_query: "
E5_DOCUMENT_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "
MULTILINGUAL_E5_MODEL = "local/multilingual-e5-small"


@dataclass(frozen=True)
class EmbeddingModelPolicy:
    """Input formatting and retrieval defaults owned by one embedding model."""

    document_prefix: str
    query_prefix: str
    retrieval_threshold: float
    clustering_similarity: float


_DEFAULT_MODEL_POLICY = EmbeddingModelPolicy(
    document_prefix="",
    query_prefix="",
    retrieval_threshold=0.45,
    clustering_similarity=0.30,
)
_NOMIC_MODEL_POLICY = EmbeddingModelPolicy(
    document_prefix=SEARCH_DOCUMENT_PREFIX,
    query_prefix=SEARCH_QUERY_PREFIX,
    retrieval_threshold=0.70,
    clustering_similarity=0.85,
)
_EMBEDDING_MODEL_POLICIES = {
    "local/minilm-l6-v2": EmbeddingModelPolicy(
        document_prefix="",
        query_prefix="",
        retrieval_threshold=0.30,
        clustering_similarity=0.30,
    ),
    "local/nomic-embed-text-v1.5": _NOMIC_MODEL_POLICY,
    # Backward-compatible provider alias. New configuration should use the
    # canonical ``local/nomic-embed-text-v1.5`` identifier.
    "local/nomic-embed-v1.5": _NOMIC_MODEL_POLICY,
    MULTILINGUAL_E5_MODEL: EmbeddingModelPolicy(
        document_prefix=E5_DOCUMENT_PREFIX,
        query_prefix=E5_QUERY_PREFIX,
        retrieval_threshold=0.7993814945220947,
        clustering_similarity=0.9284577369689941,
    ),
}


def is_multilingual_e5_model(model: str) -> bool:
    """Return whether ``model`` uses the shared multilingual E5 policy."""
    return model.strip().casefold() == MULTILINGUAL_E5_MODEL


def embedding_input(
    text: str,
    *,
    model_name: str,
    purpose: Literal["document", "query"] = "document",
) -> str:
    """Apply input formatting owned by the selected embedding model.

    ``purpose`` must be ``"document"`` or ``"query"``; any other value raises so a
    misspelled call site fails fast instead of silently writing or searching the
    wrong vector space. Unrecognized models receive the text unchanged.
    """
    policy = _EMBEDDING_MODEL_POLICIES.get(
        model_name.strip().casefold(), _DEFAULT_MODEL_POLICY
    )
    if purpose == "document":
        return policy.document_prefix + text
    if purpose == "query":
        return policy.query_prefix + text
    raise ValueError(
        f"Unknown embedding purpose {purpose!r}; expected 'document' or 'query'"
    )


def resolve_retrieval_threshold(
    requested_threshold: float | None,
    *,
    model_name: str,
) -> float:
    """Return an explicit threshold or the selected model's default."""
    if requested_threshold is not None:
        return requested_threshold
    policy = _EMBEDDING_MODEL_POLICIES.get(
        model_name.strip().casefold(), _DEFAULT_MODEL_POLICY
    )
    return policy.retrieval_threshold


def resolve_clustering_similarity(
    requested_similarity: float | None,
    *,
    model_name: str,
) -> float:
    """Return an explicit clustering similarity or the model default."""
    if requested_similarity is not None:
        return requested_similarity
    policy = _EMBEDDING_MODEL_POLICIES.get(
        model_name.strip().casefold(), _DEFAULT_MODEL_POLICY
    )
    return policy.clustering_similarity
