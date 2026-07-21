"""Canonical text and model-specific policy used for vector embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Interaction,
    UserPlaybook,
    UserProfile,
)

SEARCH_DOCUMENT_PREFIX = "search_document: "
SEARCH_QUERY_PREFIX = "search_query: "


@dataclass(frozen=True)
class EmbeddingModelPolicy:
    """Input formatting and retrieval defaults owned by one embedding model."""

    document_prefix: str
    query_prefix: str
    retrieval_threshold: float


_DEFAULT_MODEL_POLICY = EmbeddingModelPolicy(
    document_prefix="",
    query_prefix="",
    retrieval_threshold=0.45,
)
_NOMIC_MODEL_POLICY = EmbeddingModelPolicy(
    document_prefix=SEARCH_DOCUMENT_PREFIX,
    query_prefix=SEARCH_QUERY_PREFIX,
    retrieval_threshold=0.70,
)
_EMBEDDING_MODEL_POLICIES = {
    "local/minilm-l6-v2": EmbeddingModelPolicy(
        document_prefix="",
        query_prefix="",
        retrieval_threshold=0.30,
    ),
    "local/nomic-embed-text-v1.5": _NOMIC_MODEL_POLICY,
    # Backward-compatible provider alias. New configuration should use the
    # canonical ``local/nomic-embed-text-v1.5`` identifier.
    "local/nomic-embed-v1.5": _NOMIC_MODEL_POLICY,
}

EmbeddingTextEntity = (
    Interaction
    | UserProfile
    | UserPlaybook
    | AgentPlaybook
    | AgentSuccessEvaluationResult
)


def embedding_text(entity: EmbeddingTextEntity) -> str:
    """Return the exact text used for an entity's stored embedding."""
    if isinstance(entity, Interaction):
        return f"{entity.content}\n{entity.user_action_description}"
    if isinstance(entity, UserProfile):
        parts = [entity.content]
        if entity.custom_features:
            parts.append(str(entity.custom_features))
        return "\n".join(parts)
    if isinstance(entity, (UserPlaybook, AgentPlaybook)):
        return entity.trigger or entity.content
    if isinstance(entity, AgentSuccessEvaluationResult):
        return " ".join(
            part
            for part in (entity.failure_type, entity.failure_reason)
            if part and part.strip()
        )
    raise TypeError(f"Unsupported embedding text entity: {type(entity).__name__}")


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
