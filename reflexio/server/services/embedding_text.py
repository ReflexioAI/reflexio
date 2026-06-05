"""Canonical text and prefixes used for vector embeddings."""

from __future__ import annotations

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Interaction,
    UserPlaybook,
    UserProfile,
)

SEARCH_DOCUMENT_PREFIX = "search_document: "
SEARCH_QUERY_PREFIX = "search_query: "

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
        return "\n".join([entity.content, str(entity.custom_features)])
    if isinstance(entity, (UserPlaybook, AgentPlaybook)):
        return entity.trigger or entity.content
    if isinstance(entity, AgentSuccessEvaluationResult):
        return f"{entity.failure_type} {entity.failure_reason}"
    raise TypeError(f"Unsupported embedding text entity: {type(entity).__name__}")


def embedding_input(text: str, *, purpose: str = "document") -> str:
    """Apply the asymmetric search prefix used before embedding calls."""
    prefix = SEARCH_DOCUMENT_PREFIX if purpose == "document" else SEARCH_QUERY_PREFIX
    return prefix + text
