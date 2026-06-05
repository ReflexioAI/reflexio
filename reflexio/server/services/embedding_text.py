"""Canonical text and prefixes used for vector embeddings."""

from __future__ import annotations

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
    text: str, *, purpose: Literal["document", "query"] = "document"
) -> str:
    """Apply the asymmetric search prefix used before embedding calls.

    ``purpose`` must be ``"document"`` or ``"query"``; any other value raises so a
    misspelled call site fails fast instead of silently writing or searching the
    wrong vector space.
    """
    if purpose == "document":
        return SEARCH_DOCUMENT_PREFIX + text
    if purpose == "query":
        return SEARCH_QUERY_PREFIX + text
    raise ValueError(
        f"Unknown embedding purpose {purpose!r}; expected 'document' or 'query'"
    )
