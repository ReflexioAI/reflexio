"""Canonical text and model-specific policy used for vector embeddings."""

from __future__ import annotations

import re

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Interaction,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.llm.embedding_policy import (
    E5_DOCUMENT_PREFIX as E5_DOCUMENT_PREFIX,
)
from reflexio.server.llm.embedding_policy import (
    E5_QUERY_PREFIX as E5_QUERY_PREFIX,
)
from reflexio.server.llm.embedding_policy import (
    MULTILINGUAL_E5_MODEL as MULTILINGUAL_E5_MODEL,
)
from reflexio.server.llm.embedding_policy import (
    SEARCH_DOCUMENT_PREFIX as SEARCH_DOCUMENT_PREFIX,
)
from reflexio.server.llm.embedding_policy import (
    SEARCH_QUERY_PREFIX as SEARCH_QUERY_PREFIX,
)
from reflexio.server.llm.embedding_policy import (
    EmbeddingModelPolicy as EmbeddingModelPolicy,
)
from reflexio.server.llm.embedding_policy import (
    embedding_input as embedding_input,
)
from reflexio.server.llm.embedding_policy import (
    is_multilingual_e5_model as is_multilingual_e5_model,
)
from reflexio.server.llm.embedding_policy import (
    resolve_clustering_similarity as resolve_clustering_similarity,
)
from reflexio.server.llm.embedding_policy import (
    resolve_retrieval_threshold as resolve_retrieval_threshold,
)

EmbeddingTextEntity = (
    Interaction
    | UserProfile
    | UserPlaybook
    | AgentPlaybook
    | AgentSuccessEvaluationResult
)

_PLAYBOOK_REQUEST_SCAFFOLDING = re.compile(
    r"^\s*(?:(?:when|whenever|if)\s+)?"
    r"(?:a|an|the)?\s*"
    r"(?:user|customer|client|person|learner|agent|assistant)\s+"
    r"(?:(?:is|was)\s+asked|asks?|requests?|wants?|needs?|submits?|provides?|"
    r"seeks?|tries|attempts?)\b\s*"
    r"(?:(?:for|to)\s+|(?:the\s+)?(?:agent|assistant|system)\s+to\s+)?",
    re.IGNORECASE,
)
_PLAYBOOK_LOW_SIGNAL_WORDS = re.compile(
    r"(?<![-'])\b(?:a|an|the|(?i:please|kindly))\b(?![-'])",
)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


def playbook_trigger_embedding_text(trigger: str | None) -> str:
    """Return a trigger-only, low-noise playbook embedding input.

    Only generic request scaffolding and low-signal function words are removed.
    Negation, sequencing, entities, and domain terms remain intact.
    """
    if not trigger:
        return ""
    text = _PLAYBOOK_REQUEST_SCAFFOLDING.sub("", trigger, count=1)
    text = _PLAYBOOK_LOW_SIGNAL_WORDS.sub(" ", text)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return " ".join(text.split())


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
        return playbook_trigger_embedding_text(entity.trigger)
    if isinstance(entity, AgentSuccessEvaluationResult):
        return " ".join(
            part
            for part in (entity.failure_type, entity.failure_reason)
            if part and part.strip()
        )
    raise TypeError(f"Unsupported embedding text entity: {type(entity).__name__}")
