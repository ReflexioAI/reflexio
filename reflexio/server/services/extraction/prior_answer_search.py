"""Prior Knowledge context for resumable extraction agents."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from reflexio.server.services.storage.storage_base import (
    BaseStorage,
    PendingToolCallStatus,
    PriorAnswerMatch,
    build_scope_hash,
    canonical_json,
    human_feedback_scope,
)

logger = logging.getLogger(__name__)

ASK_HUMAN_TOOL_NAME = "ask_human"
MAX_KNOWLEDGE_QUERY_CHARS = 4_000
MAX_RESULT_CHARS = 1_200


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def build_knowledge_need_query(
    *,
    extractor_kind: str,
    extractor_name: str,
    extractor_config: Any,
    source: str | None,
    agent_version: str | None,
    messages: list[dict[str, Any]],
) -> str:
    """Build a compact question-like query for prior human requests."""
    definition = getattr(extractor_config, "extraction_definition_prompt", "") or ""
    recent_messages = "\n".join(
        _content_to_text(message.get("content")) for message in messages[-4:]
    )
    query = "\n".join(
        [
            f"Extractor kind: {extractor_kind}",
            f"Extractor name: {extractor_name}",
            f"Source: {source or 'unknown'}",
            f"Agent version: {agent_version or 'unknown'}",
            f"Extraction definition: {definition}",
            "Recent extraction context:",
            recent_messages,
            "Find prior human clarification questions or answers that may help this extraction.",
        ]
    )
    return query[:MAX_KNOWLEDGE_QUERY_CHARS]


def _query_embedding(storage: BaseStorage, query: str) -> list[float] | None:
    get_embedding = getattr(storage, "_get_embedding", None)
    if not callable(get_embedding):
        return None
    try:
        embedding = get_embedding(query, purpose="query")
    except TypeError:
        embedding = get_embedding(query)
    except Exception as exc:  # pragma: no cover - backend logs details elsewhere
        logger.warning(
            "event=prior_answer_query_embedding_failed error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return None
    return embedding if isinstance(embedding, list) else None


def find_prior_answer_matches(
    *,
    storage: BaseStorage,
    org_id: str,
    knowledge_need_query: str,
    limit: int = 8,
    similarity_threshold: float = 0.0,
    exclude_pending_tool_call_ids: set[str] | None = None,
) -> list[PriorAnswerMatch]:
    scope = human_feedback_scope(org_id)
    query_embedding = _query_embedding(storage, knowledge_need_query)
    matches = storage.search_prior_tool_calls(
        org_id=org_id,
        scope_hash=build_scope_hash(scope),
        tool_name=ASK_HUMAN_TOOL_NAME,
        query_embedding=query_embedding,
        now=datetime.now(UTC),
        limit=limit,
    )
    excluded = exclude_pending_tool_call_ids or set()
    if excluded:
        # On resume, the run's own just-resolved calls are injected separately
        # as explicit "Agent Builder provided additional information" messages;
        # excluding them here avoids surfacing the same answer twice.
        matches = [
            match for match in matches if match.pending_tool_call_id not in excluded
        ]
    if similarity_threshold <= 0.0:
        return matches
    return [
        match
        for match in matches
        if match.similarity is None or match.similarity >= similarity_threshold
    ]


def _format_datetime(value: datetime | None) -> str:
    return value.date().isoformat() if value else "unknown"


def _format_result(result: dict[str, Any] | None) -> str:
    text = canonical_json(result or {})
    if len(text) > MAX_RESULT_CHARS:
        return f"{text[:MAX_RESULT_CHARS]}..."
    return text


def format_prior_knowledge_context(matches: list[PriorAnswerMatch]) -> str | None:
    """Return a seed-context block, or None when there is no prior knowledge."""
    if not matches:
        return None

    resolved = [
        match for match in matches if match.status == PendingToolCallStatus.RESOLVED
    ]
    pending = [
        match for match in matches if match.status == PendingToolCallStatus.PENDING
    ]
    lines = [
        "Prior Knowledge for org-scoped human feedback",
        "Use these entries only if they are relevant to the current extraction window.",
    ]
    if resolved:
        lines.append("")
        lines.append("Resolved answers:")
        for match in resolved:
            lines.extend(
                [
                    f"- source_request_id: {match.pending_tool_call_id}",
                    f"  question: {match.question_text}",
                    f"  answer: {_format_result(match.result)}",
                    f"  resolved_at: {_format_datetime(match.resolved_at)}",
                    f"  valid_until: {_format_datetime(match.valid_until)}",
                ]
            )
    if pending:
        lines.append("")
        lines.append(
            "Pending requests: only call attach_pending_info_request with one of "
            "the pending_tool_call_id values in this section when the pending request "
            "is relevant and this run should resume when answered."
        )
        for match in pending:
            lines.extend(
                [
                    f"- pending_tool_call_id: {match.pending_tool_call_id}",
                    f"  question: {match.question_text}",
                    f"  answer_format: {match.answer_format or 'unspecified'}",
                    f"  expires_at: {_format_datetime(match.expires_at)}",
                ]
            )
    return "\n".join(lines)


def append_prior_knowledge_context(
    *,
    messages: list[dict[str, Any]],
    storage: BaseStorage,
    org_id: str,
    extractor_kind: str,
    extractor_name: str,
    extractor_config: Any,
    source: str | None,
    agent_version: str | None,
    limit: int = 8,
    similarity_threshold: float = 0.0,
    exclude_pending_tool_call_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    query = build_knowledge_need_query(
        extractor_kind=extractor_kind,
        extractor_name=extractor_name,
        extractor_config=extractor_config,
        source=source,
        agent_version=agent_version,
        messages=messages,
    )
    matches = find_prior_answer_matches(
        storage=storage,
        org_id=org_id,
        knowledge_need_query=query,
        limit=limit,
        similarity_threshold=similarity_threshold,
        exclude_pending_tool_call_ids=exclude_pending_tool_call_ids,
    )
    context = format_prior_knowledge_context(matches)
    if context is None:
        return messages
    logger.info(
        "event=prior_answer_injected org_id=%s extractor_kind=%s extractor_name=%s count=%d",
        org_id,
        extractor_kind,
        extractor_name,
        len(matches),
    )
    return [*messages, {"role": "user", "content": context}]
