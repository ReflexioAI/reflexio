"""OSS emission helpers: translate billing signals into usage_events.

The single source of truth for each billing event's name/category/fields. Plain
primitive signatures (no reflexio_ext types) so OSS call sites — the generation
service and search endpoints — can use them directly. Each is a thin, non-blocking
wrapper over the ``record_usage_event`` hook (which only enqueues). No DB I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reflexio.server.billing_signals import count_input_tokens
from reflexio.server.usage_metrics import record_usage_event

_INTERNAL = "internal"   # == BillingCallerType.INTERNAL.value (kept literal; OSS stays clean)


def record_extraction_tokens(
    *,
    org_id: str,
    billing_input_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    platform_llm: bool | None,
    platform_storage: bool | None,
    pipeline: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Emit the Learning cost facet — call only when extraction fired.

    No-op when ``billing_input_tokens <= 0``.

    Args:
        org_id: Organisation identifier.
        billing_input_tokens: Input-anchored token count (the metered basis).
        prompt_tokens: Real provider prompt tokens (COGS; not billed to customer).
        completion_tokens: Real provider completion tokens (COGS; not billed).
        platform_llm: True iff the platform supplies the LLM for this org.
        platform_storage: True iff the platform supplies storage; None defers to rollup.
        pipeline: Optional pipeline tag (e.g. ``"profile"``).
        request_id: Optional request correlation ID.
        session_id: Optional session ID.
    """
    if billing_input_tokens <= 0:
        return
    record_usage_event(
        org_id=org_id,
        event_name="extraction_tokens",
        event_category="learning",
        pipeline=pipeline,
        request_id=request_id,
        session_id=session_id,
        count_value=billing_input_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        billing_input_tokens=billing_input_tokens,
        platform_llm=platform_llm,
        platform_storage=platform_storage,
        caller_type=_INTERNAL,
    )


def record_learnings_generated(
    *,
    org_id: str,
    count: int,
    platform_llm: bool | None,
    platform_storage: bool | None,
    pipeline: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Emit the Learning value facet — number of profiles/playbooks generated.

    No-op when ``count <= 0``.

    Args:
        org_id: Organisation identifier.
        count: Number of learnings generated in this run.
        platform_llm: True iff the platform supplies the LLM for this org.
        platform_storage: True iff the platform supplies storage; None defers to rollup.
        pipeline: Optional pipeline tag (e.g. ``"playbook"``).
        request_id: Optional request correlation ID.
        session_id: Optional session ID.
    """
    if count <= 0:
        return
    record_usage_event(
        org_id=org_id,
        event_name="learnings_generated",
        event_category="learning",
        pipeline=pipeline,
        request_id=request_id,
        session_id=session_id,
        count_value=count,
        platform_llm=platform_llm,
        platform_storage=platform_storage,
        caller_type=_INTERNAL,
    )


def record_applied_learnings(
    *,
    org_id: str,
    surfaced_count: int,
    caller_type: str,
    platform_llm: bool | None,
    platform_storage: bool | None,
    pipeline: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Emit the Application line — surfaced top-K learnings.

    No-op unless ``caller_type == "production_agent"`` AND ``surfaced_count > 0``.

    Args:
        org_id: Organisation identifier.
        surfaced_count: Number of learnings surfaced in the search response.
        caller_type: Caller classification string (e.g. ``"production_agent"``).
        platform_llm: True iff the platform supplies the LLM for this org.
        platform_storage: True iff the platform supplies storage; None defers to rollup.
        pipeline: Optional pipeline tag.
        request_id: Optional request correlation ID.
        session_id: Optional session ID.
    """
    if caller_type != "production_agent" or surfaced_count <= 0:
        return
    record_usage_event(
        org_id=org_id,
        event_name="learning_applied",
        event_category="application",
        pipeline=pipeline,
        request_id=request_id,
        session_id=session_id,
        count_value=surfaced_count,
        platform_llm=platform_llm,
        platform_storage=platform_storage,
        caller_type=caller_type,
    )


def record_injection_events(
    *,
    org_id: str,
    caller_type: str,
    entities: Sequence[tuple[str, str]],
    pipeline: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    contents: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit one ``learning_injection`` event per surfaced entity.

    Channel-agnostic observability surface: per-entity rows land in the
    ``usage_events`` table regardless of which channel adapter (claude-smart,
    Codex, custom) drove the search. Aggregated by the storage layer's
    :meth:`ExtrasMixin.get_injection_stats` and the
    ``POST /api/get_injection_stats`` endpoint.

    Distinct from :func:`record_applied_learnings` (one row per search,
    billing-oriented). Per-entity rows are observability-oriented and
    answer "what was actually rendered into the context window?"

    No-op unless ``caller_type == "production_agent"`` AND ``entities`` is
    non-empty. Token cost is computed per-entity via
    :func:`reflexio.server.billing_signals.count_input_tokens` when
    ``contents`` is supplied; when omitted, ``prompt_tokens`` is 0.

    Args:
        org_id: Organisation identifier.
        caller_type: Caller classification (e.g. ``"production_agent"``).
        entities: Sequence of ``(entity_type, entity_id)`` pairs, one per
            surfaced entity. ``entity_type`` is ``"user_playbook"``,
            ``"agent_playbook"`` or ``"profile"``; ``entity_id`` is the
            storage id (string-encoded).
        pipeline: Optional pipeline tag (e.g. ``"unified_search"``).
        request_id: Optional request correlation id.
        session_id: Optional session id.
        contents: Optional parallel list of content strings, one per
            entity. When supplied, ``prompt_tokens`` is set to the
            ``cl100k_base`` token count of each content string.
        metadata: Optional free-form metadata applied to every emitted row.

    Returns:
        None. Side-effect only.
    """
    if caller_type != "production_agent" or not entities:
        return
    for idx, (entity_type, entity_id) in enumerate(entities):
        content = ""
        if contents is not None and idx < len(contents):
            content = contents[idx] or ""
        prompt_tokens = count_input_tokens(content) if content else 0
        record_usage_event(
            org_id=org_id,
            event_name="learning_injection",
            event_category="application",
            pipeline=pipeline,
            request_id=request_id,
            session_id=session_id,
            entity_type=entity_type,
            entity_id=str(entity_id),
            count_value=1,
            prompt_tokens=prompt_tokens,
            caller_type=caller_type,
            metadata=metadata,
        )
