"""OSS emission helpers: translate billing signals into usage_events.

The single source of truth for each billing event's name/category/fields. Plain
primitive signatures (no reflexio_ext types) so OSS call sites — the generation
service and search endpoints — can use them directly. Each is a thin, non-blocking
wrapper over the ``record_usage_event`` hook (which only enqueues). No DB I/O.
"""

from __future__ import annotations

from reflexio.server.usage_metrics import record_usage_event

_INTERNAL = (
    "internal"  # == BillingCallerType.INTERNAL.value (kept literal; OSS stays clean)
)


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


def record_search_request(
    *,
    org_id: str,
    caller_type: str,
    request_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Emit one analytics-only search request count for production agents.

    No-op unless ``caller_type == "production_agent"``. Unlike
    :func:`record_applied_learnings`, empty search responses still count because
    this measures requests made, not learnings surfaced.

    Args:
        org_id: Organisation identifier.
        caller_type: Caller classification string.
        request_id: Optional request correlation ID.
        session_id: Optional session ID.
    """
    if caller_type != "production_agent":
        return
    record_usage_event(
        org_id=org_id,
        event_name="search_request",
        event_category="application",
        request_id=request_id,
        session_id=session_id,
        count_value=1,
        caller_type=caller_type,
    )
