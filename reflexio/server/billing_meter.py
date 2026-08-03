"""OSS emission helpers: translate billing signals into usage_events.

The single source of truth for each billing event's name/category/fields. Plain
primitive signatures (no reflexio_ext types) so OSS call sites — the generation
service and search endpoints — can use them directly. Each is a thin, non-blocking
wrapper over the ``record_usage_event`` hook (which only enqueues). No DB I/O.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from typing import Any

from reflexio.server.usage_metrics import record_usage_event

logger = logging.getLogger(__name__)

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

    No-op when ``billing_input_tokens <= 0``. Each call mints a fresh
    ``event_key=f"tok:{uuid4()}"`` so two token emits under the same
    ``request_id`` (e.g. profile + playbook extraction in one request) never
    collapse into one billed event downstream.

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
        event_key=f"tok:{uuid.uuid4()}",
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
    user_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    source: str | None = None,
    agent_version: str | None = None,
    playbook_name: str | None = None,
    entity_type: str | None = None,
    event_key: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit the Learning value facet — number of profiles/playbooks generated.

    Intended for online extraction paths that have a known billable count but
    do not retain a complete per-record id list. Resumable finalization must
    use :func:`record_learnings_generated_records` and skip items without
    durable ids. No-op when ``count <= 0``.

    Emits a single event carrying ``event_key`` when the caller has a durable
    retry identity, otherwise synthesizes ``f"learn-batch:{uuid4()}"``. This
    gives retryable callers an idempotent aggregate event without forcing an
    unstable key on event-moment callers.

    Args:
        org_id: Organisation identifier.
        count: Number of learnings generated in this run.
        platform_llm: True iff the platform supplies the LLM for this org.
        platform_storage: True iff the platform supplies storage; None defers to rollup.
        pipeline: Optional pipeline tag (e.g. ``"playbook"``).
        user_id: Optional user ID tied to the generated learning.
        request_id: Optional request correlation ID.
        session_id: Optional session ID.
        source: Optional metering source/path label.
        agent_version: Optional agent version tied to the generated learning.
        playbook_name: Optional playbook name for playbook learnings.
        entity_type: Optional entity type (e.g. ``"profile"``).
        event_key: Optional caller-supplied, retry-stable event key.
        metadata: Optional path-specific usage metadata.
    """
    if count <= 0:
        return
    record_usage_event(
        org_id=org_id,
        event_name="learnings_generated",
        event_category="learning",
        pipeline=pipeline,
        user_id=user_id,
        request_id=request_id,
        session_id=session_id,
        source=source,
        agent_version=agent_version,
        playbook_name=playbook_name,
        entity_type=entity_type,
        event_key=event_key or f"learn-batch:{uuid.uuid4()}",
        count_value=count,
        platform_llm=platform_llm,
        platform_storage=platform_storage,
        caller_type=_INTERNAL,
        metadata=metadata,
    )


def record_learnings_generated_records(
    *,
    org_id: str,
    learning_ids: list[str],
    platform_llm: bool | None,
    platform_storage: bool | None,
    pipeline: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    source: str | None = None,
    agent_version: str | None = None,
    playbook_name: str | None = None,
    entity_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Emit the Learning value facet — one event per generated learning record.

    Entity-backed alternative to :func:`record_learnings_generated`: emits one
    ``learnings_generated`` event per id in ``learning_ids`` (``count_value=1``,
    ``event_key=f"learn:{entity_type}:{id}"``, ``entity_id=id``) instead of a
    single aggregate event, so downstream dedup can key on the learning id.
    The ``entity_type`` segment is required for collision-freedom: entity-backed
    callers draw ids from separate autoincrement primary keys in separate
    tables (e.g. ``user_playbook_id`` and ``agent_playbook_id`` both start at
    1), so the same integer id can legitimately occur in two tables — without
    the entity-type segment those would mint the same ``event_key`` and
    collapse into one event downstream. When ``entity_type`` is falsy, a
    stable ``"_"`` placeholder is used (``learn:_:{id}``) rather than emitting
    ``entity_type=None`` literally into the key. The summed ``count_value``
    across the emitted events equals ``len(learning_ids)`` — unchanged from
    the total a caller would have passed as ``count`` to
    :func:`record_learnings_generated`.

    Callers must pass real, durable ids — never fabricate one to pad the
    list. No-op when ``learning_ids`` is empty.

    Args:
        org_id: Organisation identifier.
        learning_ids: Ids of the learnings durably generated in this run
            (e.g. ``profile_id`` / ``user_playbook_id``).
        platform_llm: True iff the platform supplies the LLM for this org.
        platform_storage: True iff the platform supplies storage; None defers to rollup.
        pipeline: Optional pipeline tag (e.g. ``"playbook"``).
        user_id: Optional user ID tied to the generated learning.
        request_id: Optional request correlation ID.
        session_id: Optional session ID.
        source: Optional metering source/path label.
        agent_version: Optional agent version tied to the generated learning.
        playbook_name: Optional playbook name for playbook learnings.
        entity_type: Optional entity type (e.g. ``"profile"``).
        metadata: Optional path-specific usage metadata (shared across events).
    """
    key_entity_type = entity_type or "_"
    for learning_id in learning_ids:
        record_usage_event(
            org_id=org_id,
            event_name="learnings_generated",
            event_category="learning",
            pipeline=pipeline,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            source=source,
            agent_version=agent_version,
            playbook_name=playbook_name,
            entity_type=entity_type,
            entity_id=learning_id,
            event_key=f"learn:{key_entity_type}:{learning_id}",
            count_value=1,
            platform_llm=platform_llm,
            platform_storage=platform_storage,
            caller_type=_INTERNAL,
            metadata=metadata,
        )


def emit_learnings_generated(
    *,
    org_id: str,
    configurator: Any,
    count: int,
    source: str,
    pipeline: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    agent_version: str | None = None,
    playbook_name: str | None = None,
    entity_type: str | None = None,
    event_key: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Resolve ``platform_llm`` from config and emit the Learning value facet.

    Convenience wrapper for count-based online extraction callers. It owns the
    ``configurator.get_config()`` + ``platform_llm_from_config`` lookup so the
    call site stays a thin one-liner, and — critically — is **guarded**: the
    product path must never fail because metering failed, so config resolution
    and emission are wrapped and any exception is logged and swallowed
    (mirroring the extraction path's ``_record_billing_learning_events``).
    Resumable finalization must use :func:`emit_learnings_generated_records`.
    No-op when ``count <= 0``.

    Args:
        org_id: Organisation identifier.
        configurator: Object exposing ``get_config()`` for platform-LLM resolution.
        count: Number of learnings durably produced by this path.
        source: Metering source/path label (e.g. ``"online_extraction"``).
        pipeline: Optional pipeline tag (e.g. ``"playbook"``).
        user_id: Optional user ID tied to the generated learning.
        request_id: Optional request correlation ID.
        agent_version: Optional agent version tied to the generated learning.
        playbook_name: Optional playbook name for playbook learnings.
        entity_type: Optional entity type (e.g. ``"profile"``).
        event_key: Optional caller-supplied, retry-stable event key.
        metadata: Optional path-specific usage metadata.
    """
    if count <= 0:
        return
    try:
        from reflexio.server.billing_signals import platform_llm_from_config

        config = configurator.get_config()
        record_learnings_generated(
            org_id=org_id,
            count=count,
            platform_llm=platform_llm_from_config(config),
            platform_storage=None,
            pipeline=pipeline,
            user_id=user_id,
            request_id=request_id,
            source=source,
            agent_version=agent_version,
            playbook_name=playbook_name,
            entity_type=entity_type,
            event_key=event_key,
            metadata=metadata,
        )
    except Exception:
        logger.warning(
            "emit_learnings_generated failed for source=%s org=%s; "
            "learnings_generated event not emitted",
            source,
            org_id,
            exc_info=True,
        )


def emit_learnings_generated_records(
    *,
    org_id: str,
    configurator: Any,
    learning_ids: list[str],
    source: str,
    pipeline: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    agent_version: str | None = None,
    playbook_name: str | None = None,
    entity_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Resolve ``platform_llm`` from config and emit one event per learning id.

    Entity-backed counterpart to :func:`emit_learnings_generated`, used by
    resumable-extraction finalization for every created user learning with a
    durable id. Items without ids are not billable on that path. Online
    extraction uses the count-based :func:`record_learnings_generated` helper
    because it does not retain a safe 1:1 id per generated unit. Same guard
    semantics: config resolution and emission are wrapped and any exception is
    logged and swallowed — the product path must never fail because metering
    failed. No-op when ``learning_ids`` is empty.

    Args:
        org_id: Organisation identifier.
        configurator: Object exposing ``get_config()`` for platform-LLM resolution.
        learning_ids: Ids of the learnings durably produced by this path.
        source: Metering source/path label (e.g. ``"resumable_extraction"``).
        pipeline: Optional pipeline tag (e.g. ``"playbook"``).
        user_id: Optional user ID tied to the generated learning.
        request_id: Optional request correlation ID.
        agent_version: Optional agent version tied to the generated learning.
        playbook_name: Optional playbook name for playbook learnings.
        entity_type: Optional entity type (e.g. ``"profile"``).
        metadata: Optional path-specific usage metadata (shared across events).
    """
    if not learning_ids:
        return
    try:
        from reflexio.server.billing_signals import platform_llm_from_config

        config = configurator.get_config()
        record_learnings_generated_records(
            org_id=org_id,
            learning_ids=learning_ids,
            platform_llm=platform_llm_from_config(config),
            platform_storage=None,
            pipeline=pipeline,
            user_id=user_id,
            request_id=request_id,
            source=source,
            agent_version=agent_version,
            playbook_name=playbook_name,
            entity_type=entity_type,
            metadata=metadata,
        )
    except Exception:
        logger.warning(
            "emit_learnings_generated_records failed for source=%s org=%s; "
            "learnings_generated events not emitted",
            source,
            org_id,
            exc_info=True,
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
    Each call mints a fresh ``event_key=f"applied:{uuid4()}"`` — a distinct
    key per search-response moment, never collapsed by ``request_id``.

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
        event_key=f"applied:{uuid.uuid4()}",
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
    this measures requests made, not learnings surfaced. Each call mints a
    fresh ``event_key=f"search:{uuid4()}"`` — a distinct key per request, so
    two searches under the same ``request_id`` are never collapsed into one
    billed event downstream.

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
        event_key=f"search:{uuid.uuid4()}",
        count_value=1,
        caller_type=caller_type,
    )
