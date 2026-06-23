"""Async information tools for resumable extraction runs."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.llm.tools import (
    AsyncAccepted,
    AsyncInfoTool,
    AsyncRequestSpec,
    Tool,
)
from reflexio.server.services.storage.storage_base import (
    BaseStorage,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)
from reflexio.server.usage_metrics import record_usage_event

logger = logging.getLogger(__name__)


class AskHumanArgs(StrictStructuredOutput):
    """Ask Agent Builder for missing extraction information and continue now."""

    question: Annotated[str, Field(min_length=1)]
    answer_format: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [tag.strip() for tag in value.split(",") if tag.strip()]
        return value


class AttachPendingInfoRequestArgs(StrictStructuredOutput):
    """Attach this run to a relevant pending Prior Knowledge request."""

    pending_tool_call_id: Annotated[str, Field(min_length=1)]
    why_relevant: str | None = None


class PendingToolCallDispatcher:
    """Dispatch newly-created pending tool calls to an external channel."""

    def dispatch(self, record: PendingToolCallRecord) -> None:
        raise NotImplementedError


class NoopPendingToolCallDispatcher(PendingToolCallDispatcher):
    """Local/test dispatcher that intentionally does nothing."""

    def dispatch(self, _record: PendingToolCallRecord) -> None:
        return None


@dataclass(slots=True)
class PendingToolCallToolContext:
    storage: BaseStorage
    run_id: str
    org_id: str
    extractor_kind: str
    user_id: str | None = None
    config: PendingToolCallConfig = field(default_factory=PendingToolCallConfig)
    dispatcher: PendingToolCallDispatcher = field(
        default_factory=NoopPendingToolCallDispatcher
    )


def _build_async_request_spec(
    *,
    args: AskHumanArgs,
    ctx: PendingToolCallToolContext,
) -> AsyncRequestSpec:
    tool_config = ctx.config.for_tool("ask_human")
    scope = human_feedback_scope(ctx.org_id)
    return AsyncRequestSpec(
        tool_name="ask_human",
        dedup_key=build_pending_tool_call_dedup_key(
            tool_name="ask_human",
            question_text=args.question,
            answer_format=args.answer_format,
        ),
        scope=scope,
        question_text=args.question,
        answer_format=args.answer_format,
        args={
            "question": args.question,
            "answer_format": args.answer_format,
        },
        tags=args.tags,
        cache_until_seconds=tool_config.dedup_cache_seconds,
        valid_until_seconds=tool_config.prior_answer_valid_seconds,
    )


def _question_embedding(
    storage: BaseStorage,
    question_text: str,
) -> list[float] | None:
    get_embedding = getattr(storage, "_get_embedding", None)
    if not callable(get_embedding):
        return None
    try:
        embedding = get_embedding(question_text, purpose="document")
    except TypeError:
        embedding = get_embedding(question_text)
    except Exception as exc:  # pragma: no cover - backend logs details elsewhere
        logger.warning(
            "event=pending_tool_call_embedding_failed error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return None
    return embedding if isinstance(embedding, list) else None


def handle_ask_human(
    args: AskHumanArgs,
    ctx: PendingToolCallToolContext,
) -> AsyncAccepted:
    if not ctx.run_id:
        raise ValueError("ask_human requires a durable agent run_id")

    current = datetime.now(UTC)
    spec = _build_async_request_spec(args=args, ctx=ctx)
    tool_config = ctx.config.for_tool(spec.tool_name)
    # Observational soft cap only: intentionally read OUTSIDE the create/attach
    # transaction. It never blocks or fails extraction (a small count race here
    # would at most over/under-log the warning by one), so it does not need the
    # per-scope transaction boundary that the dedup insert/attach uses.
    unresolved_count = ctx.storage.count_unresolved_followup_dependencies(
        org_id=ctx.org_id,
        extractor_kind=ctx.extractor_kind,
        tool_name=spec.tool_name,
    )
    if unresolved_count >= tool_config.max_pending_followups_per_scope:
        logger.warning(
            "event=pending_followup_soft_cap_exceeded org_id=%s extractor_kind=%s "
            "tool_name=%s current_count=%d soft_cap=%d",
            ctx.org_id,
            ctx.extractor_kind,
            spec.tool_name,
            unresolved_count,
            tool_config.max_pending_followups_per_scope,
        )

    record = PendingToolCallRecord(
        id=f"ptc_{uuid.uuid4().hex}",
        org_id=ctx.org_id,
        user_id=ctx.user_id,
        scope=spec.scope,
        scope_hash=build_scope_hash(spec.scope),
        tool_name=spec.tool_name,
        dedup_key=spec.dedup_key,
        status=PendingToolCallStatus.PENDING,
        question_text=spec.question_text,
        args=spec.args,
        tags=spec.tags,
        answer_format=spec.answer_format,
        embedding=_question_embedding(ctx.storage, spec.question_text),
        expires_at=current + timedelta(seconds=tool_config.pending_ttl_seconds),
        cache_until=current + timedelta(seconds=spec.cache_until_seconds),
    )
    upsert = ctx.storage.create_or_attach_pending_tool_call(
        record=record,
        dependency=RunToolDependencyRecord(
            run_id=ctx.run_id,
            pending_tool_call_id=record.id,
        ),
        now=current,
    )
    record_usage_event(
        org_id=ctx.org_id,
        event_name="extraction_agent_async_info_requested",
        event_category="extraction_agent",
        user_id=ctx.user_id,
        pipeline=ctx.extractor_kind,
        outcome="created" if upsert.created else "deduped",
        metadata={
            "run_id": ctx.run_id,
            "pending_tool_call_id": upsert.pending_tool_call.id,
            "tool_name": upsert.pending_tool_call.tool_name,
        },
    )
    if upsert.created:
        ctx.dispatcher.dispatch(upsert.pending_tool_call)
        logger.info(
            "event=pending_tool_call_created org_id=%s user_id=%s "
            "extractor_kind=%s run_id=%s pending_tool_call_id=%s "
            "tool_name=%s",
            ctx.org_id,
            ctx.user_id,
            ctx.extractor_kind,
            ctx.run_id,
            upsert.pending_tool_call.id,
            upsert.pending_tool_call.tool_name,
        )
    else:
        logger.info(
            "event=extraction_agent_followup_attached org_id=%s user_id=%s "
            "extractor_kind=%s run_id=%s pending_tool_call_id=%s "
            "tool_name=%s deduped=true",
            ctx.org_id,
            ctx.user_id,
            ctx.extractor_kind,
            ctx.run_id,
            upsert.pending_tool_call.id,
            upsert.pending_tool_call.tool_name,
        )

    return AsyncAccepted(
        pending_tool_call_id=upsert.pending_tool_call.id,
        result={
            "status": "request_pending",
            "pending_tool_call_id": upsert.pending_tool_call.id,
            "instruction": (
                "Continue extraction using currently available evidence. "
                "This run will be resumed when the answer arrives."
            ),
        },
    )


def handle_attach_pending_info_request(
    args: AttachPendingInfoRequestArgs,
    ctx: PendingToolCallToolContext,
) -> dict[str, Any]:
    if not ctx.run_id:
        raise ValueError("attach_pending_info_request requires a durable agent run_id")

    record = ctx.storage.get_pending_tool_call(args.pending_tool_call_id)
    expected_scope = human_feedback_scope(ctx.org_id)
    current = datetime.now(UTC)
    if (
        record is None
        or record.org_id != ctx.org_id
        or record.tool_name != "ask_human"
        or record.scope_hash != build_scope_hash(expected_scope)
    ):
        return {
            "status": "not_found",
            "instruction": (
                "The referenced Prior Knowledge request is unavailable. "
                "Continue extraction, or call ask_human if the information is still needed."
            ),
        }

    if record.status == PendingToolCallStatus.RESOLVED:
        if record.valid_until is not None and record.valid_until <= current:
            return {
                "status": "expired",
                "instruction": (
                    "The referenced answer is no longer valid. Continue extraction, "
                    "or call ask_human if the information is still needed."
                ),
            }
        return {
            "status": "already_resolved",
            "pending_tool_call_id": record.id,
            "question": record.question_text,
            "result": record.result or {},
            "instruction": (
                "Use this result only if it is relevant to the current extraction."
            ),
        }

    if record.status != PendingToolCallStatus.PENDING or (
        record.expires_at is not None and record.expires_at <= current
    ):
        return {
            "status": "unavailable",
            "instruction": (
                "The referenced request is no longer pending. Continue extraction, "
                "or call ask_human if the information is still needed."
            ),
        }

    ctx.storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id=ctx.run_id,
            pending_tool_call_id=record.id,
        )
    )
    logger.info(
        "event=extraction_agent_followup_attached org_id=%s user_id=%s "
        "extractor_kind=%s run_id=%s pending_tool_call_id=%s "
        "tool_name=%s",
        ctx.org_id,
        ctx.user_id,
        ctx.extractor_kind,
        ctx.run_id,
        record.id,
        record.tool_name,
    )
    return {
        "status": "attached_for_followup",
        "pending_tool_call_id": record.id,
        "instruction": (
            "Continue extraction using currently available evidence. "
            "This run will be resumed when the pending answer arrives."
        ),
    }


def _handle_ask_human_tool(args: BaseModel, ctx: Any) -> AsyncAccepted:
    if not isinstance(args, AskHumanArgs):
        raise TypeError(f"Expected AskHumanArgs, got {type(args).__name__}")
    ctx = getattr(ctx, "extra_tool_context", ctx)
    if not isinstance(ctx, PendingToolCallToolContext):
        raise TypeError(
            f"Expected PendingToolCallToolContext, got {type(ctx).__name__}"
        )
    return handle_ask_human(args, ctx)


def _handle_attach_pending_info_request_tool(
    args: BaseModel, ctx: Any
) -> dict[str, Any]:
    if not isinstance(args, AttachPendingInfoRequestArgs):
        raise TypeError(
            f"Expected AttachPendingInfoRequestArgs, got {type(args).__name__}"
        )
    ctx = getattr(ctx, "extra_tool_context", ctx)
    if not isinstance(ctx, PendingToolCallToolContext):
        raise TypeError(
            f"Expected PendingToolCallToolContext, got {type(ctx).__name__}"
        )
    return handle_attach_pending_info_request(args, ctx)


def create_ask_human_tool() -> AsyncInfoTool:
    return AsyncInfoTool(
        name="ask_human",
        args_model=AskHumanArgs,
        handler=_handle_ask_human_tool,
    )


def create_attach_pending_info_request_tool() -> Tool:
    return Tool(
        name="attach_pending_info_request",
        args_model=AttachPendingInfoRequestArgs,
        handler=_handle_attach_pending_info_request_tool,
    )
