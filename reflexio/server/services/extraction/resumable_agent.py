"""Finish-tool runner for resumable classic extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import Tool, ToolLoopTrace, ToolRegistry, run_tool_loop
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
)
from reflexio.server.services.extraction.pending_tool_call_dispatch import (
    PendingToolCallToolContext,
    create_ask_human_tool,
    create_attach_pending_info_request_tool,
)
from reflexio.server.services.extraction.prior_answer_search import (
    append_prior_knowledge_context,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    BaseStorage,
    PendingToolCallRecord,
)
from reflexio.server.site_var.feature_flags import (
    is_resumable_extraction_agent_enabled as is_resumable_extraction_agent_feature_enabled,
)
from reflexio.server.usage_metrics import record_usage_event

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext

logger = logging.getLogger(__name__)

FINISH_EXTRACTION_TOOL_NAME = "finish_extraction"
PROFILE_EXTRACTOR_KIND = "profile"


def _record_agent_usage_event(
    *,
    run: AgentRunRecord,
    event_name: str,
    outcome: str | None = None,
    error_kind: str | None = None,
    count_value: int = 1,
    metadata: dict[str, Any] | None = None,
) -> None:
    record_usage_event(
        org_id=run.binding.org_id,
        event_name=event_name,
        event_category="extraction_agent",
        user_id=run.binding.user_id,
        request_id=run.binding.request_id,
        pipeline=run.binding.extractor_kind,
        source=run.binding.source,
        agent_version=run.binding.agent_version,
        outcome=outcome,
        error_kind=error_kind,
        count_value=count_value,
        metadata={"run_id": run.id, **(metadata or {})},
    )


@dataclass(slots=True)
class AgentRunResult:
    run_id: str
    output: BaseModel | None
    pending_tool_call_ids: list[str]
    messages: list[dict[str, Any]]
    trace: ToolLoopTrace
    finished_reason: str


@dataclass(slots=True)
class _FinishExtractionContext:
    output: BaseModel | None = None


@dataclass(slots=True)
class _ExtractionAgentToolContext:
    finish_context: _FinishExtractionContext
    extra_tool_context: Any | None = None


def _format_resolved_tool_result(record: PendingToolCallRecord) -> str:
    resolved_at = record.resolved_at.isoformat() if record.resolved_at else "unknown"
    return (
        "Resolved tool result for extraction follow-up\n"
        f"Tool: {record.tool_name}\n"
        f"Question: {record.question_text}\n"
        f"Resolved at: {resolved_at}\n"
        f"Result: {record.result or {}}\n\n"
        "Use this Agent Builder feedback only if it is relevant to the "
        "current extraction window. If it adds or corrects durable profile or "
        "playbook information, include that in finish_extraction."
    )


def append_resolved_tool_result_context(
    messages: list[dict[str, Any]],
    resolved_tool_calls: list[PendingToolCallRecord],
) -> list[dict[str, Any]]:
    """Append resolved async tool results as user-role Agent Builder context."""
    ordered = sorted(
        resolved_tool_calls,
        key=lambda record: record.resolved_at or datetime.max.replace(tzinfo=UTC),
    )
    return [
        *messages,
        *[
            {"role": "user", "content": _format_resolved_tool_result(record)}
            for record in ordered
        ],
    ]


def _finish_handler(args: BaseModel, ctx: Any) -> dict[str, Any]:
    ctx = getattr(ctx, "finish_context", ctx)
    if not isinstance(ctx, _FinishExtractionContext):
        raise TypeError(f"Expected _FinishExtractionContext, got {type(ctx).__name__}")
    ctx.output = args
    return {"status": "completed"}


def create_finish_extraction_tool(output_schema: type[BaseModel]) -> Tool:
    return Tool(
        name=FINISH_EXTRACTION_TOOL_NAME,
        args_model=output_schema,
        handler=_finish_handler,
    )


def _pending_tool_call_config(request_context: RequestContext) -> Any | None:
    root_config = request_context.configurator.get_config()
    return (
        getattr(root_config, "pending_tool_call_config", None)
        if root_config is not None
        else None
    )


def pending_tool_calls_enabled(request_context: RequestContext) -> bool:
    """Gate whether pending-info tools are offered.

    This does NOT gate whether the extraction loop runs — the loop is always
    the extraction path. It only governs whether the resumable human-in-the-loop
    and prior-knowledge tools may be registered alongside ``finish_extraction``.
    """
    pending_config = _pending_tool_call_config(request_context)
    return bool(
        pending_config
        and pending_config.enabled
        and is_resumable_extraction_agent_feature_enabled(request_context.org_id)
        and request_context.storage is not None
    )


def create_pending_info_tools_for_extractor_kind(extractor_kind: str) -> list[Tool]:
    """Return pending-info tools available to a given extractor kind."""
    attach_tool = create_attach_pending_info_request_tool()
    if extractor_kind == PROFILE_EXTRACTOR_KIND:
        return [attach_tool]
    return [create_ask_human_tool(), attach_tool]


def run_resumable_extraction_agent(
    *,
    request_context: RequestContext,
    client: LiteLLMClient,
    extractor_kind: str,
    user_id: str | None,
    request_id: str,
    agent_version: str | None,
    source: str | None,
    request_interaction_data_models: list[RequestInteractionDataModel],
    extractor_config: BaseModel,
    service_config: Any,
    agent_context: str,
    messages: list[dict[str, Any]],
    output_schema: type[BaseModel],
    log_label: str,
) -> AgentRunResult:
    """Run and finalize a config-gated classic extraction agent pass."""
    pending_config = _pending_tool_call_config(request_context)
    storage = request_context.storage
    if storage is None:
        raise RuntimeError(f"Resumable {extractor_kind} extraction requires storage")

    pending_tools_active = pending_tool_calls_enabled(request_context)

    run = build_extractor_agent_run_record(
        org_id=request_context.org_id,
        extractor_kind=extractor_kind,
        user_id=user_id,
        request_id=request_id,
        agent_version=agent_version,
        source=source,
        request_interaction_data_models=request_interaction_data_models,
        extractor_config=extractor_config,
        service_config=service_config,
        agent_context=agent_context,
    )
    extra_tools: list[Tool] = []
    extra_tool_context = None
    if pending_tools_active and pending_config is not None:
        messages = append_prior_knowledge_context(
            messages=messages,
            storage=storage,
            org_id=request_context.org_id,
            extractor_kind=extractor_kind,
            extractor_config=extractor_config,
            source=source,
            agent_version=agent_version,
            similarity_threshold=pending_config.for_tool(
                "attach_pending_info_request"
                if extractor_kind == PROFILE_EXTRACTOR_KIND
                else "ask_human"
            ).similarity_threshold,
        )
        extra_tool_context = PendingToolCallToolContext(
            storage=storage,
            run_id=run.id,
            org_id=request_context.org_id,
            extractor_kind=extractor_kind,
            user_id=user_id,
            config=pending_config,
        )
        extra_tools.extend(create_pending_info_tools_for_extractor_kind(extractor_kind))

    return ResumableExtractionAgent(client=client, storage=storage).start(
        run=run,
        messages=messages,
        output_schema=output_schema,
        extra_tools=extra_tools,
        extra_tool_context=extra_tool_context,
        log_label=log_label,
    )


class ResumableExtractionAgent:
    """Run a classic extractor prompt through a durable finish-tool loop."""

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        storage: BaseStorage,
        max_steps: int = 8,
        model_role: ModelRole = ModelRole.EXTRACTION_AGENT,
    ) -> None:
        self.client = client
        self.storage = storage
        self.max_steps = max_steps
        self.model_role = model_role

    def start(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        extra_tools: list[Tool] | None = None,
        extra_tool_context: Any | None = None,
        log_label: str | None = None,
    ) -> AgentRunResult:
        """Create the run row, execute the tool loop, and store completed output."""
        run = replace(
            run,
            max_steps_remaining=(
                self.max_steps
                if run.max_steps_remaining is None
                else min(run.max_steps_remaining, self.max_steps)
            ),
        )
        self.storage.create_agent_run(run)
        logger.info(
            "event=extraction_agent_started org_id=%s user_id=%s extractor_kind=%s "
            "run_id=%s request_id=%s",
            run.binding.org_id,
            run.binding.user_id,
            run.binding.extractor_kind,
            run.id,
            run.binding.request_id,
        )
        _record_agent_usage_event(run=run, event_name="extraction_agent_started")
        return self._run(
            run=run,
            messages=messages,
            output_schema=output_schema,
            extra_tools=extra_tools,
            extra_tool_context=extra_tool_context,
            log_label=log_label,
        )

    def resume(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        resolved_tool_calls: list[PendingToolCallRecord],
        extra_tools: list[Tool] | None = None,
        extra_tool_context: Any | None = None,
        log_label: str | None = None,
    ) -> AgentRunResult:
        """Resume a claimed run with resolved async tool results in context."""
        logger.info(
            "event=extraction_agent_resumed org_id=%s user_id=%s extractor_kind=%s "
            "run_id=%s request_id=%s resolved_tool_calls=%d",
            run.binding.org_id,
            run.binding.user_id,
            run.binding.extractor_kind,
            run.id,
            run.binding.request_id,
            len(resolved_tool_calls),
        )
        _record_agent_usage_event(
            run=run,
            event_name="extraction_agent_resumed",
            count_value=len(resolved_tool_calls),
            metadata={"resolved_tool_calls": len(resolved_tool_calls)},
        )
        resumed_messages = append_resolved_tool_result_context(
            messages,
            resolved_tool_calls,
        )
        return self._run(
            run=run,
            messages=resumed_messages,
            output_schema=output_schema,
            extra_tools=extra_tools,
            extra_tool_context=extra_tool_context,
            log_label=log_label,
        )

    def _run(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        extra_tools: list[Tool] | None = None,
        extra_tool_context: Any | None = None,
        log_label: str | None = None,
    ) -> AgentRunResult:
        finish_ctx = _FinishExtractionContext()
        ctx: Any = (
            _ExtractionAgentToolContext(
                finish_context=finish_ctx,
                extra_tool_context=extra_tool_context,
            )
            if extra_tool_context is not None
            else finish_ctx
        )
        registry = ToolRegistry(
            [*(extra_tools or []), create_finish_extraction_tool(output_schema)]
        )
        max_steps = self.max_steps
        if run.max_steps_remaining is not None:
            max_steps = min(max_steps, max(0, run.max_steps_remaining))

        # When only finish_extraction is registered (no human/prior-knowledge
        # tools), force that tool so the degenerate one-call loop is strictly
        # equivalent to the old single-shot response_format extraction —
        # the model cannot return a no-tool turn and yield empty output.
        #
        # When the async-info tools are also registered we cannot force a single
        # tool (the model must be free to pick ask_human vs finish_extraction),
        # but we still require *some* tool call via "required". This prevents a
        # weak tool-caller (e.g. MiniMax) from emitting the answer as plain text
        # with zero tool_calls, which the loop would otherwise treat as a no-op
        # finish and drop the output.
        tool_choice: str | dict[str, Any] = (
            {
                "type": "function",
                "function": {"name": FINISH_EXTRACTION_TOOL_NAME},
            }
            if not extra_tools
            else "required"
        )

        result = run_tool_loop(
            client=self.client,
            messages=messages,
            registry=registry,
            model_role=self.model_role,
            max_steps=max_steps,
            ctx=ctx,
            finish_tool_name=FINISH_EXTRACTION_TOOL_NAME,
            tool_choice=tool_choice,
            log_label=log_label,
        )

        committed_output = (
            finish_ctx.output.model_dump() if finish_ctx.output is not None else None
        )
        if result.finished_reason == "finish_tool" and committed_output is not None:
            self.storage.update_agent_run_status(
                run.id,
                AgentRunStatus.AGENT_COMPLETED,
                committed_output=committed_output,
                pending_tool_call_ids=result.pending_tool_call_ids,
                max_steps_remaining=result.max_steps_remaining,
            )
            logger.info(
                "event=extraction_agent_finished org_id=%s user_id=%s "
                "extractor_kind=%s run_id=%s request_id=%s "
                "pending_tool_calls=%d",
                run.binding.org_id,
                run.binding.user_id,
                run.binding.extractor_kind,
                run.id,
                run.binding.request_id,
                len(result.pending_tool_call_ids),
            )
            _record_agent_usage_event(
                run=run,
                event_name="extraction_agent_finished",
                outcome="completed",
                metadata={
                    "pending_tool_calls": len(result.pending_tool_call_ids),
                    "finished_reason": result.finished_reason,
                },
            )
        else:
            last_error = f"Extraction agent did not finish: {result.finished_reason}"
            self.storage.update_agent_run_status(
                run.id,
                AgentRunStatus.FAILED,
                max_steps_remaining=result.max_steps_remaining,
                last_error=last_error,
            )
            logger.warning(
                "event=extraction_agent_failed org_id=%s user_id=%s "
                "extractor_kind=%s run_id=%s request_id=%s "
                "finished_reason=%s has_output=%s",
                run.binding.org_id,
                run.binding.user_id,
                run.binding.extractor_kind,
                run.id,
                run.binding.request_id,
                result.finished_reason,
                finish_ctx.output is not None,
            )
            _record_agent_usage_event(
                run=run,
                event_name="extraction_agent_failed",
                outcome="failed",
                error_kind=result.finished_reason,
                metadata={"has_output": finish_ctx.output is not None},
            )

        return AgentRunResult(
            run_id=run.id,
            output=finish_ctx.output,
            pending_tool_call_ids=result.pending_tool_call_ids,
            messages=result.messages,
            trace=result.trace,
            finished_reason=result.finished_reason,
        )
