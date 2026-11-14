"""Finish-tool runner for resumable classic extraction."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.compute_admission import (
    compute_attempt_scope,
    record_compute_result,
    reserve_compute_attempt,
    retain_compute_result,
)
from reflexio.server.llm.failure import OutageWait, Transient, classify_llm_failure
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    InputBudgetExceededError,
    model_input_budget,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import (
    Tool,
    ToolLoopTrace,
    ToolRegistry,
    format_failure_label,
    run_tool_loop,
)
from reflexio.server.services.deferred_learning_plan import ExtractorBookmarkPosition
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
    frozen_extractor_config_identity,
    matches_frozen_extractor_binding,
    matches_frozen_window_binding,
)
from reflexio.server.services.extraction.answer_inputs import (
    answer_input_scope,
    snapshot_answer_inputs,
)
from reflexio.server.services.extraction.history_inputs import (
    HistoricalInputs,
    require_historical_inputs,
)
from reflexio.server.services.extraction.overflow_answers import (
    answer_witness,
    pin_extraction_answer,
)
from reflexio.server.services.extraction.pending_tool_admission import (
    pending_tool_calls_disabled_reason as pending_tool_calls_disabled_reason,
)
from reflexio.server.services.extraction.pending_tool_admission import (
    pending_tool_calls_enabled,
)
from reflexio.server.services.extraction.pending_tool_call_dispatch import (
    PendingToolCallToolContext,
    bind_pending_tool_execution,
    create_ask_human_tool,
    create_attach_pending_info_request_tool,
)
from reflexio.server.services.extraction.prior_answer_search import (
    append_prior_knowledge_context,
)
from reflexio.server.services.extraction.publication_receipts import PublicationOrigin
from reflexio.server.services.extraction.source_inputs import (
    capture_source_inputs,
    require_source_inputs,
)
from reflexio.server.services.profile.committed_output import (
    read_saved_structured_output,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    BaseStorage,
    PendingToolCallRecord,
    build_scope_hash,
)
from reflexio.server.usage_metrics import record_usage_event

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.extraction.overflow_pipeline import (
        OverflowExtractionCall,
    )
    from reflexio.server.services.extraction.package_compression import (
        CompressedSourcePackage,
    )
    from reflexio.server.services.extraction.profile_package_preparation import (
        ProfilePackagePreparation,
    )

logger = logging.getLogger(__name__)

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


class ExtractionIncompleteError(RuntimeError):
    """An extraction agent ended without structured output.

    ``error_detail`` is the tool loop's failure classification (``transient``,
    ``non_transient``, or a repair ``failure_kind``), kept structured so a
    caller choosing a log level does not parse it back out of the message.
    """

    def __init__(self, message: str, *, error_detail: str | None) -> None:
        super().__init__(message)
        self.error_detail = error_detail


@dataclass(slots=True)
class AgentRunResult:
    run_id: str
    output: BaseModel | None
    pending_tool_call_ids: list[str]
    messages: list[dict[str, Any]]
    trace: ToolLoopTrace
    finished_reason: str
    model_provenance: ModelProvenance | None = None
    # Cause of a finished_reason == "error" run (class name and kind only; see
    # ToolLoopResult.error_type). Carried so the extractor's failure message
    # names the cause instead of a bare "error".
    error_type: str | None = None
    error_detail: str | None = None
    execution_run: AgentRunRecord | None = field(default=None, repr=False)
    # Runtime-only cause of an "error" run, for chaining; never persisted.
    error: BaseException | None = field(default=None, repr=False)
    # The run was left RESUMING to wait out a provider outage. Decided once,
    # here: the circuit may close before the worker looks again.
    provider_wait: bool = False

    @property
    def failure_label(self) -> str:
        """``finished_reason`` plus the error cause, safe for messages and logs."""
        return format_failure_label(
            self.finished_reason, self.error_type, self.error_detail
        )


def encode_committed_output(
    output: BaseModel, provenance: ModelProvenance | None
) -> dict[str, Any]:
    """Persist output with provenance while old raw payloads remain readable."""
    return {
        "_reflexio_envelope_version": 1,
        "output": output.model_dump(),
        "model_provenance": asdict(provenance) if provenance is not None else None,
    }


def decode_committed_output(
    committed_output: dict[str, Any],
) -> tuple[dict[str, Any], ModelProvenance | None]:
    """Read the new envelope or a pre-provenance raw structured payload."""
    if "_reflexio_envelope_version" not in committed_output:
        return committed_output, None
    version = committed_output["_reflexio_envelope_version"]
    if version != 1:
        raise ValueError(f"Unsupported committed output envelope version: {version!r}")
    output = committed_output.get("output")
    if not isinstance(output, dict):
        raise ValueError(
            "Corrupt v1 committed output envelope: output must be an object"
        )
    raw_provenance = committed_output.get("model_provenance")
    if raw_provenance is None:
        return output, None
    if not isinstance(raw_provenance, dict):
        raise ValueError(
            "Corrupt v1 committed output envelope: model_provenance must be an object or null"
        )
    try:
        return output, ModelProvenance(**raw_provenance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Corrupt v1 committed output envelope: invalid model_provenance"
        ) from exc


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
        "playbook information, include that in your structured extraction result."
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


def _pending_tool_call_config(request_context: RequestContext) -> Any | None:
    root_config = request_context.configurator.get_config()
    return (
        getattr(root_config, "pending_tool_call_config", None)
        if root_config is not None
        else None
    )


def create_pending_info_tools_for_extractor_kind(extractor_kind: str) -> list[Tool]:
    """Return pending-info tools available to a given extractor kind."""
    attach_tool = create_attach_pending_info_request_tool()
    if extractor_kind == PROFILE_EXTRACTOR_KIND:
        return [attach_tool]
    return [create_ask_human_tool(), attach_tool]


def _require_overflow_parent(
    storage: BaseStorage, coordinator: OverflowCoordinator, run: AgentRunRecord
) -> AgentRunRecord:
    existing = storage.get_agent_run(run.id)
    if (
        run.id != f"window:{coordinator.window.window_id}"
        or existing is None
        or not matches_frozen_window_binding(
            existing,
            run,
            coordinator.window.policy["extractor"],
            project_id=coordinator.window.project_id,
        )
    ):
        raise RuntimeError("Overflow extraction child parent binding changed")
    return existing


def run_resumable_extraction_agent(
    *,
    request_context: RequestContext,
    client: LiteLLMClient,
    extractor_kind: str,
    user_id: str,
    request_id: str,
    agent_version: str | None,
    source: str | None,
    request_interaction_data_models: list[RequestInteractionDataModel],
    extractor_config: BaseModel,
    service_config: Any,
    agent_context: str,
    messages: list[dict[str, Any]],
    history_inputs: HistoricalInputs | None,
    output_schema: type[BaseModel],
    log_label: str,
    structured_output_validator: (Callable[[BaseModel], Sequence[str]] | None) = None,
    overflow_call_factory: Callable[[], OverflowExtractionCall] | None = None,
    overflow_final_validator: Callable[[BaseModel], Sequence[str]] | None = None,
) -> AgentRunResult:
    """Run and finalize a config-gated classic extraction agent pass."""
    from reflexio.server.services.extraction.overflow_coordinator import (
        current_overflow_coordinator,
    )
    from reflexio.server.services.extraction.overflow_presentation import (
        current_overflow_presentation,
    )

    coordinator = current_overflow_coordinator.get()
    if coordinator is not None and current_overflow_presentation.get() is None:
        if overflow_call_factory is None:
            raise InputBudgetConfigurationError(
                "Extraction requires a prepared-source call factory"
            )
        from reflexio.server.services.extraction.overflow_pipeline import (
            extract_overflow_window,
        )

        def execute_child(call: OverflowExtractionCall) -> AgentRunResult:
            return run_resumable_extraction_agent(
                request_context=request_context,
                client=client,
                extractor_kind=extractor_kind,
                user_id=user_id,
                request_id=request_id,
                agent_version=agent_version,
                source=source,
                request_interaction_data_models=request_interaction_data_models,
                extractor_config=extractor_config,
                service_config=service_config,
                agent_context=agent_context,
                messages=call.messages,
                history_inputs=None,  # Actual protected parent owns its historical checkpoint.
                output_schema=call.schema,
                log_label=log_label,
                structured_output_validator=call.validator,
            )

        return extract_overflow_window(
            coordinator=coordinator,
            client=client,
            build_call=overflow_call_factory,
            execute_call=execute_child,
            final_validator=overflow_final_validator,
        )

    generation_request_id = request_id
    pending_config = _pending_tool_call_config(request_context)
    storage = request_context.storage
    if storage is None:
        raise RuntimeError(f"Resumable {extractor_kind} extraction requires storage")

    pending_tools_active = pending_tool_calls_enabled(request_context)

    run = build_extractor_agent_run_record(
        org_id=request_context.org_id,
        extractor_kind=extractor_kind,
        user_id=user_id,
        generation_request_id=generation_request_id,
        agent_version=agent_version,
        source=source,
        request_interaction_data_models=request_interaction_data_models,
        extractor_config=extractor_config,
        service_config=service_config,
        agent_context=agent_context,
        output_schema_name=output_schema.__name__,
    )
    window_id = getattr(service_config, "extraction_window_id", None)
    if window_id is not None:
        run = replace(run, id=f"window:{window_id}")
    prepare_messages: Callable[[], list[dict[str, Any]]] | None = None
    extra_tools: list[Tool] = []
    extra_tool_context = None
    if pending_tools_active and pending_config is not None:
        config_identity = None
        if coordinator is not None:
            parent = _require_overflow_parent(storage, coordinator, run)
            # Preserve the v1 lookup identity after frozen binding validation,
            # including when a stable snapshot layout makes the hashes equal.
            frozen = parent.generation_request_snapshot["extractor_config"]
            config_identity = frozen_extractor_config_identity(extractor_config, frozen)
        prepare_messages = partial(
            append_prior_knowledge_context,
            messages=messages,
            storage=storage,
            org_id=request_context.org_id,
            extractor_kind=extractor_kind,
            extractor_config=extractor_config,
            extractor_config_identity=config_identity,
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
        publication_origin="initial_window"
        if window_id is not None
        else "initial_inline",
        bookmark_position=ExtractorBookmarkPosition(
            extractor_kind,
            [
                item.interaction_id
                for group in request_interaction_data_models
                for item in group.interactions
            ],
            max(
                (
                    item.created_at
                    for group in request_interaction_data_models
                    for item in group.interactions
                    if item.created_at is not None
                ),
                default=None,
            ),
            user_id,
        )
        if window_id is None
        else None,
        source_inputs=(
            capture_source_inputs(storage, run, request_interaction_data_models)
            if coordinator is None
            else None
        ),
        frozen_extractor_config=getattr(
            service_config, "frozen_extractor_config", None
        ),
        messages=messages,
        history_inputs=history_inputs,
        prepare_messages=prepare_messages,
        output_schema=output_schema,
        extra_tools=extra_tools,
        extra_tool_context=extra_tool_context,
        log_label=log_label,
        structured_output_validator=structured_output_validator,
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

    def start_profile_package(
        self,
        *,
        run: AgentRunRecord,
        preparation: ProfilePackagePreparation,
        package: CompressedSourcePackage,
        coordinator: OverflowCoordinator,
        extra_tools: list[Tool],
        extra_tool_context: Any,
        cache_only: bool = False,
    ) -> AgentRunResult:
        """Consume a verified package through the existing durable one-turn loop."""
        from reflexio.server.services.extraction.profile_package_consumer import (
            validate_profile_package,
        )
        from reflexio.server.services.profile.components.extractor import (
            _profile_evidence_validation_errors,
        )
        from reflexio.server.services.profile.profile_generation_service_utils import (
            StructuredProfilesOutput,
        )

        witness = validate_profile_package(preparation, package, coordinator)
        if self.max_steps != preparation.snapshot["agent_max_steps"]:
            raise InputBudgetConfigurationError("Profile package step limit changed")
        if coordinator.checkpoint.get("package_processing") != witness:
            raise InputBudgetConfigurationError("Profile package was not processed")
        if [tool.openai_spec() for tool in extra_tools] != preparation.snapshot[
            "tools"
        ]:
            raise InputBudgetConfigurationError("Profile package tools changed")
        with coordinator.provider_scope():
            return self._start_overflow_step(
                run=run,
                messages=preparation.agent_messages(package.presentation.text),
                output_schema=StructuredProfilesOutput,
                extra_tools=extra_tools,
                extra_tool_context=extra_tool_context,
                log_label=None,
                structured_output_validator=lambda output: (
                    _profile_evidence_validation_errors(
                        output, package.presentation.visible_spans
                    )
                ),
                package_request=(preparation, package),
                cache_only=cache_only,
            )

    def start(
        self,
        *,
        run: AgentRunRecord,
        publication_origin: PublicationOrigin = "initial_inline",
        bookmark_position: ExtractorBookmarkPosition | None = None,
        frozen_extractor_config: dict[str, Any] | None = None,
        messages: list[dict[str, Any]],
        history_inputs: HistoricalInputs | None,
        output_schema: type[BaseModel],
        source_inputs: dict[str, Any] | None = None,
        extra_tools: list[Tool] | None = None,
        extra_tool_context: Any | None = None,
        log_label: str | None = None,
        prepare_messages: Callable[[], list[dict[str, Any]]] | None = None,
        structured_output_validator: (
            Callable[[BaseModel], Sequence[str]] | None
        ) = None,
    ) -> AgentRunResult:
        """Create the run row, execute the tool loop, and store completed output."""
        from reflexio.server.services.extraction.overflow_coordinator import (
            current_overflow_coordinator,
        )
        from reflexio.server.services.extraction.overflow_presentation import (
            current_overflow_presentation,
        )

        coordinator = current_overflow_coordinator.get()
        if coordinator is not None and current_overflow_presentation.get() is not None:
            if prepare_messages is not None:
                messages = prepare_messages()
            return self._start_overflow_step(
                run=run,
                messages=messages,
                output_schema=output_schema,
                extra_tools=extra_tools,
                extra_tool_context=extra_tool_context,
                log_label=log_label,
                structured_output_validator=structured_output_validator,
            )
        from reflexio.server.services.durable_learning.user_lease import (
            adopt_agent_run_for_execution,
            agent_run_user_work,
            create_owned_agent_run,
            owned_agent_run_write,
        )
        from reflexio.server.services.storage.storage_base import LeaseLostError

        with (
            answer_input_scope(self.storage, run.id),
            agent_run_user_work(self.storage, run),
        ):
            run = replace(
                run,
                max_steps_remaining=(
                    self.max_steps
                    if run.max_steps_remaining is None
                    else min(run.max_steps_remaining, self.max_steps)
                ),
            )
            existing = (
                self.storage.get_agent_run(run.id)
                if run.id.startswith("window:")
                else None
            )
            fresh_budget = run.max_steps_remaining
            if existing is not None:
                if existing.binding != run.binding:
                    if (
                        frozen_extractor_config is None
                        or not matches_frozen_extractor_binding(
                            existing, run, frozen_extractor_config
                        )
                    ):
                        prior_fields = asdict(existing.binding)
                        current_fields = asdict(run.binding)
                        logging.getLogger("reflexio.ops").warning(
                            "event=extraction_binding_rejected run_id=%s changed_fields=%s",
                            existing.id,
                            ",".join(
                                name
                                for name in prior_fields
                                if prior_fields[name] != current_fields[name]
                            ),
                        )
                        raise RuntimeError("Extraction window agent binding changed")
                    logging.getLogger("reflexio.ops").info(
                        "event=extraction_binding_schema_compatible run_id=%s",
                        existing.id,
                    )
                # Keep the admitted hash, snapshots and pending dependencies for
                # both cache replay and failed/running continuation.
                existing = adopt_agent_run_for_execution(self.storage, existing)
                run = existing
                if existing.committed_output is not None:
                    payload, provenance = decode_committed_output(
                        existing.committed_output
                    )
                    return AgentRunResult(
                        run_id=existing.id,
                        output=read_saved_structured_output(output_schema, payload),
                        pending_tool_call_ids=existing.pending_tool_call_ids,
                        messages=[],
                        trace=ToolLoopTrace.model_validate(
                            existing.committed_output.get("trace", {"finished": True})
                        ),
                        finished_reason="structured_output",
                        model_provenance=provenance,
                        execution_run=existing,
                    )
                if existing.status not in (
                    AgentRunStatus.RUNNING,
                    AgentRunStatus.FAILED,
                ):
                    raise RuntimeError("Extraction window agent awaits resolution")
            require_source_inputs(source_inputs, run)
            if prepare_messages is not None:
                messages = prepare_messages()
            if existing is not None:
                # No durable output exists: a crash may require repeating a remote
                # call. Reuse its stable run identity; the first saved output wins.
                # The step budget is NOT continuation state here: this path keeps no
                # conversation, so the attempt restarts from the seed prompt. A
                # failed (or crashed) attempt stores 0 remaining, and reusing it
                # would run zero model turns and fail with max_steps on every retry.
                run = replace(run, max_steps_remaining=fresh_budget)
                with owned_agent_run_write(self.storage, run):
                    updated = self.storage.update_agent_run_status(
                        run.id,
                        AgentRunStatus.RUNNING,
                        max_steps_remaining=fresh_budget,
                        expected_statuses=(
                            AgentRunStatus.RUNNING,
                            AgentRunStatus.FAILED,
                        ),
                        expected_claimed_by=run.claimed_by,
                        expected_claimed_at=run.claimed_at,
                    )
                    if updated is None:
                        raise LeaseLostError(
                            "Extraction restart lost its adopted revision"
                        )
                    run = updated
            else:
                run = create_owned_agent_run(self.storage, run)
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
                publication_origin=publication_origin,
                bookmark_position=bookmark_position,
                source_inputs=source_inputs,
                messages=messages,
                history_inputs=history_inputs,
                output_schema=output_schema,
                dependency_inputs={},
                extra_tools=extra_tools,
                extra_tool_context=extra_tool_context,
                log_label=log_label,
                structured_output_validator=structured_output_validator,
            )

    def _start_overflow_step(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        extra_tools: list[Tool] | None,
        extra_tool_context: Any | None,
        log_label: str | None,
        structured_output_validator: Callable[[BaseModel], Sequence[str]] | None,
        package_request: tuple[ProfilePackagePreparation, CompressedSourcePackage]
        | None = None,
        cache_only: bool = False,
    ) -> AgentRunResult:
        """Stage a child receipt; only the full window owns parent completion."""
        from reflexio.server.services.extraction.overflow_coordinator import (
            current_overflow_coordinator,
        )
        from reflexio.server.services.extraction.overflow_presentation import (
            current_overflow_presentation,
        )

        coordinator = current_overflow_coordinator.get()
        legacy_presentation = current_overflow_presentation.get()
        if package_request is not None:
            presentation = package_request[1].presentation
            source_digest = presentation.source_digest
        else:
            if legacy_presentation is None:
                raise RuntimeError("Overflow extraction presentation is unavailable")
            presentation = legacy_presentation
            source_digest = legacy_presentation.prepared.source_digest
        if (
            coordinator is None
            or presentation is None
            or source_digest != coordinator.source_digest
        ):
            raise RuntimeError("Overflow extraction presentation binding changed")
        if not presentation.is_present_in(messages):
            raise InputBudgetConfigurationError(
                "Extraction template omits prepared source"
            )
        _require_overflow_parent(self.storage, coordinator, run)
        checkpoint = coordinator.checkpoint
        selected_schema = checkpoint.get("output_schema_name")
        if selected_schema is not None and selected_schema != output_schema.__name__:
            raise InputBudgetConfigurationError(
                "Overflow extraction output schema changed"
            )
        if selected_schema is None:
            if cache_only:
                raise InputBudgetConfigurationError("Profile package schema is missing")
            checkpoint["output_schema_name"] = output_schema.__name__
            coordinator.save_checkpoint(checkpoint)
        registry = ToolRegistry(list(extra_tools or []))
        key = coordinator.step_key(
            "extract-package-v1" if package_request is not None else "extract-v1",
            {
                "messages": messages,
                "schema": output_schema.model_json_schema(),
                "tools": [tool.openai_spec() for tool in (extra_tools or [])],
            },
        )
        if package_request is None:
            if legacy_presentation is None:
                raise RuntimeError("Overflow extraction presentation is unavailable")
            coordinator.register_extraction_child(legacy_presentation.chunk.text, key)
        else:
            checkpoint = coordinator.checkpoint
            child = {
                "key": key,
                "package_digest": package_request[1].presentation.package_digest,
                "contract_digest": package_request[0].contract_digest,
            }
            previous = checkpoint.get("package_child")
            if previous is not None and previous != child:
                raise InputBudgetConfigurationError("Profile package child changed")
            if previous is None:
                if cache_only:
                    raise InputBudgetConfigurationError(
                        "Profile package child is missing"
                    )
                checkpoint["package_child"] = child
                coordinator.save_checkpoint(checkpoint)
        saved = coordinator.completed_step(key)
        if saved is not None:
            payload = saved["output"]
            if package_request is not None:
                coordinator.validate_profile_attempts(
                    "profile_agent", payload.get("attempt_ids"), key
                )
            output = read_saved_structured_output(output_schema, payload["output"])
            if (
                package_request is not None
                and structured_output_validator is not None
                and structured_output_validator(output)
            ):
                raise InputBudgetConfigurationError("Saved profile quotes changed")
            return AgentRunResult(
                run_id=run.id,
                output=output,
                pending_tool_call_ids=[],
                messages=[],
                trace=ToolLoopTrace.model_validate(payload["trace"]),
                finished_reason="structured_output",
                model_provenance=ModelProvenance(**saved["model_provenance"]),
                execution_run=coordinator.saved_run,
            )
        if cache_only:
            raise InputBudgetConfigurationError("Profile package child is unfinished")
        return self._run_overflow_child(
            coordinator=coordinator,
            run=run,
            key=key,
            messages=messages,
            output_schema=output_schema,
            registry=registry,
            extra_tool_context=extra_tool_context,
            log_label=log_label,
            structured_output_validator=structured_output_validator,
            preparation=package_request[0] if package_request is not None else None,
        )

    def _run_overflow_child(
        self,
        *,
        coordinator: OverflowCoordinator,
        run: AgentRunRecord,
        key: str,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        registry: ToolRegistry,
        extra_tool_context: Any,
        log_label: str | None,
        structured_output_validator: Callable[[BaseModel], Sequence[str]] | None,
        preparation: ProfilePackagePreparation | None,
    ) -> AgentRunResult:
        """One shared durable loop for legacy chunks and source packages."""
        # A pending child is resumed by the coordinator's protocol, never by
        # restarting its seed prompt and attaching another human request.
        from reflexio.server.services.extraction.overflow_answers import (
            resume_waiting_child,
        )
        from reflexio.server.services.extraction.overflow_coordinator import (
            OverflowSuspendedError,
        )

        resume_waiting_child(coordinator, key)
        active = coordinator.checkpoint.get("active_child")
        if active is not None and active["key"] != key:
            raise InputBudgetConfigurationError(
                "Overflow extraction active child binding changed"
            )
        remaining = active["max_steps_remaining"] if active else self.max_steps
        current_messages = active["messages"] if active else messages
        child_attempts = active.get("attempt_ids") if active is not None else []
        if preparation is not None and active is not None:
            coordinator.validate_profile_attempts("profile_agent", child_attempts, key)
        if preparation is not None and (
            type(remaining) is not int
            or not 0 <= remaining <= self.max_steps
            or current_messages[: len(messages)] != messages
        ):
            raise InputBudgetConfigurationError("Profile package continuation changed")
        trace = (
            ToolLoopTrace.model_validate(active["trace"]) if active else ToolLoopTrace()
        )
        while remaining > 0:
            # One model turn at a time preserves synchronous tool progress before
            # a deadline yield; the original per-child step limit is not reset.
            with (
                coordinator.profile_request(
                    preparation, "agent", current_messages, receipt_key=key
                )
                if preparation is not None
                else nullcontext()
            ) as attempts:
                execution_run = coordinator.saved_run
                if execution_run is None:
                    raise InputBudgetConfigurationError(
                        "Pending tools require the admitted overflow parent"
                    )
                result = run_tool_loop(
                    client=self.client,
                    messages=current_messages,
                    registry=registry,
                    model_role=self.model_role,
                    max_steps=1,
                    ctx=bind_pending_tool_execution(
                        extra_tool_context,
                        storage=self.storage,
                        run=execution_run,
                    ),
                    response_format=output_schema,
                    structured_output_validator=structured_output_validator,
                    tool_choice="auto",
                    log_label=log_label,
                )
            if preparation is not None:
                if attempts is None:
                    raise InputBudgetConfigurationError(
                        "Profile turn lacks an attempt scope"
                    )
                child_attempts.extend(attempts)
            # A turn the provider never answered did no work, so it spends no
            # step: otherwise every attempt during an outage burns one and the
            # window is held once the child's steps run out. Answered dispatch
            # IDs are recorded below; one that failed during an outage already
            # returned its attempt (``OverflowCoordinator._settle_failure``).
            if not (
                result.finished_reason == "error"
                and result.error is not None
                and isinstance(
                    classify_llm_failure(result.error), OutageWait | Transient
                )
            ):
                remaining -= 1
            trace.turns.extend(result.trace.turns)
            trace.finished = result.trace.finished
            if (
                result.finished_reason == "structured_output"
                and result.structured_output is not None
                and not result.pending_tool_call_ids
            ):
                break
            checkpoint = coordinator.checkpoint
            checkpoint["active_child"] = {
                "key": key,
                "messages": result.messages,
                "max_steps_remaining": remaining,
                "output_schema_name": output_schema.__name__,
                "trace": trace.model_dump(mode="json"),
                **({"attempt_ids": child_attempts} if preparation is not None else {}),
            }
            if result.pending_tool_call_ids:
                checkpoint["waiting_child"] = {
                    **checkpoint["active_child"],
                    "pending_tool_call_ids": result.pending_tool_call_ids,
                }
            coordinator.save_checkpoint(checkpoint)
            if result.pending_tool_call_ids:
                raise OverflowSuspendedError("Extraction child awaits human/tool input")
            if result.finished_reason != "max_steps":
                raise ExtractionIncompleteError(
                    f"Overflow extraction child did not finish: {result.failure_label}",
                    error_detail=result.error_detail,
                ) from result.error
            current_messages = result.messages
        else:
            raise InputBudgetExceededError(
                "Overflow extraction child step budget exhausted"
            )
        if preparation is not None:
            coordinator.validate_profile_attempts("profile_agent", child_attempts, key)
        coordinator.complete_step(
            key,
            {
                "output": result.structured_output.model_dump(mode="json"),
                "trace": trace.model_dump(mode="json"),
                **({"attempt_ids": child_attempts} if preparation is not None else {}),
            },
            result.provenance or ModelProvenance(),
        )
        return AgentRunResult(
            run_id=run.id,
            output=result.structured_output,
            pending_tool_call_ids=[],
            messages=result.messages,
            trace=trace,
            finished_reason=result.finished_reason,
            model_provenance=result.provenance,
            execution_run=coordinator.saved_run,
        )

    def resume(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        history_inputs: HistoricalInputs | None,
        output_schema: type[BaseModel],
        source_inputs: dict[str, Any] | None = None,
        resolved_tool_calls: list[PendingToolCallRecord],
        before_model: Callable[[AgentRunRecord], AgentRunRecord] | None = None,
        extra_tools: list[Tool] | None = None,
        extra_tool_context: Any | None = None,
        log_label: str | None = None,
        structured_output_validator: (
            Callable[[BaseModel], Sequence[str]] | None
        ) = None,
    ) -> AgentRunResult:
        """Resume a claimed run with resolved async tool results in context."""
        from reflexio.server.services.durable_learning.user_lease import (
            adopt_agent_run_for_execution,
            agent_run_user_work,
        )

        with (
            answer_input_scope(self.storage, run.id),
            agent_run_user_work(self.storage, run),
        ):
            run = adopt_agent_run_for_execution(self.storage, run)
            # Refuse missing/incomplete source proof before charging an attempt.
            require_source_inputs(source_inputs, run)
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
            # Snapshot supplied inputs before provider work, rather than the
            # answers that happen to be resolved when that work completes.
            # Normalize full answer qualifiers without copying its text again.
            for record in resolved_tool_calls:
                pin_extraction_answer(record)
            dependency_inputs = {
                record.id: build_scope_hash(answer_witness(asdict(record)))
                for record in resolved_tool_calls
            }
            resumed_messages = append_resolved_tool_result_context(
                messages,
                resolved_tool_calls,
            )
            return self._run(
                run=run,
                publication_origin="resume",
                source_inputs=source_inputs,
                messages=resumed_messages,
                history_inputs=history_inputs,
                before_model=before_model,
                output_schema=output_schema,
                dependency_inputs=dependency_inputs,
                extra_tools=extra_tools,
                extra_tool_context=extra_tool_context,
                log_label=log_label,
                structured_output_validator=structured_output_validator,
            )

    @compute_attempt_scope()
    def _run(
        self,
        *,
        run: AgentRunRecord,
        publication_origin: PublicationOrigin = "initial_inline",
        bookmark_position: ExtractorBookmarkPosition | None = None,
        messages: list[dict[str, Any]],
        history_inputs: HistoricalInputs | None,
        output_schema: type[BaseModel],
        source_inputs: dict[str, Any] | None = None,
        dependency_inputs: dict[str, str],
        before_model: Callable[[AgentRunRecord], AgentRunRecord] | None = None,
        extra_tools: list[Tool] | None = None,
        extra_tool_context: Any | None = None,
        log_label: str | None = None,
        structured_output_validator: (
            Callable[[BaseModel], Sequence[str]] | None
        ) = None,
    ) -> AgentRunResult:
        captured_bookmark = deepcopy(bookmark_position)
        captured_sources = deepcopy(source_inputs)
        require_source_inputs(captured_sources, run)
        captured_history = require_historical_inputs(history_inputs)
        if captured_history.profiles or captured_history.playbooks:
            self.storage.validate_agent_history_inputs(
                run, captured_history.profiles, captured_history.playbooks
            )
        reserve_compute_attempt()
        if before_model is not None:
            run = before_model(run)
        max_steps = self.max_steps
        if run.max_steps_remaining is not None:
            max_steps = min(max_steps, max(0, run.max_steps_remaining))

        # The extractor delivers its result as a native structured response
        # (response_format=output_schema): a turn with no tool call IS the
        # committed output, parsed into output_schema. ask_human /
        # attach_pending_info_request stay real tools the model may call
        # mid-run, so tool_choice="auto" — forcing a tool would suppress the
        # structured finish and is what produced the no_tool_call failures.
        #
        # No retry loop: a plain (no-tool) turn is now the SUCCESS terminus
        # (finished_reason="structured_output"), not a dropped output, and the
        # client performs one configured malformed-output retry or corrective
        # repair turn. The
        # async-info tool handlers read ctx via getattr(ctx, "extra_tool_context",
        # ctx), so the bare context object can be passed directly.
        registry = ToolRegistry(list(extra_tools or []))
        extra_tool_context = bind_pending_tool_execution(
            extra_tool_context, storage=self.storage, run=run
        )
        with model_input_budget():
            result = run_tool_loop(
                client=self.client,
                messages=messages,
                registry=registry,
                model_role=self.model_role,
                max_steps=max_steps,
                ctx=extra_tool_context,
                response_format=output_schema,
                structured_output_validator=structured_output_validator,
                tool_choice="auto",
                log_label=log_label,
            )
        record_compute_result()

        output = result.structured_output
        committed_output = (
            encode_committed_output(output, result.provenance)
            if output is not None
            else None
        )
        if committed_output is not None:
            committed_output["trace"] = result.trace.model_dump(mode="json")
        active_statuses = (AgentRunStatus.RUNNING, AgentRunStatus.RESUMING)
        # A resume runs under a worker's claim. Fence its terminal writes on
        # that claim, not on status alone: once the run is requeued (claim TTL
        # expiry, or governed erasure withdrawing an answer this worker already
        # loaded) a NEW claim puts it back in RESUMING, and a status-only write
        # would let this stale worker overwrite it. A fenced write that matches
        # nothing returns None, which the late-output branch below discards.
        from reflexio.server.services.durable_learning.user_lease import (
            owned_agent_run_write,
        )
        from reflexio.server.services.storage.storage_base import LeaseLostError

        provider_wait = False
        claim_fence: dict[str, Any] = {
            "expected_claimed_by": run.claimed_by,
            "expected_claimed_at": run.claimed_at,
        }
        if (
            result.finished_reason == "structured_output"
            and committed_output is not None
        ):
            # Identity belongs to this output, not its conversation or owner.
            # Saved-output adoption/replay returns without minting an ID.
            # These input digests are evidence, never execution authority.
            committed_output["publication"] = {
                "id": uuid.uuid4().hex,
                "origin": publication_origin,
                "bookmark": asdict(captured_bookmark)
                if captured_bookmark is not None
                else None,
                "dependency_inputs": dict(dependency_inputs),
                "answer_inputs": snapshot_answer_inputs(self.storage, run.id),
                "source_inputs": captured_sources,
                "history_inputs": asdict(captured_history),
            }
            try:
                with self.storage.agent_completion_input_scope(
                    run,
                    committed_output,
                    captured_history.profiles,
                    captured_history.playbooks,
                ) as admitted_output:
                    stored_run = self.storage.update_agent_run_status(
                        run.id,
                        AgentRunStatus.AGENT_COMPLETED,
                        committed_output=admitted_output,
                        pending_tool_call_ids=result.pending_tool_call_ids,
                        max_steps_remaining=result.max_steps_remaining,
                        expected_statuses=active_statuses,
                        **claim_fence,
                    )
            except LeaseLostError:
                stored_run = None
            if (
                stored_run is None
                or stored_run.status != AgentRunStatus.AGENT_COMPLETED
            ):
                logger.warning(
                    "event=extraction_agent_late_output_discarded org_id=%s "
                    "user_id=%s extractor_kind=%s run_id=%s request_id=%s "
                    "stored_status=%s",
                    run.binding.org_id,
                    run.binding.user_id,
                    run.binding.extractor_kind,
                    run.id,
                    run.binding.request_id,
                    stored_run.status if stored_run is not None else None,
                )
                return AgentRunResult(
                    run_id=run.id,
                    output=None,
                    pending_tool_call_ids=[],
                    messages=result.messages,
                    trace=result.trace,
                    finished_reason="late_output_discarded",
                    model_provenance=None,
                )
            retain_compute_result()
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
        elif (
            run.status == AgentRunStatus.RESUMING
            and result.error is not None
            and isinstance(classify_llm_failure(result.error), OutageWait)
        ):
            # The provider is down for everyone. A resumed run's FAILED would be
            # terminal, so leave it RESUMING: its worker refunds the attempt and
            # waits for the provider instead. (A window run is re-adopted on
            # retry whatever its status, so it takes the ordinary path below.)
            stored_run = None
            provider_wait = True
            logger.warning(
                "event=extraction_agent_provider_wait org_id=%s run_id=%s "
                "error_type=%s",
                run.binding.org_id,
                run.id,
                result.error_type,
            )
        else:
            last_error = f"Extraction agent did not finish: {result.failure_label}"
            try:
                with owned_agent_run_write(self.storage, run):
                    stored_run = self.storage.update_agent_run_status(
                        run.id,
                        AgentRunStatus.FAILED,
                        max_steps_remaining=result.max_steps_remaining,
                        last_error=last_error,
                        expected_statuses=active_statuses,
                        **claim_fence,
                    )
            except LeaseLostError:
                stored_run = None
            # A losing claim cannot establish a terminal failure of the run
            # now owned by another worker. Keep its discarded attempt diagnostic.
            logger.warning(
                "event=extraction_agent_failed release_blocking=%s org_id=%s user_id=%s "
                "extractor_kind=%s run_id=%s request_id=%s "
                "finished_reason=%s error_type=%s error_detail=%s has_output=%s",
                "true"
                if stored_run is not None and stored_run.status == AgentRunStatus.FAILED
                else "false",
                run.binding.org_id,
                run.binding.user_id,
                run.binding.extractor_kind,
                run.id,
                run.binding.request_id,
                result.finished_reason,
                result.error_type,
                result.error_detail,
                output is not None,
            )
            _record_agent_usage_event(
                run=run,
                event_name="extraction_agent_failed",
                outcome="failed",
                error_kind=result.finished_reason,
                metadata={
                    "has_output": output is not None,
                    "error_type": result.error_type,
                    "error_detail": result.error_detail,
                },
            )

        return AgentRunResult(
            run_id=run.id,
            output=output,
            pending_tool_call_ids=result.pending_tool_call_ids,
            messages=result.messages,
            trace=result.trace,
            finished_reason=result.finished_reason,
            model_provenance=result.provenance,
            error_type=result.error_type,
            error_detail=result.error_detail,
            execution_run=stored_run,
            error=result.error,
            provider_wait=provider_wait,
        )
