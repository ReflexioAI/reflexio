"""Follow-up resume worker for resolved async extraction tool calls."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import PlaybookConfig, ProfileExtractorConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.extraction.agent_run_records import build_scope_hash
from reflexio.server.services.extraction.pending_tool_call_dispatch import (
    PendingToolCallToolContext,
)
from reflexio.server.services.extraction.prior_answer_search import (
    append_prior_knowledge_context,
)
from reflexio.server.services.extraction.resumable_agent import (
    AgentRunResult,
    ResumableExtractionAgent,
    create_pending_info_tools_for_extractor_kind,
)
from reflexio.server.services.playbook.components.extractor import PlaybookExtractor
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookList,
    construct_expert_playbook_extraction_messages,
    construct_playbook_extraction_messages_from_sessions,
    has_expert_content,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.profile.components.extractor import (
    MAX_EXISTING_PROFILES_FOR_CONTEXT,
    ProfileExtractor,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
    construct_profile_extraction_messages_from_sessions,
)
from reflexio.server.services.profile.service import (
    ProfileGenerationService,
    ProfileGenerationServiceConfig,
)
from reflexio.server.services.service_utils import (
    extract_interactions_from_request_interaction_data_models,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    BaseStorage,
    PendingToolCallRecord,
    PendingToolCallStatus,
)
from reflexio.server.services.tagging.tagging_scheduler import schedule_tagging
from reflexio.server.site_var.site_var_manager import SiteVarManager
from reflexio.server.tracing import sentry_tags

logger = logging.getLogger(__name__)


class ResumeWorkerError(RuntimeError):
    """Base error for retryable resume-worker failures."""


def _next_retry_at(attempt_count: int) -> datetime:
    delay_seconds = min(300, max(1, 2 ** max(0, attempt_count - 1)))
    return datetime.now(UTC) + timedelta(seconds=delay_seconds)


def _create_llm_client(request_context: RequestContext) -> LiteLLMClient:
    # The tool loop re-resolves the model per call via ModelRole.EXTRACTION_AGENT
    # (see run_tool_loop); this client's model name is only a fallback. We resolve
    # the GENERATION model here so the client still carries a sensible default and
    # the org's api_key_config, which the loop reuses for provider routing.
    model_setting = SiteVarManager().get_site_var("llm_model_setting")
    site_var = model_setting if isinstance(model_setting, dict) else {}
    config = request_context.configurator.get_config()
    api_key_config = config.api_key_config if config else None
    llm_config = config.llm_config if config else None
    model_name = resolve_model_name(
        ModelRole.GENERATION,
        site_var_value=site_var.get("default_generation_model_name"),
        config_override=llm_config.generation_model_name if llm_config else None,
        api_key_config=api_key_config,
    )
    return LiteLLMClient(LiteLLMConfig(model=model_name, api_key_config=api_key_config))


def _storage(request_context: RequestContext) -> BaseStorage:
    storage = request_context.storage
    if storage is None:
        raise ResumeWorkerError("Storage not configured")
    return storage


def _request_interaction_models_from_ids(
    storage: BaseStorage,
    interaction_ids: list[int],
    *,
    fallback_source: str | None,
    fallback_agent_version: str | None,
) -> list[RequestInteractionDataModel]:
    interactions = storage.get_interactions_by_ids(interaction_ids)
    if not interactions:
        raise ResumeWorkerError("No source interactions found for resumed run")

    by_request: dict[str, list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        by_request[interaction.request_id].append(interaction)

    models: list[RequestInteractionDataModel] = []
    for request_id, request_interactions in by_request.items():
        ordered = sorted(request_interactions, key=lambda item: item.created_at)
        request = storage.get_request(request_id)
        if request is None:
            first = ordered[0]
            request = Request(
                request_id=request_id,
                user_id=first.user_id,
                created_at=first.created_at,
                source=fallback_source or "",
                agent_version=fallback_agent_version or "",
                session_id=request_id,
            )
        models.append(
            RequestInteractionDataModel(
                session_id=request.session_id,
                request=request,
                interactions=ordered,
            )
        )
    return sorted(
        models,
        key=lambda model: min(
            (interaction.created_at for interaction in model.interactions),
            default=0,
        ),
    )


def _select_current_extractor_config(
    request_context: RequestContext,
    run: AgentRunRecord,
) -> ProfileExtractorConfig | PlaybookConfig:
    root_config = request_context.configurator.get_config()
    if run.binding.extractor_kind == "profile":
        config = getattr(root_config, "profile_extractor_config", None)
    elif run.binding.extractor_kind == "playbook":
        config = getattr(root_config, "user_playbook_extractor_config", None)
    else:
        raise ResumeWorkerError(
            f"Unsupported extractor kind {run.binding.extractor_kind!r}"
        )

    if config is not None and isinstance(
        config, ProfileExtractorConfig | PlaybookConfig
    ):
        return config
    raise ResumeWorkerError(
        f"Current extractor config not found for {run.binding.extractor_kind}"
    )


def _log_config_hash_drift(
    run: AgentRunRecord,
    extractor_config: BaseModel,
) -> None:
    current_hash = build_scope_hash(extractor_config.model_dump(mode="json"))
    if run.binding.extractor_config_hash and (
        current_hash != run.binding.extractor_config_hash
    ):
        logger.info(
            "event=resumable_extraction_config_hash_drift run_id=%s extractor_kind=%s "
            "stored_hash=%s current_hash=%s",
            run.id,
            run.binding.extractor_kind,
            run.binding.extractor_config_hash,
            current_hash,
        )


class ExtractionResumeWorker:
    """Claim and resume finalized extraction runs with resolved tool results."""

    def __init__(
        self,
        *,
        request_context: RequestContext,
        llm_client: LiteLLMClient | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.request_context = request_context
        self.storage = _storage(request_context)
        self.client = llm_client or _create_llm_client(request_context)
        self.worker_id = worker_id or f"resume_worker_{uuid.uuid4().hex}"

    def drain(self, *, max_runs: int = 10) -> int:
        """Resume up to ``max_runs`` ready rows for this request context."""
        resumed = 0
        for _ in range(max_runs):
            if self.run_once() is None:
                break
            resumed += 1
        return resumed

    def run_once(self) -> AgentRunRecord | None:
        config = self.request_context.configurator.get_config()
        pending_config = config.pending_tool_call_config
        finalization_retry = self.storage.claim_finalization_failed_agent_run(
            org_id=self.request_context.org_id,
            worker_id=self.worker_id,
            claim_ttl_seconds=pending_config.resume_claim_ttl_seconds,
        )
        if finalization_retry is not None:
            return self._retry_finalization(finalization_retry)

        run = self.storage.claim_ready_agent_run(
            org_id=self.request_context.org_id,
            worker_id=self.worker_id,
            claim_ttl_seconds=pending_config.resume_claim_ttl_seconds,
        )
        if run is None:
            return None

        if run.max_steps_remaining is not None and run.max_steps_remaining <= 0:
            return self.storage.update_agent_run_status(
                run.id,
                AgentRunStatus.FAILED,
                last_error="Resumable extraction max-step budget exhausted",
            )

        try:
            resolved_calls = self._load_resolved_tool_calls(run)
            if not resolved_calls:
                raise ResumeWorkerError(
                    f"Run {run.id} has no resolved, unconsumed tool calls"
                )
            items, pending_tool_call_ids = self._resume_run(run, resolved_calls)
        except Exception as exc:
            with sentry_tags(
                subsystem="extraction",
                op="resume_run",
                org_id=self.request_context.org_id,
                run_id=run.id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=resumable_extraction_resume_failed run_id=%s",
                    run.id,
                )
            failed_status = (
                AgentRunStatus.FAILED
                if run.resume_attempts >= pending_config.max_resume_attempts
                else AgentRunStatus.RESUME_READY
            )
            return self.storage.update_agent_run_status(
                run.id,
                failed_status,
                next_resume_at=_next_retry_at(run.resume_attempts),
                last_error=str(exc),
            )

        try:
            self.storage.update_agent_run_status(run.id, AgentRunStatus.FINALIZING)
            self._finalize_items(run, items)
            self._schedule_finalized_tagging(run)
            self.storage.consume_run_tool_dependencies(run.id)
            finalized_status = (
                AgentRunStatus.FINALIZED_PENDING_TOOL
                if pending_tool_call_ids
                else AgentRunStatus.FINALIZED
            )
            return self.storage.update_agent_run_status(
                run.id,
                finalized_status,
                pending_tool_call_ids=pending_tool_call_ids,
            )
        except Exception as exc:
            with sentry_tags(
                subsystem="extraction",
                op="finalize_run",
                org_id=self.request_context.org_id,
                run_id=run.id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=resumable_extraction_finalization_failed run_id=%s",
                    run.id,
                )
            next_attempt_count = run.finalization_attempts + 1
            failed_status = (
                AgentRunStatus.FAILED
                if next_attempt_count >= pending_config.max_finalization_attempts
                else AgentRunStatus.FINALIZATION_FAILED
            )
            return self.storage.update_agent_run_status(
                run.id,
                failed_status,
                next_resume_at=_next_retry_at(next_attempt_count),
                last_error=str(exc),
                increment_finalization_attempts=True,
            )

    def _retry_finalization(self, run: AgentRunRecord) -> AgentRunRecord | None:
        config = self.request_context.configurator.get_config()
        pending_config = config.pending_tool_call_config
        try:
            items, pending_tool_call_ids = self._items_from_committed_output(run)
            self._finalize_items(run, items)
            self._schedule_finalized_tagging(run)
            self.storage.consume_run_tool_dependencies(run.id)
            finalized_status = (
                AgentRunStatus.FINALIZED_PENDING_TOOL
                if pending_tool_call_ids
                else AgentRunStatus.FINALIZED
            )
            return self.storage.update_agent_run_status(
                run.id,
                finalized_status,
                pending_tool_call_ids=pending_tool_call_ids,
            )
        except Exception as exc:
            with sentry_tags(
                subsystem="extraction",
                op="finalize_run_retry",
                org_id=self.request_context.org_id,
                run_id=run.id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=resumable_extraction_finalization_retry_failed run_id=%s",
                    run.id,
                )
            next_attempt_count = run.finalization_attempts + 1
            failed_status = (
                AgentRunStatus.FAILED
                if next_attempt_count >= pending_config.max_finalization_attempts
                else AgentRunStatus.FINALIZATION_FAILED
            )
            return self.storage.update_agent_run_status(
                run.id,
                failed_status,
                next_resume_at=_next_retry_at(next_attempt_count),
                last_error=str(exc),
                increment_finalization_attempts=True,
            )

    def _load_resolved_tool_calls(
        self,
        run: AgentRunRecord,
    ) -> list[PendingToolCallRecord]:
        records: list[PendingToolCallRecord] = []
        for dependency in self.storage.list_run_tool_dependencies(run.id):
            if dependency.consumed_at is not None or dependency.resolved_at is None:
                continue
            record = self.storage.get_pending_tool_call(dependency.pending_tool_call_id)
            if record and record.status == PendingToolCallStatus.RESOLVED:
                records.append(record)
        return sorted(
            records,
            key=lambda record: record.resolved_at or datetime.max.replace(tzinfo=UTC),
        )

    def _extra_tools_for_run(
        self,
        run: AgentRunRecord,
    ) -> tuple[list[Any], PendingToolCallToolContext | None]:
        pending_config = (
            self.request_context.configurator.get_config().pending_tool_call_config
        )
        tools = create_pending_info_tools_for_extractor_kind(run.binding.extractor_kind)
        return tools, PendingToolCallToolContext(
            storage=self.storage,
            run_id=run.id,
            org_id=self.request_context.org_id,
            extractor_kind=run.binding.extractor_kind,
            user_id=run.binding.user_id,
            config=pending_config,
        )

    def _resume_run(
        self,
        run: AgentRunRecord,
        resolved_calls: list[PendingToolCallRecord],
    ) -> tuple[list[Any], list[str]]:
        request_interaction_data_models = _request_interaction_models_from_ids(
            self.storage,
            run.binding.source_interaction_ids,
            fallback_source=run.binding.source,
            fallback_agent_version=run.binding.agent_version,
        )
        extractor_config = _select_current_extractor_config(self.request_context, run)
        _log_config_hash_drift(run, extractor_config)
        if run.binding.extractor_kind == "profile":
            return self._resume_profile(
                run,
                extractor_config,
                request_interaction_data_models,
                resolved_calls,
            )
        if run.binding.extractor_kind == "playbook":
            return self._resume_playbook(
                run,
                extractor_config,
                request_interaction_data_models,
                resolved_calls,
            )
        raise ResumeWorkerError(
            f"Unsupported extractor kind {run.binding.extractor_kind!r}"
        )

    def _resume_profile(
        self,
        run: AgentRunRecord,
        extractor_config: ProfileExtractorConfig | PlaybookConfig,
        request_interaction_data_models: list[RequestInteractionDataModel],
        resolved_calls: list[PendingToolCallRecord],
    ) -> tuple[list[Any], list[str]]:
        if not isinstance(extractor_config, ProfileExtractorConfig):
            raise ResumeWorkerError("Expected profile extractor config")
        if run.binding.user_id is None:
            raise ResumeWorkerError("Profile resume requires user_id")

        existing_profiles = self.storage.get_user_profile(run.binding.user_id)
        context_profiles = sorted(
            existing_profiles,
            key=lambda profile: profile.last_modified_timestamp,
            reverse=True,
        )[:MAX_EXISTING_PROFILES_FOR_CONTEXT]
        agent_context = self.request_context.configurator.get_agent_context()
        service_config = ProfileGenerationServiceConfig(
            user_id=run.binding.user_id,
            request_id=run.binding.request_id,
            source=run.binding.source,
            existing_data=existing_profiles,
            auto_run=False,
            force_extraction=True,
        )
        extractor = ProfileExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=agent_context,
        )
        messages = construct_profile_extraction_messages_from_sessions(
            prompt_manager=self.request_context.prompt_manager,
            request_interaction_data_models=request_interaction_data_models,
            agent_context_prompt=agent_context,
            context_prompt=(
                extractor_config.context_prompt.strip()
                if extractor_config.context_prompt
                else ""
            ),
            extraction_definition_prompt=extractor_config.extraction_definition_prompt.strip(),
            existing_profiles=context_profiles,
        )
        result = self._resume_agent(
            run=run,
            messages=self._messages_with_prior_knowledge(
                run=run,
                messages=messages,
                extractor_config=extractor_config,
                resolved_calls=resolved_calls,
            ),
            output_schema=StructuredProfilesOutput,
            resolved_calls=resolved_calls,
            log_label="Profile extraction resume",
        )
        if not isinstance(result.output, StructuredProfilesOutput):
            raise ResumeWorkerError(
                f"Profile resume did not finish: {result.finished_reason}"
            )
        raw_profiles = [
            profile.model_dump() for profile in result.output.profiles or []
        ]
        return (
            extractor._convert_raw_to_user_profiles(
                raw_profiles=raw_profiles,
                user_id=run.binding.user_id,
                request_id=run.binding.request_id,
            ),
            result.pending_tool_call_ids,
        )

    def _resume_playbook(
        self,
        run: AgentRunRecord,
        extractor_config: ProfileExtractorConfig | PlaybookConfig,
        request_interaction_data_models: list[RequestInteractionDataModel],
        resolved_calls: list[PendingToolCallRecord],
    ) -> tuple[list[Any], list[str]]:
        if not isinstance(extractor_config, PlaybookConfig):
            raise ResumeWorkerError("Expected playbook extractor config")

        agent_context = self.request_context.configurator.get_agent_context()
        service_config = PlaybookGenerationServiceConfig(
            request_id=run.binding.request_id,
            agent_version=run.binding.agent_version or "",
            user_id=run.binding.user_id,
            source=run.binding.source,
            auto_run=False,
            force_extraction=True,
        )
        extractor = PlaybookExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=agent_context,
        )
        source_interaction_ids = [
            interaction.interaction_id
            for data_model in request_interaction_data_models
            for interaction in data_model.interactions
            if interaction.interaction_id
        ]
        playbook_definition = (
            extractor_config.extraction_definition_prompt.strip()
            if extractor_config.extraction_definition_prompt
            else ""
        )
        all_interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )
        prompt_manager = self.request_context.prompt_manager
        if has_expert_content(all_interactions):
            messages = construct_expert_playbook_extraction_messages(
                prompt_manager=prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=agent_context,
                extraction_definition_prompt=playbook_definition,
            )
        else:
            root_config = self.request_context.configurator.get_config()
            tool_can_use = ""
            if root_config and root_config.tool_can_use:
                tool_can_use = "\n".join(
                    [
                        f"{tool.tool_name}: {tool.tool_description}"
                        for tool in root_config.tool_can_use
                    ]
                )
            messages = construct_playbook_extraction_messages_from_sessions(
                prompt_manager=prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=agent_context,
                extraction_definition_prompt=playbook_definition,
                tool_can_use=tool_can_use,
            )
        result = self._resume_agent(
            run=run,
            messages=self._messages_with_prior_knowledge(
                run=run,
                messages=messages,
                extractor_config=extractor_config,
                resolved_calls=resolved_calls,
            ),
            output_schema=StructuredPlaybookList,
            resolved_calls=resolved_calls,
            log_label="Playbook extraction resume",
        )
        if not isinstance(result.output, StructuredPlaybookList):
            raise ResumeWorkerError(
                f"Playbook resume did not finish: {result.finished_reason}"
            )
        return (
            extractor._process_structured_response_list(
                result.output,
                source_interaction_ids=source_interaction_ids,
            ),
            result.pending_tool_call_ids,
        )

    def _messages_with_prior_knowledge(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        extractor_config: BaseModel,
        resolved_calls: list[PendingToolCallRecord] | None = None,
    ) -> list[dict[str, Any]]:
        pending_config = (
            self.request_context.configurator.get_config().pending_tool_call_config
        )
        # Prior-knowledge context is always injected within the resumable path.
        return append_prior_knowledge_context(
            messages=messages,
            storage=self.storage,
            org_id=self.request_context.org_id,
            extractor_kind=run.binding.extractor_kind,
            extractor_config=extractor_config,
            source=run.binding.source,
            agent_version=run.binding.agent_version,
            similarity_threshold=pending_config.for_tool(
                "ask_human"
            ).similarity_threshold,
            # Avoid re-surfacing the run's own just-resolved answers, which are
            # injected separately as explicit resolved-tool-result messages.
            exclude_pending_tool_call_ids={call.id for call in resolved_calls or []},
        )

    def _resume_agent(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        resolved_calls: list[PendingToolCallRecord],
        log_label: str,
    ) -> AgentRunResult:
        extra_tools, extra_context = self._extra_tools_for_run(run)
        return ResumableExtractionAgent(
            client=self.client,
            storage=self.storage,
        ).resume(
            run=run,
            messages=messages,
            output_schema=output_schema,
            resolved_tool_calls=resolved_calls,
            extra_tools=extra_tools,
            extra_tool_context=extra_context,
            log_label=log_label,
        )

    def _items_from_committed_output(
        self,
        run: AgentRunRecord,
    ) -> tuple[list[Any], list[str]]:
        if run.committed_output is None:
            raise ResumeWorkerError(
                f"Run {run.id} cannot retry finalization without committed output"
            )
        request_interaction_data_models = _request_interaction_models_from_ids(
            self.storage,
            run.binding.source_interaction_ids,
            fallback_source=run.binding.source,
            fallback_agent_version=run.binding.agent_version,
        )
        extractor_config = _select_current_extractor_config(self.request_context, run)
        if run.binding.extractor_kind == "profile":
            return self._profile_items_from_output(
                run,
                extractor_config,
                run.committed_output,
            )
        if run.binding.extractor_kind == "playbook":
            return self._playbook_items_from_output(
                run,
                extractor_config,
                request_interaction_data_models,
                run.committed_output,
            )
        raise ResumeWorkerError(
            f"Unsupported extractor kind {run.binding.extractor_kind!r}"
        )

    def _profile_items_from_output(
        self,
        run: AgentRunRecord,
        extractor_config: ProfileExtractorConfig | PlaybookConfig,
        committed_output: dict[str, Any],
    ) -> tuple[list[Any], list[str]]:
        if not isinstance(extractor_config, ProfileExtractorConfig):
            raise ResumeWorkerError("Expected profile extractor config")
        if run.binding.user_id is None:
            raise ResumeWorkerError("Profile finalization retry requires user_id")
        output = StructuredProfilesOutput.model_validate(committed_output)
        agent_context = self.request_context.configurator.get_agent_context()
        service_config = ProfileGenerationServiceConfig(
            user_id=run.binding.user_id,
            request_id=run.binding.request_id,
            source=run.binding.source,
            existing_data=self.storage.get_user_profile(run.binding.user_id),
            auto_run=False,
            force_extraction=True,
        )
        extractor = ProfileExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=agent_context,
        )
        raw_profiles = [profile.model_dump() for profile in output.profiles or []]
        return (
            extractor._convert_raw_to_user_profiles(
                raw_profiles=raw_profiles,
                user_id=run.binding.user_id,
                request_id=run.binding.request_id,
            ),
            run.pending_tool_call_ids,
        )

    def _playbook_items_from_output(
        self,
        run: AgentRunRecord,
        extractor_config: ProfileExtractorConfig | PlaybookConfig,
        request_interaction_data_models: list[RequestInteractionDataModel],
        committed_output: dict[str, Any],
    ) -> tuple[list[Any], list[str]]:
        if not isinstance(extractor_config, PlaybookConfig):
            raise ResumeWorkerError("Expected playbook extractor config")
        output = StructuredPlaybookList.model_validate(committed_output)
        agent_context = self.request_context.configurator.get_agent_context()
        service_config = PlaybookGenerationServiceConfig(
            request_id=run.binding.request_id,
            agent_version=run.binding.agent_version or "",
            user_id=run.binding.user_id,
            source=run.binding.source,
            auto_run=False,
            force_extraction=True,
        )
        extractor = PlaybookExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=agent_context,
        )
        source_interaction_ids = [
            interaction.interaction_id
            for data_model in request_interaction_data_models
            for interaction in data_model.interactions
            if interaction.interaction_id
        ]
        return (
            extractor._process_structured_response_list(
                output,
                source_interaction_ids=source_interaction_ids,
            ),
            run.pending_tool_call_ids,
        )

    def _finalize_items(self, run: AgentRunRecord, items: list[Any]) -> None:
        if run.binding.extractor_kind == "profile":
            service = ProfileGenerationService(
                llm_client=self.client,
                request_context=self.request_context,
            )
            service.service_config = ProfileGenerationServiceConfig(
                user_id=run.binding.user_id or "",
                request_id=run.binding.request_id,
                source=run.binding.source,
                auto_run=False,
                force_extraction=True,
            )
            service._finalize_extracted_items(items)
            return
        if run.binding.extractor_kind == "playbook":
            service = PlaybookGenerationService(
                llm_client=self.client,
                request_context=self.request_context,
            )
            service.service_config = PlaybookGenerationServiceConfig(
                request_id=run.binding.request_id,
                agent_version=run.binding.agent_version or "",
                user_id=run.binding.user_id,
                source=run.binding.source,
                auto_run=False,
                force_extraction=True,
            )
            service._finalize_extracted_items(items)
            return
        raise ResumeWorkerError(
            f"Unsupported extractor kind {run.binding.extractor_kind!r}"
        )

    def _schedule_finalized_tagging(self, run: AgentRunRecord) -> None:
        user_id = run.binding.user_id
        if not user_id:
            return
        if run.binding.extractor_kind not in ("profile", "playbook"):
            return

        # Defer tagging off this worker's drain loop. The pass is idempotent
        # (skips already-tagged entities), so running both profile and playbook
        # tagging is safe regardless of this run's extractor kind.
        try:
            schedule_tagging(
                org_id=self.request_context.org_id,
                user_id=user_id,
                agent_version=run.binding.agent_version or "",
                request_context=self.request_context,
                llm_client=self.client,
            )
        except Exception:
            logger.exception(
                "Failed to schedule tagging for finalized %s run %s",
                run.binding.extractor_kind,
                run.id,
            )
