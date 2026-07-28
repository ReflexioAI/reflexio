from __future__ import annotations

import logging
import os
from collections import Counter
from typing import TYPE_CHECKING

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.models.config_schema import PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.llm.token_accounting import RunTokenTotals, sum_trace_tokens
from reflexio.server.services.deferred_learning_plan import ExtractorBookmarkAdvance
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.extraction.resumable_agent import (
    run_resumable_extraction_agent,
)
from reflexio.server.services.extractor_config_utils import get_extractor_name
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
)
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.playbook.playbook_evidence import (
    as_playbook_content,
    build_playbook_extraction_contract,
    candidate_rejection_reason,
    resolve_verbatim_source_span,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookEvidenceSource,
    StructuredExtractedPlaybookList,
    StructuredPlaybookContent,
    StructuredPlaybookEvidence,
    StructuredPlaybookList,
    build_playbook_prompt_context,
    construct_expert_playbook_extraction_messages,
    construct_playbook_extraction_messages_from_sessions,
    ensure_playbook_content,
    has_expert_content,
    uses_evidence_grounded_extraction,
)
from reflexio.server.services.service_utils import (
    extract_interactions_from_request_interaction_data_models,
    log_llm_messages,
)
from reflexio.server.site_var.site_var_manager import SiteVarManager

if TYPE_CHECKING:
    from reflexio.server.services.playbook.service import (
        PlaybookGenerationServiceConfig,
    )

logger = logging.getLogger(__name__)


"""
Extract agent evolvement playbook entries from agent to improve its performance through self evolvement.
Make better decisions on what to improve next time.
"""


class PlaybookExtractor:
    """
    Extract agent evolvement playbook entries from agent interactions to improve its performance.

    This class analyzes agent-user interactions and generates structured playbook entries
    to help the agent make better decisions.
    """

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        extractor_config: PlaybookConfig,
        service_config: PlaybookGenerationServiceConfig,
        agent_context: str,
    ):
        """
        Initialize the playbook extractor.

        Args:
            request_context: Request context with storage and prompt manager
            llm_client: Unified LLM client supporting both OpenAI and Claude
            extractor_config: Playbook configuration from YAML
            service_config: Runtime service configuration with request data
            agent_context: Context about the agent
        """
        self.request_context: RequestContext = request_context
        self.client: LiteLLMClient = llm_client
        self.config: PlaybookConfig = extractor_config
        self.service_config: PlaybookGenerationServiceConfig = service_config
        self.agent_context: str = agent_context
        self._last_resumable_run_id: str | None = None
        self._last_resumable_token_totals: RunTokenTotals | None = None
        self._last_model_provenance = None

        # Get LLM config overrides from configuration
        config = self.request_context.configurator.get_config()
        llm_config = config.llm_config if config else None

        # Resolve model names: config override -> site var -> auto-detect
        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}
        api_key_config = self.request_context.configurator.get_config().api_key_config

        self.should_run_model_name = resolve_model_name(
            ModelRole.SHOULD_RUN,
            site_var_value=site_var.get("should_run_model_name"),
            config_override=llm_config.should_run_model_name if llm_config else None,
            api_key_config=api_key_config,
        )
        self.default_generation_model_name = resolve_model_name(
            ModelRole.GENERATION,
            site_var_value=site_var.get("default_generation_model_name"),
            config_override=llm_config.generation_model_name if llm_config else None,
            api_key_config=api_key_config,
        )

    def _create_state_manager(self) -> OperationStateManager:
        """
        Create an OperationStateManager for this extractor.

        Returns:
            OperationStateManager configured for playbook_extractor
        """
        return OperationStateManager(
            self.request_context.storage,  # type: ignore[reportArgumentType]
            self.request_context.org_id,
            "playbook_extractor",
        )

    def _get_interactions(self) -> list[RequestInteractionDataModel] | None:
        """
        Get interactions for this extractor based on its config.

        Handles:
        - Getting window parameters (extractor override or global fallback)
        - Source filtering based on extractor config
        - Time range filtering for rerun flows

        Note: Stride checking is handled upstream by BaseGenerationService._filter_configs_by_stride()
        before the extractor is created.

        Returns:
            List of request interaction data models, or None if source filter skips this extractor
        """
        # Get global config values
        config = self.request_context.configurator.get_config()
        global_window_size = getattr(config, "window_size", None) if config else None
        global_stride_size = getattr(config, "stride_size", None) if config else None

        # Get effective window_size for this extractor
        window_size, _ = get_extractor_window_params(
            self.config,
            global_window_size,
            global_stride_size,
        )

        # Get effective source filter (None = get ALL sources)
        should_skip, effective_source = get_effective_source_filter(
            self.config,
            self.service_config.source,
        )
        if should_skip:
            return None

        storage = self.request_context.storage

        # Only filter by agent_version during rerun (non-auto_run) mode
        rerun_agent_version = (
            self.service_config.agent_version
            if not self.service_config.auto_run
            else None
        )

        # Get window interactions with time range filter
        session_data_models, _ = storage.get_last_k_interactions_grouped(  # type: ignore[reportOptionalMemberAccess]
            user_id=self.service_config.user_id,
            k=window_size,
            sources=effective_source,
            start_time=self.service_config.rerun_start_time,
            end_time=self.service_config.rerun_end_time,
            agent_version=rerun_agent_version,
        )
        return session_data_models

    # ===============================
    # public methods
    # ===============================

    def run(self) -> list[UserPlaybook] | ExtractionOutcome[UserPlaybook]:
        """
        Run playbook extraction on request interaction groups.

        This extractor handles its own data collection:
        1. Gets interactions based on its config (window size, source filtering)
        2. Applies time range filter for rerun flows
        3. Defers the stride-bookmark advance onto the outcome (applied in persist)

        Returns:
            An empty list when there are no interactions to process; otherwise an
            ExtractionOutcome carrying the extracted playbook entries, the
            resumable run_id (when set), and the deferred bookmark advance.
        """
        # Collect interactions using extractor's own window_size/stride_size settings
        request_interaction_data_models = self._get_interactions()
        if not request_interaction_data_models:
            # No interactions or stride_size not met
            return []

        # should_generate check is handled at the service level (consolidated across all extractors)

        user_playbooks = self.extract_playbook_entries(request_interaction_data_models)

        # Defer the stride-bookmark advance onto the outcome instead of
        # self-advancing here (F1): applied downstream — inside the persist
        # fence on the durable path, or in ``.run()``'s persist half — so it
        # stays atomic with the playbook row writes. Only produced when output
        # was generated (bookmark-iff-rows).
        bookmark_advance: ExtractorBookmarkAdvance | None = None
        if user_playbooks:
            bookmark_advance = ExtractorBookmarkAdvance(
                extractor_name=get_extractor_name(self.config),
                processed_interactions=extract_interactions_from_request_interaction_data_models(
                    request_interaction_data_models
                ),
                user_id=self.service_config.user_id,
            )

        # Always return an ExtractionOutcome so the bookmark advance rides along
        # even in the non-resumable case; a resumable run also surfaces its
        # run_id for _agent_runs finalization.
        return ExtractionOutcome.completed(
            user_playbooks,
            run_id=self._last_resumable_run_id,
            token_totals=self._last_resumable_token_totals,
            bookmark_advance=bookmark_advance,
            model_provenance=self._last_model_provenance,
        )

    def extract_playbook_entries(
        self, request_interaction_data_models: list[RequestInteractionDataModel]
    ) -> list[UserPlaybook]:
        """
        Extract playbook entries from the given request interaction groups using structured output.

        Args:
            request_interaction_data_models: List of request interaction groups

        Returns:
            list[UserPlaybook]: List of extracted user playbook entries
        """
        all_interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )
        expert_mode = has_expert_content(all_interactions)
        prompt_manager = self.request_context.prompt_manager
        strict_evidence = uses_evidence_grounded_extraction(
            prompt_manager, expert=expert_mode
        )
        prompt_context = build_playbook_prompt_context(
            request_interaction_data_models,
            expert=expert_mode,
            label_turns=strict_evidence,
        )

        # Check if mock mode is enabled
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            logger.info("Mock mode: generating mock playbook entry")
            mock_response = self._generate_mock_playbook_list(
                request_interaction_data_models, prompt_context.evidence_sources
            )
            logger.debug(
                "Mock playbook list: %d entries — %s",
                len(mock_response.playbooks),
                [entry.content for entry in mock_response.playbooks],
            )
            return self._process_structured_response_list(
                mock_response, evidence_sources=prompt_context.evidence_sources
            )

        # Get tool_can_use from root config
        root_config = self.request_context.configurator.get_config()
        tool_can_use_str = ""
        if root_config and root_config.tool_can_use:
            tool_can_use_str = "\n".join(
                [
                    f"{tool.tool_name}: {tool.tool_description}"
                    for tool in root_config.tool_can_use
                ]
            )

        # Check if interactions contain expert content — use expert extraction path
        playbook_definition = (
            self.config.extraction_definition_prompt.strip()
            if self.config.extraction_definition_prompt
            else ""
        )
        contract = build_playbook_extraction_contract(
            prompt_manager,
            expert=expert_mode,
            evidence_sources=prompt_context.evidence_sources,
        )

        if expert_mode:
            logger.info("Expert content detected, using expert extraction path")
            messages = construct_expert_playbook_extraction_messages(
                prompt_manager=prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=self.agent_context,
                extraction_definition_prompt=playbook_definition,
                prompt_context=prompt_context,
            )
        else:
            messages = construct_playbook_extraction_messages_from_sessions(
                prompt_manager=prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=self.agent_context,
                extraction_definition_prompt=playbook_definition,
                tool_can_use=tool_can_use_str,
                prompt_context=prompt_context,
            )
        log_llm_messages(logger, "Playbook extraction", messages)

        result = run_resumable_extraction_agent(
            request_context=self.request_context,
            client=self.client,
            extractor_kind="playbook",
            user_id=self.service_config.user_id,
            request_id=self.service_config.request_id,
            agent_version=self.service_config.agent_version,
            source=self.service_config.source,
            request_interaction_data_models=request_interaction_data_models,
            extractor_config=self.config,
            service_config=self.service_config,
            agent_context=self.agent_context,
            messages=messages,
            output_schema=contract.schema,
            log_label="Playbook extraction",
            # Opt strict normal extraction into the client's one bounded
            # corrective repair turn. The validator only fires when the whole
            # batch is ungrounded; individually bad candidates are filtered
            # below without spending a repair turn.
            structured_output_validator=contract.validator,
        )
        self._last_resumable_run_id = result.run_id
        self._last_resumable_token_totals = sum_trace_tokens(result.trace)
        self._last_model_provenance = result.model_provenance
        if not isinstance(
            result.output, StructuredPlaybookList | StructuredExtractedPlaybookList
        ):
            logger.warning(
                "Playbook extraction did not finish: %s",
                result.finished_reason,
            )
            if result.finished_reason in {"error", "no_tool_call"}:
                raise RuntimeError(
                    "Playbook extraction failed without structured output: "
                    f"{result.finished_reason}"
                )
            return []
        if contract.strict:
            return self._process_structured_response_list(
                result.output,
                evidence_sources=prompt_context.evidence_sources,
            )
        return self._process_structured_response_list(
            result.output,
            source_interaction_ids=[
                interaction.interaction_id for interaction in all_interactions
            ],
        )

    def _generate_mock_playbook_list(
        self,
        request_interaction_data_models: list[RequestInteractionDataModel],
        evidence_sources: dict[str, PlaybookEvidenceSource],
    ) -> StructuredPlaybookList:
        """
        Generate mock structured playbook list for testing purposes.

        Args:
            request_interaction_data_models: List of request interaction groups

        Returns:
            StructuredPlaybookList: Mock structured playbook list with one entry
        """
        # Extract flat interactions from sessions
        interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )

        # Generate concise playbook based on playbook definition
        playbook_definition = (
            self.config.extraction_definition_prompt.strip()
            if self.config.extraction_definition_prompt
            else "agent behavior"
        )

        # Build trigger from interaction context
        trigger = "similar interactions occur"
        if interactions:
            last_interaction = interactions[-1]
            if last_interaction.content:
                content_preview = last_interaction.content[:50]
                trigger = f"user says something like '{content_preview}'"

        evidence: list[StructuredPlaybookEvidence] = []
        for turn_ref, source in reversed(evidence_sources.items()):
            if source.interaction_id and source.evidence_texts:
                evidence = [
                    StructuredPlaybookEvidence(
                        turn_ref=turn_ref, source_span=source.evidence_texts[0]
                    )
                ]
                break

        entry = StructuredPlaybookContent(
            content=f"When {trigger}, improve on {playbook_definition} by adjusting the current approach.",
            trigger=trigger,
            rationale="The referenced turn contains the task signal used to construct this mock rule.",
            evidence_kind="verified-success",
            future_task_class="similar interactions",
            improvement_mechanism="reuses the grounded behavior from the referenced turn",
            reader_angle="behavior",
            evidence=evidence,
        )
        return StructuredPlaybookList(playbooks=[entry])

    def _process_structured_response_list(
        self,
        response: StructuredPlaybookList | StructuredExtractedPlaybookList,
        source_interaction_ids: list[int] | None = None,
        evidence_sources: dict[str, PlaybookEvidenceSource] | None = None,
    ) -> list[UserPlaybook]:
        """
        Process a structured playbook list from the LLM into UserPlaybook entries.

        Filters out entries with no usable content or valid evidence and emits
        one UserPlaybook per valid entry. In production, each entry receives
        only the source interaction IDs resolved from its cited local turns.

        Args:
            response (StructuredPlaybookList): Parsed Pydantic model from structured output
            source_interaction_ids: Legacy source IDs used by private callers without
                an evidence map.
            evidence_sources: Prompt-local turn references mapped to persisted sources.

        Returns:
            list[UserPlaybook]: Zero or more user playbook entries
        """
        user_playbooks: list[UserPlaybook] = []
        rejection_counts: Counter[str] = Counter()
        for entry in map(as_playbook_content, response.playbooks):
            if evidence_sources is not None:
                # Sole rejection gate for the evidence-grounded path;
                # ``_build_user_playbook`` trusts the candidate from here on.
                rejection_reason = candidate_rejection_reason(entry, evidence_sources)
                if rejection_reason is not None:
                    rejection_counts[rejection_reason] += 1
                    logger.info(
                        "event=playbook_candidate_rejected reason=%s",
                        rejection_reason,
                    )
                    continue
            playbook = self._build_user_playbook(
                entry,
                source_interaction_ids=source_interaction_ids or [],
                evidence_sources=evidence_sources,
            )
            if playbook is not None:
                user_playbooks.append(playbook)

        if rejection_counts:
            logger.info(
                "event=playbook_evidence_validation_summary rejected=%d reasons=%s",
                sum(rejection_counts.values()),
                ",".join(
                    f"{reason}:{count}"
                    for reason, count in sorted(rejection_counts.items())
                ),
            )

        if not user_playbooks:
            logger.info(
                "No playbook entries can be generated for the given interactions"
            )
        else:
            interaction_count = (
                len({source.interaction_id for source in evidence_sources.values()})
                if evidence_sources is not None
                else len(source_interaction_ids or [])
            )
            logger.info(
                "Extracted %d playbook entries from %d interactions",
                len(user_playbooks),
                interaction_count,
            )
        return user_playbooks

    def _build_user_playbook(
        self,
        entry: StructuredPlaybookContent,
        source_interaction_ids: list[int],
        evidence_sources: dict[str, PlaybookEvidenceSource] | None = None,
    ) -> UserPlaybook | None:
        """
        Convert one StructuredPlaybookContent entry into a UserPlaybook.

        Args:
            entry (StructuredPlaybookContent): A single parsed playbook entry from the LLM
            source_interaction_ids (list[int]): IDs of interactions used to generate this entry
            evidence_sources: Prompt-local turn references mapped to persisted
                sources.

        Returns:
            UserPlaybook | None: The constructed playbook, or None if the entry has no usable content
        """
        if not entry.has_content:
            return None

        resolved_source_ids = list(source_interaction_ids)
        source_span = entry.source_span
        if evidence_sources is not None:
            # Defence in depth: callers pre-filter, but this method is the last
            # step before a row becomes persistable, so it re-checks rather than
            # trusting the caller. The check is a pure function over the entry.
            rejection_reason = candidate_rejection_reason(entry, evidence_sources)
            if rejection_reason is not None:
                logger.info(
                    "event=playbook_candidate_rejected reason=%s",
                    rejection_reason,
                )
                return None
            resolved_source_ids = []
            spans: list[str] = []
            for evidence in entry.evidence:
                source = evidence_sources[evidence.turn_ref.strip()]
                if source.interaction_id not in resolved_source_ids:
                    resolved_source_ids.append(source.interaction_id)
                span = resolve_verbatim_source_span(
                    evidence.source_span.strip(), source.evidence_texts
                )
                if span is None:  # defensive against evidence changing after validation
                    return None
                if span not in spans:
                    spans.append(span)
            source_span = "\n\n".join(spans)

        playbook_content = ensure_playbook_content(entry.content, entry)

        return UserPlaybook(
            playbook_name=get_extractor_name(self.config),
            user_id=self.service_config.user_id,
            agent_version=self.service_config.agent_version,
            request_id=self.service_config.request_id,
            content=playbook_content,
            trigger=entry.trigger,
            rationale=entry.rationale,
            source_interaction_ids=resolved_source_ids,
            source_span=source_span,
            notes=entry.notes,
            reader_angle=entry.reader_angle or entry.evidence_kind,
        )
