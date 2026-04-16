from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.models.config_schema import PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
)
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
    StructuredPlaybookList,
    construct_expert_playbook_extraction_messages,
    construct_playbook_extraction_messages_from_sessions,
    ensure_playbook_content,
    has_expert_content,
)
from reflexio.server.services.service_utils import (
    extract_interactions_from_request_interaction_data_models,
    log_llm_messages,
    log_model_response,
)
from reflexio.server.site_var.site_var_manager import SiteVarManager

if TYPE_CHECKING:
    from reflexio.server.services.playbook.playbook_generation_service import (
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

        Note: Batch interval checking is handled upstream by BaseGenerationService._filter_configs_by_batch_interval()
        before the extractor is created.

        Returns:
            List of request interaction data models, or None if source filter skips this extractor
        """
        # Get global config values
        config = self.request_context.configurator.get_config()
        global_batch_size = getattr(config, "batch_size", None) if config else None
        global_batch_interval = (
            getattr(config, "batch_interval", None) if config else None
        )

        # Get effective batch_size for this extractor
        batch_size, _ = get_extractor_window_params(
            self.config,
            global_batch_size,
            global_batch_interval,
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
            k=batch_size,
            sources=effective_source,
            start_time=self.service_config.rerun_start_time,
            end_time=self.service_config.rerun_end_time,
            agent_version=rerun_agent_version,
        )
        return session_data_models

    def _update_operation_state(
        self, request_interaction_data_models: list[RequestInteractionDataModel]
    ) -> None:
        """
        Update operation state after processing interactions.

        Args:
            request_interaction_data_models: The interactions that were processed
        """
        all_interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )
        mgr = self._create_state_manager()
        mgr.update_extractor_bookmark(
            extractor_name=self.config.extractor_name,
            processed_interactions=all_interactions,
            user_id=self.service_config.user_id,
        )

    # ===============================
    # public methods
    # ===============================

    def run(self) -> list[UserPlaybook]:
        """
        Run playbook extraction on request interaction groups.

        This extractor handles its own data collection:
        1. Gets interactions based on its config (window size, source filtering)
        2. Applies time range filter for rerun flows
        3. Updates operation state after processing

        Returns:
            list[UserPlaybook]: List of extracted user playbook entries
        """
        # Collect interactions using extractor's own batch_size/batch_interval settings
        request_interaction_data_models = self._get_interactions()
        if not request_interaction_data_models:
            # No interactions or batch_interval not met
            return []

        # should_generate check is handled at the service level (consolidated across all extractors)

        user_playbooks = self.extract_playbook_entries(request_interaction_data_models)

        # Update operation state after successful processing
        if user_playbooks:
            self._update_operation_state(request_interaction_data_models)

        return user_playbooks

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
        # Collect source interaction IDs
        source_interaction_ids = [
            interaction.interaction_id
            for ridm in request_interaction_data_models
            for interaction in ridm.interactions
            if interaction.interaction_id
        ]

        # Check if mock mode is enabled
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            logger.info("Mock mode: generating mock playbook entry")
            mock_response = self._generate_mock_playbook_list(
                request_interaction_data_models
            )
            logger.debug(
                "Mock playbook list: %d entries — %s",
                len(mock_response.playbooks),
                [entry.content for entry in mock_response.playbooks],
            )
            return self._process_structured_response_list(
                mock_response, source_interaction_ids=source_interaction_ids
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
        all_interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )
        playbook_definition = (
            self.config.extraction_definition_prompt.strip()
            if self.config.extraction_definition_prompt
            else ""
        )

        if has_expert_content(all_interactions):
            logger.info("Expert content detected, using expert extraction path")
            messages = construct_expert_playbook_extraction_messages(
                prompt_manager=self.request_context.prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=self.agent_context,
                extraction_definition_prompt=playbook_definition,
            )
        elif self.service_config.is_incremental:
            from reflexio.server.services.playbook.playbook_service_utils import (
                construct_incremental_playbook_extraction_messages,
            )

            # Flatten previously_extracted (list of list[UserPlaybook]) into single list
            previously_extracted_flat = []
            for playbook_list in self.service_config.previously_extracted:
                if isinstance(playbook_list, list):
                    previously_extracted_flat.extend(playbook_list)

            messages = construct_incremental_playbook_extraction_messages(
                prompt_manager=self.request_context.prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=self.agent_context,
                extraction_definition_prompt=playbook_definition,
                previously_extracted=previously_extracted_flat,
                tool_can_use=tool_can_use_str,
            )
        else:
            messages = construct_playbook_extraction_messages_from_sessions(
                prompt_manager=self.request_context.prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=self.agent_context,
                extraction_definition_prompt=playbook_definition,
                tool_can_use=tool_can_use_str,
            )
        log_llm_messages(logger, "Playbook extraction", messages)

        try:
            response = self.client.generate_chat_response(
                messages=messages,
                model=self.default_generation_model_name,
                response_format=StructuredPlaybookList,
                parse_structured_output=True,
            )
            log_model_response(logger, "Playbook structured response", response)

            return self._process_structured_response_list(
                response,  # type: ignore[reportArgumentType]
                source_interaction_ids=source_interaction_ids,
            )
        except Exception as exc:
            # Log full traceback so non-OpenAI providers that drift from the
            # StructuredPlaybookList schema (e.g. silent regression to a
            # legacy single-entry shape) are debuggable from CI logs instead
            # of being swallowed as an empty extraction result.
            logger.exception(
                "Playbook extraction failed (%s): %s",
                type(exc).__name__,
                exc,
            )
            return []

    def _generate_mock_playbook_list(
        self, request_interaction_data_models: list[RequestInteractionDataModel]
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

        entry = StructuredPlaybookContent(
            content=f"When {trigger}, improve on {playbook_definition} by adjusting the current approach.",
            trigger=trigger,
        )
        return StructuredPlaybookList(playbooks=[entry])

    def _process_structured_response_list(
        self,
        response: StructuredPlaybookList,
        source_interaction_ids: list[int],
    ) -> list[UserPlaybook]:
        """
        Process a structured playbook list from the LLM into UserPlaybook entries.

        Filters out entries with no usable content and emits one UserPlaybook per
        valid entry. All emitted entries share the same source_interaction_ids
        because they were extracted from the same window in a single LLM call.

        Args:
            response (StructuredPlaybookList): Parsed Pydantic model from structured output
            source_interaction_ids (list[int]): IDs of interactions used to generate these entries

        Returns:
            list[UserPlaybook]: Zero or more user playbook entries
        """
        user_playbooks: list[UserPlaybook] = []
        for entry in response.playbooks:
            playbook = self._build_user_playbook(entry, source_interaction_ids)
            if playbook is not None:
                user_playbooks.append(playbook)

        if not user_playbooks:
            logger.info(
                "No playbook entries can be generated for the given interactions"
            )
        else:
            logger.info(
                "Extracted %d playbook entries from %d interactions",
                len(user_playbooks),
                len(source_interaction_ids),
            )
        return user_playbooks

    def _build_user_playbook(
        self,
        entry: StructuredPlaybookContent,
        source_interaction_ids: list[int],
    ) -> UserPlaybook | None:
        """
        Convert one StructuredPlaybookContent entry into a UserPlaybook.

        Args:
            entry (StructuredPlaybookContent): A single parsed playbook entry from the LLM
            source_interaction_ids (list[int]): IDs of interactions used to generate this entry

        Returns:
            UserPlaybook | None: The constructed playbook, or None if the entry has no usable content
        """
        if not entry.has_content:
            return None

        playbook_content = ensure_playbook_content(entry.content, entry)

        return UserPlaybook(
            playbook_name=self.config.extractor_name,
            user_id=self.service_config.user_id,
            agent_version=self.service_config.agent_version,
            request_id=self.service_config.request_id,
            content=playbook_content,
            trigger=entry.trigger,
            rationale=entry.rationale,
            blocking_issue=entry.blocking_issue,
            source_interaction_ids=source_interaction_ids,
        )
