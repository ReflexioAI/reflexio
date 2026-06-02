from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    UserProfile,
)
from reflexio.models.config_schema import ProfileExtractorConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.extraction.resumable_agent import (
    run_resumable_extraction_agent,
)
from reflexio.server.services.extraction.tools import new_profile_id
from reflexio.server.services.extractor_config_utils import get_extractor_name
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
)
from reflexio.server.services.operation_state_utils import OperationStateManager

if TYPE_CHECKING:
    from reflexio.server.services.profile.profile_generation_service import (
        ProfileGenerationServiceConfig,
    )
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileTimeToLive,
    StructuredProfilesOutput,
    calculate_expiration_timestamp,
    construct_profile_extraction_messages_from_sessions,
)
from reflexio.server.services.service_utils import (
    extract_interactions_from_request_interaction_data_models,
    format_sessions_to_history_string,
    log_llm_messages,
)
from reflexio.server.site_var.site_var_manager import SiteVarManager

logger = logging.getLogger(__name__)
PROFILE_EXTRACTION_TIMEOUT_SECONDS = 300
PROFILE_EXTRACTION_MAX_RETRIES = 2

# Maximum number of existing profiles to include in extraction prompt for context
MAX_EXISTING_PROFILES_FOR_CONTEXT = 5


class ProfileExtractor:
    """
    Extract user profile information from interactions.

    This class analyzes user interactions to extract new user profile information.
    It focuses purely on extraction — deduplication against existing profiles
    is handled separately by ProfileDeduplicator.
    """

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        extractor_config: ProfileExtractorConfig,
        service_config: ProfileGenerationServiceConfig,
        agent_context: str,
    ):
        """
        Initialize the profile extractor.

        Args:
            request_context: Request context with storage and prompt manager
            llm_client: Unified LLM client supporting both OpenAI and Claude
            extractor_config: Profile extractor configuration from YAML
            service_config: Runtime service configuration with request data
            agent_context: Context about the agent
        """
        self.request_context = request_context
        self.client = llm_client
        self.config: ProfileExtractorConfig = extractor_config
        self.service_config: ProfileGenerationServiceConfig = service_config
        self.agent_context = agent_context
        self._last_resumable_run_id: str | None = None

        # Get LLM config overrides from configuration
        config = self.request_context.configurator.get_config()
        llm_config = config.llm_config if config else None

        # Resolve model names: config override → site var → auto-detect
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
            OperationStateManager configured for profile_extractor
        """
        return OperationStateManager(
            self.request_context.storage,  # type: ignore[reportArgumentType]
            self.request_context.org_id,
            "profile_extractor",
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

        # Get window interactions with time range filter
        session_data_models, _ = storage.get_last_k_interactions_grouped(  # type: ignore[reportOptionalMemberAccess]
            user_id=self.service_config.user_id,
            k=window_size,
            sources=effective_source,
            start_time=self.service_config.rerun_start_time,
            end_time=self.service_config.rerun_end_time,
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
            extractor_name=get_extractor_name(self.config),
            processed_interactions=all_interactions,
            user_id=self.service_config.user_id,
        )

    def run(self) -> list[UserProfile] | ExtractionOutcome[UserProfile] | None:
        """
        Extract profiles from request interaction groups.

        This extractor handles its own data collection:
        1. Gets interactions based on its config (window size, source filtering)
        2. Applies time range filter for rerun flows
        3. Calls LLM to extract profiles
        4. Converts raw extraction to UserProfile objects
        5. Updates operation state after processing

        Returns:
            Optional[list[UserProfile]]: List of extracted profiles, or None if no profiles found
        """
        # Collect interactions using extractor's own window_size/stride_size settings
        request_interaction_data_models = self._get_interactions()
        if not request_interaction_data_models:
            return None

        # Limit existing profiles to most recent for context
        existing_profiles = self.service_config.existing_data or []
        context_profiles = sorted(
            existing_profiles,
            key=lambda p: p.last_modified_timestamp,
            reverse=True,
        )[:MAX_EXISTING_PROFILES_FOR_CONTEXT]

        try:
            raw_profiles = self._generate_raw_updates_from_sessions(
                request_interaction_data_models=request_interaction_data_models,
                existing_profiles=context_profiles,
            )
        except Exception as e:
            logger.error(
                "event=profile_extract_failed user_id=%s extractor_name=%s error_type=%s error=%s",
                self.service_config.user_id,
                get_extractor_name(self.config),
                type(e).__name__,
                str(e),
            )
            raise RuntimeError(
                f"Profile extraction failed for user {self.service_config.user_id}"
            ) from e

        logger.info("Generated raw profiles: %s", raw_profiles)
        if isinstance(raw_profiles, ExtractionOutcome):
            user_profiles = self._convert_raw_to_user_profiles(
                raw_profiles=raw_profiles.items,
                user_id=self.service_config.user_id,
                request_id=self.service_config.request_id,
            )
            self._update_operation_state(request_interaction_data_models)
            return ExtractionOutcome.completed(
                user_profiles, run_id=raw_profiles.run_id
            )
        user_profiles = self._convert_raw_to_user_profiles(
            raw_profiles=raw_profiles or [],
            user_id=self.service_config.user_id,
            request_id=self.service_config.request_id,
        )
        if raw_profiles:
            # Update operation state (bookmark) only when output was produced.
            self._update_operation_state(request_interaction_data_models)

        # A resumable run must always surface its run_id so the generation
        # service can finalize the _agent_runs row (FINALIZED_PENDING_TOOL when
        # the agent created a follow-up ask and finished with empty output).
        # Dropping the run_id here would orphan the run in AGENT_COMPLETED and
        # sever the resolve -> resume chain. Mirrors PlaybookExtractor.run().
        if self._last_resumable_run_id:
            return ExtractionOutcome.completed(
                user_profiles,
                run_id=self._last_resumable_run_id,
            )
        return user_profiles or None

    def _convert_raw_to_user_profiles(
        self,
        raw_profiles: list[dict],
        user_id: str,
        request_id: str,
    ) -> list[UserProfile]:
        """
        Convert raw profile dicts from LLM to UserProfile objects.

        Args:
            raw_profiles: List of profile dicts with content, time_to_live, and optional metadata
            user_id: User ID
            request_id: Request ID

        Returns:
            List of UserProfile objects
        """
        new_profiles = []
        for profile_content in raw_profiles:
            if (
                not isinstance(profile_content, dict)
                or "content" not in profile_content
            ):
                logger.warning("Invalid profile content: %s", profile_content)
                continue

            # Get all custom features by excluding content and time_to_live
            custom_features = {
                k: v
                for k, v in profile_content.items()
                if k not in ["content", "time_to_live"]
            }

            now_ts = int(datetime.now(UTC).timestamp())
            ttl = ProfileTimeToLive(profile_content.get("time_to_live", "infinity"))

            added_profile = UserProfile(
                profile_id=new_profile_id(),
                user_id=user_id,
                content=profile_content["content"],
                last_modified_timestamp=now_ts,
                generated_from_request_id=request_id,
                profile_time_to_live=ttl,
                expiration_timestamp=calculate_expiration_timestamp(now_ts, ttl),
                custom_features=custom_features or None,
                extractor_names=None,
            )

            new_profiles.append(added_profile)
        return new_profiles

    def _generate_raw_updates_from_sessions(
        self,
        request_interaction_data_models: list[RequestInteractionDataModel],
        existing_profiles: list[UserProfile],
    ) -> list[dict]:
        """
        Generate raw profile extractions from request interaction groups.

        Args:
            request_interaction_data_models: List of request interaction groups
            existing_profiles: List of existing user profiles for context

        Returns:
            list[dict]: List of profile dicts with content, time_to_live, and optional metadata
        """
        # Check if mock mode is enabled
        mock_env_for_raw = os.getenv("MOCK_LLM_RESPONSE", "")
        if mock_env_for_raw.lower() == "true":
            return self._generate_mock_profiles(
                request_interaction_data_models=request_interaction_data_models,
            )

        messages = construct_profile_extraction_messages_from_sessions(
            prompt_manager=self.request_context.prompt_manager,
            request_interaction_data_models=request_interaction_data_models,
            agent_context_prompt=self.agent_context,
            context_prompt=(
                self.config.context_prompt.strip() if self.config.context_prompt else ""
            ),
            extraction_definition_prompt=self.config.extraction_definition_prompt.strip(),
            metadata_definition_prompt=(
                self.config.metadata_definition_prompt.strip()
                if self.config.metadata_definition_prompt
                else None
            ),
            existing_profiles=existing_profiles,
        )

        messages_dict = messages
        session_count = len(request_interaction_data_models)
        interaction_count = sum(
            len(data_model.interactions)
            for data_model in request_interaction_data_models
        )
        history_chars = len(
            format_sessions_to_history_string(request_interaction_data_models)
        )
        logger.info(
            "event=profile_extract_llm_start user_id=%s extractor_name=%s sessions=%d interactions=%d history_chars=%d existing_profiles=%d model=%s timeout=%d max_retries=%d response_format=%s",
            self.service_config.user_id,
            get_extractor_name(self.config),
            session_count,
            interaction_count,
            history_chars,
            len(existing_profiles),
            self.default_generation_model_name,
            PROFILE_EXTRACTION_TIMEOUT_SECONDS,
            PROFILE_EXTRACTION_MAX_RETRIES,
            "StructuredProfilesOutput",
        )

        log_llm_messages(logger, "Profile extraction", messages_dict)

        result = run_resumable_extraction_agent(
            request_context=self.request_context,
            client=self.client,
            extractor_kind="profile",
            user_id=self.service_config.user_id,
            request_id=self.service_config.request_id,
            agent_version=None,
            source=self.service_config.source,
            request_interaction_data_models=request_interaction_data_models,
            extractor_config=self.config,
            service_config=self.service_config,
            agent_context=self.agent_context,
            messages=messages_dict,
            output_schema=StructuredProfilesOutput,
            log_label="Profile extraction",
        )
        self._last_resumable_run_id = result.run_id
        if not isinstance(result.output, StructuredProfilesOutput):
            logger.warning(
                "Profile extraction did not finish: %s", result.finished_reason
            )
            return []
        return [p.model_dump() for p in (result.output.profiles or [])]

    def _generate_mock_profiles(
        self,
        request_interaction_data_models: list[RequestInteractionDataModel],
    ) -> list[dict]:
        """
        Generate mock profile extractions for testing.

        Args:
            request_interaction_data_models: List of request interaction groups

        Returns:
            list[dict]: Mock profile dicts
        """
        interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )

        if not interactions:
            return []

        sample_content = (
            interactions[-1].content[:50]
            if interactions[-1].content
            else "sample interaction"
        )

        # Capture additional context that contains helpful keywords
        highlight_keywords = {
            "software",
            "solution",
            "product",
            "company",
            "service",
        }
        highlighted_snippet = next(
            (
                interaction.content[:80]
                for interaction in reversed(interactions)
                if interaction.content
                and any(
                    keyword in interaction.content.lower()
                    for keyword in highlight_keywords
                )
            ),
            "",
        )

        summary_parts = [f"User mentioned: {sample_content}"]
        if highlighted_snippet and highlighted_snippet not in sample_content:
            summary_parts.append(f"Key context: {highlighted_snippet}")

        mock_profile = {
            "content": " ".join(summary_parts),
            "time_to_live": "one_month",
        }

        # If metadata definition exists, add mock metadata
        if self.config.metadata_definition_prompt:
            mock_profile["metadata"] = "mock_metadata_value"

        return [mock_profile]
