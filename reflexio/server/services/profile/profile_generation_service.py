"""Service to generate user profiles from interactions"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    DeleteUserProfileRequest,
    DowngradeProfilesResponse,
    ManualProfileGenerationRequest,
    ManualProfileGenerationResponse,
    ProfileChangeLog,
    RerunProfileGenerationRequest,
    RerunProfileGenerationResponse,
    Status,
    UpgradeProfilesResponse,
    UserProfile,
)
from reflexio.models.config_schema import ProfileExtractorConfig
from reflexio.server.services.base_generation_service import (
    BaseGenerationService,
    StatusChangeOperation,
)
from reflexio.server.services.profile.profile_extractor import ProfileExtractor
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
    ProfileGenerationServiceConstants,
)
from reflexio.server.services.service_utils import (
    format_sessions_to_history_string,
)

logger = logging.getLogger(__name__)


@dataclass
class ProfileGenerationServiceConfig:
    """Runtime configuration for profile generation service shared across all extractors.

    Attributes:
        user_id: The user ID
        request_id: The request ID
        source: Source of the interactions (triggering source)
        existing_data: Existing profiles for the user
        allow_manual_trigger: Whether to allow extractors with manual_trigger=True
        output_pending_status: Whether to output profiles with PENDING status
        extractor_names: Optional list of extractor names to filter which extractors run
        rerun_start_time: Optional start time filter for rerun flows (Unix timestamp)
        rerun_end_time: Optional end time filter for rerun flows (Unix timestamp)
        auto_run: True for regular flow (checks batch_interval), False for rerun/manual (skips batch_interval)
    """

    user_id: str
    request_id: str
    source: str | None = None
    existing_data: Any = None
    allow_manual_trigger: bool = False
    output_pending_status: bool = False
    extractor_names: list[str] | None = None
    rerun_start_time: int | None = None
    rerun_end_time: int | None = None
    auto_run: bool = True
    force_extraction: bool = False
    is_incremental: bool = False
    previously_extracted: list[list[UserProfile]] = field(default_factory=list)


class ProfileGenerationService(
    BaseGenerationService[
        ProfileExtractorConfig,
        ProfileExtractor,
        ProfileGenerationServiceConfig,
        ProfileGenerationRequest,
    ]
):
    """Service to generate user profiles from interactions"""

    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        allow_manual_trigger: bool = False,
        output_pending_status: bool = False,
    ) -> None:
        """
        Initialize the profile generation service.

        Args:
            llm_client: Unified LLM client supporting both OpenAI and Claude
            request_context: Request context with storage, configurator, and org_id
            allow_manual_trigger: Whether to allow extractors with manual_trigger=True
            output_pending_status: Whether to output profiles with PENDING status (for rerun)
        """
        super().__init__(llm_client=llm_client, request_context=request_context)
        self.allow_manual_trigger = allow_manual_trigger
        self.output_pending_status = output_pending_status

    def _load_generation_service_config(
        self, request: ProfileGenerationRequest
    ) -> ProfileGenerationServiceConfig:
        """
        Extract parameters from ProfileGenerationRequest.

        Args:
            request: ProfileGenerationRequest containing request interaction groups and metadata

        Returns:
            ProfileGenerationServiceConfig object
        """
        # Get existing profiles for the user
        # When output_pending_status is True (rerun mode), only include pending profiles as existing data
        # This allows the LLM to generate fresh profiles instead of just mentioning current ones
        if self.output_pending_status:
            existing_profiles = self.storage.get_user_profile(  # type: ignore[reportOptionalMemberAccess]
                request.user_id, status_filter=[Status.PENDING]
            )
        else:
            existing_profiles = self.storage.get_user_profile(request.user_id)  # type: ignore[reportOptionalMemberAccess]

        return ProfileGenerationServiceConfig(
            user_id=request.user_id,
            request_id=request.request_id,
            source=request.source,
            existing_data=existing_profiles,
            allow_manual_trigger=self.allow_manual_trigger,
            output_pending_status=self.output_pending_status,
            extractor_names=getattr(request, "extractor_names", None),
            rerun_start_time=request.rerun_start_time,
            rerun_end_time=request.rerun_end_time,
            auto_run=request.auto_run,
            force_extraction=request.force_extraction,
        )

    def _process_results(self, results: list[list[UserProfile]]) -> None:
        """
        Process, deduplicate, and apply all extracted profiles. Called once after all extractors complete.

        Args:
            results: List of profile lists from extractors (one list per successful extractor)
        """
        user_id = self.service_config.user_id  # type: ignore[reportOptionalMemberAccess]
        source = self.service_config.source  # type: ignore[reportOptionalMemberAccess]
        request_id = self.service_config.request_id  # type: ignore[reportOptionalMemberAccess]

        # Flatten all new profiles
        all_new_profiles = [p for result in results if result for p in result]
        existing_ids_to_delete: list[str] = []
        superseded_profiles: list[UserProfile] = []

        # Always run deduplicator when enabled and there are new profiles
        if all_new_profiles:
            from reflexio.server.site_var.feature_flags import is_deduplicator_enabled

            if is_deduplicator_enabled(self.org_id):
                from reflexio.server.services.profile.profile_deduplicator import (
                    ProfileDeduplicator,
                )

                deduplicator = ProfileDeduplicator(
                    request_context=self.request_context,
                    llm_client=self.client,
                )
                all_new_profiles, existing_ids_to_delete, superseded_profiles = (
                    deduplicator.deduplicate(all_new_profiles, user_id, request_id)
                )
                logger.info(
                    "Profile updates after deduplication: %d profiles, %d existing to delete",
                    len(all_new_profiles),
                    len(existing_ids_to_delete),
                )

        # Set source and status for all profiles
        for profile in all_new_profiles:
            profile.source = source
            profile.status = Status.PENDING if self.output_pending_status else None

        # Save new profiles
        if all_new_profiles:
            try:
                self.storage.add_user_profile(user_id, all_new_profiles)  # type: ignore[reportOptionalMemberAccess]
            except Exception as e:
                logger.error(
                    "Failed to save profiles for user id: %s due to %s, exception type: %s",
                    user_id,
                    str(e),
                    type(e).__name__,
                )
                return

        # Delete superseded existing profiles
        if existing_ids_to_delete:
            for profile_id in existing_ids_to_delete:
                try:
                    self.storage.delete_user_profile(  # type: ignore[reportOptionalMemberAccess]
                        DeleteUserProfileRequest(
                            user_id=user_id,
                            profile_id=profile_id,
                        )
                    )
                except Exception as e:  # noqa: PERF203
                    logger.error(
                        "Failed to delete superseded profile %s for user %s: %s",
                        profile_id,
                        user_id,
                        str(e),
                    )

        # Create profile changelog post-deduplication
        if all_new_profiles or superseded_profiles:
            try:
                profile_change_log = ProfileChangeLog(
                    id=0,  # Auto-generated by storage
                    user_id=user_id,
                    request_id=request_id,
                    created_at=int(datetime.now(UTC).timestamp()),
                    added_profiles=all_new_profiles,
                    removed_profiles=superseded_profiles,
                    mentioned_profiles=[],
                )
                self.storage.add_profile_change_log(profile_change_log)  # type: ignore[reportOptionalMemberAccess]
            except Exception as e:
                logger.error(
                    "Failed to add profile change log for user %s: %s",
                    user_id,
                    str(e),
                )

    def check_and_update_profiles(self, profiles: list[UserProfile]) -> None:
        """check if the profiles are expired and update them if they are"""
        raise NotImplementedError

    def _load_extractor_configs(self) -> list[ProfileExtractorConfig]:
        """
        Load profile extractor configs from configurator.

        Returns:
            list[ProfileExtractorConfig]: List of profile extractor configuration objects from YAML
        """
        return self.configurator.get_config().profile_extractor_configs  # type: ignore[reportReturnType]

    def _create_extractor(
        self,
        extractor_config: ProfileExtractorConfig,
        service_config: ProfileGenerationServiceConfig,
    ) -> ProfileExtractor:
        """
        Create a ProfileExtractor instance from configuration.

        Args:
            extractor_config: ProfileExtractorConfig configuration object from YAML
            service_config: ProfileGenerationServiceConfig containing runtime parameters

        Returns:
            ProfileExtractor instance
        """
        return ProfileExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=self.configurator.get_agent_context(),
        )

    def _build_should_run_prompt(
        self,
        scoped_configs: list[ProfileExtractorConfig],
        session_data_models: list[RequestInteractionDataModel],
    ) -> str | None:
        """
        Build prompt for consolidated should_extract_profile check.

        Combines all enabled extractors' profile definitions and override conditions
        into a single criteria block for one LLM call.

        Args:
            scoped_configs: Profile extractor configs that had scoped interactions
            session_data_models: Deduplicated request interaction data models

        Returns:
            str | None: The rendered prompt, or None if no criteria to check
        """
        new_interactions = format_sessions_to_history_string(session_data_models)
        agent_context = self.configurator.get_agent_context()
        prompt_manager = self.request_context.prompt_manager

        # Combine all extractor criteria into a numbered list.
        # Each extractor can contribute:
        # 1) extraction_definition_prompt (what to extract)
        # 2) should_extract_profile_prompt_override (custom extraction condition)
        combined_criteria_items = []
        for i, config in enumerate(scoped_configs, 1):
            criteria_parts = []
            if config.extraction_definition_prompt:
                criteria_parts.append(
                    f"definition: {config.extraction_definition_prompt.strip()}"
                )
            if config.should_extract_profile_prompt_override:
                criteria_parts.append(
                    "condition: "
                    f"{config.should_extract_profile_prompt_override.strip()}"
                )

            if criteria_parts:
                combined_criteria_items.append(f"{i}. {'; '.join(criteria_parts)}")

        combined_criteria = (
            "\n".join(combined_criteria_items) if combined_criteria_items else ""
        )
        if not combined_criteria:
            return None

        return prompt_manager.render_prompt(
            ProfileGenerationServiceConstants.PROFILE_SHOULD_GENERATE_PROMPT_ID,
            {
                "agent_context_prompt": agent_context,
                "should_extract_profile_prompt": combined_criteria,
                "new_interactions": new_interactions,
            },
        )

    def _update_config_for_incremental(self, previously_extracted: list) -> None:
        """Update service_config for incremental profile extraction."""
        self.service_config.is_incremental = True  # type: ignore[reportOptionalMemberAccess]
        self.service_config.previously_extracted = list(previously_extracted)  # type: ignore[reportOptionalMemberAccess]

    def _get_extractor_state_service_name(self) -> str:
        """
        Get the service name for batch_interval bookmark lookups.

        Returns:
            str: "profile_extractor" for OperationStateManager batch_interval checks
        """
        return "profile_extractor"

    def _get_service_name(self) -> str:
        """
        Get the name of the service for logging and operation state tracking.

        Returns:
            Service name string - "rerun_profile_generation" for rerun operations,
            "profile_generation" for regular operations
        """
        if self.output_pending_status:
            return "rerun_profile_generation"
        return "profile_generation"

    def _get_base_service_name(self) -> str:
        """
        Get the base service name for OperationStateManager keys.

        Returns:
            str: "profile_generation"
        """
        return "profile_generation"

    def _should_track_in_progress(self) -> bool:
        """
        Profile generation should track in-progress state to prevent duplicates.

        Returns:
            bool: True - profile generation tracks in-progress state
        """
        return True

    def _get_lock_scope_id(self, request: ProfileGenerationRequest) -> str | None:
        """
        Get the scope ID for lock key construction.

        Profile generation is user-scoped, so returns user_id.

        Args:
            request: The ProfileGenerationRequest

        Returns:
            str: The user_id from the request
        """
        return request.user_id

    # ===============================
    # Rerun hook implementations (override base class methods)
    # ===============================

    def _get_rerun_user_ids(self, request: RerunProfileGenerationRequest) -> list[str]:
        """Get user IDs to process. Extractors collect their own data.

        Identifies unique user_ids with matching requests via storage-level filtering.

        Args:
            request: RerunProfileGenerationRequest with optional filters

        Returns:
            List of user IDs to process
        """
        return self.storage.get_rerun_user_ids(  # type: ignore[reportOptionalMemberAccess]
            user_id=request.user_id,
            start_time=(
                int(request.start_time.timestamp()) if request.start_time else None
            ),
            end_time=(int(request.end_time.timestamp()) if request.end_time else None),
            source=request.source,
        )

    def _build_rerun_request_params(
        self, request: RerunProfileGenerationRequest
    ) -> dict:
        """Build request params dict for operation state tracking.

        Args:
            request: Original rerun request

        Returns:
            Dictionary of request parameters
        """
        return {
            "user_id": request.user_id,
            "start_time": (
                request.start_time.isoformat() if request.start_time else None
            ),
            "end_time": request.end_time.isoformat() if request.end_time else None,
            "source": request.source,
            "extractor_names": request.extractor_names,
        }

    def _create_run_request_for_item(
        self,
        user_id: str,
        request: RerunProfileGenerationRequest | ManualProfileGenerationRequest,
    ) -> ProfileGenerationRequest:
        """Create ProfileGenerationRequest for a single user.

        Handles both rerun and manual request types.

        Args:
            user_id: The user ID to process
            request: The original rerun or manual request

        Returns:
            ProfileGenerationRequest for this user with filter constraints
        """
        # Handle rerun requests (have start_time/end_time datetime objects)
        if isinstance(request, RerunProfileGenerationRequest):
            return ProfileGenerationRequest(
                user_id=user_id,
                request_id=f"rerun_{uuid.uuid4().hex[:8]}",
                source=request.source,
                extractor_names=request.extractor_names,
                rerun_start_time=(
                    int(request.start_time.timestamp()) if request.start_time else None
                ),
                rerun_end_time=(
                    int(request.end_time.timestamp()) if request.end_time else None
                ),
                auto_run=False,
            )
        # Handle manual requests (ManualProfileGenerationRequest)
        return ProfileGenerationRequest(
            user_id=user_id,
            request_id=f"manual_{uuid.uuid4().hex[:8]}",
            source=request.source,
            extractor_names=request.extractor_names,
            auto_run=False,
        )

    def _create_rerun_response(
        self, success: bool, msg: str, count: int
    ) -> RerunProfileGenerationResponse:
        """Create RerunProfileGenerationResponse.

        Args:
            success: Whether the operation succeeded
            msg: Status message
            count: Number of profiles generated

        Returns:
            RerunProfileGenerationResponse
        """
        return RerunProfileGenerationResponse(
            success=success,
            msg=msg,
            profiles_generated=count,
        )

    def _get_generated_count(
        self,
        request: RerunProfileGenerationRequest,
        processed_user_ids: list[str] | None = None,
    ) -> int:
        """Get the count of profiles generated during rerun.

        Counts profiles with PENDING status for each processed user.

        Args:
            request: The rerun request object
            processed_user_ids: List of user IDs processed in the batch

        Returns:
            Number of profiles generated
        """
        user_ids = processed_user_ids or [request.user_id]
        total = 0
        for user_id in user_ids:
            profiles = self.storage.get_user_profile(  # type: ignore[reportOptionalMemberAccess]
                user_id=user_id,
                status_filter=[Status.PENDING],
            )
            total += len(profiles)
        return total

    # ===============================
    # Upgrade/Downgrade hook implementations (override base class methods)
    # ===============================

    def _has_items_with_status(
        self,
        status: Status | None,
        request: ProfileGenerationRequest,  # noqa: ARG002
    ) -> bool:
        """Check if profiles exist with given status.

        Args:
            status: The status to check for (None for CURRENT)
            request: The upgrade/downgrade request object

        Returns:
            bool: True if any matching profiles exist
        """
        user_ids = self.storage.get_user_ids_with_status(status=status)  # type: ignore[reportOptionalMemberAccess]
        return bool(user_ids)

    def _delete_items_by_status(
        self,
        status: Status,
        request: ProfileGenerationRequest,  # noqa: ARG002
    ) -> int:
        """Delete profiles with given status.

        Args:
            status: The status of profiles to delete
            request: The upgrade/downgrade request object

        Returns:
            int: Number of profiles deleted
        """
        return self.storage.delete_all_profiles_by_status(status=status)  # type: ignore[reportOptionalMemberAccess]

    def _update_items_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        request: ProfileGenerationRequest,  # noqa: ARG002
        user_ids: list[str] | None = None,  # noqa: ARG002
    ) -> int:
        """Update profiles from old_status to new_status.

        Args:
            old_status: The current status to match (None for CURRENT)
            new_status: The new status to set (None for CURRENT)
            request: The upgrade/downgrade request object
            user_ids: Optional pre-computed list of user IDs to filter by

        Returns:
            int: Number of profiles updated
        """
        return self.storage.update_all_profiles_status(  # type: ignore[reportOptionalMemberAccess]
            old_status, new_status, user_ids=user_ids
        )

    def _get_affected_user_ids_for_upgrade(
        self, request: ProfileGenerationRequest
    ) -> list[str] | None:
        """Get user IDs to filter by for upgrade operations.

        Args:
            request: The upgrade request object

        Returns:
            Optional[list[str]]: List of user IDs with PENDING profiles, or None for no filtering
        """
        if hasattr(request, "only_affected_users") and request.only_affected_users:  # type: ignore[reportAttributeAccessIssue]
            return self.storage.get_user_ids_with_status(Status.PENDING)  # type: ignore[reportOptionalMemberAccess]
        return None

    def _get_affected_user_ids_for_downgrade(
        self, request: ProfileGenerationRequest
    ) -> list[str] | None:
        """Get user IDs to filter by for downgrade operations.

        Args:
            request: The downgrade request object

        Returns:
            Optional[list[str]]: List of user IDs with ARCHIVED profiles, or None for no filtering
        """
        if hasattr(request, "only_affected_users") and request.only_affected_users:  # type: ignore[reportAttributeAccessIssue]
            return self.storage.get_user_ids_with_status(Status.ARCHIVED)  # type: ignore[reportOptionalMemberAccess]
        return None

    def _create_status_change_response(
        self,
        operation: StatusChangeOperation,
        success: bool,
        counts: dict,
        msg: str,
    ) -> UpgradeProfilesResponse | DowngradeProfilesResponse:
        """Create upgrade or downgrade response object for profiles.

        Args:
            operation: The operation type (UPGRADE or DOWNGRADE)
            success: Whether the operation succeeded
            counts: Dictionary of counts
            msg: Status message

        Returns:
            UpgradeProfilesResponse or DowngradeProfilesResponse
        """
        if operation == StatusChangeOperation.UPGRADE:
            return UpgradeProfilesResponse(
                success=success,
                profiles_deleted=counts.get("deleted", 0),
                profiles_archived=counts.get("archived", 0),
                profiles_promoted=counts.get("promoted", 0),
                message=msg,
            )
        # DOWNGRADE
        return DowngradeProfilesResponse(
            success=success,
            profiles_demoted=counts.get("demoted", 0),
            profiles_restored=counts.get("restored", 0),
            message=msg,
        )

    # ===============================
    # Manual Regular Generation (window-sized, CURRENT output)
    # ===============================

    def run_manual_regular(
        self, request: ManualProfileGenerationRequest
    ) -> ManualProfileGenerationResponse:
        """
        Run profile generation with window-sized interactions and CURRENT output.

        This is a manual trigger that behaves like regular generation
        (uses batch_size, outputs CURRENT profiles) but only runs
        profile extraction (not feedback or agent success).

        Each extractor collects its own data using its configured batch_size.
        Uses progress tracking via OperationStateManager.

        Args:
            request: ManualProfileGenerationRequest with optional user_id, source, and extractor_names

        Returns:
            ManualProfileGenerationResponse with success status and count
        """
        state_manager = self._create_state_manager()

        try:
            # Check for existing in-progress operation
            error = state_manager.check_in_progress()
            if error:
                return ManualProfileGenerationResponse(
                    success=False, msg=error, profiles_generated=0
                )

            # 1. Get users to process
            if request.user_id:
                user_ids = [request.user_id]
            else:
                user_ids = self.storage.get_all_user_ids()  # type: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

            if not user_ids:
                return ManualProfileGenerationResponse(
                    success=True,
                    msg="No users found to process",
                    profiles_generated=0,
                )

            # 2. Run batch with progress tracking
            request_params = {
                "user_id": request.user_id,
                "source": request.source,
                "extractor_names": request.extractor_names,
                "mode": "manual_regular",
            }
            users_processed, _ = self._run_batch_with_progress(
                user_ids=user_ids,
                request=request,  # type: ignore[reportArgumentType]
                request_params=request_params,
                state_manager=state_manager,
            )

            # 3. Count generated profiles (CURRENT status = None)
            total_profiles = self._count_manual_generated(request)

            return ManualProfileGenerationResponse(
                success=True,
                msg=f"Generated {total_profiles} profiles for {users_processed} user(s)",
                profiles_generated=total_profiles,
            )

        except Exception as e:
            state_manager.mark_progress_failed(str(e))
            return ManualProfileGenerationResponse(
                success=False,
                msg=f"Failed to run manual profile generation: {str(e)}",
                profiles_generated=0,
            )

    def _count_manual_generated(self, request: ManualProfileGenerationRequest) -> int:
        """
        Count profiles generated during manual regular generation.

        Counts profiles with CURRENT status (None), optionally filtered by user_id.

        Args:
            request: The manual generation request object

        Returns:
            Number of profiles with CURRENT status
        """
        profiles = self.storage.get_user_profile(  # type: ignore[reportOptionalMemberAccess]
            user_id=request.user_id,  # type: ignore[reportArgumentType]
            status_filter=[None],  # CURRENT profiles
        )
        return len(profiles)
