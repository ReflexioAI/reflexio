from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    DowngradeUserPlaybooksResponse,
    ManualPlaybookGenerationRequest,
    ManualPlaybookGenerationResponse,
    RerunPlaybookGenerationRequest,
    RerunPlaybookGenerationResponse,
    Status,
    UpgradeUserPlaybooksResponse,
    UserPlaybook,
)
from reflexio.models.config_schema import PlaybookConfig
from reflexio.server.operation_limiter import run_with_operation_limit
from reflexio.server.services.base_generation_service import (
    BaseGenerationService,
    StatusChangeOperation,
)
from reflexio.server.services.playbook.playbook_aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_extractor import PlaybookExtractor
from reflexio.server.services.playbook.playbook_service_constants import (
    PlaybookServiceConstants,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
    PlaybookGenerationRequest,
    format_expert_comparison_pairs,
    has_expert_content,
)
from reflexio.server.services.polarity_utils import (
    warn_if_polarity_content_mismatch,
)
from reflexio.server.services.service_utils import (
    extract_interactions_from_request_interaction_data_models,
    format_sessions_to_history_string,
)

logger = logging.getLogger(__name__)


@dataclass
class PlaybookGenerationServiceConfig:
    """Runtime configuration for playbook generation service shared across all extractors.

    Attributes:
        request_id: The request ID
        agent_version: The agent version
        user_id: The user ID for per-user playbook extraction
        source: Source of the interactions
        allow_manual_trigger: Whether to allow extractors with manual_trigger=True
        rerun_start_time: Optional start time filter for rerun flows (Unix timestamp)
        rerun_end_time: Optional end time filter for rerun flows (Unix timestamp)
        auto_run: True for regular flow (checks stride_size), False for rerun/manual (skips stride_size)
        extractor_names: Optional list of extractor names to run (derived from playbook_name)
    """

    request_id: str
    agent_version: str
    user_id: str | None = None
    source: str | None = None
    allow_manual_trigger: bool = False
    rerun_start_time: int | None = None
    rerun_end_time: int | None = None
    auto_run: bool = True
    force_extraction: bool = False
    extractor_names: list[str] | None = None


class PlaybookGenerationService(
    BaseGenerationService[
        PlaybookConfig,
        PlaybookExtractor,
        PlaybookGenerationServiceConfig,
        PlaybookGenerationRequest,
    ]
):
    """
    Service for generating playbook entries from user interactions.
    Runs the configured PlaybookExtractor for each generation request.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
        allow_manual_trigger: bool = False,
        output_pending_status: bool = False,
        skip_aggregation: bool = False,
    ) -> None:
        """
        Initialize the playbook generation service.

        Args:
            llm_client: Unified LLM client supporting both OpenAI and Claude
            request_context: Request context with storage, configurator, and org_id
            allow_manual_trigger: Whether to allow extractors with manual_trigger=True
            output_pending_status: Whether to output entries with PENDING status (for rerun)
            skip_aggregation: Whether to skip playbook aggregation (extract only, no agent playbooks)
        """
        super().__init__(llm_client=llm_client, request_context=request_context)
        self.allow_manual_trigger = allow_manual_trigger
        self.output_pending_status = output_pending_status
        self.skip_aggregation = skip_aggregation

    def _load_generation_service_config(
        self, request: PlaybookGenerationRequest
    ) -> PlaybookGenerationServiceConfig:
        """
        Extract request parameters from PlaybookGenerationRequest.

        Args:
            request: PlaybookGenerationRequest containing evaluation parameters

        Returns:
            PlaybookGenerationServiceConfig object
        """
        return PlaybookGenerationServiceConfig(
            request_id=request.request_id,
            agent_version=request.agent_version,
            user_id=request.user_id,
            source=request.source,
            allow_manual_trigger=self.allow_manual_trigger,
            rerun_start_time=request.rerun_start_time,
            rerun_end_time=request.rerun_end_time,
            auto_run=request.auto_run,
            force_extraction=request.force_extraction,
            extractor_names=[request.playbook_name] if request.playbook_name else None,
        )

    def _configured_playbook_config(self) -> PlaybookConfig | None:
        root_config = self.configurator.get_config()
        return getattr(root_config, "user_playbook_extractor_config", None)

    def _load_extractor_config(self) -> PlaybookConfig | None:
        """
        Load the configured user playbook extractor from configurator.

        Returns:
            PlaybookConfig | None: The configured user playbook extractor, if enabled.
        """
        return self._configured_playbook_config()

    def _create_extractor(
        self,
        extractor_config: PlaybookConfig,
        service_config: PlaybookGenerationServiceConfig,
    ) -> PlaybookExtractor:
        """
        Create a PlaybookExtractor instance from configuration.

        Args:
            extractor_config: PlaybookConfig configuration object from YAML
            service_config: PlaybookGenerationServiceConfig containing runtime parameters

        Returns:
            PlaybookExtractor instance
        """
        return PlaybookExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=self.configurator.get_agent_context(),
        )

    def _build_should_run_prompt(
        self,
        scoped_config: PlaybookConfig,
        session_data_models: list[RequestInteractionDataModel],
    ) -> str | None:
        """
        Build prompt for consolidated should_generate check.

        Renders the configured playbook definition for one LLM call.

        Args:
            scoped_config: Playbook extractor config that had scoped interactions
            session_data_models: Deduplicated request interaction data models

        Returns:
            str | None: The rendered prompt, or None if no definitions to check
        """
        # Check for expert content — use expert-specific should-generate prompt
        all_interactions = extract_interactions_from_request_interaction_data_models(
            session_data_models
        )
        if has_expert_content(all_interactions):
            agent_context = self.configurator.get_agent_context()
            comparison_pairs = format_expert_comparison_pairs(session_data_models)
            return self.request_context.prompt_manager.render_prompt(
                PlaybookServiceConstants.PLAYBOOK_SHOULD_GENERATE_EXPERT_PROMPT_ID,
                {
                    "agent_context_prompt": agent_context,
                    "comparison_pairs": comparison_pairs,
                },
            )

        new_interactions = format_sessions_to_history_string(session_data_models)

        combined_definition = (
            scoped_config.extraction_definition_prompt.strip()
            if scoped_config.extraction_definition_prompt
            else ""
        )

        if not combined_definition:
            return None

        # Get tool_can_use from root config
        root_config = self.request_context.configurator.get_config()
        tool_can_use_str = ""
        if root_config and root_config.tool_can_use:
            tool_can_use_str = "\n".join(
                f"{tool.tool_name}: {tool.tool_description}"
                for tool in root_config.tool_can_use
            )

        agent_context = self.configurator.get_agent_context()
        prompt_manager = self.request_context.prompt_manager

        return prompt_manager.render_prompt(
            PlaybookServiceConstants.PLAYBOOK_SHOULD_GENERATE_PROMPT_ID,
            {
                "agent_context_prompt": agent_context,
                "extraction_definition_prompt": combined_definition,
                "new_interactions": new_interactions,
                "tool_can_use": tool_can_use_str,
            },
        )

    def _get_precheck_interaction_query_kwargs(self) -> dict:
        """Return agent_version filter for non-auto runs."""
        return {
            "agent_version": (
                self.service_config.agent_version  # type: ignore[reportOptionalMemberAccess]
                if not self.service_config.auto_run  # type: ignore[reportOptionalMemberAccess]
                else None
            ),
        }

    def _process_results(self, results: list[list[UserPlaybook]]) -> None:
        """
        Process, deduplicate, and save all results. Called once after all extractors complete.

        Args:
            results: List of UserPlaybook results from extractors (one list per extractor)
        """
        all_playbooks = []
        for result in results:
            if isinstance(result, list):
                all_playbooks.extend(result)
        self._finalize_extracted_items(all_playbooks)

    def _finalize_extracted_items(self, all_playbooks: list[UserPlaybook]) -> None:
        """Deduplicate, persist, and aggregate extracted user playbook items."""
        # Deduplicate against existing entries in DB when deduplicator is enabled
        existing_ids_to_delete: list[int] = []
        from reflexio.server.site_var.feature_flags import is_deduplicator_enabled

        if is_deduplicator_enabled(self.org_id):
            from reflexio.server.services.playbook.playbook_consolidator import (
                PlaybookConsolidator,
            )

            playbook_config = self._configured_playbook_config()
            dedup_config = (
                playbook_config.deduplication_config if playbook_config else None
            )

            consolidator = PlaybookConsolidator(
                request_context=self.request_context,
                llm_client=self.client,
                dedup_config=dedup_config,
            )
            deduplicated_playbooks, existing_ids_to_delete = consolidator.deduplicate(
                [all_playbooks],
                self.service_config.request_id,  # type: ignore[reportOptionalMemberAccess]
                self.service_config.agent_version,  # type: ignore[reportOptionalMemberAccess]
                user_id=self.service_config.user_id,  # type: ignore[reportOptionalMemberAccess]
            )
            logger.info(
                "User playbook entries after deduplication: %d",
                len(deduplicated_playbooks),
            )
            if deduplicated_playbooks:
                all_playbooks = deduplicated_playbooks

        # Set status and source for all entries
        for playbook in all_playbooks:
            playbook.status = Status.PENDING if self.output_pending_status else None
            playbook.source = self.service_config.source  # type: ignore[reportOptionalMemberAccess]
            warn_if_polarity_content_mismatch(playbook)

        logger.info("All user playbook entries: %s", all_playbooks)

        logger.info(
            "Successfully completed %d %s playbook generation for request id: %s",
            len(all_playbooks),
            self._get_service_name(),
            self.service_config.request_id,  # type: ignore[reportOptionalMemberAccess]
        )

        # Save results
        if all_playbooks:
            try:
                self.storage.save_user_playbooks(all_playbooks)  # type: ignore[reportOptionalMemberAccess]
                self._enqueue_user_playbook_optimization(all_playbooks)

                # Delete superseded existing entries only after save succeeds
                if existing_ids_to_delete:
                    try:
                        deleted_count = self.storage.delete_user_playbooks_by_ids(  # type: ignore[reportOptionalMemberAccess]
                            existing_ids_to_delete
                        )
                        logger.info(
                            "Deleted %d superseded existing entries", deleted_count
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to delete superseded existing entries: %s",
                            str(e),
                        )
            except Exception as e:
                logger.error(
                    "Failed to save %s results for request id: %s due to %s, exception type: %s",
                    self._get_service_name(),
                    self.service_config.request_id,  # type: ignore[reportOptionalMemberAccess]
                    str(e),
                    type(e).__name__,
                )

            # Trigger playbook aggregation
            if not self.output_pending_status and not self.skip_aggregation:
                logger.info("Trigger playbook aggregation")
                self._trigger_playbook_aggregation()

    def _enqueue_user_playbook_optimization(
        self, saved_playbooks: list[UserPlaybook]
    ) -> None:
        config = self.configurator.get_config().playbook_optimizer_config
        if (
            not config.enabled
            or not config.optimize_user_playbooks
            or not saved_playbooks
        ):
            return
        from reflexio.server.services.playbook_optimizer import (
            PlaybookOptimizationScheduler,
            PlaybookOptimizationTarget,
            PlaybookOptimizer,
        )

        scheduler = PlaybookOptimizationScheduler.get_instance()
        for playbook in saved_playbooks:
            if (
                not playbook.user_playbook_id
                or playbook.status is not None
                or not playbook.source_interaction_ids
            ):
                continue
            target = PlaybookOptimizationTarget(
                kind="user_playbook", target_id=playbook.user_playbook_id
            )
            scheduler.enqueue(
                org_id=self.request_context.org_id,
                target=target,
                callback=lambda target=target: PlaybookOptimizer(
                    self.request_context, self.client
                ).optimize(target),
                jitter_seconds=config.scheduler_jitter_seconds,
                abort_cooldown_threshold=config.abort_cooldown_threshold,
                cooldown_after_aborts_seconds=config.cooldown_after_aborts_seconds,
            )

    def _get_extractor_state_service_name(self) -> str:
        """
        Get the service name for stride_size bookmark lookups.

        Returns:
            str: "playbook_extractor" for OperationStateManager stride_size checks
        """
        return "playbook_extractor"

    def _get_service_name(self) -> str:
        """
        Get the name of the service for logging and operation state tracking.

        Returns:
            Service name string - "rerun_playbook_generation" for rerun operations,
            "playbook_generation" for regular operations
        """
        if self.output_pending_status:
            return "rerun_playbook_generation"
        return "playbook_generation"

    def _get_base_service_name(self) -> str:
        """
        Get the base service name for OperationStateManager keys.

        Returns:
            str: "playbook_generation"
        """
        return "playbook_generation"

    def _should_track_in_progress(self) -> bool:
        """
        Playbook generation should track in-progress state to prevent duplicates.

        Returns:
            bool: True - playbook generation tracks in-progress state
        """
        return True

    def _get_lock_scope_id(self, request: PlaybookGenerationRequest) -> str | None:  # noqa: ARG002
        """
        Get the scope ID for lock key construction.

        Playbook generation is org-scoped, so returns None (no user scope).

        Args:
            request: The PlaybookGenerationRequest

        Returns:
            None: Playbook uses org-level scope only
        """
        return None

    def _trigger_playbook_aggregation(self) -> None:
        """
        Trigger playbook aggregation for playbook types that have aggregator config.
        This is called after raw user playbook entries are saved to check if aggregation should run.
        """
        playbook_config = self._configured_playbook_config()
        if not playbook_config or not playbook_config.aggregation_config:
            return

        playbook_name = playbook_config.extractor_name
        logger.info("Triggering aggregation for playbook_name: %s", playbook_name)

        # Create aggregator request
        aggregator_request = PlaybookAggregatorRequest(
            agent_version=self.service_config.agent_version,  # type: ignore[reportOptionalMemberAccess]
            playbook_name=playbook_name,
        )

        # Initialize and run aggregator (synchronous)
        aggregator = PlaybookAggregator(
            llm_client=self.client,
            request_context=self.request_context,
            agent_version=self.service_config.agent_version,  # type: ignore[reportOptionalMemberAccess]
        )
        try:
            run_with_operation_limit(
                org_id=self.request_context.org_id,
                operation="aggregation",
                fn=lambda: aggregator.run(aggregator_request),
            )
        except TimeoutError:
            logger.info(
                "Skipping inline aggregation for playbook_name=%s agent_version=%s: aggregation limiter is saturated",
                playbook_name,
                self.service_config.agent_version,  # type: ignore[reportOptionalMemberAccess]
            )

    # ===============================
    # Rerun hook implementations (override base class methods)
    # ===============================

    def _pre_process_rerun(self, request: RerunPlaybookGenerationRequest) -> None:
        """Delete existing pending raw entries before generating new ones.

        This ensures that each rerun starts fresh without accumulating pending entries
        from previous reruns.

        Args:
            request: RerunPlaybookGenerationRequest with optional agent_version and playbook_name filters
        """
        deleted_count = self.storage.delete_all_user_playbooks_by_status(  # type: ignore[reportOptionalMemberAccess]
            status=Status.PENDING,
            agent_version=request.agent_version,
            playbook_name=request.playbook_name,
        )
        logger.info(
            "Deleted %d existing pending raw entries before rerun (agent_version=%s, playbook_name=%s)",
            deleted_count,
            request.agent_version,
            request.playbook_name,
        )

    def _get_rerun_user_ids(self, request: RerunPlaybookGenerationRequest) -> list[str]:
        """Get user IDs to process. Extractors collect their own data.

        Identifies unique user_ids with matching requests via storage-level filtering.

        Args:
            request: RerunPlaybookGenerationRequest with optional time and source filters

        Returns:
            List of user IDs to process
        """
        return self.storage.get_rerun_user_ids(  # type: ignore[reportOptionalMemberAccess]
            user_id=None,
            start_time=(
                int(request.start_time.timestamp()) if request.start_time else None
            ),
            end_time=(int(request.end_time.timestamp()) if request.end_time else None),
            source=request.source,
            agent_version=request.agent_version,
        )

    def _build_rerun_request_params(
        self, request: RerunPlaybookGenerationRequest
    ) -> dict:
        """Build request params dict for operation state tracking.

        Args:
            request: Original rerun request

        Returns:
            Dictionary of request parameters
        """
        return {
            "agent_version": request.agent_version,
            "start_time": (
                request.start_time.isoformat() if request.start_time else None
            ),
            "end_time": request.end_time.isoformat() if request.end_time else None,
            "playbook_name": request.playbook_name,
        }

    def _create_run_request_for_item(
        self,
        user_id: str,
        request: RerunPlaybookGenerationRequest | ManualPlaybookGenerationRequest,
    ) -> PlaybookGenerationRequest:
        """Create PlaybookGenerationRequest for a single user.

        Handles both rerun and manual request types.

        Args:
            user_id: The user ID to process
            request: The original rerun or manual request

        Returns:
            PlaybookGenerationRequest for this user with filter constraints
        """
        # Handle rerun requests (have start_time/end_time datetime objects)
        if isinstance(request, RerunPlaybookGenerationRequest):
            return PlaybookGenerationRequest(
                request_id=f"rerun_playbook_{uuid.uuid4().hex[:8]}",
                agent_version=request.agent_version,
                user_id=user_id,
                source=request.source,
                rerun_start_time=(
                    int(request.start_time.timestamp()) if request.start_time else None
                ),
                rerun_end_time=(
                    int(request.end_time.timestamp()) if request.end_time else None
                ),
                playbook_name=request.playbook_name,
                auto_run=False,
            )
        # Handle manual requests (ManualPlaybookGenerationRequest)
        return PlaybookGenerationRequest(
            request_id=f"manual_{uuid.uuid4().hex[:8]}",
            agent_version=request.agent_version,
            user_id=user_id,
            source=request.source,
            auto_run=False,
        )

    def _create_rerun_response(
        self, success: bool, msg: str, count: int
    ) -> RerunPlaybookGenerationResponse:
        """Create RerunPlaybookGenerationResponse.

        Args:
            success: Whether the operation succeeded
            msg: Status message
            count: Number of entries generated

        Returns:
            RerunPlaybookGenerationResponse
        """
        return RerunPlaybookGenerationResponse(
            success=success,
            msg=msg,
            playbooks_generated=count,
        )

    def _get_generated_count(
        self,
        request: RerunPlaybookGenerationRequest,
        processed_user_ids: list[str] | None = None,  # noqa: ARG002
    ) -> int:
        """Get the count of entries generated during rerun.

        Counts entries with pending status, filtered by agent_version and optionally playbook_name.

        Args:
            request: The rerun request object

        Returns:
            Number of entries generated
        """
        playbooks = self.storage.get_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
            playbook_name=request.playbook_name,
            agent_version=request.agent_version,
            status_filter=[Status.PENDING],
            limit=10000,
        )
        return len(playbooks)

    # ===============================
    # Manual Regular Generation (window-sized, CURRENT output)
    # ===============================

    def run_manual_regular(
        self, request: ManualPlaybookGenerationRequest
    ) -> ManualPlaybookGenerationResponse:
        """
        Run playbook generation with window-sized interactions and CURRENT output.

        Processes entries per-user. Each extractor collects its own data
        using its configured window_size.
        Uses progress tracking via OperationStateManager.

        Args:
            request: ManualPlaybookGenerationRequest with agent_version, optional source and playbook_name

        Returns:
            ManualPlaybookGenerationResponse with success status and count
        """
        state_manager = self._create_state_manager()

        try:
            # Check for existing in-progress operation
            error = state_manager.check_in_progress()
            if error:
                return ManualPlaybookGenerationResponse(
                    success=False, msg=error, playbooks_generated=0
                )

            # 1. Get user_ids with recent interactions
            requests_dict = self.storage.get_sessions(  # type: ignore[reportOptionalMemberAccess]
                user_id=None,  # All users
                top_k=1000,  # Get recent sessions to find users
            )

            # Get unique user_ids
            user_ids_set: set[str] = set()
            for session_requests in requests_dict.values():
                for rig in session_requests:
                    # Apply source filter if provided
                    if request.source and rig.request.source != request.source:
                        continue
                    user_ids_set.add(rig.request.user_id)

            user_ids = list(user_ids_set)

            if not user_ids:
                return ManualPlaybookGenerationResponse(
                    success=True,
                    msg="No interactions found to process",
                    playbooks_generated=0,
                )

            # 2. Run batch with progress tracking
            request_params = {
                "agent_version": request.agent_version,
                "source": request.source,
                "playbook_name": request.playbook_name,
                "mode": "manual_regular",
            }
            self._run_batch_with_progress(
                user_ids=user_ids,
                request=request,  # type: ignore[reportArgumentType]
                request_params=request_params,
                state_manager=state_manager,
            )

            # 3. Count generated entries (CURRENT status = None)
            total_playbooks = self._count_manual_generated(request)

            return ManualPlaybookGenerationResponse(
                success=True,
                msg=f"Generated {total_playbooks} playbook entries",
                playbooks_generated=total_playbooks,
            )

        except Exception as e:
            state_manager.mark_progress_failed(str(e))
            return ManualPlaybookGenerationResponse(
                success=False,
                msg=f"Failed to generate playbook entries: {str(e)}",
                playbooks_generated=0,
            )

    def _count_manual_generated(self, request: ManualPlaybookGenerationRequest) -> int:
        """
        Count entries generated during manual regular generation.

        Counts entries with CURRENT status (None), filtered by agent_version
        and optionally playbook_name.

        Args:
            request: The manual generation request object

        Returns:
            Number of entries with CURRENT status
        """
        playbooks = self.storage.get_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
            playbook_name=request.playbook_name,
            agent_version=request.agent_version,
            status_filter=[None],  # CURRENT entries
            limit=10000,
        )
        return len(playbooks)

    # ===============================
    # Upgrade/Downgrade hook implementations (override base class methods)
    # ===============================

    def _has_items_with_status(
        self, status: Status | None, request: PlaybookGenerationRequest
    ) -> bool:
        """Check if raw entries exist with given status.

        Args:
            status: The status to check for (None for CURRENT)
            request: The upgrade/downgrade request object

        Returns:
            bool: True if any matching raw entries exist
        """
        return self.storage.has_user_playbooks_with_status(  # type: ignore[reportOptionalMemberAccess]
            status=status,
            agent_version=getattr(request, "agent_version", None),
            playbook_name=getattr(request, "playbook_name", None),
        )

    def _delete_items_by_status(
        self, status: Status, request: PlaybookGenerationRequest
    ) -> int:
        """Delete raw entries with given status.

        Args:
            status: The status of raw entries to delete
            request: The upgrade/downgrade request object

        Returns:
            int: Number of raw entries deleted
        """
        return self.storage.delete_all_user_playbooks_by_status(  # type: ignore[reportOptionalMemberAccess]
            status=status,
            agent_version=getattr(request, "agent_version", None),
            playbook_name=getattr(request, "playbook_name", None),
        )

    def _update_items_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        request: PlaybookGenerationRequest,
        user_ids: list[str] | None = None,  # noqa: ARG002
    ) -> int:
        """Update raw entries from old_status to new_status with request filters.

        Args:
            old_status: The current status to match (None for CURRENT)
            new_status: The new status to set (None for CURRENT)
            request: The upgrade/downgrade request object with filters
            user_ids: Optional pre-computed list of user IDs (not used for playbook service)

        Returns:
            int: Number of raw entries updated
        """
        # Note: user_ids is ignored for playbook service as it uses agent_version/playbook_name filters
        return self.storage.update_all_user_playbooks_status(  # type: ignore[reportOptionalMemberAccess]
            old_status=old_status,
            new_status=new_status,
            agent_version=getattr(request, "agent_version", None),
            playbook_name=getattr(request, "playbook_name", None),
        )

    def _create_status_change_response(
        self,
        operation: StatusChangeOperation,
        success: bool,
        counts: dict,
        msg: str,
    ) -> UpgradeUserPlaybooksResponse | DowngradeUserPlaybooksResponse:
        """Create upgrade or downgrade response object for raw entries.

        Args:
            operation: The operation type (UPGRADE or DOWNGRADE)
            success: Whether the operation succeeded
            counts: Dictionary of counts
            msg: Status message

        Returns:
            UpgradeUserPlaybooksResponse or DowngradeUserPlaybooksResponse
        """
        if operation == StatusChangeOperation.UPGRADE:
            return UpgradeUserPlaybooksResponse(
                success=success,
                user_playbooks_deleted=counts.get("deleted", 0),
                user_playbooks_archived=counts.get("archived", 0),
                user_playbooks_promoted=counts.get("promoted", 0),
                message=msg,
            )
        # DOWNGRADE
        return DowngradeUserPlaybooksResponse(
            success=success,
            user_playbooks_demoted=counts.get("demoted", 0),
            user_playbooks_restored=counts.get("restored", 0),
            message=msg,
        )
