from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import UTC, datetime

from reflexio.defaults import resolve_agent_version
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    PublishUserInteractionRequest,
    Request,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation.delayed_group_evaluator import (
    GroupEvaluationScheduler,
)
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.playbook.playbook_generation_service import (
    PlaybookGenerationService,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
)
from reflexio.server.services.profile.profile_generation_service import (
    ProfileGenerationService,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
)

logger = logging.getLogger(__name__)
# Stale lock timeout - if cleanup started > 10 min ago and still "in_progress", assume it crashed
CLEANUP_STALE_LOCK_SECONDS = 600
# Timeout for the outer generation service parallel execution
GENERATION_SERVICE_TIMEOUT_SECONDS = 600


@dataclass
class GenerationServiceResult:
    """Result of a GenerationService.run call.

    Exposes the internally generated request_id plus any warnings so callers
    (CLI, API) can report back to users where their publish landed.

    Attributes:
        request_id (str | None): The UUID assigned to this publish call, or
            None when ``run()`` returned early before generating one (e.g.
            empty request, missing ``user_id``, or no interactions).
        warnings (list[str]): Non-fatal warnings raised by individual
            generation services during the run.
    """

    request_id: str | None = None
    warnings: list[str] = field(default_factory=list)


class GenerationService:
    """
    Main service for orchestrating profile, playbook, and agent success evaluation generation.

    This service coordinates multiple generation services (profile, playbook, agent success)
    and manages the overall interaction processing workflow.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        request_context: RequestContext,
    ) -> None:
        """
        Initialize the generation service.

        Args:
            llm_client: Pre-configured LLM client for making API calls.
            request_context: Request context with storage and configurator.
        """
        self.client = llm_client
        self.storage = request_context.storage
        self.org_id = request_context.org_id
        self.configurator = request_context.configurator
        self.request_context = request_context

    # ===============================
    # public methods
    # ===============================

    def run(
        self, publish_user_interaction_request: PublishUserInteractionRequest
    ) -> GenerationServiceResult:
        """
        Process a user interaction request by storing interactions and triggering generation services.

        Profile and playbook generation services run inline in parallel. Agent success
        evaluation is deferred via GroupEvaluationScheduler when a session_id is present,
        so the full session can be evaluated after a period of inactivity.

        Each generation service (profile, playbook) handles its own:
        - Data collection based on extractor-specific configs
        - Batch interval checking based on extractor-specific settings
        - Operation state tracking per extractor

        Args:
            publish_user_interaction_request: The incoming user interaction request

        Returns:
            GenerationServiceResult: The request_id assigned to this publish call
                and any non-fatal warnings raised by individual generation services.
        """
        result = GenerationServiceResult()

        if not publish_user_interaction_request:
            logger.error("Received None publish_user_interaction_request")
            return result

        user_id = publish_user_interaction_request.user_id
        if not user_id:
            logger.error("Received None user_id in publish_user_interaction_request")
            return result

        # Check if cleanup is needed before adding new interactions
        self._cleanup_old_interactions_if_needed()

        try:
            # Always generate a new UUID for request_id
            request_id = str(uuid.uuid4())
            result.request_id = request_id

            new_interactions: list[Interaction] = (
                GenerationService.get_interaction_from_publish_user_interaction_request(
                    publish_user_interaction_request, request_id
                )
            )

            if not new_interactions:
                logger.info(
                    "No interactions from the publish user interaction request: %s, get all interactions for the user: %s",
                    request_id,
                    user_id,
                )
                return result

            # Resolve agent_version: explicit > env var > default
            agent_version = resolve_agent_version(
                publish_user_interaction_request.agent_version
            )

            # Store Request
            new_request = Request(
                request_id=request_id,
                user_id=user_id,
                source=publish_user_interaction_request.source,
                agent_version=agent_version,
                session_id=publish_user_interaction_request.session_id or None,
            )
            self.storage.add_request(new_request)  # type: ignore[reportOptionalMemberAccess]

            # Add interactions to storage (bulk insert with batched embedding generation)
            self.storage.add_user_interactions_bulk(  # type: ignore[reportOptionalMemberAccess]
                user_id=user_id, interactions=new_interactions
            )

            # Extract source (empty string treated as None)
            source = publish_user_interaction_request.source or None

            # Create generation services and requests
            # Each service writes to separate storage tables and has no dependencies on others
            profile_generation_service = ProfileGenerationService(
                llm_client=self.client, request_context=self.request_context
            )
            profile_generation_request = ProfileGenerationRequest(
                user_id=user_id,
                request_id=request_id,
                source=source,
                force_extraction=publish_user_interaction_request.force_extraction,
            )

            playbook_generation_service = PlaybookGenerationService(
                llm_client=self.client,
                request_context=self.request_context,
                skip_aggregation=publish_user_interaction_request.skip_aggregation,
            )
            playbook_generation_request = PlaybookGenerationRequest(
                request_id=request_id,
                agent_version=agent_version,
                user_id=user_id,
                source=source,
                force_extraction=publish_user_interaction_request.force_extraction,
            )

            # Run profile and playbook generation services in parallel
            # Each service creates its own internal ThreadPoolExecutor for extractors
            # This is safe because we create separate, independent pool instances
            # Uses manual executor management to avoid blocking on shutdown(wait=True)
            # when threads are hung on LLM calls
            executor = ThreadPoolExecutor(max_workers=2)
            try:
                futures = [
                    executor.submit(
                        profile_generation_service.run, profile_generation_request
                    ),
                    executor.submit(
                        playbook_generation_service.run, playbook_generation_request
                    ),
                ]

                # Collect results and handle any exceptions
                # Each service failure is logged but doesn't block others
                service_names = ["profile_generation", "playbook_generation"]
                for future, service_name in zip(futures, service_names, strict=True):
                    try:
                        future.result(timeout=GENERATION_SERVICE_TIMEOUT_SECONDS)
                    except FuturesTimeoutError:  # noqa: PERF203
                        msg = f"{service_name} timed out after {GENERATION_SERVICE_TIMEOUT_SECONDS}s"
                        logger.error("%s for request %s", msg, request_id)
                        result.warnings.append(msg)
                    except Exception as e:
                        msg = f"{service_name} failed: {e}"
                        logger.error(
                            "Generation service failed for request %s: %s, exception type: %s",
                            request_id,
                            str(e),
                            type(e).__name__,
                        )
                        result.warnings.append(msg)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            # Schedule delayed group evaluation if session_id is present
            session_id = new_request.session_id
            if session_id:
                scheduler = GroupEvaluationScheduler.get_instance()
                key = (self.org_id, user_id, session_id)

                def make_callback(
                    _org_id: str,
                    _user_id: str,
                    _sid: str,
                    _av: str,
                    _src: str | None,
                    _rc: RequestContext,
                    _llm: LiteLLMClient,
                ) -> Callable[[], None]:
                    def callback() -> None:
                        run_group_evaluation(
                            org_id=_org_id,
                            user_id=_user_id,
                            session_id=_sid,
                            agent_version=_av,
                            source=_src,
                            request_context=_rc,
                            llm_client=_llm,
                        )

                    return callback

                scheduler.schedule(
                    key,
                    make_callback(
                        self.org_id,
                        user_id,
                        session_id,
                        agent_version,
                        source,
                        self.request_context,
                        self.client,
                    ),
                )

            return result

        except Exception as e:
            # log exception
            logger.error(
                "Failed to refresh user profile for user id: %s due to %s, exception type: %s",
                user_id,
                e,
                type(e).__name__,
            )
            raise e

    # ===============================
    # private methods
    # ===============================

    def _cleanup_old_interactions_if_needed(self) -> None:
        """
        Check total interaction count and cleanup oldest interactions if threshold exceeded.
        Uses OperationStateManager simple lock to prevent race conditions.
        """
        from reflexio.server import (
            INTERACTION_CLEANUP_DELETE_COUNT,
            INTERACTION_CLEANUP_THRESHOLD,
        )

        if INTERACTION_CLEANUP_THRESHOLD <= 0:
            return  # Cleanup disabled

        try:
            total_count = self.storage.count_all_interactions()  # type: ignore[reportOptionalMemberAccess]
            if total_count < INTERACTION_CLEANUP_THRESHOLD:
                return  # No cleanup needed

            mgr = OperationStateManager(
                self.storage,  # type: ignore[reportArgumentType]
                self.org_id,
                "interaction_cleanup",  # type: ignore[reportArgumentType]
            )
            if not mgr.acquire_simple_lock(stale_seconds=CLEANUP_STALE_LOCK_SECONDS):
                return

            try:
                # Perform cleanup
                deleted = self.storage.delete_oldest_interactions(  # type: ignore[reportOptionalMemberAccess]
                    INTERACTION_CLEANUP_DELETE_COUNT
                )
                logger.info(
                    "Cleaned up %d oldest interactions (total was %d, threshold %d)",
                    deleted,
                    total_count,
                    INTERACTION_CLEANUP_THRESHOLD,
                )
            finally:
                mgr.release_simple_lock()

        except Exception as e:
            logger.error("Failed to cleanup old interactions: %s", e)
            # Don't raise - cleanup failure shouldn't block normal operation

    # ===============================
    # static methods
    # ===============================

    @staticmethod
    def get_interaction_from_publish_user_interaction_request(
        publish_user_interaction_request: PublishUserInteractionRequest,
        request_id: str,
    ) -> list[Interaction]:
        """get interaction from publish user interaction request

        Args:
            publish_user_interaction_request (PublishUserInteractionRequest): The publish user interaction request
            request_id (str): The request ID generated by the service

        Returns:
            list[Interaction]: List of interactions created from the request
        """
        interaction_data_list = publish_user_interaction_request.interaction_data_list

        user_id = publish_user_interaction_request.user_id
        # Always use server-side UTC timestamp to ensure consistency
        server_timestamp = int(datetime.now(UTC).timestamp())
        return [
            Interaction(
                # interaction_id is auto-generated by DB
                user_id=user_id,
                request_id=request_id,
                created_at=server_timestamp,  # Use server UTC timestamp
                content=interaction_data.content,
                role=interaction_data.role,
                user_action=interaction_data.user_action,
                user_action_description=interaction_data.user_action_description,
                interacted_image_url=interaction_data.interacted_image_url,
                image_encoding=interaction_data.image_encoding,
                shadow_content=interaction_data.shadow_content,
                expert_content=interaction_data.expert_content,
                tools_used=interaction_data.tools_used,
            )
            for interaction_data in interaction_data_list
        ]
