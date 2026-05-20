from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from reflexio.defaults import resolve_agent_version
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    PublishUserInteractionRequest,
    Request,
)
from reflexio.models.config_schema import Config
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
from reflexio.server.services.reflection.reflection_service import ReflectionService
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionServiceRequest,
)
from reflexio.server.services.storage.retention import (
    delete_count_for_retention,
    get_row_retention_limits,
)
from reflexio.server.usage_metrics import record_usage_event

if TYPE_CHECKING:
    from reflexio.server.services.search.agentic_search_service import (
        AgenticSearchService,
    )
    from reflexio.server.services.unified_search_service import UnifiedSearchService

logger = logging.getLogger(__name__)
# Stale lock timeout - if cleanup started > 10 min ago and still "in_progress", assume it crashed
CLEANUP_STALE_LOCK_SECONDS = 600
# Timeout for the outer generation service parallel execution
GENERATION_SERVICE_TIMEOUT_SECONDS = 600
_STALL_WARNING_PREFIX = "Reflexio learning is paused"


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
        - Stride checking based on extractor-specific settings
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

        # Check if cleanup is needed before adding new interactions.
        self._cleanup_storage_tables_if_needed()

        publish_start = time.perf_counter()
        # Resolve agent_version: explicit > env var > default. Resolved here
        # (before the try) so success and failure telemetry share the same value.
        agent_version = resolve_agent_version(
            publish_user_interaction_request.agent_version
        )

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

            record_usage_event(
                org_id=self.org_id,
                user_id=user_id,
                request_id=request_id,
                session_id=publish_user_interaction_request.session_id or None,
                source=publish_user_interaction_request.source,
                agent_version=agent_version,
                event_name="publish_request_received",
                event_category="publish",
                outcome="received",
                count_value=len(new_interactions),
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

            if (
                not publish_user_interaction_request.override_learning_stall
                and (stall_warning := self._active_learning_stall_warning()) is not None
            ):
                result.warnings.append(stall_warning)
                logger.warning("%s; skipping automatic extraction", stall_warning)
                record_usage_event(
                    org_id=self.org_id,
                    user_id=user_id,
                    request_id=request_id,
                    session_id=new_request.session_id,
                    source=source,
                    agent_version=agent_version,
                    event_name="publish_request_succeeded",
                    event_category="publish",
                    outcome="success",
                    count_value=len(new_interactions),
                    duration_ms=int((time.perf_counter() - publish_start) * 1000),
                    metadata={"warning_count": len(result.warnings)},
                )
                return result

            # Reflection runs as its own sliding-window step BEFORE the
            # extractor pool spins up, so any replacements it makes are
            # visible to the extractors when they retrieve existing
            # profile/playbook context. Wrapped in a broad except so a
            # reflection bug never breaks the publish.
            self._maybe_run_reflection(
                user_id=user_id, agent_version=agent_version, source=source
            )

            # Dispatch to the agentic pipeline when the config flag is set.
            # Classic path (default) falls through to the ProfileGenerationService
            # + PlaybookGenerationService fan-out below.
            root_config = self.configurator.get_config()
            if (
                root_config is not None
                and getattr(root_config, "extraction_backend", "classic") == "agentic"
            ):
                from reflexio.server.services.extraction.agentic_adapter import (
                    AgenticExtractionRunner,
                )

                runner = AgenticExtractionRunner(
                    llm_client=self.client,
                    request_context=self.request_context,
                )
                result.warnings.extend(
                    runner.run(
                        publish_request=publish_user_interaction_request,
                        request_id=request_id,
                        new_interactions=new_interactions,
                        new_request=new_request,
                        config=root_config,
                    )
                )
                record_usage_event(
                    org_id=self.org_id,
                    user_id=user_id,
                    request_id=request_id,
                    session_id=new_request.session_id,
                    source=source,
                    agent_version=agent_version,
                    backend="agentic",
                    event_name="publish_request_succeeded",
                    event_category="publish",
                    outcome="success",
                    count_value=len(new_interactions),
                    duration_ms=int((time.perf_counter() - publish_start) * 1000),
                    metadata={"warning_count": len(result.warnings)},
                )
                return result

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
                # Each thread needs its own context copy — Context.run() is non-reentrant
                futures = [
                    executor.submit(
                        contextvars.copy_context().run,
                        profile_generation_service.run,
                        profile_generation_request,
                    ),
                    executor.submit(
                        contextvars.copy_context().run,
                        playbook_generation_service.run,
                        playbook_generation_request,
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

            record_usage_event(
                org_id=self.org_id,
                user_id=user_id,
                request_id=request_id,
                session_id=new_request.session_id,
                source=source,
                agent_version=agent_version,
                backend="classic",
                event_name="publish_request_succeeded",
                event_category="publish",
                outcome="success",
                count_value=len(new_interactions),
                duration_ms=int((time.perf_counter() - publish_start) * 1000),
                metadata={"warning_count": len(result.warnings)},
            )
            return result

        except Exception as e:
            record_usage_event(
                org_id=self.org_id,
                user_id=user_id,
                request_id=result.request_id,
                session_id=publish_user_interaction_request.session_id or None,
                source=publish_user_interaction_request.source,
                agent_version=agent_version,
                event_name="publish_request_failed",
                event_category="publish",
                outcome="failed",
                duration_ms=int((time.perf_counter() - publish_start) * 1000),
                error_kind=type(e).__name__,
            )
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

    def _maybe_run_reflection(
        self, *, user_id: str, agent_version: str, source: str | None
    ) -> None:
        """Best-effort reflection pass before extraction.

        Any failure is caught and logged so the surrounding publish
        flow (extraction + delayed evaluation) is unaffected.
        """
        try:
            service = ReflectionService(
                request_context=self.request_context,
                llm_client=self.client,
            )
            service.run(
                ReflectionServiceRequest(
                    user_id=user_id,
                    agent_version=agent_version,
                    source=source,
                )
            )
        except Exception as exc:  # noqa: BLE001 — must not break publish
            logger.warning(
                "reflection step failed for user %s: %s (error_type=%s)",
                user_id,
                exc,
                type(exc).__name__,
            )

    def _cleanup_storage_tables_if_needed(self) -> None:
        """Best-effort publish-boundary cleanup for capped storage tables."""
        limits = {
            target_name: limit
            for target_name, limit in get_row_retention_limits().items()
            if limit > 0
        }
        if not limits:
            return

        try:
            mgr = OperationStateManager(
                self.storage,  # type: ignore[reportArgumentType]
                self.org_id,
                "storage_table_cleanup",  # type: ignore[reportArgumentType]
            )
            if not mgr.acquire_simple_lock(stale_seconds=CLEANUP_STALE_LOCK_SECONDS):
                return

            try:
                for target_name, limit in limits.items():
                    # Isolate per-target failures so one bad table does not
                    # short-circuit cleanup for every subsequent target.
                    try:
                        self._cleanup_retention_target(target_name, limit)
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            "Failed to cleanup retention target %s: %s",
                            target_name,
                            e,
                        )
            finally:
                mgr.release_simple_lock()

        except Exception as e:
            logger.error("Failed to cleanup storage tables: %s", e)
            # Don't raise - cleanup failure shouldn't block normal operation

    def _active_learning_stall_warning(self) -> str | None:
        """Return a warning when extraction should not auto-retry.

        Plugin publishes should still store raw interactions while the local
        LLM provider is blocked by auth or billing, but they
        must not keep invoking extraction on every publish. Only callers that
        pass ``override_learning_stall=True`` bypass this check so an explicit
        retry after reauth can clear the stall state on a successful provider
        call.
        """
        try:
            stall_state = self.storage.get_stall_state()  # type: ignore[reportOptionalMemberAccess]
        except (AttributeError, NotImplementedError):
            return None
        except Exception as exc:  # noqa: BLE001 - stall telemetry must not block publish.
            logger.debug("Failed to read stall_state before extraction: %s", exc)
            return None

        if not getattr(stall_state, "stalled", False):
            return None
        reason = getattr(stall_state, "reason", None) or "unknown"
        suffix = (
            "reauthenticate the active coding-agent provider, then run an explicit "
            "override retry to resume."
            if reason == "auth_error"
            else "wait for the limit/reset condition to clear, then run an explicit "
            "override retry to resume."
        )
        return f"{_STALL_WARNING_PREFIX} ({reason}); {suffix}"

    def _cleanup_retention_target(self, target_name: str, limit: int) -> None:
        total_count = self.storage.count_retention_target_rows(target_name)  # type: ignore[reportOptionalMemberAccess]
        if total_count < limit:
            return
        delete_count = delete_count_for_retention(total_count)
        deleted = self.storage.delete_oldest_retention_target_rows(  # type: ignore[reportOptionalMemberAccess]
            target_name,
            delete_count,
        )
        logger.info(
            "Cleaned up %d oldest %s row(s) (total was %d, limit %d)",
            deleted,
            target_name,
            total_count,
            limit,
        )

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
        # Honor the client-provided ``created_at`` — InteractionData defaults
        # it to client-side ``now()`` on construction, so it's always populated.
        # Apps that publish backdated conversations (e.g., a benchmark replay
        # of 2023 chats run in 2026) need the wall-clock time preserved so the
        # extraction agent has a real temporal anchor for relative-time
        # references like "X weeks ago" / "yesterday". Stamping server-now here
        # would erase that anchor and force every event onto today's date.
        return [
            Interaction(
                # interaction_id is auto-generated by DB
                user_id=user_id,
                request_id=request_id,
                created_at=interaction_data.created_at,
                content=interaction_data.content,
                role=interaction_data.role,
                user_action=interaction_data.user_action,
                user_action_description=interaction_data.user_action_description,
                interacted_image_url=interaction_data.interacted_image_url,
                image_encoding=interaction_data.image_encoding,
                shadow_content=interaction_data.shadow_content,
                expert_content=interaction_data.expert_content,
                tools_used=interaction_data.tools_used,
                citations=interaction_data.citations,
            )
            for interaction_data in interaction_data_list
        ]


def build_extraction_service(
    config: Config,
    *,
    llm_client: LiteLLMClient,
    request_context: RequestContext,
) -> ProfileGenerationService:
    """Return the classic profile extraction service.

    The agentic extraction path is handled directly by
    ``AgenticExtractionRunner`` inside ``GenerationService.run`` and does not
    go through this factory.  This function exists for the classic dispatcher
    path only.

    Args:
        config (Config): Top-level ``Config`` (unused; kept for API consistency).
        llm_client (LiteLLMClient): Configured ``LiteLLMClient``.
        request_context (RequestContext): Current request context.

    Returns:
        ProfileGenerationService: Classic profile extraction service.
    """
    del config  # unused — agentic path bypasses this factory
    return ProfileGenerationService(
        llm_client=llm_client, request_context=request_context
    )


def build_search_service(
    config: Config,
    *,
    llm_client: LiteLLMClient,
    request_context: RequestContext,
) -> UnifiedSearchService | AgenticSearchService:
    """Dispatch to the classic or agentic search service.

    Selected by ``config.search_backend``. Classic returns a
    ``UnifiedSearchService``; agentic returns the Phase-4 pipeline.

    Args:
        config (Config): Top-level ``Config``. Reads ``search_backend``.
        llm_client (LiteLLMClient): Configured ``LiteLLMClient``.
        request_context (RequestContext): Current request context.

    Returns:
        Object holding ``llm_client`` and ``request_context`` — either a
        classic ``UnifiedSearchService`` or the agentic service.
    """
    if config.search_backend == "agentic":
        from reflexio.server.services.search.agentic_search_service import (  # type: ignore[import-not-found]
            AgenticSearchService,
        )

        return AgenticSearchService(
            llm_client=llm_client, request_context=request_context
        )
    from reflexio.server.services.unified_search_service import UnifiedSearchService

    return UnifiedSearchService(llm_client=llm_client, request_context=request_context)
