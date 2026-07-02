"""
Base class for generation services
"""

import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.services.base_generation import (
    ConfigFilterMixin,
    ExtractionRunLifecycleMixin,
    ShouldRunPrecheckMixin,
    StatusChangeMixin,
    UsageBillingMixin,
)
from reflexio.server.services.base_generation import (
    StatusChangeOperation as StatusChangeOperation,  # re-export for back-compat
)
from reflexio.server.services.extractor_config_utils import (
    get_extractor_name,
)
from reflexio.server.services.operation_state_utils import OperationStateManager


class ExtractorExecutionError(RuntimeError):
    """Raised when the configured extractor fails for a request/user context."""


logger = logging.getLogger(__name__)


# Cheap-signal thresholds for the pre-LLM should_run filter. Tuned for
# coding-assistant traffic where most turns are either slash commands
# or tool scaffolding, and where the LLM should_run gate costs 5–7s
# even when it ultimately votes False.
_MIN_USER_CONTENT_LEN = 30
# Heuristic match for reflexio's own extractor system prompts that
# sometimes leak into the corpus via the claude-code LLM provider's
# self-invocation. Kept conservative — false positives just mean one
# real interaction gets skipped this cycle (it'll re-enter at the next
# publish), false negatives are what we're actually trying to avoid.
_EXTRACTOR_PROMPT_PREFIXES = (
    "you are a detector",
    "you are an user signal",
    "you are a signal detection",
    "you are an extractor",
)
# Matches a single slash-command token at the start of a message. The
# ``:`` allows plugin-namespaced commands like ``/claude-smart:tag``.
_SLASH_COMMAND_TOKEN_RE = re.compile(r"^/[A-Za-z0-9_:-]+\s*")


def _is_pure_slash_command(content: str) -> bool:
    """Whether ``content`` is a bare slash command with no substantive text.

    ``/learn`` and ``/claude-smart:tag`` return True. ``/btw some note``
    and ``/claude-smart:tag fix the foo`` return False because the text
    after the command token carries user signal the extractors should see.
    """
    stripped = content.lstrip()
    if not stripped.startswith("/"):
        return False
    remainder = _SLASH_COMMAND_TOKEN_RE.sub("", stripped, count=1)
    return not remainder.strip()


def _iter_user_contents(
    session_data_models: list[RequestInteractionDataModel],
) -> list[str]:
    """Collect the ``content`` of every User-role interaction, order-preserving."""
    out: list[str] = []
    for model in session_data_models:
        out.extend(
            interaction.content
            for interaction in model.interactions
            if interaction.role == "User" and interaction.content
        )
    return out


def _cheap_should_run_reject(
    session_data_models: list[RequestInteractionDataModel],
) -> str | None:
    """Cheap pre-filter for the consolidated should_run LLM gate.

    Returns a short reason string when we can cheaply decide the batch
    has no learnable signal — the caller logs the reason and skips the
    LLM call. Returns None when we cannot decide cheaply and the LLM
    should run.

    Rejection rules:
        - No user message at least ``_MIN_USER_CONTENT_LEN`` chars long
          (purely short commands / confirmations).
        - Every user message is a bare slash-command dispatch with no
          substantive trailing text (e.g. ``/commit``, ``/review``,
          ``/claude-smart:tag``). Slash commands that carry user text
          after the token (e.g. ``/btw some note``) are kept.
        - Any user message begins with a known extractor-prompt prefix
          (reflexio talking to itself via the claude-code LLM provider).

    Args:
        session_data_models: The deduplicated per-session interaction
            batch built by ``_collect_scoped_interactions_for_precheck``.

    Returns:
        str | None: Reason code for the reject, or None to fall through.
    """
    user_contents = _iter_user_contents(session_data_models)
    if not user_contents:
        return "no_user_turns"

    for content in user_contents:
        lowered = content.lstrip().lower()
        if any(lowered.startswith(p) for p in _EXTRACTOR_PROMPT_PREFIXES):
            return "extractor_prompt_echo"

    if not any(len(c.strip()) >= _MIN_USER_CONTENT_LEN for c in user_contents):
        return "all_user_turns_too_short"

    if all(_is_pure_slash_command(c) for c in user_contents):
        return "all_slash_commands"

    return None


# Timeout for individual extractor execution (safety net if LLM provider ignores its own timeout)
EXTRACTOR_TIMEOUT_SECONDS = 300

TExtractorConfig = TypeVar("TExtractorConfig")
TExtractor = TypeVar("TExtractor")
TGenerationServiceConfig = TypeVar("TGenerationServiceConfig")
TRequest = TypeVar("TRequest")


@dataclass(frozen=True)
class PreparedGenerationRun(Generic[TExtractorConfig]):  # noqa: UP046
    extractor_config: TExtractorConfig
    extractor_name: str
    identifier: str


# Unified base class for all generation services (evaluation, playbook, profile)
class BaseGenerationService(
    UsageBillingMixin[TExtractorConfig, TGenerationServiceConfig],
    ExtractionRunLifecycleMixin[TExtractorConfig, TGenerationServiceConfig],
    ShouldRunPrecheckMixin[TExtractorConfig, TGenerationServiceConfig],
    ConfigFilterMixin[TExtractorConfig, TGenerationServiceConfig],
    StatusChangeMixin[TRequest],
    ABC,
    Generic[TExtractorConfig, TExtractor, TGenerationServiceConfig, TRequest],  # noqa: UP046
):
    # Only profile/playbook GENERATION services bill the ② Learning line;
    # reflection/consolidation/aggregation/evaluation are bundled, not metered.
    # Default is False so any future subclass is safe by default (opt-IN).
    EMITS_LEARNING_BILLING: bool = False
    """
    Base class for generation services that run one configured extractor.

    This unified class supports two types of services:
    1. Evaluation services (playbook, agent success) - process interactions and save UserPlaybook
    2. Profile services - process interactions with existing data and apply updates

    Type Parameters:
        TExtractorConfig: The extractor configuration type from YAML (e.g., PlaybookConfig, ProfileExtractorConfig)
        TExtractor: The extractor type (e.g., PlaybookExtractor, ProfileExtractor, AgentSuccessEvaluator)
        TGenerationServiceConfig: The runtime service configuration type (e.g., PlaybookGenerationServiceConfig, ProfileGenerationServiceConfig)
        TRequest: The request type (e.g., ProfileGenerationRequest, PlaybookGenerationRequest, AgentSuccessEvaluationRequest)

    Child classes must implement:
    - _load_extractor_config(): Load extractor configuration from configurator
    - _load_generation_service_config(): Extract parameters from request and return GenerationServiceConfig
    - _create_extractor(): Create extractor instance with extractor config and service config
    - _get_service_name(): Get service name for logging
    - _process_results(): Process and save results (can access self.service_config)
    """

    def __init__(
        self, llm_client: LiteLLMClient, request_context: RequestContext
    ) -> None:
        """
        Initialize the base generation service.

        Args:
            llm_client: Unified LLM client supporting both OpenAI and Claude
            request_context: Request context with storage, configurator, and org_id
        """
        self.client = llm_client
        self.storage = request_context.storage
        self.org_id = request_context.org_id
        self.configurator = request_context.configurator
        self.request_context = request_context
        self.service_config: TGenerationServiceConfig | None = None
        self._is_batch_mode: bool = False
        self._last_extractor_run_stats: dict[str, int] = {
            "total": 0,
            "failed": 0,
            "timed_out": 0,
        }
        self._last_extraction_run_ids: list[str] = []
        self._last_token_totals: RunTokenTotals | None = None
        # Window fetched by the should-run gate (_collect_scoped_interactions_for_precheck),
        # stashed so the billing path (_extraction_input_text) can reuse it instead of
        # re-querying storage. None when the gate did not run (bypass paths).
        self._last_precheck_sessions: list[Any] | None = None

    @abstractmethod
    def _load_extractor_config(self) -> TExtractorConfig | None:
        """
        Load extractor configuration from the configurator.

        Returns:
            Extractor configuration object from YAML, or None when disabled.
        """

    @abstractmethod
    def _load_generation_service_config(
        self, request: TRequest
    ) -> TGenerationServiceConfig:
        """
        Extract parameters from request object and return GenerationServiceConfig.

        Args:
            request: The request object

        Returns:
            GenerationServiceConfig object (e.g., PlaybookGenerationServiceConfig, ProfileGenerationServiceConfig)
        """

    @abstractmethod
    def _create_extractor(
        self,
        extractor_config: TExtractorConfig,
        service_config: TGenerationServiceConfig,
    ) -> TExtractor:
        """
        Create an extractor instance from extractor config and service config.

        Args:
            extractor_config: The extractor configuration object from YAML (e.g., PlaybookConfig, ProfileExtractorConfig)
            service_config: The runtime service configuration object (e.g., PlaybookGenerationServiceConfig, ProfileGenerationServiceConfig)

        Returns:
            An extractor instance
        """

    @abstractmethod
    def _get_service_name(self) -> str:
        """
        Get the name of the service for logging purposes.

        Returns:
            Service name string
        """

    @abstractmethod
    def _get_base_service_name(self) -> str:
        """
        Get the base service name for OperationStateManager keys.

        This is the service identity used for progress/lock key construction,
        independent of whether the operation is a rerun or regular run.

        Returns:
            Base service name (e.g., "profile_generation", "playbook_generation")
        """

    @abstractmethod
    def _process_results(self, results: list) -> None:
        """
        Process and save all results from extractors. Called once after all extractors complete.

        Responsible for flattening, deduplication (if applicable), and saving results.
        Can access self.service_config for context.

        Args:
            results: List of all results from extractors (one per successful extractor)
        """

    def _finalize_extracted_items(self, items: list) -> None:
        """Persist already-flattened extracted items through the service path."""
        if items:
            self._process_results([items])

    @abstractmethod
    def _should_track_in_progress(self) -> bool:
        """
        Return True if this service should track in-progress state to prevent duplicates.

        Profile and Feedback services should return True to prevent duplicate generation
        when back-to-back requests arrive. AgentSuccess services should return False
        as they process per-request and don't have the same duplication issue.

        Returns:
            bool: True if in-progress tracking should be enabled
        """

    @abstractmethod
    def _get_lock_scope_id(self, request: TRequest) -> str | None:
        """
        Get the scope ID for lock key construction.

        Profile services return user_id (per-user lock), playbook services return None (per-org lock).

        Args:
            request: The generation request

        Returns:
            Optional[str]: Scope ID (e.g., user_id) or None for org-level scope
        """

    # ===============================
    # In-progress state management via OperationStateManager
    # ===============================

    def _create_state_manager(self) -> OperationStateManager:
        """Create an OperationStateManager for this service.

        Returns:
            OperationStateManager instance configured for this service
        """
        return OperationStateManager(
            self.storage,  # type: ignore[reportArgumentType]
            self.org_id,
            self._get_base_service_name(),  # type: ignore[reportArgumentType]
        )

    def _serialize_request_for_queue(self, request: TRequest) -> dict | None:
        """Serialize a request for the pending-request queue.

        Default implementation handles Pydantic ``BaseModel`` requests via
        ``model_dump(mode="json")``. Override in subclasses whose requests
        are not Pydantic models.

        The queued payload is what the rerun loop will run when this request
        comes off the queue — so it MUST capture every field the run needs to
        reproduce the original publish (user_id, request_id, agent_version,
        source, force_extraction, etc.). Without this, the rerun runs with the
        wrong holder's request and the queued user's interactions are silently
        skipped (R2).

        Returns ``None`` to opt out — the queue then stores only the
        request_id and the rerun falls back to the original holder's request,
        which is the pre-fix behaviour. Use only for services where the
        per-request payload doesn't differ between concurrent callers.
        """
        # Pydantic BaseModel — handles the common case (PlaybookGenerationRequest,
        # ProfileGenerationRequest).
        model_dump = getattr(request, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(mode="json")
            except Exception:  # pragma: no cover — defensive
                logger.warning(
                    "Failed to model_dump %s request for queue; "
                    "rerun will fall back to original holder's request",
                    self._get_service_name(),
                )
                return None
            if isinstance(dumped, dict):
                return dumped
        return None

    def _deserialize_request_from_queue(
        self,
        payload: dict,
        original_request: TRequest,
    ) -> TRequest:
        """Reconstruct a request object from a queued payload.

        Default implementation calls ``type(original_request).model_validate(payload)``
        for Pydantic-backed requests. Override in subclasses with non-Pydantic
        request types.

        Args:
            payload: The dict previously produced by ``_serialize_request_for_queue``
            original_request: The request the lock holder ran with — used as a
                fallback type and for any fields the payload doesn't carry
        """
        request_cls = type(original_request)
        model_validate = getattr(request_cls, "model_validate", None)
        if callable(model_validate):
            try:
                rebuilt = model_validate(payload)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "Failed to model_validate queued payload for %s: %s; "
                    "falling back to original request",
                    self._get_service_name(),
                    exc,
                )
                return original_request
            # Narrow the object type to TRequest — model_validate on
            # type(original_request) returns the same class, so the cast is
            # safe in practice. Pyright can't see through getattr, so we
            # use isinstance to satisfy the type checker.
            if isinstance(rebuilt, request_cls):
                return rebuilt  # type: ignore[reportReturnType]
        return original_request

    def run(self, request: TRequest) -> None:
        """
        Run the generation service for the given request.

        This is the main entry point that:
        1. If in-progress tracking is enabled, handles lock acquisition/release
        2. Validates and extracts parameters from the request into GenerationServiceConfig
        3. Runs extractors sequentially (each extractor handles its own data collection)
        4. Processes results
        5. Re-runs if new requests came in during generation

        Args:
            request: The request object containing parameters
        """
        # Check if this service tracks in-progress state
        if not self._should_track_in_progress():
            self._run_generation(request)
            return

        # Get scope ID and request ID for in-progress tracking
        scope_id = self._get_lock_scope_id(request)
        my_request_id = getattr(request, "request_id", None) or str(uuid.uuid4())

        state_manager = self._create_state_manager()

        # Try to acquire lock — pass the serialized payload so blocked
        # publishes land in the queue with their own data attached. This is
        # the fix for R2: without the payload, the
        # rerun re-uses the holder's request and the queued users' batches
        # never get extracted.
        my_payload = self._serialize_request_for_queue(request)
        if not state_manager.acquire_lock(
            my_request_id, scope_id=scope_id, payload=my_payload
        ):
            return  # Another operation is running, we've enqueued ourselves

        current_request: TRequest = request

        # Re-run loop: drain the pending queue (FIFO) until empty
        try:
            while True:
                self._run_generation(current_request)

                # If in batch mode and cancellation was requested, clear lock
                # to prevent queued pending requests from running, then stop
                if self._is_batch_mode and state_manager.is_cancellation_requested():
                    state_manager.clear_lock(scope_id=scope_id)
                    logger.info(
                        "Cancellation detected in run() for %s, cleared lock to prevent pending re-runs",
                        self._get_service_name(),
                    )
                    break

                # Pop the next queued request (if any). Returns the queued
                # request's ID + payload so the rerun runs against THAT
                # publish's data, not the original holder's.
                next_entry = state_manager.release_lock_pop_queue(
                    my_request_id, scope_id=scope_id
                )

                if next_entry is None:
                    break  # Queue empty — we're done

                next_request_id = next_entry["request_id"]
                next_payload = next_entry.get("payload")

                logger.info(
                    "Draining queued %s request: prev_request_id=%s, next_request_id=%s, "
                    "payload_present=%s",
                    self._get_service_name(),
                    my_request_id,
                    next_request_id,
                    next_payload is not None,
                )

                # Reconstruct the queued request. If the payload is missing
                # (legacy state row from a pre-fix server), fall back to the
                # original request — matches pre-fix behaviour.
                if next_payload:
                    current_request = self._deserialize_request_from_queue(
                        next_payload, request
                    )
                else:
                    current_request = request

                my_request_id = next_request_id

        except Exception:
            # Clear lock on error to prevent deadlock
            state_manager.clear_lock(scope_id=scope_id)
            raise

    def _run_generation(self, request: TRequest) -> None:
        """
        Run the actual generation logic.

        Orchestrates validation, config loading, extractor execution, and result
        processing by delegating to _prepare_generation_run and _execute_extractor.

        Args:
            request: The request object containing parameters
        """
        if not request:
            logger.error("Received None request for %s", self._get_service_name())
            return

        generation_start = time.perf_counter()
        try:
            prepared = self._prepare_generation_run(request)
            if prepared is None:
                return

            self._record_generation_event(
                event_name="generation_started",
                outcome="started",
                count_value=1,
                metadata={
                    "identifier": prepared.identifier,
                    "extractor_name": prepared.extractor_name,
                },
            )
            self._last_extraction_run_ids = []
            self._last_token_totals = None
            result = self._execute_extractor(
                prepared.extractor_config, prepared.identifier
            )
            generated_count = self._count_generated_results(result)

            try:
                if result:
                    self._process_results([result])
                self._finalize_extraction_runs()
            except Exception as exc:
                self._mark_extraction_runs_finalization_failed(exc)
                raise

            self._record_generation_event(
                event_name="generation_succeeded",
                outcome="success",
                count_value=generated_count,
                duration_ms=int((time.perf_counter() - generation_start) * 1000),
                metadata={
                    "identifier": prepared.identifier,
                    "extractor_name": prepared.extractor_name,
                    "extractor_failed": bool(
                        self._last_extractor_run_stats.get("failed")
                    ),
                    "extractor_timed_out": bool(
                        self._last_extractor_run_stats.get("timed_out")
                    ),
                },
            )
            self._record_billing_learning_events(
                prepared=prepared, generated_count=generated_count
            )

        except Exception as e:
            self._record_generation_event(
                event_name="generation_failed",
                outcome="failed",
                duration_ms=int((time.perf_counter() - generation_start) * 1000),
                error_kind=type(e).__name__,
            )
            logger.error(
                "Failed to run %s due to %s, exception type: %s",
                self._get_service_name(),
                str(e),
                type(e).__name__,
            )
            if isinstance(e, ExtractorExecutionError):
                raise

    def _prepare_generation_run(
        self, request: TRequest
    ) -> PreparedGenerationRun[TExtractorConfig] | None:
        """
        Validate request, load config, filter extractor config, and run pre-extraction checks.

        Loads the generation service config from the request, loads and filters the
        extractor config by source, manual trigger, and stride_size, then runs the
        pre-extraction gate.

        Args:
            request: The request object containing parameters

        Returns:
            PreparedGenerationRun when generation should proceed, otherwise None.
        """
        # Reset BEFORE the should-run gate runs. The gate
        # (_collect_scoped_interactions_for_precheck) stashes its fetched window
        # here for the billing path to reuse. On bypass paths (auto_run=False,
        # force_extraction, skip_should_run_check, mock mode) the gate never runs,
        # so this stays None and billing falls back to its own fetch.
        self._last_precheck_sessions = None

        self.service_config = self._load_generation_service_config(request)

        extractor_config = self._load_extractor_config()
        if extractor_config is None:
            logger.warning("No %s extractor config found", self._get_service_name())
            return None

        extractor_config = self._filter_extractor_config_by_service_config(
            extractor_config, self.service_config
        )

        if extractor_config is None:
            source = getattr(self.service_config, "source", "N/A")
            source_display = source or "N/A"
            logger.info(
                "No %s extractor config enabled for source: %s",
                self._get_service_name(),
                source_display,
            )
            return None

        extractor_config = self._filter_config_by_stride(extractor_config)
        if extractor_config is None:
            logger.info(
                "Extractor config did not pass stride_size check for %s",
                self._get_service_name(),
            )
            return None

        identifier = getattr(self.service_config, "user_id", None) or getattr(
            self.service_config, "request_id", "unknown"
        )
        extractor_name = get_extractor_name(extractor_config)

        should_run = self._should_run_before_extraction(extractor_config)
        self._record_generation_event(
            event_name="generation_gate_evaluated",
            outcome="should_run" if should_run else "should_skip",
            count_value=1,
            metadata={
                "identifier": identifier,
                "extractor_name": extractor_name,
            },
        )

        if not should_run:
            logger.info(
                "Pre-extraction check returned False for %s identifier=%s, skipping",
                self._get_service_name(),
                identifier,
            )
            return None

        return PreparedGenerationRun(
            extractor_config=extractor_config,
            extractor_name=extractor_name,
            identifier=identifier,
        )

    # ===============================
    # Batch with progress (shared by rerun + manual)
    # ===============================

    def _run_batch_with_progress(
        self,
        user_ids: list[str],
        request: TRequest,
        request_params: dict,
        state_manager: OperationStateManager,
    ) -> tuple[int, int]:
        """Run a batch of users with progress tracking.

        Shared logic for both run_rerun() and run_manual_regular().
        Initializes progress, processes each user, and finalizes.
        Checks for cancellation before each user.

        Args:
            user_ids: List of user IDs to process
            request: The original request object
            request_params: Parameters dict for progress state
            state_manager: OperationStateManager instance

        Returns:
            Tuple of (users_processed, total_generated)
        """
        total_users = len(user_ids)
        self._is_batch_mode = True

        # Initialize progress
        state_manager.initialize_progress(
            total_users=total_users,
            request_params=request_params,
        )

        try:
            # Process each user
            users_processed = 0
            processed_user_ids: list[str] = []
            for user_id in user_ids:
                # Check for cancellation before starting next user
                if state_manager.is_cancellation_requested():
                    logger.info(
                        "Cancellation requested for %s, stopping after %d/%d users",
                        self._get_base_service_name(),
                        users_processed,
                        total_users,
                    )
                    state_manager.mark_cancelled()
                    return users_processed, self._get_generated_count(
                        request, processed_user_ids=processed_user_ids
                    )

                state_manager.set_current_item(user_id)

                try:
                    run_request = self._create_run_request_for_item(user_id, request)
                    self.run(run_request)
                    users_processed += 1
                    processed_user_ids.append(user_id)

                    state_manager.update_progress(
                        item_id=user_id,
                        count=0,  # Extractors collect their own data
                        success=True,
                        total_users=total_users,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to process user %s for %s: %s",
                        user_id,
                        self._get_base_service_name(),
                        str(e),
                    )
                    state_manager.update_progress(
                        item_id=user_id,
                        count=0,
                        success=False,
                        total_users=total_users,
                        error=str(e),
                    )
                    continue

            # Get generated count and finalize
            total_generated = self._get_generated_count(
                request, processed_user_ids=processed_user_ids
            )
            state_manager.finalize_progress(users_processed, total_generated)

            return users_processed, total_generated
        finally:
            self._is_batch_mode = False

    # ===============================
    # Rerun methods (optional - override to enable rerun functionality)
    # ===============================

    def _get_rerun_user_ids(self, request: TRequest) -> list[str]:
        """Get user IDs to process during rerun.

        Override this method to enable rerun functionality for the service.
        Returns a list of user IDs that have interactions matching the request filters.
        Each extractor collects its own data using its configured window_size.

        Args:
            request: The rerun request object

        Returns:
            List of user IDs to process
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _build_rerun_request_params(self, request: TRequest) -> dict:
        """Build request params dict for operation state tracking.

        Override this method to enable rerun functionality for the service.

        Args:
            request: The rerun request object

        Returns:
            Dictionary of request parameters for state tracking
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _create_run_request_for_item(self, user_id: str, request: TRequest) -> TRequest:
        """Create the request object to pass to self.run() for a single user.

        Override this method to enable rerun functionality for the service.
        Each extractor collects its own data using its configured window_size.

        Args:
            user_id: The user ID to process
            request: The original rerun request object

        Returns:
            A request object suitable for self.run()
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _create_rerun_response(self, success: bool, msg: str, count: int) -> Any:
        """Create the rerun response object.

        Override this method to enable rerun functionality for the service.

        Args:
            success: Whether the operation succeeded
            msg: Status message
            count: Number of items generated

        Returns:
            A response object (e.g., RerunProfileGenerationResponse)
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _get_generated_count(
        self,
        request: TRequest,
        processed_user_ids: list[str] | None = None,
    ) -> int:
        """Get the count of generated items (profiles or playbooks) after rerun.

        Override this method to enable rerun functionality for the service.

        Args:
            request: The rerun request object (for filtering)
            processed_user_ids: List of user IDs that were successfully processed
                in the batch. Provided by _run_batch_with_progress so overrides
                don't need to handle user_id=None from batch requests.

        Returns:
            Number of items generated during rerun
        """
        raise NotImplementedError("Rerun not supported by this service")

    def _pre_process_rerun(self, request: TRequest) -> None:  # noqa: B027
        """Hook called before processing rerun items.

        Override in subclasses to perform cleanup or preparation before rerun.
        Default implementation does nothing.

        Args:
            request: The rerun request object
        """

    def run_rerun(self, request: TRequest) -> Any:
        """Run the rerun workflow for the service.

        This template method orchestrates the rerun process:
        1. Check for existing in-progress operations
        2. Get user IDs to process
        3. Pre-process hook
        4. Run batch with progress tracking
        5. Return response

        Child classes must implement the hook methods to enable rerun functionality:
        - _get_rerun_user_ids()
        - _build_rerun_request_params()
        - _create_run_request_for_item()
        - _create_rerun_response()

        Args:
            request: The rerun request object

        Returns:
            A response object with success status, message, and count
        """
        state_manager = self._create_state_manager()

        try:
            # 1. Check for existing in-progress operation
            error = state_manager.check_in_progress()
            if error:
                return self._create_rerun_response(False, error, 0)

            # 2. Get user IDs to process
            user_ids = self._get_rerun_user_ids(request)
            if not user_ids:
                return self._create_rerun_response(
                    False, "No interactions found matching the specified filters", 0
                )

            # 3. Pre-process hook (e.g., delete existing pending items)
            self._pre_process_rerun(request)

            # 4. Run batch with progress tracking
            users_processed, total_generated = self._run_batch_with_progress(
                user_ids=user_ids,
                request=request,
                request_params=self._build_rerun_request_params(request),
                state_manager=state_manager,
            )

            msg = f"Completed for {users_processed} user(s)"
            return self._create_rerun_response(True, msg, total_generated)

        except Exception as e:
            state_manager.mark_progress_failed(str(e))
            return self._create_rerun_response(
                False,
                f"Failed to run {self._get_base_service_name()}: {str(e)}",
                0,
            )
