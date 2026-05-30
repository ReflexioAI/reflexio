"""
Base class for generation services
"""

import contextvars
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Generic, TypeVar

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Status
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.extractor_config_utils import (
    filter_extractor_configs,
    get_extractor_name,
)
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
    should_extractor_run_by_stride,
)
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.service_utils import log_llm_messages, log_model_response
from reflexio.server.services.storage.storage_base import AgentRunStatus
from reflexio.server.usage_metrics import record_usage_event


class StatusChangeOperation(StrEnum):
    """Operation type for upgrade/downgrade responses."""

    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"


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
    ABC,
    Generic[TExtractorConfig, TExtractor, TGenerationServiceConfig, TRequest],  # noqa: UP046
):
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

    def _usage_pipeline(self) -> str | None:
        service_name = self._get_service_name()
        if "profile" in service_name:
            return "profile"
        if "playbook" in service_name:
            return "playbook"
        if "evaluation" in service_name:
            return "evaluation"
        return None

    def _usage_context(self) -> dict[str, Any]:
        service_config = self.service_config
        return {
            "org_id": self.org_id,
            "user_id": getattr(service_config, "user_id", None),
            "request_id": getattr(service_config, "request_id", None),
            "source": getattr(service_config, "source", None),
            "agent_version": getattr(service_config, "agent_version", None),
            "pipeline": self._usage_pipeline(),
        }

    def _record_generation_event(
        self,
        *,
        event_name: str,
        outcome: str,
        count_value: int = 1,
        duration_ms: int | None = None,
        error_kind: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record_usage_event(
            **self._usage_context(),
            event_name=event_name,
            event_category="generation",
            outcome=outcome,
            count_value=count_value,
            duration_ms=duration_ms,
            error_kind=error_kind,
            metadata=metadata,
        )

    @staticmethod
    def _count_generated_results(result: Any) -> int:
        if isinstance(result, list):
            return len(result)
        return 1 if result else 0

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

    def _filter_extractor_config_by_service_config(
        self,
        extractor_config: TExtractorConfig,
        service_config: TGenerationServiceConfig,
    ) -> TExtractorConfig | None:
        """
        Filter the extractor config based on request_sources_enabled, manual_trigger,
        and explicit extractor name filters.
        """
        filtered = filter_extractor_configs(
            extractor_configs=[extractor_config],
            source=getattr(service_config, "source", None),
            allow_manual_trigger=getattr(service_config, "allow_manual_trigger", False),
            extractor_names=getattr(service_config, "extractor_names", None),
        )
        return filtered[0] if filtered else None

    def _get_extractor_state_service_name(self) -> str | None:
        """
        Get the service name used for extractor state (stride_size bookmark) lookups.

        Override in subclasses that support stride_size-based pre-filtering to return
        the OperationStateManager service name (e.g., "profile_extractor", "playbook_extractor").
        Returns None by default, meaning stride_size pre-filtering is skipped.

        Returns:
            Optional[str]: Service name for OperationStateManager, or None to skip stride_size pre-filtering
        """
        return None

    def _filter_config_by_stride(
        self, extractor_config: TExtractorConfig
    ) -> TExtractorConfig | None:
        """
        Filter extractor config by stride_size check before the should_run LLM call.

        Skips filtering when:
        - _get_extractor_state_service_name() returns None (service doesn't support stride_size)
        - auto_run is False (rerun/manual flows skip stride_size)

        Args:
            extractor_config: Extractor config after source/manual_trigger filtering

        Returns:
            Extractor config when it passes the stride_size check, otherwise None.
        """
        state_service_name = self._get_extractor_state_service_name()
        if state_service_name is None:
            return extractor_config

        if not getattr(self.service_config, "auto_run", True):
            return extractor_config

        if getattr(self.service_config, "force_extraction", False):
            return extractor_config

        root_config = self.request_context.configurator.get_config()
        global_window_size = (
            getattr(root_config, "window_size", None) if root_config else None
        )
        global_stride_size = (
            getattr(root_config, "stride_size", None) if root_config else None
        )

        state_manager = OperationStateManager(
            self.storage,  # type: ignore[reportArgumentType]
            self.org_id,
            state_service_name,  # type: ignore[reportArgumentType]
        )

        name = get_extractor_name(extractor_config)
        _, stride_size = get_extractor_window_params(
            extractor_config, global_window_size, global_stride_size
        )

        # Resolve effective source filter for this extractor
        should_skip, effective_source = get_effective_source_filter(
            extractor_config, getattr(self.service_config, "source", None)
        )
        if should_skip:
            return None

        (
            _,
            new_interactions,
        ) = state_manager.get_extractor_state_with_new_interactions(
            extractor_name=name,
            user_id=getattr(self.service_config, "user_id", None),
            sources=effective_source,
        )
        new_count = sum(len(ri.interactions) for ri in new_interactions)

        if should_extractor_run_by_stride(new_count, stride_size):
            return extractor_config

        logger.info(
            "Stride pre-filter: skipping extractor '%s' (new=%d, stride_size=%s)",
            name,
            new_count,
            stride_size,
        )
        return None

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
        skipped (R2 / reflexio-enterprise#59).

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
        # the fix for R2 / reflexio-enterprise#59: without the payload, the
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

    def _execute_extractor(
        self,
        extractor_config: TExtractorConfig,
        identifier: str,
    ) -> Any | None:
        """
        Run the configured extractor with timeout and error handling.

        The extractor runs in a thread pool with a timeout guard so providers that
        ignore their own timeout cannot block generation forever.

        Args:
            extractor_config: Filtered extractor config to execute
            identifier: Logging context identifier (user_id or request_id)

        Returns:
            Extractor result, or None if the extractor succeeded with no output.

        Raises:
            ExtractorExecutionError: If the extractor fails with an exception or timeout.
        """
        if (
            self.service_config is None
        ):  # pragma: no cover — set by _prepare_generation_run
            raise RuntimeError("service_config must be set before executing extractor")

        self._last_extractor_run_stats = {"total": 1, "failed": 0, "timed_out": 0}
        extractor = self._create_extractor(extractor_config, self.service_config)
        executor: ThreadPoolExecutor | None = None
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            # Copy context so correlation ID propagates to worker thread
            ctx = contextvars.copy_context()
            future = executor.submit(ctx.run, extractor.run)  # type: ignore[reportAttributeAccessIssue]
            result = future.result(timeout=EXTRACTOR_TIMEOUT_SECONDS)
            if isinstance(result, ExtractionOutcome):
                if result.run_id:
                    self._last_extraction_run_ids.append(result.run_id)
                if result.status == "completed" and result.items:
                    return result.items
                logger.info(
                    "No results generated for %s identifier: %s",
                    self._get_service_name(),
                    identifier,
                )
                return None
            if result:
                return result
            logger.info(
                "No results generated for %s identifier: %s",
                self._get_service_name(),
                identifier,
            )
            return None
        except FuturesTimeoutError as exc:
            self._last_extractor_run_stats = {"total": 1, "failed": 1, "timed_out": 1}
            error_msg = (
                f"Extractor timed out after {EXTRACTOR_TIMEOUT_SECONDS} seconds "
                f"for {self._get_service_name()} identifier={identifier}"
            )
            logger.error(error_msg)
            raise ExtractorExecutionError(error_msg) from exc
        except Exception as exc:
            self._last_extractor_run_stats = {"total": 1, "failed": 1, "timed_out": 0}
            error_msg = (
                f"Extractor failed for {self._get_service_name()} "
                f"identifier={identifier}: {exc} (type={type(exc).__name__})"
            )
            logger.error(error_msg)
            raise ExtractorExecutionError(error_msg) from exc
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def _finalize_extraction_runs(self) -> None:
        if self.storage is None:
            return
        for run_id in self._last_extraction_run_ids:
            run = self.storage.get_agent_run(run_id)
            if run is None:
                continue
            status = (
                AgentRunStatus.FINALIZED_PENDING_TOOL
                if run.pending_tool_call_ids
                else AgentRunStatus.FINALIZED
            )
            self.storage.update_agent_run_status(
                run_id,
                status,
                pending_tool_call_ids=run.pending_tool_call_ids,
            )

    def _mark_extraction_runs_finalization_failed(self, exc: Exception) -> None:
        if self.storage is None:
            return
        root_config = self.request_context.configurator.get_config()
        pending_config = getattr(root_config, "pending_tool_call_config", None)
        for run_id in self._last_extraction_run_ids:
            run = self.storage.get_agent_run(run_id)
            if run is None or run.committed_output is None:
                continue
            next_attempt_count = run.finalization_attempts + 1
            max_attempts = (
                pending_config.max_finalization_attempts
                if pending_config is not None
                else 3
            )
            status = (
                AgentRunStatus.FAILED
                if next_attempt_count >= max_attempts
                else AgentRunStatus.FINALIZATION_FAILED
            )
            delay_seconds = min(300, max(1, 2 ** max(0, next_attempt_count - 1)))
            self.storage.update_agent_run_status(
                run_id,
                status,
                next_resume_at=datetime.now(UTC) + timedelta(seconds=delay_seconds),
                last_error=str(exc),
                increment_finalization_attempts=True,
            )

    def _should_run_before_extraction(self, extractor_config: TExtractorConfig) -> bool:
        """
        Pre-extraction check called before extractor execution.

        Template method that:
        1. Skips for non-auto runs and mock mode
        2. Returns True immediately when service_config.force_extraction=True
           (bypasses cheap pre-filter and LLM should_run vote)
        3. Collects scoped interactions via _collect_scoped_interactions_for_precheck
        4. Delegates prompt building to _build_should_run_prompt (subclass hook)
        5. Makes a single LLM call to determine if extraction should proceed

        Override _build_should_run_prompt in subclasses to provide service-specific
        criteria and prompt construction. Default returns True (always run) when
        no prompt hook is provided.

        Args:
            extractor_config: Enabled extractor config that will be run

        Returns:
            bool: True if extraction should proceed, False to skip
        """
        # Skip for non-auto runs (rerun/manual flows always run)
        if not getattr(self.service_config, "auto_run", True):
            return True

        # Skip for mock mode
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            return True

        # `force_extraction=True` is the caller's explicit "no gates" signal —
        # corrections, manual /learn, anything time-sensitive. Bypass the
        # cheap pre-filter (slash-only / too-short rejects) and the LLM
        # should_run vote so the extractor always runs on this batch.
        if getattr(self.service_config, "force_extraction", False):
            return True

        # Skip if org config disables the pre-extraction check
        root_config = self.request_context.configurator.get_config()
        if root_config and root_config.skip_should_run_check:
            logger.info(
                "skip_should_run_check is enabled for %s, bypassing pre-extraction check",
                self._get_service_name(),
            )
            return True

        # Collect scoped interactions
        session_data_models, scoped_config = (
            self._collect_scoped_interactions_for_precheck(extractor_config)
        )
        if not session_data_models:
            logger.info(
                "No interactions found for consolidated should_generate check for %s",
                self._get_service_name(),
            )
            return False

        # Cheap pre-filter: reject batches that are structurally unable
        # to yield signal (slash-commands only, too-short user turns,
        # extractor-prompt echoes) without burning a 5–7s LLM call. See
        # _cheap_should_run_reject for the rule set.
        reject_reason = _cheap_should_run_reject(session_data_models)
        if reject_reason is not None:
            logger.info(
                "Cheap pre-filter rejected %s should_run: reason=%s identifier=%s",
                self._get_service_name(),
                reject_reason,
                getattr(self.service_config, "user_id", None) or "unknown",
            )
            return False

        # Build prompt via subclass hook
        prompt = self._build_should_run_prompt(scoped_config, session_data_models)
        if not prompt:
            return True  # No prompt means no check needed, proceed

        # Resolve model and make LLM call
        should_run_model = self._resolve_should_run_model()
        identifier = getattr(self.service_config, "user_id", None) or "unknown"
        try:
            should_start = time.perf_counter()
            logger.info(
                "event=consolidated_should_run_start service=%s identifier=%s model=%s extractor=%s",
                self._get_service_name(),
                identifier,
                should_run_model,
                get_extractor_name(extractor_config),
            )
            log_llm_messages(
                logger,
                "Should extract check",
                [{"role": "user", "content": prompt}],
            )

            content = self.client.generate_chat_response(
                messages=[{"role": "user", "content": prompt}],
                model=should_run_model,
            )
            log_model_response(
                logger,
                f"Consolidated {self._get_service_name()} should_run response",
                content,
            )
            decision = bool(content and "true" in content.lower())  # type: ignore[reportAttributeAccessIssue]
            logger.info(
                "event=consolidated_should_run_end service=%s identifier=%s elapsed_seconds=%.3f decision=%s",
                self._get_service_name(),
                identifier,
                time.perf_counter() - should_start,
                decision,
            )
            return decision
        except Exception as exc:
            logger.error(
                "Consolidated should_generate check failed for %s: %s, defaulting to run",
                self._get_service_name(),
                str(exc),
            )
            return True

    def _build_should_run_prompt(
        self,
        scoped_config: TExtractorConfig,  # noqa: ARG002
        session_data_models: list[RequestInteractionDataModel],  # noqa: ARG002
    ) -> str | None:
        """
        Build the prompt for the consolidated should_run LLM check.

        Override in subclasses to provide service-specific criteria building
        and prompt rendering. Return None if no check is needed (always proceed).

        Args:
            scoped_config: Extractor config that had scoped interactions
            session_data_models: Deduplicated request interaction data models

        Returns:
            Optional[str]: The rendered prompt string, or None to skip the check
        """
        return None

    def _collect_scoped_interactions_for_precheck(
        self, extractor_config: TExtractorConfig
    ) -> tuple[list[RequestInteractionDataModel], TExtractorConfig]:
        """
        Collect interactions for consolidated pre-check using extractor-scoped filters.

        Mirrors each extractor's source/window scope so the consolidated gate
        does not skip valid extraction because of an unrelated fixed interaction slice.

        Args:
            extractor_config: Enabled extractor config after request-level filtering

        Returns:
            tuple: (session data models, extractor config)
        """
        root_config = self.request_context.configurator.get_config()
        global_window_size = (
            getattr(root_config, "window_size", None) if root_config else None
        )
        global_stride_size = (
            getattr(root_config, "stride_size", None) if root_config else None
        )

        extra_kwargs = self._get_precheck_interaction_query_kwargs()

        should_skip, effective_source = get_effective_source_filter(
            extractor_config, getattr(self.service_config, "source", None)
        )
        if should_skip:
            return [], extractor_config

        window_size, _ = get_extractor_window_params(
            extractor_config, global_window_size, global_stride_size
        )
        session_data_models, _ = self.storage.get_last_k_interactions_grouped(  # type: ignore[reportOptionalMemberAccess]
            user_id=getattr(self.service_config, "user_id", None),
            k=window_size,
            sources=effective_source,
            start_time=getattr(self.service_config, "rerun_start_time", None),
            end_time=getattr(self.service_config, "rerun_end_time", None),
            **extra_kwargs,
        )

        return session_data_models, extractor_config

    def _get_precheck_interaction_query_kwargs(self) -> dict:
        """
        Return extra keyword arguments for get_last_k_interactions_grouped in precheck.

        Override in subclasses that need additional query parameters
        (e.g., agent_version for playbook services).

        Returns:
            dict: Extra kwargs to pass to get_last_k_interactions_grouped
        """
        return {}

    def _resolve_should_run_model(self) -> str:
        """
        Resolve the model name for should_run/should_generate LLM checks.

        Uses LLM config override if available, falls back to site var setting.

        Returns:
            str: Model name for the should_run check
        """
        from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
        from reflexio.server.site_var.site_var_manager import SiteVarManager

        root_config = self.request_context.configurator.get_config()
        llm_config = root_config.llm_config if root_config else None
        api_key_config = root_config.api_key_config if root_config else None

        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}

        return resolve_model_name(
            ModelRole.SHOULD_RUN,
            site_var_value=site_var.get("should_run_model_name"),
            config_override=llm_config.should_run_model_name if llm_config else None,
            api_key_config=api_key_config,
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

    # ===============================
    # Upgrade/Downgrade methods (optional - override to enable)
    # ===============================

    def _has_items_with_status(self, status: Status | None, request: TRequest) -> bool:
        """Check if items exist with given status and filters from request.

        Override this method to enable upgrade/downgrade functionality for the service.

        Args:
            status: The status to check for (None for CURRENT)
            request: The upgrade/downgrade request object with filters

        Returns:
            bool: True if any matching items exist
        """
        raise NotImplementedError("Upgrade/downgrade not supported by this service")

    def _delete_items_by_status(self, status: Status, request: TRequest) -> int:
        """Delete items with given status matching request filters.

        Override this method to enable upgrade/downgrade functionality for the service.

        Args:
            status: The status of items to delete
            request: The upgrade/downgrade request object with filters

        Returns:
            int: Number of items deleted
        """
        raise NotImplementedError("Upgrade/downgrade not supported by this service")

    def _update_items_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        request: TRequest,
        user_ids: list[str] | None = None,
    ) -> int:
        """Update items from old_status to new_status with request filters.

        Override this method to enable upgrade/downgrade functionality for the service.

        Args:
            old_status: The current status to match (None for CURRENT)
            new_status: The new status to set (None for CURRENT)
            request: The upgrade/downgrade request object with filters
            user_ids: Optional pre-computed list of user IDs to filter by

        Returns:
            int: Number of items updated
        """
        raise NotImplementedError("Upgrade/downgrade not supported by this service")

    def _get_affected_user_ids_for_upgrade(self, request: TRequest) -> list[str] | None:  # noqa: ARG002
        """Get user IDs to filter by for upgrade operations.

        Override this method to support the only_affected_users flag.
        By default returns None (no filtering).

        Args:
            request: The upgrade request object

        Returns:
            Optional[list[str]]: List of user IDs to filter by, or None for no filtering
        """
        return None

    def _get_affected_user_ids_for_downgrade(
        self,
        request: TRequest,  # noqa: ARG002
    ) -> list[str] | None:
        """Get user IDs to filter by for downgrade operations.

        Override this method to support the only_affected_users flag.
        By default returns None (no filtering).

        Args:
            request: The downgrade request object

        Returns:
            Optional[list[str]]: List of user IDs to filter by, or None for no filtering
        """
        return None

    def _create_status_change_response(
        self,
        operation: StatusChangeOperation,
        success: bool,
        counts: dict,
        msg: str,
    ) -> Any:
        """Create upgrade or downgrade response object based on operation type.

        Override this method to enable upgrade/downgrade functionality for the service.

        Args:
            operation: The operation type (UPGRADE or DOWNGRADE)
            success: Whether the operation succeeded
            counts: Dictionary of counts (upgrade: deleted/archived/promoted, downgrade: demoted/restored)
            msg: Status message

        Returns:
            A response object (e.g., UpgradeProfilesResponse, DowngradeUserPlaybooksResponse)
        """
        raise NotImplementedError("Upgrade/downgrade not supported by this service")

    def run_upgrade(self, request: TRequest) -> Any:
        """Run the upgrade workflow for the service.

        This template method orchestrates the upgrade process:
        1. Validate that pending items exist
        2. Delete old archived items
        3. Archive current items (None → ARCHIVED)
        4. Promote pending items (PENDING → None/CURRENT)

        Child classes must implement the hook methods to enable upgrade functionality:
        - _has_items_with_status()
        - _delete_items_by_status()
        - _update_items_status()
        - _create_status_change_response()

        Args:
            request: The upgrade request object with optional filters

        Returns:
            A response object with success status, counts, and message
        """
        try:
            # 1. Validate pending items exist
            if not self._has_items_with_status(Status.PENDING, request):
                return self._create_status_change_response(
                    StatusChangeOperation.UPGRADE,
                    False,
                    {"deleted": 0, "archived": 0, "promoted": 0},
                    "No pending items found to upgrade",
                )

            # Get affected user IDs once (child class determines the logic)
            affected_user_ids = self._get_affected_user_ids_for_upgrade(request)

            # 2. Delete old archived items (skip if archive_current=False)
            deleted = 0
            archived = 0
            if getattr(request, "archive_current", True):
                deleted = self._delete_items_by_status(Status.ARCHIVED, request)

                # 3. Archive current items (None → ARCHIVED)
                archived = self._update_items_status(
                    None, Status.ARCHIVED, request, user_ids=affected_user_ids
                )

            # 4. Promote pending items (PENDING → None)
            promoted = self._update_items_status(
                Status.PENDING, None, request, user_ids=affected_user_ids
            )

            msg = f"Upgraded: {promoted} promoted, {archived} archived, {deleted} old archived deleted"
            return self._create_status_change_response(
                StatusChangeOperation.UPGRADE,
                True,
                {"deleted": deleted, "archived": archived, "promoted": promoted},
                msg,
            )

        except Exception as e:
            return self._create_status_change_response(
                StatusChangeOperation.UPGRADE,
                False,
                {"deleted": 0, "archived": 0, "promoted": 0},
                f"Failed to upgrade: {str(e)}",
            )

    def run_downgrade(self, request: TRequest) -> Any:
        """Run the downgrade workflow for the service.

        This template method orchestrates the downgrade process:
        1. Validate that archived items exist
        2. Demote current items (None → ARCHIVE_IN_PROGRESS)
        3. Restore archived items (ARCHIVED → None/CURRENT)
        4. Complete archiving (ARCHIVE_IN_PROGRESS → ARCHIVED)

        Child classes must implement the hook methods to enable downgrade functionality:
        - _has_items_with_status()
        - _update_items_status()
        - _create_status_change_response()

        Args:
            request: The downgrade request object with optional filters

        Returns:
            A response object with success status, counts, and message
        """
        try:
            # 1. Validate archived items exist
            if not self._has_items_with_status(Status.ARCHIVED, request):
                return self._create_status_change_response(
                    StatusChangeOperation.DOWNGRADE,
                    False,
                    {"demoted": 0, "restored": 0},
                    "No archived items found to restore",
                )

            # Get affected user IDs once (child class determines the logic)
            affected_user_ids = self._get_affected_user_ids_for_downgrade(request)

            # 2. Demote current (None → ARCHIVE_IN_PROGRESS)
            demoted = self._update_items_status(
                None, Status.ARCHIVE_IN_PROGRESS, request, user_ids=affected_user_ids
            )

            # 3. Restore archived (ARCHIVED → None)
            restored = self._update_items_status(
                Status.ARCHIVED, None, request, user_ids=affected_user_ids
            )

            # 4. Complete archiving (ARCHIVE_IN_PROGRESS → ARCHIVED)
            self._update_items_status(
                Status.ARCHIVE_IN_PROGRESS,
                Status.ARCHIVED,
                request,
                user_ids=affected_user_ids,
            )

            msg = f"Downgraded: {demoted} archived, {restored} restored"
            return self._create_status_change_response(
                StatusChangeOperation.DOWNGRADE,
                True,
                {"demoted": demoted, "restored": restored},
                msg,
            )

        except Exception as e:
            return self._create_status_change_response(
                StatusChangeOperation.DOWNGRADE,
                False,
                {"demoted": 0, "restored": 0},
                f"Failed to downgrade: {str(e)}",
            )
