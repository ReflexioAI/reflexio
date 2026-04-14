"""
Base class for generation services
"""

import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from enum import StrEnum
from typing import Any, Generic, TypeVar

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Status
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extractor_config_utils import (
    filter_extractor_configs,
    get_extractor_name,
)
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
    should_extractor_run_by_batch_interval,
)
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.service_utils import log_llm_messages, log_model_response


class StatusChangeOperation(StrEnum):
    """Operation type for upgrade/downgrade responses."""

    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"


class ExtractorExecutionError(RuntimeError):
    """Raised when all extractors fail for a request/user context."""


logger = logging.getLogger(__name__)

# Timeout for individual extractor execution (safety net if LLM provider ignores its own timeout)
EXTRACTOR_TIMEOUT_SECONDS = 300

TExtractorConfig = TypeVar("TExtractorConfig")
TExtractor = TypeVar("TExtractor")
TGenerationServiceConfig = TypeVar("TGenerationServiceConfig")
TRequest = TypeVar("TRequest")


# Unified base class for all generation services (evaluation, playbook, profile)
class BaseGenerationService(
    ABC,
    Generic[TExtractorConfig, TExtractor, TGenerationServiceConfig, TRequest],
):
    """
    Base class for generation services that run multiple extractors sequentially.

    This unified class supports two types of services:
    1. Evaluation services (playbook, agent success) - process interactions and save UserPlaybook
    2. Profile services - process interactions with existing data and apply updates

    Type Parameters:
        TExtractorConfig: The extractor configuration type from YAML (e.g., PlaybookConfig, ProfileExtractorConfig)
        TExtractor: The extractor type (e.g., PlaybookExtractor, ProfileExtractor, AgentSuccessEvaluator)
        TGenerationServiceConfig: The runtime service configuration type (e.g., PlaybookGenerationServiceConfig, ProfileGenerationServiceConfig)
        TRequest: The request type (e.g., ProfileGenerationRequest, PlaybookGenerationRequest, AgentSuccessEvaluationRequest)

    Child classes must implement:
    - _load_extractor_configs(): Load extractor configurations from configurator
    - _load_generation_service_config(): Extract parameters from request and return GenerationServiceConfig
    - _create_extractor(): Create extractor instances with extractor config and service config
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

    @abstractmethod
    def _load_extractor_configs(self) -> list[TExtractorConfig]:
        """
        Load extractor configurations from the configurator.

        Returns:
            List of extractor configuration objects (from YAML)
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

    def _filter_extractor_configs_by_service_config(
        self,
        extractor_configs: list[TExtractorConfig],
        service_config: TGenerationServiceConfig,
    ) -> list[TExtractorConfig]:
        """
        Filter extractor configs based on request_sources_enabled and manual_trigger fields.

        Args:
            extractor_configs: List of extractor configuration objects from YAML
            service_config: Runtime service configuration containing the source and allow_manual_trigger flag

        Returns:
            Filtered list of extractor configs that should run for the given source and trigger mode
        """
        # Extract filtering parameters from service_config
        source = getattr(service_config, "source", None)
        allow_manual_trigger = getattr(service_config, "allow_manual_trigger", False)
        extractor_names = getattr(service_config, "extractor_names", None)

        return filter_extractor_configs(
            extractor_configs=extractor_configs,
            source=source,
            allow_manual_trigger=allow_manual_trigger,
            extractor_names=extractor_names,
        )

    def _get_extractor_state_service_name(self) -> str | None:
        """
        Get the service name used for extractor state (batch_interval bookmark) lookups.

        Override in subclasses that support batch_interval-based pre-filtering to return
        the OperationStateManager service name (e.g., "profile_extractor", "playbook_extractor").
        Returns None by default, meaning batch_interval pre-filtering is skipped.

        Returns:
            Optional[str]: Service name for OperationStateManager, or None to skip batch_interval pre-filtering
        """
        return None

    def _filter_configs_by_batch_interval(
        self, extractor_configs: list[TExtractorConfig]
    ) -> list[TExtractorConfig]:
        """
        Filter extractor configs by batch_interval check before the should_run LLM call.

        Skips filtering when:
        - _get_extractor_state_service_name() returns None (service doesn't support batch_interval)
        - auto_run is False (rerun/manual flows skip batch_interval)

        For each config, resolves batch_size/batch_interval params and checks if enough new
        interactions exist since the last run. Only configs that pass batch_interval are returned.

        Args:
            extractor_configs: List of extractor configs after source/manual_trigger filtering

        Returns:
            List of extractor configs that pass the batch_interval check
        """
        state_service_name = self._get_extractor_state_service_name()
        if state_service_name is None:
            return extractor_configs

        if not getattr(self.service_config, "auto_run", True):
            return extractor_configs

        if getattr(self.service_config, "force_extraction", False):
            return extractor_configs

        root_config = self.request_context.configurator.get_config()
        global_batch_size = (
            getattr(root_config, "batch_size", None) if root_config else None
        )
        global_batch_interval = (
            getattr(root_config, "batch_interval", None) if root_config else None
        )

        state_manager = OperationStateManager(
            self.storage,  # type: ignore[reportArgumentType]
            self.org_id,
            state_service_name,  # type: ignore[reportArgumentType]
        )

        passing_configs: list[TExtractorConfig] = []
        for config in extractor_configs:
            name = get_extractor_name(config)
            _, batch_interval_size = get_extractor_window_params(
                config, global_batch_size, global_batch_interval
            )

            # Resolve effective source filter for this extractor
            should_skip, effective_source = get_effective_source_filter(
                config, getattr(self.service_config, "source", None)
            )
            if should_skip:
                continue

            (
                _,
                new_interactions,
            ) = state_manager.get_extractor_state_with_new_interactions(
                extractor_name=name,
                user_id=getattr(self.service_config, "user_id", None),
                sources=effective_source,
            )
            new_count = sum(len(ri.interactions) for ri in new_interactions)

            if should_extractor_run_by_batch_interval(new_count, batch_interval_size):
                passing_configs.append(config)
            else:
                logger.info(
                    "Batch interval pre-filter: skipping extractor '%s' (new=%d, batch_interval=%s)",
                    name,
                    new_count,
                    batch_interval_size,
                )

        return passing_configs

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

        # Try to acquire lock
        if not state_manager.acquire_lock(my_request_id, scope_id=scope_id):
            return  # Another operation is running, we've updated pending_request_id

        # Re-run loop: keep running until no new requests come in
        try:
            while True:
                self._run_generation(request)

                # If in batch mode and cancellation was requested, clear lock
                # to prevent queued pending requests from running, then stop
                if self._is_batch_mode and state_manager.is_cancellation_requested():
                    state_manager.clear_lock(scope_id=scope_id)
                    logger.info(
                        "Cancellation detected in run() for %s, cleared lock to prevent pending re-runs",
                        self._get_service_name(),
                    )
                    break

                # Check if another request came in during our run
                pending_request_id = state_manager.release_lock(
                    my_request_id, scope_id=scope_id
                )

                logger.info(
                    "Released in-progress lock for %s: request_id=%s, pending_request_id=%s",
                    self._get_service_name(),
                    my_request_id,
                    pending_request_id,
                )

                if not pending_request_id:
                    break  # No pending request, we're done

                # Another request came in, update my_request_id and re-run
                my_request_id = pending_request_id

        except Exception:
            # Clear lock on error to prevent deadlock
            state_manager.clear_lock(scope_id=scope_id)
            raise

    def _run_generation(self, request: TRequest) -> None:
        """
        Run the actual generation logic.

        Orchestrates validation, config loading, extractor execution, and result
        processing by delegating to _prepare_generation_run and _execute_extractors.

        Args:
            request: The request object containing parameters
        """
        if not request:
            logger.error("Received None request for %s", self._get_service_name())
            return

        try:
            extractor_configs, identifier = self._prepare_generation_run(request)
            if extractor_configs is None:
                return

            all_results = self._execute_extractors(
                extractor_configs, request, identifier
            )

            if all_results:
                self._process_results(all_results)

        except Exception as e:
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
    ) -> tuple[list[TExtractorConfig] | None, str]:
        """
        Validate request, load config, filter extractor configs, and run pre-extraction checks.

        Loads the generation service config from the request, loads and filters extractor
        configs by source, manual trigger, and batch_interval, then runs the pre-extraction gate.

        Args:
            request: The request object containing parameters

        Returns:
            tuple: (extractor_configs, identifier) where extractor_configs is None if
                generation should be skipped, or the filtered list of configs to execute.
                identifier is a string for logging context.
        """
        self.service_config = self._load_generation_service_config(request)

        extractor_configs = self._load_extractor_configs()
        if not extractor_configs:
            logger.warning("No %s extractor configs found", self._get_service_name())
            return None, ""

        extractor_configs = self._filter_extractor_configs_by_service_config(
            extractor_configs, self.service_config
        )

        if not extractor_configs:
            source = getattr(self.service_config, "source", "N/A")
            source_display = source or "N/A"
            logger.info(
                "No %s extractor configs enabled for source: %s",
                self._get_service_name(),
                source_display,
            )
            return None, ""

        extractor_configs = self._filter_configs_by_batch_interval(extractor_configs)
        if not extractor_configs:
            logger.info(
                "No extractor configs passed batch_interval check for %s",
                self._get_service_name(),
            )
            return None, ""

        identifier = getattr(self.service_config, "user_id", None) or getattr(
            self.service_config, "request_id", "unknown"
        )

        if not self._should_run_before_extraction(extractor_configs):
            logger.info(
                "Pre-extraction check returned False for %s identifier=%s, skipping",
                self._get_service_name(),
                identifier,
            )
            return None, identifier

        return extractor_configs, identifier

    def _execute_extractors(
        self,
        extractor_configs: list[TExtractorConfig],
        request: TRequest,
        identifier: str,
    ) -> list:
        """
        Run extractors sequentially with timeout and error handling.

        Each extractor runs independently in a thread pool with a timeout guard.
        After the first extractor, the service config is reloaded so subsequent
        extractors see updated state. Updates _last_extractor_run_stats and raises
        ExtractorExecutionError if all extractors fail.

        Args:
            extractor_configs: Filtered list of extractor configs to execute
            request: The original request (used to reload service config between extractors)
            identifier: Logging context identifier (user_id or request_id)

        Returns:
            list: Results from successful extractors (may be empty if none produced output
                but at least one succeeded without error)

        Raises:
            ExtractorExecutionError: If every extractor fails with an exception
        """
        if (
            self.service_config is None
        ):  # pragma: no cover — set by _prepare_generation_run
            raise RuntimeError("service_config must be set before executing extractors")

        all_results: list = []
        previously_extracted: list = []
        run_stats = {"total": len(extractor_configs), "failed": 0, "timed_out": 0}

        for i, config in enumerate(extractor_configs):
            if i > 0 and previously_extracted:
                self.service_config = self._load_generation_service_config(request)
                self._update_config_for_incremental(previously_extracted)

            extractor = self._create_extractor(config, self.service_config)
            executor: ThreadPoolExecutor | None = None
            try:
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(extractor.run)  # type: ignore[reportAttributeAccessIssue]
                result = future.result(timeout=EXTRACTOR_TIMEOUT_SECONDS)
                if result:
                    all_results.append(result)
                    previously_extracted.append(result)
            except FuturesTimeoutError:
                run_stats["failed"] += 1
                run_stats["timed_out"] += 1
                logger.error(
                    "Extractor timed out after %d seconds for %s identifier=%s",
                    EXTRACTOR_TIMEOUT_SECONDS,
                    self._get_service_name(),
                    identifier,
                )
                continue
            except Exception as e:
                run_stats["failed"] += 1
                logger.error(
                    "Extractor failed for %s identifier=%s: %s (type=%s)",
                    self._get_service_name(),
                    identifier,
                    str(e),
                    type(e).__name__,
                )
                continue
            finally:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)

        self._last_extractor_run_stats = run_stats

        if not all_results:
            all_extractors_failed = (
                run_stats["total"] > 0 and run_stats["failed"] == run_stats["total"]
            )
            if all_extractors_failed:
                error_msg = (
                    f"All extractors failed for {self._get_service_name()} "
                    f"identifier={identifier}"
                )
                logger.error(error_msg)
                raise ExtractorExecutionError(error_msg)
            logger.info(
                "No results generated for %s identifier: %s",
                self._get_service_name(),
                identifier,
            )

        return all_results

    def _should_run_before_extraction(
        self, extractor_configs: list[TExtractorConfig]
    ) -> bool:
        """
        Pre-extraction check called before the sequential extraction loop.

        Template method that:
        1. Skips for non-auto runs and mock mode
        2. Collects scoped interactions via _collect_scoped_interactions_for_precheck
        3. Delegates prompt building to _build_should_run_prompt (subclass hook)
        4. Makes a single LLM call to determine if extraction should proceed

        Override _build_should_run_prompt in subclasses to provide service-specific
        criteria and prompt construction. Default returns True (always run) when
        no prompt hook is provided.

        Args:
            extractor_configs: List of enabled extractor configs that will be run

        Returns:
            bool: True if extraction should proceed, False to skip
        """
        # Skip for non-auto runs (rerun/manual flows always run)
        if not getattr(self.service_config, "auto_run", True):
            return True

        # Skip for mock mode
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
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
        session_data_models, scoped_configs = (
            self._collect_scoped_interactions_for_precheck(extractor_configs)
        )
        if not session_data_models:
            logger.info(
                "No interactions found for consolidated should_generate check for %s",
                self._get_service_name(),
            )
            return False

        # Build prompt via subclass hook
        prompt = self._build_should_run_prompt(scoped_configs, session_data_models)
        if not prompt:
            return True  # No prompt means no check needed, proceed

        # Resolve model and make LLM call
        should_run_model = self._resolve_should_run_model()
        identifier = getattr(self.service_config, "user_id", None) or "unknown"
        try:
            should_start = time.perf_counter()
            logger.info(
                "event=consolidated_should_run_start service=%s identifier=%s model=%s extractors=%d",
                self._get_service_name(),
                identifier,
                should_run_model,
                len(extractor_configs),
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
        scoped_configs: list[TExtractorConfig],  # noqa: ARG002
        session_data_models: list[RequestInteractionDataModel],  # noqa: ARG002
    ) -> str | None:
        """
        Build the prompt for the consolidated should_run LLM check.

        Override in subclasses to provide service-specific criteria building
        and prompt rendering. Return None if no check is needed (always proceed).

        Args:
            scoped_configs: Extractor configs that had scoped interactions
            session_data_models: Deduplicated request interaction data models

        Returns:
            Optional[str]: The rendered prompt string, or None to skip the check
        """
        return None

    def _collect_scoped_interactions_for_precheck(
        self, extractor_configs: list[TExtractorConfig]
    ) -> tuple[list[RequestInteractionDataModel], list[TExtractorConfig]]:
        """
        Collect interactions for consolidated pre-check using extractor-scoped filters.

        Mirrors each extractor's source/window scope so the consolidated gate
        does not skip valid extraction because of an unrelated fixed interaction slice.

        Args:
            extractor_configs: Enabled extractor configs after request-level filtering

        Returns:
            tuple: (deduplicated session data models, extractor configs that had scoped interactions)
        """
        root_config = self.request_context.configurator.get_config()
        global_batch_size = (
            getattr(root_config, "batch_size", None) if root_config else None
        )
        global_batch_interval = (
            getattr(root_config, "batch_interval", None) if root_config else None
        )

        deduped_sessions: dict[str, RequestInteractionDataModel] = {}
        scoped_configs: list[TExtractorConfig] = []
        extra_kwargs = self._get_precheck_interaction_query_kwargs()

        for config in extractor_configs:
            should_skip, effective_source = get_effective_source_filter(
                config, getattr(self.service_config, "source", None)
            )
            if should_skip:
                continue

            batch_size, _ = get_extractor_window_params(
                config, global_batch_size, global_batch_interval
            )
            fetch_k = batch_size
            session_data_models, _ = self.storage.get_last_k_interactions_grouped(  # type: ignore[reportOptionalMemberAccess]
                user_id=getattr(self.service_config, "user_id", None),
                k=fetch_k,
                sources=effective_source,
                start_time=getattr(self.service_config, "rerun_start_time", None),
                end_time=getattr(self.service_config, "rerun_end_time", None),
                **extra_kwargs,
            )
            if not session_data_models:
                continue

            scoped_configs.append(config)
            for data_model in session_data_models:
                request_id = getattr(data_model.request, "request_id", None)
                dedupe_key = (
                    request_id
                    or data_model.session_id
                    or f"scoped_group_{len(deduped_sessions)}"
                )
                if dedupe_key not in deduped_sessions:
                    deduped_sessions[dedupe_key] = data_model

        return list(deduped_sessions.values()), scoped_configs

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

    def _update_config_for_incremental(self, previously_extracted: list) -> None:  # noqa: B027
        """
        Update service_config for incremental extraction after the first extractor.

        Override in subclasses that support incremental extraction to set
        is_incremental and previously_extracted on the service config.
        Default implementation does nothing.

        Args:
            previously_extracted: List of results from previous extractors
        """

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
        Each extractor collects its own data using its configured batch_size.

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
        Each extractor collects its own data using its configured batch_size.

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
