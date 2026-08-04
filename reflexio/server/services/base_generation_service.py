"""
Base class for generation services
"""

import logging
import re
import time
import unicodedata
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.services.base_generation import (
    BatchProgressMixin,
    ConfigFilterMixin,
    ExtractionRunLifecycleMixin,
    ShouldRunPrecheckMixin,
    StatusChangeMixin,
    UsageBillingMixin,
)
from reflexio.server.services.base_generation import (
    StatusChangeOperation as StatusChangeOperation,  # re-export for back-compat
)
from reflexio.server.services.deferred_learning_plan import (
    ExtractorBookmarkAdvance,
    GenerationComputePlan,
)
from reflexio.server.services.extractor_config_utils import (
    get_extractor_name,
)
from reflexio.server.services.operation_state_utils import OperationStateManager


class ExtractorExecutionError(RuntimeError):
    """Raised when the configured extractor fails for a request/user context."""


logger = logging.getLogger(__name__)


# Cheap-signal threshold for the pre-LLM should_run filter. Tuned for
# coding-assistant traffic where most turns are either slash commands
# or tool scaffolding, and where the LLM should_run gate costs 5–7s
# even when it ultimately votes False. Non-Latin letters and numbers
# count twice because many scripts carry more meaning per character.
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
    """Collect non-empty user-authored content, preserving interaction order.

    Role matching is case-insensitive because API and CLI publishers use
    different casing. These contents feed user-text-only heuristics; they do
    not determine stride or window eligibility.
    """
    out: list[str] = []
    for model in session_data_models:
        out.extend(
            interaction.content
            for interaction in model.interactions
            if interaction.role.casefold() == "user" and interaction.content
        )
    return out


def _weighted_content_length(content: str) -> int:
    """Count original non-Latin letters and numbers twice.

    Normalize each code point only for script classification so compatibility
    forms such as full-width Latin stay unweighted without changing the
    original content length.
    """
    stripped = content.strip()

    def is_non_latin_letter_or_number(char: str) -> bool:
        normalized_chars = [
            normalized_char
            for normalized_char in unicodedata.normalize("NFKC", char)
            if unicodedata.category(normalized_char)[0] in {"L", "N"}
        ]
        return bool(normalized_chars) and all(
            not normalized_char.isascii()
            and "LATIN" not in unicodedata.name(normalized_char, "")
            for normalized_char in normalized_chars
        )

    non_latin_bonus = sum(1 for char in stripped if is_non_latin_letter_or_number(char))
    return len(stripped) + non_latin_bonus


def _cheap_should_run_reject(
    session_data_models: list[RequestInteractionDataModel],
) -> str | None:
    """Cheap pre-filter for the consolidated should_run LLM gate.

    Returns a short reason string when we can cheaply decide the batch
    has no learnable signal — the caller logs the reason and skips the
    LLM call. Returns None when we cannot decide cheaply and the LLM
    should run.

    When the scoped batch has no non-empty user-authored content, these
    user-text-only heuristics are inapplicable and the batch falls through to
    the LLM gate. Stride and window eligibility are evaluated independently.

    Rejection rules when user-authored content is present:
        - No user message reaches ``_MIN_USER_CONTENT_LEN`` weighted
          characters (purely short commands / confirmations). Non-Latin
          letters and numbers count twice.
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
        return None

    for content in user_contents:
        lowered = content.lstrip().lower()
        if any(lowered.startswith(p) for p in _EXTRACTOR_PROMPT_PREFIXES):
            return "extractor_prompt_echo"

    if not any(
        len(content.strip()) >= _MIN_USER_CONTENT_LEN
        or _weighted_content_length(content) >= _MIN_USER_CONTENT_LEN
        for content in user_contents
    ):
        return "all_user_turns_too_short"

    if all(_is_pure_slash_command(c) for c in user_contents):
        return "all_slash_commands"

    return None


# Timeout for individual extractor execution (safety net if LLM provider ignores its own timeout)
EXTRACTOR_TIMEOUT_SECONDS = 300
# A two-rung LLM ladder can legitimately consume two independent 125-second
# provider bounds during one tool-driven extraction step. Leave enough room for
# that fallback and one follow-up while staying below the 600-second outer
# generation-service guard.
FALLBACK_EXTRACTOR_TIMEOUT_SECONDS = 540

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
    BatchProgressMixin[TRequest],
    UsageBillingMixin[TExtractorConfig, TGenerationServiceConfig],
    ExtractionRunLifecycleMixin[TExtractorConfig, TGenerationServiceConfig],
    ShouldRunPrecheckMixin[TExtractorConfig, TGenerationServiceConfig],
    ConfigFilterMixin[TExtractorConfig, TGenerationServiceConfig],
    StatusChangeMixin[TRequest],
    ABC,
    Generic[TExtractorConfig, TExtractor, TGenerationServiceConfig, TRequest],  # noqa: UP046
):
    # Only profile/playbook GENERATION services emit extraction-run billing here.
    # Non-extraction learning mutation paths emit their value facet at their own
    # durable-success point.
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
        # Stride-bookmark advance deferred off the extractor (F1); captured in
        # ``_execute_extractor`` and applied in the persist half of the run.
        self._last_bookmark_advance: ExtractorBookmarkAdvance | None = None
        self._last_model_provenance: ModelProvenance | None = None
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
        lock_request_id = getattr(request, "request_id", None) or str(uuid.uuid4())

        state_manager = self._create_state_manager()

        # Try to acquire lock — pass the serialized payload so blocked
        # publishes land in the queue with their own data attached. This is
        # the fix for R2: without the payload, the
        # rerun re-uses the holder's request and the queued users' batches
        # never get extracted.
        my_payload = self._serialize_request_for_queue(request)
        if not state_manager.acquire_lock(
            lock_request_id, scope_id=scope_id, payload=my_payload
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
                    lock_request_id, scope_id=scope_id
                )

                if next_entry is None:
                    break  # Queue empty — we're done

                next_lock_request_id = next_entry["request_id"]
                next_payload = next_entry.get("payload")

                logger.info(
                    "Draining queued %s request: prev_lock_request_id=%s, next_lock_request_id=%s, "
                    "payload_present=%s",
                    self._get_service_name(),
                    lock_request_id,
                    next_lock_request_id,
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

                lock_request_id = next_lock_request_id

        except Exception:
            # Clear lock on error to prevent deadlock
            state_manager.clear_lock(scope_id=scope_id)
            raise

    def _run_generation(self, request: TRequest) -> None:
        """Run one generation synchronously: compute -> persist -> emit.

        Thin wrapper preserving the pre-split behavior for the synchronous
        ``.run()`` / manual / rerun callers: compute (extractor + dedup +
        embeddings), persist (row writes + the extractor stride-bookmark
        advance) and post-commit side-effects (telemetry + billing) run together
        with **no** external ``commit_scope``. Only the durable worker (Task 8/9)
        drives ``compute_generation`` / ``persist_generation`` /
        ``emit_generation_side_effects`` separately across its fenced scope.

        The outer ``except`` records ``generation_failed`` and re-raises
        ``ExtractorExecutionError`` exactly as before.

        Args:
            request: The request object containing parameters
        """
        generation_start = time.perf_counter()
        try:
            plan = self.compute_generation(request)
            if plan is None:
                return
            self.persist_generation(plan)
            self.emit_generation_side_effects(plan)
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

    def compute_generation(self, request: TRequest) -> GenerationComputePlan | None:
        """Compute half of one generation run — NO learning DB write, NO fence.

        Runs the prepare gate, the extractor (thread-pool LLM tool-loop), dedup
        + embedding resolution (``_resolve_write_plan``), and drives the
        ``agent_run`` rows to their terminal state (``_finalize_extraction_runs``
        — agent_run only, §4.3). Snapshots the billing inputs onto the returned
        plan so ``emit_generation_side_effects`` never depends on the reused
        instance's mutable ``_last_*``.

        Returns a resolved ``GenerationComputePlan`` for persist/emit, or
        ``None`` when the prepare gate closes (nothing to run). Raises
        ``ExtractorExecutionError`` on extractor failure — the caller
        (``_run_generation`` / the durable worker) records ``generation_failed``.
        """
        if not request:
            logger.error("Received None request for %s", self._get_service_name())
            return None

        generation_start = time.perf_counter()
        prepared = self._prepare_generation_run(request)
        if prepared is None:
            return None

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
        self._last_bookmark_advance = None
        self._last_model_provenance = None
        result = self._execute_extractor(prepared.extractor_config, prepared.identifier)
        generated_count = self._count_generated_results(result)

        try:
            write_plan = self._resolve_write_plan([result]) if result else None
            self._finalize_extraction_runs()
        except Exception as exc:
            self._mark_extraction_runs_finalization_failed(exc)
            raise

        return GenerationComputePlan(
            prepared=prepared,
            generated_count=generated_count,
            write_plan=write_plan,
            bookmark_advance=self._last_bookmark_advance,
            generation_start=generation_start,
            extraction_run_ids=list(self._last_extraction_run_ids),
            token_totals=self._last_token_totals,
        )

    def persist_generation(self, plan: GenerationComputePlan) -> None:
        """Persist half — apply the write-plan + the extractor bookmark advance.

        This is the ONLY part that runs inside the durable worker's fenced
        ``commit_scope`` (and inline on the synchronous ``.run()`` path). It
        issues only the resolved write-plan's row writes (embeddings already
        precomputed in compute for Tasks 6-7) and the extractor stride-bookmark
        advance — **no** events, **no** billing, **no** LLM / embedding compute.

        The bookmark advance is applied HERE so that BOTH the synchronous
        ``.run()`` path and the durable persist fence advance the bookmark
        atomically with the row writes it corresponds to (F1) — and it is
        applied in exactly one place, so it is never double-applied.
        """
        if plan.write_plan is not None:
            self._persist_write_plan(plan.write_plan)
        self._apply_bookmark_advance(plan.bookmark_advance)

    def emit_generation_side_effects(self, plan: GenerationComputePlan) -> None:
        """Post-commit side-effects — telemetry + ② Learning billing.

        Runs only for a fence-winning job (the durable worker calls it after the
        scope commits; ``.run()`` calls it inline). Reads the plan's compute-time
        snapshot (``generated_count`` / ``prepared`` / ``generation_start``) so a
        fence-lost job never emits.

        Billing purity note (round-2 finding): ``_record_billing_learning_events``
        also reads ``self._last_token_totals`` / ``self._last_precheck_sessions``,
        and the ``generation_succeeded`` metadata reads ``self._last_extractor_run_stats``.
        These are NOT re-plumbed to the plan — the money helper lives in a
        separate mixin behind a documented patch seam (``_usage_billing.py`` SINK-3)
        and re-plumbing it would touch the money path for zero behavior change.
        It is safe under the **single-use-instance invariant**: every generation
        service instance runs exactly one compute→persist→emit for one job with
        no interleaving (``.run()`` is synchronous; the durable orchestration in
        Task 8 holds one ``(instance, plan)`` pair per profile/playbook service
        and never re-runs compute on that instance before emit), so ``self._last_*``
        is exactly as compute left it when emit runs. The plan still snapshots the
        billing inputs so a future durable reuse can re-plumb locally.
        """
        self._record_generation_event(
            event_name="generation_succeeded",
            outcome="success",
            count_value=plan.generated_count,
            duration_ms=int((time.perf_counter() - plan.generation_start) * 1000),
            metadata={
                "identifier": plan.prepared.identifier,
                "extractor_name": plan.prepared.extractor_name,
                "extractor_failed": bool(self._last_extractor_run_stats.get("failed")),
                "extractor_timed_out": bool(
                    self._last_extractor_run_stats.get("timed_out")
                ),
            },
        )
        self._record_billing_learning_events(
            prepared=plan.prepared, generated_count=plan.generated_count
        )

    @abstractmethod
    def _resolve_write_plan(self, results: list) -> Any | None:
        """Compute-half of item finalization — resolve a write-plan, NO DB write.

        Runs the dedup + source/status assignment + embedding precompute and
        returns a resolved write-plan (or ``None`` when there is nothing to
        write). Issues NO learning DB write — the write is the persist half's
        job (``_persist_write_plan``), so this stays inside the compute purity
        contract (the compute-write-tripwire contract test).

        The durable-split ``ProfileGenerationService`` /
        ``PlaybookGenerationService`` return a real ``ProfileWritePlan`` /
        ``PlaybookWritePlan`` (Tasks 6-7). ``AgentSuccessEvaluationService`` was
        never split into a durable persist path — it keeps a concrete override
        that writes in compute via its permanent ``_finalize_extracted_items``
        wrapper (its results never flow through the durable worker fence).
        """

    @abstractmethod
    def _persist_write_plan(self, plan: Any) -> None:
        """Persist-half of item finalization — apply the resolved write-plan.

        Issues only the fence-critical row writes for a plan produced by
        ``_resolve_write_plan`` (embeddings already precomputed in compute), with
        NO LLM / embedding / dedup. This is the only item-finalization work that
        runs inside the durable worker's fenced ``commit_scope``.

        ``AgentSuccessEvaluationService`` (never split) implements this as a
        no-op — its ``_resolve_write_plan`` performs the write in compute and
        returns ``None`` so ``persist_generation`` never invokes this.
        """

    def _apply_bookmark_advance(self, advance: ExtractorBookmarkAdvance | None) -> None:
        """Apply the deferred extractor stride-bookmark advance (F1).

        The extractor emits its bookmark advance on the ``ExtractionOutcome``
        rather than self-advancing; ``_execute_extractor`` captures it and
        ``compute_generation`` snapshots it onto the ``GenerationComputePlan``.
        This applies it via ``update_extractor_bookmark`` using the same
        OperationStateManager service name the extractor used, so the bookmark
        key is byte-identical. No-op when the extractor produced no output
        (``advance`` is ``None``) or the service has no stride bookmark.

        Args:
            advance: The bookmark advance from ``plan.bookmark_advance``.
        """
        if advance is None or self.storage is None:
            return
        service_name = self._get_extractor_state_service_name()
        if service_name is None:
            return
        manager = OperationStateManager(self.storage, self.org_id, service_name)
        manager.update_extractor_bookmark(
            extractor_name=advance.extractor_name,
            processed_interactions=advance.processed_interactions,
            user_id=advance.user_id,
        )

    def _record_skip(
        self, reason: str, *, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record that generation was skipped BEFORE the pre-extraction gate ran.

        Emitted on the early-return paths of :meth:`_prepare_generation_run`. Those
        paths previously logged at ``info``/``warning`` and wrote nothing durable, so
        "a gate skipped this" was indistinguishable from "extraction ran and found
        nothing" and from "extraction was never invoked" — all three looked identical
        from the outside (publish returns success, no ``_agent_runs`` row, no usage
        event). Diagnosing one such case took several hours and four wrong theories.

        Uses the same ``generation_gate_evaluated`` event name as the gate itself so a
        single query returns every decision, with ``outcome`` naming which gate fired:
        ``skip_no_extractor_config`` / ``skip_source_not_enabled`` /
        ``skip_stride_not_met`` from here, and ``should_run`` / ``should_skip`` from
        the pre-extraction gate.

        Args:
            reason (str): Short slug for the gate that skipped, without the ``skip_``
                prefix (added here so every outcome sorts together).
            metadata (dict[str, Any] | None): Extra context for the event, e.g. the
                request source that failed the source filter.
        """
        self._record_generation_event(
            event_name="generation_gate_evaluated",
            outcome=f"skip_{reason}",
            count_value=1,
            metadata={
                "service": self._get_service_name(),
                **(metadata or {}),
            },
        )

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
            self._record_skip("no_extractor_config")
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
            self._record_skip("source_not_enabled", metadata={"source": source_display})
            return None

        extractor_config = self._filter_config_by_stride(extractor_config)
        if extractor_config is None:
            logger.info(
                "Extractor config did not pass stride_size check for %s",
                self._get_service_name(),
            )
            self._record_skip("stride_not_met")
            return None

        identifier = getattr(self.service_config, "user_id", None) or getattr(
            self.service_config, "request_id", "unknown"
        )
        extractor_name = get_extractor_name(extractor_config)

        should_run = self._should_run_before_extraction(extractor_config)
        # Exactly ONE event per gate decision. The gate reports a more specific
        # skip reason when it has one (e.g. the cheap pre-filter, which never
        # reaches the LLM); recording our generic ``should_skip`` alongside it
        # would double-count and imply an LLM verdict that never happened.
        # ``getattr`` because a subclass may override the gate and not set it.
        specific_outcome = getattr(self, "_last_gate_skip_outcome", None)
        self._record_generation_event(
            event_name="generation_gate_evaluated",
            outcome=(
                "should_run" if should_run else (specific_outcome or "should_skip")
            ),
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
