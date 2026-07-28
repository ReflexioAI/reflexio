"""Extraction-run lifecycle helpers for ``BaseGenerationService`` (Tier-1b decomposition).

``ExtractionRunLifecycleMixin`` holds the four concrete methods that run the
configured extractor and drive the agent-run rows through their terminal states:
``_execute_extractor`` (thread-pool run + timeout guard), ``_fail_active_extraction_runs``
(timeout cleanup), ``_finalize_extraction_runs`` (success finalization), and
``_mark_extraction_runs_finalization_failed`` (failure/backoff).

``_execute_extractor`` writes the per-run billing accumulators — ``_last_extractor_run_stats``,
``_last_extraction_run_ids`` (appended on an ``ExtractionOutcome`` run id), and
``_last_token_totals`` (set ONLY under ``isinstance(result, ExtractionOutcome)``). Those
writes and their guards are moved VERBATIM: the ``isinstance`` write-guard is the double-bill
idempotency backstop (a non-``ExtractionOutcome`` result must NOT overwrite the accumulator),
and ``_finalize_extraction_runs`` / ``_mark_extraction_runs_finalization_failed`` are invoked
from ``_run_generation`` (still on the base) via ``self.`` MRO — the atomicity invariant
(``_mark_..._failed`` on a ``_process_results`` failure, not finalize) depends on both bodies
being unchanged.

``ExtractorExecutionError`` and ``EXTRACTOR_TIMEOUT_SECONDS`` live in
``base_generation_service`` and are imported function-locally to avoid a circular import
(the base module imports this package at load time). Method bodies are otherwise moved
verbatim from the former monolithic ``base_generation_service.py``.
"""

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.services.deferred_learning_plan import ExtractorBookmarkAdvance
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.extractor_config_utils import get_extractor_name
from reflexio.server.services.storage.storage_base import AgentRunStatus, BaseStorage

logger = logging.getLogger(__name__)

TExtractorConfig = TypeVar("TExtractorConfig")
TGenerationServiceConfig = TypeVar("TGenerationServiceConfig")


class ExtractionRunLifecycleMixin(Generic[TExtractorConfig, TGenerationServiceConfig]):  # noqa: UP046
    """Extractor execution + agent-run terminal-state transitions.

    Mixed into ``BaseGenerationService`` ahead of the other mixins and ``ABC``; the
    per-run ``self`` attributes these methods read/write (``service_config``,
    ``storage``, ``request_context`` and the billing accumulators
    ``_last_extractor_run_stats`` / ``_last_extraction_run_ids`` / ``_last_token_totals``)
    are initialised on the base ``__init__`` — the annotation-only stubs below give
    pyright the types without introducing shared class-level mutable state.
    """

    # Annotation-only stubs for base-owned attributes these helpers read/write (init'd
    # in the base ``__init__``). NEVER assign here — a class-level default would be a
    # shared-state footgun on a mixin.
    service_config: TGenerationServiceConfig | None
    storage: BaseStorage | None
    request_context: RequestContext
    _last_extractor_run_stats: dict[str, int]
    _last_extraction_run_ids: list[str]
    _last_token_totals: RunTokenTotals | None
    _last_bookmark_advance: ExtractorBookmarkAdvance | None
    _last_model_provenance: ModelProvenance | None

    if TYPE_CHECKING:
        # Abstract on the base ABC (stays there per SINK-2); declared here type-only so
        # pyright can resolve the ``self._create_extractor()`` / ``self._get_service_name()``
        # calls. No runtime attribute is added, so ``__abstractmethods__`` is unaffected.
        def _create_extractor(
            self,
            extractor_config: TExtractorConfig,
            service_config: TGenerationServiceConfig,
        ) -> Any: ...
        def _get_service_name(self) -> str: ...

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
        from reflexio.server.services.base_generation_service import (
            EXTRACTOR_TIMEOUT_SECONDS,
            ExtractorExecutionError,
        )

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
                self._last_token_totals = result.token_totals
                # Capture the deferred stride-bookmark advance (F1); applied
                # later in ``persist_generation`` (durable fence) or in
                # ``_run_generation``'s persist half for the synchronous path.
                self._last_bookmark_advance = result.bookmark_advance
                self._last_model_provenance = result.model_provenance
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
            self._fail_active_extraction_runs(
                extractor_kind=get_extractor_name(extractor_config),
                last_error=error_msg,
            )
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

    def _fail_active_extraction_runs(
        self,
        *,
        extractor_kind: str,
        last_error: str,
    ) -> None:
        """Mark active agent-run rows failed when the service timeout fires."""
        if self.storage is None or self.service_config is None:
            return
        generation_request_id = getattr(self.service_config, "request_id", None)
        if not generation_request_id:
            return
        user_id = getattr(self.service_config, "user_id", None)
        try:
            failed_count = self.storage.fail_running_agent_runs_for_request(
                org_id=self.request_context.org_id,
                extractor_kind=extractor_kind,
                user_id=user_id,
                request_id=generation_request_id,
                last_error=last_error,
            )
        except NotImplementedError:
            return
        except Exception as exc:  # noqa: BLE001 - keep timeout error primary
            logger.warning(
                "Failed to mark timed-out %s agent runs failed: %s",
                extractor_kind,
                exc,
            )
            return
        if failed_count:
            logger.warning(
                "Marked %d timed-out %s agent run(s) failed for generation_request_id=%s",
                failed_count,
                extractor_kind,
                generation_request_id,
            )

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
