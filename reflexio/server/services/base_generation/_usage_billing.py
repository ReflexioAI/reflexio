"""Usage-metric + ② Learning billing helpers for ``BaseGenerationService`` (Tier-1b decomposition).

``UsageBillingMixin`` holds the money-critical Bucket A: the usage-event helpers
(``_usage_pipeline``, ``_usage_context``, ``_record_generation_event``) and the
② Learning billing path (``_extraction_input_text``, ``_record_billing_learning_events``,
plus the pure ``_count_generated_results`` staticmethod).

These helpers only READ the per-run accumulators (``_last_token_totals``,
``_last_precheck_sessions``, ``service_config``, ``EMITS_LEARNING_BILLING``) — they
never write them, so moving them introduces no ordering/race change. The billing
emit-ordering (``record_learnings_generated`` BEFORE ``record_extraction_tokens``,
success-path only) lives in ``_run_generation`` on the base and is unchanged.

SINK-3 (money-critical): the ``billing_meter`` / ``billing_signals`` imports inside
``_record_billing_learning_events`` stay FUNCTION-LOCAL — the Phase-A drain test
patches the terminal emitters at their source module (``reflexio.server.billing_meter``),
and hoisting the imports to module top would both break that patch seam and risk a
circular import that the ``:392`` exception-swallow would hide silently. Both
exception swallows (``_record_billing_learning_events`` and ``_extraction_input_text``)
are moved VERBATIM. Method bodies are otherwise moved unchanged from the former
monolithic ``base_generation_service.py``.
"""

import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
)
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.usage_metrics import record_usage_event

if TYPE_CHECKING:
    from reflexio.server.services.base_generation_service import PreparedGenerationRun

logger = logging.getLogger(__name__)

TExtractorConfig = TypeVar("TExtractorConfig")
TGenerationServiceConfig = TypeVar("TGenerationServiceConfig")


class UsageBillingMixin(Generic[TExtractorConfig, TGenerationServiceConfig]):  # noqa: UP046
    """Usage-event emission + the ② Learning billing path.

    Mixed into ``BaseGenerationService`` ahead of the other mixins and ``ABC``; the
    per-run ``self`` attributes these methods read (``service_config``, ``storage``,
    ``org_id``, ``request_context`` and the billing accumulators
    ``_last_token_totals`` / ``_last_precheck_sessions``) are initialised on the base
    ``__init__`` — the annotation-only stubs below give pyright the types without
    introducing shared class-level mutable state.
    """

    # Annotation-only stubs for base-owned attributes these helpers read (init'd in
    # the base ``__init__``). NEVER assign here — a class-level default would be a
    # shared-state footgun on a mixin.
    service_config: TGenerationServiceConfig | None
    storage: BaseStorage | None
    org_id: str
    request_context: RequestContext
    # Read-only in this bucket: written by the extraction lifecycle / precheck mixins.
    _last_token_totals: RunTokenTotals | None
    _last_precheck_sessions: list[Any] | None
    # Class-level opt-in flag defined on the base; annotation-only here so pyright can
    # resolve ``self.EMITS_LEARNING_BILLING`` without shadowing the base default.
    EMITS_LEARNING_BILLING: bool

    if TYPE_CHECKING:
        # Abstract on the base ABC (stays there per SINK-2); declared here type-only so
        # pyright can resolve the ``self._get_service_name()`` call. No runtime
        # attribute is added, so ``__abstractmethods__`` is unaffected.
        def _get_service_name(self) -> str: ...
        # Concrete on ``ShouldRunPrecheckMixin`` (resolved via MRO); declared type-only
        # so pyright can resolve the ``self._get_precheck_interaction_query_kwargs()`` call.
        def _get_precheck_interaction_query_kwargs(self) -> dict: ...

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

    def _extraction_input_text(self, prepared: "PreparedGenerationRun[Any]") -> str:
        """Return the interaction content fed to the extractor as a single string.

        Replays the same storage query the extractor uses (same user_id, source,
        window params) so ``count_input_tokens`` sees the same text the LLM saw.
        Returns an empty string when storage is unavailable or no interactions exist.

        Args:
            prepared: The prepared generation run, used to access extractor_config
                for window/source parameters.

        Returns:
            str: Formatted interaction history string, or "" if nothing is found.
        """
        from reflexio.server.services.service_utils import (
            format_sessions_to_history_string,
        )

        if self.storage is None or self.service_config is None:
            return ""
        # Reuse the window the should-run gate already fetched
        # (_collect_scoped_interactions_for_precheck stashed it). Same query params,
        # so the token count is byte-identical — and we avoid a second storage read.
        if self._last_precheck_sessions is not None:
            return format_sessions_to_history_string(self._last_precheck_sessions)
        # Fallback storage read: used ONLY when the gate did not pre-fetch
        # (bypass paths: auto_run=False, force_extraction, skip_should_run_check,
        # mock mode). Re-fetches the extractor's input window so billing token
        # counting sees exactly the text the LLM saw.
        try:
            root_config = self.request_context.configurator.get_config()
            global_window_size = (
                getattr(root_config, "window_size", None) if root_config else None
            )
            global_stride_size = (
                getattr(root_config, "stride_size", None) if root_config else None
            )
            window_size, _ = get_extractor_window_params(
                prepared.extractor_config, global_window_size, global_stride_size
            )
            should_skip, effective_source = get_effective_source_filter(
                prepared.extractor_config, getattr(self.service_config, "source", None)
            )
            if should_skip:
                return ""
            session_data_models, _ = self.storage.get_last_k_interactions_grouped(
                user_id=getattr(self.service_config, "user_id", None),
                k=window_size,
                sources=effective_source,
                start_time=getattr(self.service_config, "rerun_start_time", None),
                end_time=getattr(self.service_config, "rerun_end_time", None),
                **self._get_precheck_interaction_query_kwargs(),
            )
            return format_sessions_to_history_string(session_data_models)
        except Exception:
            logger.warning(
                "_extraction_input_text failed; billing_input_tokens will be 0",
                exc_info=True,
            )
            return ""

    def _record_billing_learning_events(
        self, *, prepared: "PreparedGenerationRun[Any]", generated_count: int
    ) -> None:
        """Emit the ② Learning billing events. Called ONLY when extraction fired.

        Emits ``learnings_generated`` (value facet) and ``extraction_tokens``
        (cost facet) via the OSS emission helpers. ``platform_storage`` is left
        ``None`` here and resolved enterprise-side at rollup (Phase 1).

        Gated by ``EMITS_LEARNING_BILLING`` — only online profile/playbook
        extraction services opt in here. Resumable-extraction finalization emits
        the same value facet separately. Derived mutation paths emit no additional
        ``learnings_generated`` events.

        Args:
            prepared: The prepared generation run (used for input-text computation).
            generated_count: Number of learnings produced by this extraction run.
        """
        if not self.EMITS_LEARNING_BILLING:
            return

        try:
            from reflexio.server.billing_meter import (
                record_extraction_tokens,
                record_learnings_generated,
            )
            from reflexio.server.billing_signals import (
                count_input_tokens,
                platform_llm_from_config,
            )

            config = self.request_context.configurator.get_config()
            platform_llm = platform_llm_from_config(config)
            ctx = self._usage_context()

            # session_id is intentionally not passed: the generation path has no
            # session_id source. _usage_context() never includes it, and neither
            # the Profile/Playbook service configs nor their requests carry a
            # session_id (unlike the Application-line path in server/api.py, which
            # reads request_id/session_id off the search request payload). Learning
            # events therefore meter without session attribution by design.

            # ② Learning — value: learnings generated (helper no-ops on count <= 0).
            record_learnings_generated(
                org_id=ctx["org_id"],
                count=generated_count,
                platform_llm=platform_llm,
                platform_storage=None,
                pipeline=ctx.get("pipeline"),
                user_id=ctx.get("user_id"),
                request_id=ctx.get("request_id"),
                source=ctx.get("source"),
                agent_version=ctx.get("agent_version"),
            )

            # ② Learning — cost: input-anchored extraction tokens + real provider tokens.
            totals = self._last_token_totals or RunTokenTotals()
            billing_input_tokens = count_input_tokens(
                self._extraction_input_text(prepared)
            )
            record_extraction_tokens(
                org_id=ctx["org_id"],
                billing_input_tokens=billing_input_tokens,
                prompt_tokens=totals.prompt_tokens,
                completion_tokens=totals.completion_tokens,
                platform_llm=platform_llm,
                platform_storage=None,
                pipeline=ctx.get("pipeline"),
                request_id=ctx.get("request_id"),
            )
        except Exception:
            logger.warning(
                "_record_billing_learning_events failed; billing learning events not emitted",
                exc_info=True,
            )

    @staticmethod
    def _count_generated_results(result: Any) -> int:
        if isinstance(result, list):
            return len(result)
        return 1 if result else 0
