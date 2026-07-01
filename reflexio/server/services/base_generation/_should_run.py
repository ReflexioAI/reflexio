"""Should-run precheck helpers for ``BaseGenerationService`` (Tier-1b decomposition).

``ShouldRunPrecheckMixin`` holds the pre-extraction ``should_run`` gate and its
template hooks: ``_should_run_before_extraction`` (the gate), the overridable
``_build_should_run_prompt`` / ``_collect_scoped_interactions_for_precheck`` /
``_get_precheck_interaction_query_kwargs`` hooks, and ``_resolve_should_run_model``
(the enterprise string-path patch target — it stays MRO-resolvable on the base
because the base is ahead of this mixin in every subclass MRO and the module path
``base_generation_service`` + the ``BaseGenerationService`` class name are unchanged).

``_collect_scoped_interactions_for_precheck`` writes ``self._last_precheck_sessions``
— the Precheck→Billing (INV-3) seam that ``_extraction_input_text`` later reads. The
cheap pre-filter helper ``_cheap_should_run_reject`` remains a module function in
``base_generation_service`` (tests pin it there); the gate imports it function-locally
to avoid a circular import and to keep the ``base_generation_service`` string-path
patch effective. Method bodies are moved verbatim from the former monolithic
``base_generation_service.py``.
"""

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extractor_config_utils import get_extractor_name
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
)
from reflexio.server.services.service_utils import log_llm_messages, log_model_response
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)

TExtractorConfig = TypeVar("TExtractorConfig")
TGenerationServiceConfig = TypeVar("TGenerationServiceConfig")


class ShouldRunPrecheckMixin(Generic[TExtractorConfig, TGenerationServiceConfig]):  # noqa: UP046
    """Pre-extraction ``should_run`` gate + precheck template hooks.

    Mixed into ``BaseGenerationService`` ahead of ``ABC``; the per-run ``self``
    attributes these methods read (``service_config``, ``storage``, ``client``,
    ``request_context``, ``_last_precheck_sessions``) are initialised on the base
    ``__init__`` — the annotation-only stubs below give pyright the types without
    introducing shared class-level mutable state.
    """

    # Annotation-only stubs for base-owned attributes these helpers read (init'd in
    # the base ``__init__``). NEVER assign here — a class-level default would be a
    # shared-state footgun on a mixin.
    service_config: TGenerationServiceConfig | None
    storage: BaseStorage | None
    client: LiteLLMClient
    request_context: RequestContext
    # The Precheck→Billing (INV-3) seam: written by
    # ``_collect_scoped_interactions_for_precheck``, read by ``_extraction_input_text``.
    _last_precheck_sessions: list[Any] | None

    if TYPE_CHECKING:
        # Abstract on the base ABC (stays there per SINK-2); declared here type-only
        # so pyright can resolve the ``self._get_service_name()`` call. No runtime
        # attribute is added, so ``__abstractmethods__`` is unaffected.
        def _get_service_name(self) -> str: ...

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
        from reflexio.server.services.base_generation_service import (
            _cheap_should_run_reject,
        )

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
            # Stash for the billing path to reuse (same window the gate saw).
            self._last_precheck_sessions = []
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

        # Stash for the billing path (_extraction_input_text) to reuse instead of
        # re-querying storage for billing_input_tokens.
        self._last_precheck_sessions = session_data_models
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
