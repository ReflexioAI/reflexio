"""Config-filtering helpers for ``BaseGenerationService`` (Tier-1b decomposition).

``ConfigFilterMixin`` holds the concrete, non-abstract config-filter helpers:
``_filter_extractor_config_by_service_config`` (source/manual-trigger filtering),
``_get_extractor_state_service_name`` (stride bookmark service name, overridable),
and ``_filter_config_by_stride`` (pre-LLM stride_size gate). The abstract config
LOADERS (``_load_extractor_config``, ``_load_generation_service_config``,
``_create_extractor``) stay physically on the base ABC — an ``@abstractmethod`` on a
plain mixin is dropped from ``__abstractmethods__``, so this mixin carries only the
concrete helpers. Method bodies are moved verbatim from the former monolithic
``base_generation_service.py``.
"""

import logging
from typing import Generic, TypeVar

from reflexio.server.api_endpoints.request_context import RequestContext
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
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)

TExtractorConfig = TypeVar("TExtractorConfig")
TGenerationServiceConfig = TypeVar("TGenerationServiceConfig")


class ConfigFilterMixin(Generic[TExtractorConfig, TGenerationServiceConfig]):  # noqa: UP046
    """Concrete config-filter helpers mixed into ``BaseGenerationService``.

    Mixed into the base ahead of ``ABC``; the per-run ``self`` attributes these
    helpers read (``service_config``, ``storage``, ``org_id``, ``request_context``)
    are initialised on the base ``__init__`` — the annotation-only stubs below give
    pyright the types without introducing shared class-level mutable state.
    """

    # Annotation-only stubs for base-owned attributes these helpers read (init'd in
    # the base ``__init__``). NEVER assign here — a class-level default would be a
    # shared-state footgun on a mixin.
    service_config: TGenerationServiceConfig | None
    storage: BaseStorage | None
    org_id: str
    request_context: RequestContext

    def _filter_extractor_config_by_service_config(
        self,
        extractor_config: TExtractorConfig,
        service_config: TGenerationServiceConfig,
    ) -> TExtractorConfig | None:
        """
        Filter the extractor config based on request_sources_enabled and manual_trigger.
        """
        filtered = filter_extractor_configs(
            extractor_configs=[extractor_config],
            source=getattr(service_config, "source", None),
            allow_manual_trigger=getattr(service_config, "allow_manual_trigger", False),
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
