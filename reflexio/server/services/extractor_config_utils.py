"""
Utility functions for filtering extractor configurations.
"""

import logging

logger = logging.getLogger(__name__)


def get_extractor_name[TExtractorConfig](config: TExtractorConfig) -> str:
    """
    Get the display name for an extractor config.

    Checks extractor_name, playbook_name, and evaluation_name attributes in order.

    Args:
        config: Extractor configuration object (e.g., ProfileExtractorConfig, PlaybookConfig, AgentSuccessConfig)

    Returns:
        str: The extractor name, or "unknown" if none found
    """
    return getattr(
        config,
        "extractor_name",
        getattr(config, "playbook_name", getattr(config, "evaluation_name", "unknown")),
    )


def filter_extractor_configs[TExtractorConfig](
    extractor_configs: list[TExtractorConfig],
    source: str | None = None,
    allow_manual_trigger: bool = False,
    extractor_names: list[str] | None = None,
) -> list[TExtractorConfig]:
    """
    Filter extractor configs based on source, manual_trigger, and extractor names.

    This is a standalone utility function that can be used by both BaseGenerationService
    and GenerationService to filter extractor configurations consistently.

    Args:
        extractor_configs: List of extractor configuration objects (e.g., ProfileExtractorConfig,
            PlaybookConfig, AgentSuccessConfig)
        source: Request source for filtering by request_sources_enabled. If None, source
            filtering is skipped.
        allow_manual_trigger: Whether to allow extractors with manual_trigger=True.
            If False, extractors with manual_trigger=True will be skipped.
        extractor_names: Optional list of extractor names to filter by. If provided,
            only extractors with names in this list will be included.

    Returns:
        Filtered list of extractor configs that should run for the given parameters
    """
    filtered_configs = []

    for config in extractor_configs:
        # Check if config has request_sources_enabled attribute
        if hasattr(config, "request_sources_enabled"):
            sources_enabled = config.request_sources_enabled  # type: ignore[reportAttributeAccessIssue]
            # Skip if source filtering applies and source is not in enabled list
            if sources_enabled and source and source not in sources_enabled:
                logger.debug(
                    "Skipping extractor '%s' - source '%s' not in enabled sources %s",
                    get_extractor_name(config),
                    source,
                    sources_enabled,
                )
                continue

        # Check manual_trigger: skip if manual_trigger=True and allow_manual_trigger=False
        manual_trigger = getattr(config, "manual_trigger", False)
        if manual_trigger and not allow_manual_trigger:
            logger.debug(
                "Skipping extractor '%s' - manual_trigger=True and allow_manual_trigger=False",
                get_extractor_name(config),
            )
            continue

        # Filter by extractor_names if specified
        if extractor_names:
            name = get_extractor_name(config)
            if name and name not in extractor_names:
                logger.debug(
                    "Skipping extractor '%s' - not in specified extractor_names %s",
                    name,
                    extractor_names,
                )
                continue

        filtered_configs.append(config)

    return filtered_configs
