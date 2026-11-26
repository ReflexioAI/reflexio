"""
Utility functions for filtering extractor configurations.
"""

import logging

from reflexio.models.config_schema import (
    SINGLETON_AGENT_SUCCESS_EVALUATION_NAME,
    SINGLETON_PROFILE_EXTRACTOR_NAME,
    SINGLETON_USER_PLAYBOOK_NAME,
    AgentSuccessConfig,
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
)

logger = logging.getLogger(__name__)


def get_extractor_name(config: object) -> str:
    """
    Get the display name for an extractor config.

    Typed extractor configs use fixed singleton identities. For other objects,
    checks extractor_name, playbook_name, and evaluation_name attributes in order.

    Args:
        config: Extractor configuration object (e.g., ProfileExtractorConfig, PlaybookConfig, AgentSuccessConfig)

    Returns:
        str: The extractor name, or "unknown" if none found
    """
    if isinstance(config, ProfileExtractorConfig):
        return SINGLETON_PROFILE_EXTRACTOR_NAME
    if isinstance(config, UserPlaybookExtractorConfig):
        return SINGLETON_USER_PLAYBOOK_NAME
    if isinstance(config, AgentSuccessConfig):
        return SINGLETON_AGENT_SUCCESS_EVALUATION_NAME
    return (
        getattr(config, "extractor_name", None)
        or getattr(config, "playbook_name", None)
        or getattr(config, "evaluation_name", "unknown")
        or "unknown"
    )


def filter_extractor_configs[TExtractorConfig](
    extractor_configs: list[TExtractorConfig],
    source: str | None = None,
) -> list[TExtractorConfig]:
    """
    Filter extractor configs by their configured request sources.

    This is a standalone utility function that can be used by both BaseGenerationService
    and GenerationService to filter extractor configurations consistently.

    Args:
        extractor_configs: List of extractor configuration objects (e.g., ProfileExtractorConfig,
            PlaybookConfig, AgentSuccessConfig)
        source: Request source for filtering by request_sources_enabled. If None, source
            filtering is skipped.

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

        filtered_configs.append(config)

    return filtered_configs
