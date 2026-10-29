"""
Utility functions for per-extractor interaction data collection.

These functions are shared across ProfileExtractor, PlaybookExtractor, and AgentSuccessEvaluator
to handle per-extractor stride checking, source filtering, and sliding window iteration.
"""

import logging

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.config_schema import DEFAULT_STRIDE_SIZE, DEFAULT_WINDOW_SIZE
from reflexio.server.services.extractor_config_utils import get_extractor_name

logger = logging.getLogger(__name__)


def get_extractor_window_params(
    extractor_config: object,
    global_window_size: int | None,
    global_stride_size: int | None,
) -> tuple[int, int]:
    """
    Get effective window_size and stride_size for a specific extractor.

    Uses extractor's override values if set, otherwise falls back to global values,
    then to defaults (DEFAULT_WINDOW_SIZE / DEFAULT_STRIDE_SIZE from config_schema).

    Args:
        extractor_config: Extractor configuration object
        global_window_size: Global window_size from config
        global_stride_size: Global stride_size from config

    Returns:
        Tuple of (window_size, stride_size) for this extractor
    """
    window_size_override = getattr(extractor_config, "window_size_override", None)
    stride_size_override = getattr(extractor_config, "stride_size_override", None)

    window_size = (
        window_size_override
        if window_size_override is not None
        else (
            global_window_size
            if global_window_size is not None
            else DEFAULT_WINDOW_SIZE
        )
    )
    stride_size = (
        stride_size_override
        if stride_size_override is not None
        else (
            global_stride_size
            if global_stride_size is not None
            else DEFAULT_STRIDE_SIZE
        )
    )

    return window_size, stride_size


def get_effective_source_filter(
    extractor_config: object,
    triggering_source: str | None,
) -> tuple[bool, list[str] | None]:
    """
    Get effective source filter for an extractor.

    NOTE: By the time this function is called, the extractor has already been filtered
    by filter_extractor_configs() in base_generation_service.py. That filter ensures:
    - If sources_enabled is set and triggering_source is not in it, the extractor is skipped
    - So if we reach here with sources_enabled set, triggering_source MUST be in the list
      (unless triggering_source is None, which happens in rerun flows)

    This function still includes a safety check for edge cases.

    Args:
        extractor_config: Extractor configuration with request_sources_enabled field
        triggering_source: The source from the triggering request (None for rerun flows)

    Returns:
        Tuple of (should_skip, source_filter):
        - (False, None): If request_sources_enabled = None or empty list → get ALL sources, no filtering
        - (False, [triggering_source]): If sources_enabled is set AND triggering_source in list
        - (False, sources_enabled): If sources_enabled is set AND triggering_source is None → return all enabled sources
        - (True, None): Safety skip if triggering_source not in sources_enabled (shouldn't happen)
    """
    sources_enabled = getattr(extractor_config, "request_sources_enabled", None)

    # If no sources_enabled configured or empty list, extractor wants ALL sources
    if not sources_enabled:
        return (False, None)  # No source filtering - get all

    # If triggering_source is None (rerun without specific source),
    # return all enabled sources so caller can filter by them
    if triggering_source is None:
        return (False, sources_enabled)  # Return list of enabled sources

    # Safety check: triggering_source should be in sources_enabled
    # (filter_extractor_configs should have already filtered this out)
    if triggering_source not in sources_enabled:
        logger.warning(
            "Skipping extractor '%s' - triggering_source '%s' not in sources_enabled %s "
            "(this should have been filtered earlier)",
            get_extractor_name(extractor_config),
            triggering_source,
            sources_enabled,
        )
        return (True, None)  # Skip this extractor

    # triggering_source is in sources_enabled, use it for filtering
    return (False, [triggering_source])


def should_extractor_run_by_stride(
    new_interaction_count: int,
    stride_size: int | None,
) -> bool:
    """
    Determine if an extractor should run based on its stride_size configuration.

    Args:
        new_interaction_count: Number of new interactions since last run
        stride_size: Configured stride_size for this extractor

    Returns:
        True if extractor should run, False otherwise
    """
    if new_interaction_count <= 0:
        return False

    if stride_size is None or stride_size <= 0:
        return True  # No stride configured, always run

    return new_interaction_count >= stride_size


def filter_interactions_by_source(
    request_interaction_data_models: list[RequestInteractionDataModel],
    source_filter: str | list[str] | None,
) -> list[RequestInteractionDataModel]:
    """
    Filter request interaction data models by source.

    Args:
        request_interaction_data_models: Data models to filter
        source_filter:
            - None: return all (no filtering)
            - str: filter to only this source
            - list[str]: filter to any of these sources

    Returns:
        Filtered list of RequestInteractionDataModel
    """
    if source_filter is None:
        return request_interaction_data_models

    if isinstance(source_filter, str):
        allowed_sources = {source_filter}
    else:
        allowed_sources = set(source_filter)

    return [
        rim
        for rim in request_interaction_data_models
        if rim.request.source in allowed_sources
    ]
