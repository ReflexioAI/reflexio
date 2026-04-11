"""
Utility functions for per-extractor interaction data collection.

These functions are shared across ProfileExtractor, PlaybookExtractor, and AgentSuccessEvaluator
to handle per-extractor batch_interval checking, source filtering, and sliding window iteration.
"""

import logging

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.config_schema import DEFAULT_BATCH_INTERVAL, DEFAULT_BATCH_SIZE
from reflexio.server.services.extractor_config_utils import get_extractor_name

logger = logging.getLogger(__name__)


def get_extractor_window_params[TExtractorConfig](
    extractor_config: TExtractorConfig,
    global_batch_size: int | None,
    global_batch_interval: int | None,
) -> tuple[int, int]:
    """
    Get effective batch_size and batch_interval for a specific extractor.

    Uses extractor's override values if set, otherwise falls back to global values,
    then to defaults (batch_size=10, batch_interval=5).

    Args:
        extractor_config: Extractor configuration object
        global_batch_size: Global batch_size from config
        global_batch_interval: Global batch_interval from config

    Returns:
        Tuple of (batch_size, batch_interval_size) for this extractor
    """
    batch_size_override = getattr(extractor_config, "batch_size_override", None)
    batch_interval_override = getattr(extractor_config, "batch_interval_override", None)

    batch_size = (
        batch_size_override
        if batch_size_override is not None
        else (
            global_batch_size if global_batch_size is not None else DEFAULT_BATCH_SIZE
        )
    )
    batch_interval_size = (
        batch_interval_override
        if batch_interval_override is not None
        else (
            global_batch_interval
            if global_batch_interval is not None
            else DEFAULT_BATCH_INTERVAL
        )
    )

    return batch_size, batch_interval_size


def get_effective_source_filter[TExtractorConfig](
    extractor_config: TExtractorConfig,
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


def should_extractor_run_by_batch_interval(
    new_interaction_count: int,
    batch_interval_size: int | None,
) -> bool:
    """
    Determine if an extractor should run based on its batch_interval configuration.

    Args:
        new_interaction_count: Number of new interactions since last run
        batch_interval_size: Configured batch_interval size for this extractor

    Returns:
        True if extractor should run, False otherwise
    """
    if new_interaction_count <= 0:
        return False

    if batch_interval_size is None or batch_interval_size <= 0:
        return True  # No batch_interval configured, always run

    return new_interaction_count >= batch_interval_size


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


from collections.abc import Iterator


def iter_sliding_windows(
    request_interaction_data_models: list[RequestInteractionDataModel],
    batch_size: int,
    batch_interval_size: int | None,
) -> Iterator[tuple[int, list[RequestInteractionDataModel]]]:
    """
    Yield sliding windows of RequestInteractionDataModel based on interaction count.

    Windows are created based on total interaction count, not number of data models.
    Each window contains complete RequestInteractionDataModel objects (never splits a model).

    Args:
        request_interaction_data_models: List of RequestInteractionDataModel to window over
        batch_size: Target number of interactions per window
        batch_interval_size: Step size in interactions between window starts (must be >= 1)

    Yields:
        Tuples of (window_index, list_of_request_interaction_data_models)

    Example:
        With 3 models having [10, 20, 15] interactions, batch_size=25, batch_interval_size=15:
        - Window 0: models[0], models[1] (30 interactions, starts at 0)
        - Window 1: models[1], models[2] (35 interactions, starts at 15)
    """
    if not request_interaction_data_models:
        return

    if batch_size <= 0:
        # Invalid batch_size, yield single window with all data
        yield (0, request_interaction_data_models)
        return

    # Default batch_interval to batch_size if not valid
    effective_batch_interval = (
        batch_interval_size
        if batch_interval_size and batch_interval_size > 0
        else batch_size
    )

    # Build cumulative interaction counts for each model
    # cumulative[i] = total interactions from models[0] to models[i-1] (exclusive)
    cumulative: list[int] = [0]
    for rim in request_interaction_data_models:
        cumulative.append(cumulative[-1] + len(rim.interactions))

    total_interactions = cumulative[-1]

    if total_interactions == 0:
        return

    # If all data fits in one window, yield single window
    if total_interactions <= batch_size:
        yield (0, request_interaction_data_models)
        return

    window_idx = 0
    window_start = 0  # Starting interaction index

    while window_start < total_interactions:
        window_end = window_start + batch_size

        # Find models that overlap with [window_start, window_end)
        # A model overlaps if its start < window_end AND its end > window_start
        selected_models: list[RequestInteractionDataModel] = []

        for i, rim in enumerate(request_interaction_data_models):
            model_start = cumulative[i]
            model_end = cumulative[i + 1]

            # Model overlaps with window if: model_start < window_end AND model_end > window_start
            if model_start < window_end and model_end > window_start:
                selected_models.append(rim)

        if selected_models:
            yield (window_idx, selected_models)

        window_idx += 1
        window_start += effective_batch_interval

        # Prevent infinite loop if batch_interval is 0 (shouldn't happen with validation above)
        if effective_batch_interval <= 0:
            break
