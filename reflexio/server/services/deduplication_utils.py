"""
Shared utilities for deduplication services.

This module contains base classes and utility functions used by both
ProfileDeduplicator and PlaybookDeduplicator.
"""

import logging
from abc import ABC
from datetime import UTC, datetime

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.site_var.site_var_manager import SiteVarManager

logger = logging.getLogger(__name__)

# Format used for "Last Modified" timestamps shown to deduplication LLMs.
# Includes hours and minutes so same-day contradictions (morning vs evening)
# can be distinguished.
DEDUP_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M UTC"
DEDUP_TIMESTAMP_FALLBACK = "unknown"


def format_dedup_timestamp(ts: int) -> str:
    """Format a Unix timestamp as a UTC date string for deduplication prompts.

    Wraps ``datetime.fromtimestamp`` in a try/except so a single malformed
    timestamp (negative, zero, or out-of-range integer) cannot abort an entire
    deduplication batch. Returns ``DEDUP_TIMESTAMP_FALLBACK`` on failure.

    Args:
        ts (int): Unix timestamp (seconds since epoch).

    Returns:
        str: Human-readable UTC timestamp like ``"2026-04-11 14:30 UTC"``,
            or ``"unknown"`` if the value is unparseable.
    """
    try:
        return datetime.fromtimestamp(ts, tz=UTC).strftime(DEDUP_TIMESTAMP_FORMAT)
    except (OverflowError, ValueError, OSError, TypeError) as exc:
        logger.warning("Failed to format dedup timestamp %r: %s", ts, exc)
        return DEDUP_TIMESTAMP_FALLBACK


def parse_item_id(item_id: str) -> tuple[str, int] | None:
    """
    Parse a prompt-format item ID like 'NEW-0' or 'EXISTING-1' into its prefix and index.

    Args:
        item_id (str): Item ID string in the format 'PREFIX-N' (e.g., 'NEW-0', 'EXISTING-1')

    Returns:
        Optional[tuple[str, int]]: A tuple of (prefix, index) where prefix is 'NEW' or 'EXISTING',
            or None if the item ID is invalid
    """
    parts = item_id.rsplit("-", 1)
    if len(parts) != 2:
        logger.warning("Invalid item ID format: %s", item_id)
        return None
    prefix, idx_str = parts
    prefix = prefix.upper()
    if prefix not in ("NEW", "EXISTING"):
        logger.warning("Invalid prefix in item ID: %s", item_id)
        return None
    try:
        return prefix, int(idx_str)
    except ValueError:
        logger.warning("Invalid index in item ID: %s", item_id)
        return None


# ===============================
# Base Deduplicator ABC
# ===============================


class BaseDeduplicator(ABC):  # noqa: B024
    """
    Abstract base class for deduplicators that use LLM-based semantic matching.

    Provides shared initialization (LLM client, model name).
    Subclasses implement their own deduplicate() method with domain-specific
    prompt building, hybrid search, and result construction.
    """

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
    ):
        """
        Initialize the deduplicator.

        Args:
            request_context: Request context with storage and prompt manager
            llm_client: Unified LLM client for LLM calls
        """
        self.request_context = request_context
        self.client = llm_client

        # Resolve model name: site var → auto-detect
        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}
        api_key_config = self.request_context.configurator.get_config().api_key_config

        self.model_name = resolve_model_name(
            ModelRole.GENERATION,
            site_var_value=site_var.get("default_generation_model_name"),
            api_key_config=api_key_config,
        )
