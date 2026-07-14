"""
Shared utilities for deduplication services.

This module contains base classes and utility functions used by both
ProfileConsolidator and PlaybookConsolidator.
"""

import logging
from abc import ABC
from datetime import UTC, datetime
from typing import Any, cast

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.embedding_text import embedding_input
from reflexio.server.site_var.site_var_manager import SiteVarManager

logger = logging.getLogger(__name__)


def resolve_dedup_query_embeddings(
    storage: Any,
    client: LiteLLMClient,
    query_texts: list[str],
    *,
    entity_label: str,
) -> list[list[float] | None]:
    """Embed dedup-search queries with the model that indexed the store.

    Prefers the storage's own embedding path (correct model + "query" prefix);
    falls back to ``client`` pinned to the storage's model/dimensions. Letting
    the client resolve its default embedding model would be wrong here: it
    picks the OSS default, which the enterprise embedding daemon rejects
    (409 model conflict) and which would not match the indexed vectors anyway.

    Args:
        storage: Storage backend (duck-typed: ``_get_embedding`` preferred,
            else ``embedding_model_name`` / ``embedding_dimensions``).
        client: Shared LLM client, used only in the fallback path.
        query_texts: Query strings to embed.
        entity_label: Log prefix identifying the caller, e.g. "Profile".

    Returns:
        One embedding (or None) per query text, in input order. On any
        failure, all entries are None so the caller degrades to text-only
        search. A backend that signals embedding-service unavailability with
        an empty vector (e.g. the Supabase query path) is normalized to None
        so search falls back to its own embedding/FTS path instead of sending
        an empty vector to the database.
    """
    try:
        get_storage_embedding = getattr(storage, "_get_embedding", None)
        if callable(get_storage_embedding):
            logger.info(
                "%s dedup query embeddings: source=storage model=%s",
                entity_label,
                getattr(storage, "embedding_model_name", "unknown"),
            )
            embeddings = [
                cast(
                    "list[float] | None",
                    get_storage_embedding(query_text, purpose="query"),
                )
                for query_text in query_texts
            ]
        else:
            embedding_model_name = storage.embedding_model_name
            embedding_dimensions = storage.embedding_dimensions
            logger.info(
                "%s dedup query embeddings: source=llm_client model=%s",
                entity_label,
                embedding_model_name,
            )
            embeddings = list(
                client.get_embeddings(
                    [
                        embedding_input(query_text, purpose="query")
                        for query_text in query_texts
                    ],
                    model=embedding_model_name,
                    dimensions=embedding_dimensions,
                )
            )
    except Exception as e:
        logger.warning("Failed to generate embeddings for dedup search: %s", e)
        return [None] * len(query_texts)
    return [emb or None for emb in embeddings]


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

    Weak models sometimes echo the rendered display label with its brackets
    (``[NEW-0]``); strip a single surrounding pair so those outputs parse
    instead of being silently dropped.

    Args:
        item_id (str): Item ID string in the format 'PREFIX-N' (e.g., 'NEW-0', 'EXISTING-1')

    Returns:
        Optional[tuple[str, int]]: A tuple of (prefix, index) where prefix is 'NEW' or 'EXISTING',
            or None if the item ID is invalid
    """
    stripped = item_id.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    parts = stripped.rsplit("-", 1)
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
