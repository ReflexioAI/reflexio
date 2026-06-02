"""Read-path relevance floor: drop retrieved items below a cross-encoder score.

Scores ``(query, item.content)`` pairs with the local cross-encoder (raw logits),
drops anything below ``floor``, returns survivors sorted by score desc, capped at
``top_k``. On reranker unavailability, degrades to the retrieved order (logged) —
never crashes, never silently returns the full unfiltered list.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from reflexio.server.llm.rerank import score_pairs
from reflexio.server.llm.rerank.cross_encoder_reranker import (
    CrossEncoderUnavailableError,
)

logger = logging.getLogger(__name__)


def apply_relevance_floor[T](
    query: str,
    items: list[T],
    floor: float,
    top_k: int,
    *,
    arm: str,
    content_of: Callable[[T], str] = lambda item: item.content,  # type: ignore[attr-defined]
) -> list[T]:
    """Filter ``items`` to those scoring >= ``floor`` against ``query``.

    Args:
        query: The search query.
        items: Retrieved candidates (each must expose ``.content`` or supply ``content_of``).
        floor: Minimum raw cross-encoder logit to survive.
        top_k: Max items to return after flooring.
        arm: Label for logging (e.g. "profiles", "user_playbooks").
        content_of: Extracts the text to score for an item.

    Returns:
        Survivors sorted by score descending, capped at ``top_k``. ``[]`` if none qualify.
        On reranker unavailability, returns ``items[:top_k]`` unchanged (logged).
    """
    if not items:
        return []
    try:
        scores = score_pairs(query, [content_of(item) for item in items])
    except CrossEncoderUnavailableError:
        logger.warning(
            "event=relevance_floor_unavailable arm=%s items=%d (returning unfiltered top_k)",
            arm,
            len(items),
        )
        return items[:top_k]

    ranked = sorted(
        zip(items, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    survivors = [item for item, score in ranked if score >= floor]
    dropped = len(ranked) - len(survivors)
    if dropped:
        logger.info(
            "event=relevance_floor arm=%s kept=%d dropped=%d floor=%.2f",
            arm,
            len(survivors),
            dropped,
            floor,
        )
    return survivors[:top_k]
