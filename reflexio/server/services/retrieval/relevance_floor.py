"""Read-path relevance floor: drop retrieved items below a cross-encoder score.

Scores ``(query, item.content)`` pairs with the local cross-encoder (raw logits),
drops anything below ``floor``, returns survivors sorted by score desc, capped at
``top_k``. On reranker unavailability, degrades to the retrieved order (logged) —
never crashes, never silently returns the full unfiltered list.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from reflexio.server.llm.rerank import score_pairs
from reflexio.server.llm.rerank.cross_encoder_reranker import (
    CrossEncoderUnavailableError,
)
from reflexio.server.tracing import profile_step

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelevanceFloorResult:
    items: list[Any]
    scores: list[float] | None


def _floor_and_sort[T](
    items: list[T], scores: list[float], floor: float
) -> list[tuple[T, float]]:
    """Return ``(item, score)`` pairs scoring >= ``floor``, sorted descending."""
    ranked = sorted(
        zip(items, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [(item, score) for item, score in ranked if score >= floor]


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
    with profile_step(
        "search.rerank.relevance_floor",
        arm=arm,
        items=len(items),
    ) as span:
        try:
            scores = score_pairs(query, [content_of(item) for item in items])
        except CrossEncoderUnavailableError:
            span.set_data("available", False)
            logger.warning(
                "event=relevance_floor_unavailable arm=%s items=%d (returning unfiltered top_k)",
                arm,
                len(items),
            )
            return items[:top_k]
        span.set_data("available", True)

        survivors = [item for item, _score in _floor_and_sort(items, scores, floor)]
        dropped = len(items) - len(survivors)
        span.set_data("kept", len(survivors))
        span.set_data("dropped", dropped)
        if dropped:
            logger.info(
                "event=relevance_floor arm=%s kept=%d dropped=%d floor=%.2f",
                arm,
                len(survivors),
                dropped,
                floor,
            )
        return survivors[:top_k]


def apply_relevance_floors(
    query: str,
    arms: Sequence[tuple[str, list[Any], float]],
    top_k: int,
    *,
    content_of: Callable[[Any], str] = lambda item: item.content,
) -> list[RelevanceFloorResult]:
    """Floor every arm with a single cross-encoder batch.

    CPU cross-encoder inference does not parallelize across threads, so
    scoring all arms in one ``score_pairs`` call beats one call per arm:
    the model runs one batched forward pass and per-call overhead is paid
    once.

    Args:
        query: The search query.
        arms: ``(arm_name, items, floor)`` per arm; items must expose
            ``.content`` or supply ``content_of``.
        top_k: Max items to return per arm after flooring.
        content_of: Extracts the text to score for an item.

    Returns:
        One result per arm, in input order. Available reranker results are
        sorted by score descending and left uncapped with the paired raw logits.
        On reranker unavailability, every arm returns the original full item
        pool with a ``None`` score sentinel (logged).
    """
    if not any(items for _, items, _ in arms):
        return [RelevanceFloorResult([], []) for _ in arms]
    arm_names = [name for name, _, _ in arms]
    contents: list[str] = []
    for _, items, _ in arms:
        contents.extend(content_of(item) for item in items)
    with profile_step(
        "search.rerank.relevance_floor",
        arm="all",
        arms=arm_names,
        items=len(contents),
        top_k=top_k,
    ) as span:
        try:
            scores = score_pairs(query, contents)
        except CrossEncoderUnavailableError:
            span.set_data("available", False)
            logger.warning(
                "event=relevance_floor_unavailable arms=%s items=%d (returning unfiltered pool)",
                arm_names,
                len(contents),
            )
            return [RelevanceFloorResult(list(items), None) for _, items, _ in arms]
        span.set_data("available", True)

        results: list[RelevanceFloorResult] = []
        offset = 0
        for name, items, floor in arms:
            arm_scores = scores[offset : offset + len(items)]
            offset += len(items)
            survivors = _floor_and_sort(items, arm_scores, floor)
            dropped = len(items) - len(survivors)
            span.set_data(f"kept_{name}", len(survivors))
            span.set_data(f"dropped_{name}", dropped)
            if dropped:
                logger.info(
                    "event=relevance_floor arm=%s kept=%d dropped=%d floor=%.2f",
                    name,
                    len(survivors),
                    dropped,
                    floor,
                )
            results.append(
                RelevanceFloorResult(
                    [item for item, _score in survivors],
                    [score for _item, score in survivors],
                )
            )
        return results
