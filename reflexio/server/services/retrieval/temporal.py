"""Temporal post-processing for search results.

Applies query-derived temporal signals (extracted by the query reformulator
alongside the rewrite — see ``ReformulationResult``) to ranked entity lists:

- ``window_bounds``: relative day offsets → absolute datetimes for the
  per-arm ``start_time``/``end_time`` SQL filters.
- ``filter_current``: drop superseded / TTL-expired entities for
  current-value questions.
- ``freshness_collapse``: within near-duplicate groups of competing facts,
  the freshest wins — deterministically fixing "stale fact with the same
  wording outranks its fresh update", which LLM ordering alone misses.
- ``sort_by_recency``: absolute timestamp ordering for "current/latest X"
  questions.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

from reflexio.server.services.profile.profile_generation_service_utils import (
    check_string_token_overlap,
)

_DUPLICATE_OVERLAP_THRESHOLD = 0.6


def entity_timestamp(entity: Any) -> int:
    """The entity's primary timestamp (0 when missing).

    Profiles carry ``last_modified_timestamp``; playbooks carry
    ``created_at`` — same fields the recency decay reads.
    """
    return int(
        getattr(entity, "last_modified_timestamp", None)
        or getattr(entity, "created_at", None)
        or 0
    )


def window_bounds(
    start_days_ago: float | None,
    end_days_ago: float | None,
    now: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Convert relative day offsets into absolute (start, end) datetimes.

    Args:
        start_days_ago: Older bound ("in the last 7 days" → 7).
        end_days_ago: Newer bound ("before this month" → ~30, no start).
        now: Reference time (real now when omitted).

    Returns:
        tuple: (start_time, end_time), either side None when unbounded.
    """
    now = now or datetime.now(UTC)
    start = now - timedelta(days=start_days_ago) if start_days_ago is not None else None
    end = now - timedelta(days=end_days_ago) if end_days_ago is not None else None
    if start and end and start > end:
        start, end = end, start
    return start, end


def filter_current(entities: list[Any], now: int) -> list[Any]:
    """Drop entities that are no longer current (current-value questions).

    Removes entities with a ``superseded_by`` link and profiles whose TTL
    expired. Defensive at orchestration level — storage usually excludes
    tombstoned rows already, but the field can be set on live rows.

    Args:
        entities: Ranked entities of one arm.
        now: Epoch seconds for TTL-expiry comparison.

    Returns:
        list: Entities still considered current, order preserved.
    """
    kept = []
    for entity in entities:
        if getattr(entity, "superseded_by", None) is not None:
            continue
        expiration = getattr(entity, "expiration_timestamp", None)
        if expiration is not None and expiration < now:
            continue
        kept.append(entity)
    return kept


def freshness_collapse(entities: list[Any]) -> list[Any]:
    """Within near-duplicate groups of competing facts, freshest wins.

    Greedy grouping in ranked order by token overlap of (trigger + content);
    each group stays anchored at its best-ranked member's position but is
    internally ordered newest-first. Relevance order across unrelated
    entities is untouched.

    Args:
        entities: Ranked entities of one arm.

    Returns:
        list: Entities with near-duplicate groups collapsed newest-first.
    """
    groups: list[list[Any]] = []
    for entity in entities:
        text = _compare_text(entity)
        for group in groups:
            if check_string_token_overlap(
                text, _compare_text(group[0]), _DUPLICATE_OVERLAP_THRESHOLD
            ):
                group.append(entity)
                break
        else:
            groups.append([entity])
    collapsed: list[Any] = []
    for group in groups:
        collapsed.extend(sorted(group, key=entity_timestamp, reverse=True))
    return collapsed


def sort_by_recency(entities: list[Any]) -> list[Any]:
    """Order entities newest-first (for explicit current/latest questions)."""
    return sorted(entities, key=entity_timestamp, reverse=True)


def _compare_text(entity: Any) -> str:
    """Trigger + content, punctuation-normalized for token-overlap grouping."""
    trigger = getattr(entity, "trigger", None) or ""
    content = getattr(entity, "content", "") or ""
    return re.sub(r"[^\w\s]", " ", f"{trigger} {content}").strip()
