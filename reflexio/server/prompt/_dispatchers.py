"""Heuristic question-shape classifiers for conditional prompt rendering.

The search-agent prompt is split: the main prompt covers Patterns A-H dispatch
plus all pattern recipes EXCEPT Pattern D. Pattern D's recipe lives in a
separate prompt file and is appended only when the question is classified as
list-shape. This isolation prevents Pattern D rule changes from bleeding into
non-list questions (the dominant regression mode in v1.20.0/v1.21.0/v1.22.0/
v1.23.0 search iterations).

False negatives fall through to Pattern A behaviour (safe default — the
main prompt's dispatch test still routes the agent to the correct pattern).
False positives are tolerable: Pattern D recipe in context but the agent's
own dispatch test still picks the right pattern.
"""

from __future__ import annotations

import re

# Patterns that signal list/enumeration shape.
# Order matters only for early-exit; any match returns True.
_LIST_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Which sports", "what books", "which TV series", "which fantasy novels"
    # — plural noun (≥4 chars to avoid matching short verbs like "is", "was",
    # "does"), optionally preceded by up to 2 intermediate words (adjectives /
    # short qualifiers like "TV", "other").
    re.compile(r"\b(which|what)\s+(?:\w+\s+){0,2}\w{3,}s\b", re.IGNORECASE),
    # Counting / total / list-all operators
    re.compile(
        r"\b(how many|how much|total|list (all|every)|all the|every|count)\b",
        re.IGNORECASE,
    ),
    # Exclusion-shaped lists ("besides X", "other than Y", "not Z-related")
    re.compile(r"\b(besides|other than)\b", re.IGNORECASE),
    # "What other places", "what other movies"
    re.compile(r"\bother\s+\w+s?\b", re.IGNORECASE),
    # Superlatives over a category
    re.compile(r"\b(most|least|highest|lowest|best|worst)\s+\w+", re.IGNORECASE),
)


def is_pattern_d_question(query: str) -> bool:
    """Return True when the question's shape suggests Pattern D (list/aggregation).

    Args:
        query (str): The user question text.

    Returns:
        bool: True if any list/enumeration heuristic matches; False otherwise.
    """
    return any(p.search(query) for p in _LIST_PATTERNS)


__all__ = ["is_pattern_d_question"]
