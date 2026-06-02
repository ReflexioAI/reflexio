"""Tests for the Pattern D heuristic classifier and conditional render path."""

from __future__ import annotations

import pytest

from reflexio.server.prompt._dispatchers import is_pattern_d_question


@pytest.mark.parametrize(
    "query",
    [
        "Which sports does John like?",
        "What books has Tim read?",
        "Which sports does John like besides basketball?",
        "What other places has user visited?",
        "How many cities has user mentioned?",
        "List all the bands user enjoys",
        "Which TV series does Tim watch?",
        "What countries has user been to?",
    ],
)
def test_dispatcher_flags_list_questions(query: str) -> None:
    """List/enumeration shapes must classify as Pattern D."""
    assert is_pattern_d_question(query)


@pytest.mark.parametrize(
    "query",
    [
        "When did John sign with the Wolves?",
        "What is John's role on the team?",
        "Has Tim been to North Carolina?",
        "Who is John's coach?",
        "What date did Tim leave for Ireland?",
        "Where was Tim last weekend?",
        "How long has user been surfing?",
    ],
)
def test_dispatcher_rejects_single_fact_questions(query: str) -> None:
    """Single-fact / non-list shapes must not classify as Pattern D."""
    assert not is_pattern_d_question(query)


# NOTE: The agentic search-agent pipeline (``render_search_prompt``, the
# ``search_agent`` prompt bank, per-pattern A-H recipes, and Pattern D
# conditional rehydration) was removed in the "post-horizon reflection +
# polarity-aware playbook lifecycle" work. The ``is_pattern_d_question``
# classifier survives, so its tests above are retained; the render-path tests
# that exercised the deleted ``render_search_prompt`` API were removed.
