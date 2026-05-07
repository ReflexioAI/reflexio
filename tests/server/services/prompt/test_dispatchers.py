"""Tests for the Pattern D heuristic classifier and conditional render path."""

from __future__ import annotations

import pytest

from reflexio.server.prompt._dispatchers import is_pattern_d_question
from reflexio.server.prompt.prompt_manager import PromptManager


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


@pytest.mark.parametrize("letter", list("abcdefgh"))
def test_render_includes_all_pattern_recipes(letter: str) -> None:
    """Full-load: render_search_prompt inlines every pattern recipe (A-H).

    The split is purely a file-organization choice; runtime output is
    byte-equivalent to embedding all 8 recipes inline. Each pattern's
    placeholder must be substituted with its recipe content.
    """
    pm = PromptManager()
    rendered = pm.render_search_prompt(
        query="Which sports does John like besides basketball?",
        max_steps="12",
        enable_agent_answer="true",
    )
    assert f"<<<PATTERN_{letter.upper()}_RECIPE>>>" not in rendered
    # Each pattern's recipe content includes a heading "### Pattern <X>".
    assert f"### Pattern {letter.upper()}" in rendered


def test_render_includes_all_recipes_for_single_fact_question() -> None:
    """Full-load is unconditional: even non-list questions see every recipe."""
    pm = PromptManager()
    rendered = pm.render_search_prompt(
        query="When did John sign with the Wolves?",
        max_steps="12",
        enable_agent_answer="true",
    )
    for letter in "ABCDEFGH":
        assert f"<<<PATTERN_{letter}_RECIPE>>>" not in rendered
        assert f"### Pattern {letter}" in rendered


def test_render_search_prompt_includes_pattern_d_active_version() -> None:
    """The Pattern D recipe loaded should be the active version of
    search_agent/patterns/d (currently v1.0.0 — mandatory rehydration).
    """
    pm = PromptManager()
    rendered = pm.render_search_prompt(
        query="What books has user read?",
        max_steps="12",
        enable_agent_answer="true",
    )
    # v1.0.0 = mandatory-rehydration shipping recipe; v1.1.0 (rehydrate-first
    # experiment) is on disk but inactive. Verify the active version's
    # signature phrase appears.
    assert "MANDATORY rehydration" in rendered
