"""Agentic-vs-classic search comparison harness (scaffolding only).

Mirrors the extraction comparison harness; actual quality numbers require
``REFLEXIO_EVAL_REAL_JUDGE=1`` + real LLM keys.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_agentic_vs_classic_search(search_case, search_judge):
    """For each golden case, the stubbed judge returns a parseable score."""
    classic_out = {"ranked_ids": []}
    agentic_out = {"ranked_ids": []}

    c_score = search_judge.score(expected=search_case, actual=classic_out)
    a_score = search_judge.score(expected=search_case, actual=agentic_out)

    assert c_score.answer_correctness >= 0.0
    assert a_score.answer_correctness >= 0.0
    assert c_score.rationale
    assert a_score.rationale
