"""Agentic-vs-classic extraction comparison harness.

Scaffolding only: ``classic_out`` and ``agentic_out`` are stubbed empty
because actual backend quality numbers require ``REFLEXIO_EVAL_REAL_JUDGE=1``
with a real LLM. The harness exists so the golden-set loader, judge wiring,
and test parametrization are proven green in CI.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_agentic_vs_classic_extraction(extraction_case, extraction_judge):
    """For each golden case, the stubbed judge returns a parseable score."""
    classic_out = {"profiles": [], "playbooks": []}
    agentic_out = {"profiles": [], "playbooks": []}

    c_score = extraction_judge.score(expected=extraction_case, actual=classic_out)
    a_score = extraction_judge.score(expected=extraction_case, actual=agentic_out)

    assert c_score.signal_f1 >= 0.0
    assert a_score.signal_f1 >= 0.0
    assert c_score.rationale
    assert a_score.rationale
