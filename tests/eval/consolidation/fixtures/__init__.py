"""Illustrative fixtures for the consolidation decision-eval harness.

These cases are *scaffolding*, not a curated eval set — they exist only
to exercise the harness end-to-end. See ``README.md`` in the parent
directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.eval.consolidation.case import ConsolidationEvalCase

_FIXTURE_DIR = Path(__file__).parent


def load_illustrative_cases() -> list[ConsolidationEvalCase]:
    """Load the tiny illustrative case set as ``ConsolidationEvalCase`` objects.

    Returns:
        The parsed cases, sorted by ``id`` for stable ordering.
    """
    raw = json.loads((_FIXTURE_DIR / "illustrative_cases.json").read_text())
    cases = [ConsolidationEvalCase.model_validate(c) for c in raw]
    return sorted(cases, key=lambda c: c.id)


__all__ = ["load_illustrative_cases"]
