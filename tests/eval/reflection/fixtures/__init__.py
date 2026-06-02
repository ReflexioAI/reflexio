"""Illustrative fixtures for the reflection decision-eval harness.

These cases are *scaffolding*, not a curated eval set — they exist only
to exercise the harness. See ``README.md`` in this directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.eval.reflection.case import ReflectionEvalCase

_FIXTURE_DIR = Path(__file__).parent


def load_illustrative_cases() -> list[ReflectionEvalCase]:
    """Load the tiny illustrative case set as ``ReflectionEvalCase`` objects.

    Returns:
        The parsed cases, sorted by ``id`` for stable ordering.
    """
    raw = json.loads((_FIXTURE_DIR / "illustrative_cases.json").read_text())
    cases = [ReflectionEvalCase.model_validate(c) for c in raw]
    return sorted(cases, key=lambda c: c.id)
