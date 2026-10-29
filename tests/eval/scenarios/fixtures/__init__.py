"""Synthetic multi-round extraction/consolidation scenarios.

Composition and contradiction-avoidance cases exercise stable evidence references
and complete batch projection into subsequent rounds. They are harness fixtures,
not a curated quality dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.eval.scenarios.case import MemoryScenario

_FIXTURE_DIR = Path(__file__).parent


def load_scenarios() -> list[MemoryScenario]:
    """Load the linchpin scenarios as validated ``MemoryScenario`` objects.

    Reads the sibling ``scenarios.json`` and validates each entry into a
    :class:`MemoryScenario` (which in turn validates its ``seed_book`` and
    ``rounds``).

    Returns:
        The parsed scenarios, sorted by ``id`` for stable ordering.
    """
    raw = json.loads((_FIXTURE_DIR / "scenarios.json").read_text())
    scenarios = [MemoryScenario.model_validate(s) for s in raw]
    return sorted(scenarios, key=lambda s: s.id)


__all__ = ["load_scenarios"]
