"""Linchpin fixtures for the multi-round memory-scenario harness.

These scenarios are *scaffolding*, not a curated eval set — they exist to
exercise the runner end to end across the three memory components
(extract -> consolidate -> reflect) on shared, domain-agnostic fixtures.
Each one isolates a single behavior the harness must demonstrate:

- ``compose_grows_skill`` — a learn round whose new avoid-rule for a
  different sub-aspect ``unify``-composes into the seeded do-rule, growing
  one skill (Option-B compose).
- ``no_self_contradiction`` — a learn round implying the opposite advice on
  the SAME trigger must ``differentiate`` (two refined triggers), never
  collapse into a self-contradicting merge.
- ``reflect_corrects`` — an unrelated learn round (``independent``) followed
  by a reflect round that tightens a cited rule the window shows misfired.

See ``case.py`` for the scenario schema.
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
