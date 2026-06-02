"""Multi-round memory-scenario eval harness (AI-judged).

Chains the existing live providers (extract -> consolidate -> reflect)
across rounds, applying each real component decision to an accumulating
in-memory ``book`` via a thin, test-only apply shim (``book.py``), so the
next round reads accumulated state. Each round is judged by the existing
component judge with its native verdict; the only aggregate is a boolean
``scenario_passed`` roll-up.

This is the deterministic multi-round coverage the SWE-bench A/B pair
lacks: it exercises all three memory components in sequence on shared
fixtures.
"""

from __future__ import annotations

from tests.eval.scenarios.case import (
    BookRule,
    MemoryScenario,
    ScenarioRound,
)

__all__ = [
    "BookRule",
    "MemoryScenario",
    "ScenarioRound",
]
