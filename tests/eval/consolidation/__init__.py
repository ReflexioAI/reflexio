"""Consolidation decision-eval harness (AI-judged).

Scaffolding to measure whether the playbook consolidator makes the right
decision for a NEW candidate fragment against the EXISTING rows search
surfaces — ``unify`` / ``reject_new`` / ``differentiate`` / ``independent`` —
and to catch regressions when the ``playbook_consolidation`` prompt changes.

The headline metric is **AI-judge agreement** (does the produced decision
match intent?), augmented with a consolidation-specific judge dimension —
**self-contradiction** (did a ``unify`` merge rules that contradict on the
same situation, or collapse distinct do/avoid rules?). That dimension is
deliberately LLM-judged: the mechanical polarity guard was retired, so no
classifier lives here.

This package contains *bounded scaffolding plus a tiny illustrative
fixture* — it is NOT a curated eval dataset. Real cases are curated later;
the fixture exists only to exercise the harness end-to-end.
"""

from __future__ import annotations

from tests.eval.consolidation.case import (
    CandidatePlaybook,
    ConsolidationEvalCase,
    ExistingPlaybook,
    GoldKind,
    kind_for_decision,
)

__all__ = [
    "CandidatePlaybook",
    "ConsolidationEvalCase",
    "ExistingPlaybook",
    "GoldKind",
    "kind_for_decision",
]
