"""AI-judged evaluation of complete audited consolidation batches.

Create/Replace/Remove actions and mixed batches are scored against original
source observations and expected memory intent. The synthetic fixtures exercise
the harness, not production quality. Contradiction checks cover the entire output
memory; action names alone do not establish semantic correctness.
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
