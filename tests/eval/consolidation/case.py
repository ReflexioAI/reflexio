"""Source-bearing inputs and complete audited batches for consolidation evals."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from reflexio.server.services.playbook.components.reconciliation import (
    Decisions,
    Reference,
)

GoldKind = Literal["create", "replace", "remove", "mixed"]


class ExistingPlaybook(BaseModel):
    id: int
    content: str = ""
    trigger: str | None = None
    rationale: str = ""
    evidence_refs: list[Reference] = Field(default_factory=list)


class CandidatePlaybook(BaseModel):
    new_id: str
    content: str = ""
    trigger: str | None = None
    rationale: str = ""
    evidence_refs: list[Reference] = Field(default_factory=list)


class EvidenceMessage(BaseModel):
    turn: int
    role: str
    content: str


class EvidenceSession(BaseModel):
    id: str
    messages: list[EvidenceMessage] = Field(min_length=1)


class ConsolidationEvalCase(BaseModel):
    id: str
    agent_context: str = ""
    existing: list[ExistingPlaybook] = Field(default_factory=list)
    candidate: CandidatePlaybook
    sessions: list[EvidenceSession] = Field(default_factory=list)
    gold_kind: GoldKind
    notes: str | None = None


def kind_for_decision(batch: Decisions) -> str:
    """Describe the complete batch; never discard existing-only decisions."""
    if not batch.decisions:
        raise ValueError("A candidate evaluation requires a nonempty decision batch")
    kinds = {decision.kind for decision in batch.decisions}
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"
