"""Case schema + kind mapping for the consolidation decision-eval harness.

A :class:`ConsolidationEvalCase` captures the inputs a single consolidation
decision needs (the agent context, the EXISTING playbook rows search
surfaced, and the one NEW candidate fragment under consideration) plus the
*intended* outcome as a coarse ``gold_kind``.

Mapping a live decision to a kind
---------------------------------
Unlike the reflection harness — where the live decision carries **no mode
label** and the outcome must be reconstructed from which replacement fields
are set (see ``reflection.case.label_for_decision``) — a consolidation
decision is an explicit discriminated union. Each concrete decision
(:class:`UnifyDecision`, :class:`RejectNewDecision`,
:class:`DifferentiateDecision`, :class:`IndependentDecision`) carries a
``kind`` literal that *is* the outcome. So the mapper here is trivial: it
just reads ``decision.kind``. There is no heuristic and no precedence to
resolve — the kind is authoritative.

==========================================  ===================================
Concrete decision                           Mapped kind
==========================================  ===================================
``UnifyDecision``                           ``unify``
``RejectNewDecision``                       ``reject_new``
``DifferentiateDecision``                   ``differentiate``
``IndependentDecision``                     ``independent``
==========================================  ===================================

The interesting question for the harness is therefore not *which kind* the
decision is (that is explicit) but *whether that kind matches intent* and,
for a ``unify``, whether the merge introduced a self-contradiction. The
former is the AI judge's headline metric; the latter is the consolidation-
specific judge dimension. Neither lives here — this module only models the
case inputs and the deterministic kind read-off.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from reflexio.server.services.playbook.components.consolidator import (
    ConsolidationDecision,
)

# The eval's coarse classification of a consolidation decision's *intent*.
# These are exactly the four discriminator literals of the live
# ``ConsolidationDecision`` union, so a produced decision's kind is always a
# valid ``GoldKind`` value.
GoldKind = Literal[
    "unify",
    "reject_new",
    "differentiate",
    "independent",
]


class ExistingPlaybook(BaseModel):
    """An EXISTING playbook row the consolidator weighs the candidate against.

    Kept deliberately loose (a small readable model rather than the full
    ``UserPlaybook`` entity) so fixtures stay compact. Mirrors how the live
    consolidator references EXISTING rows by their integer id.

    Attributes:
        id: Stable integer id of the existing row (the consolidator's
            ``archive_existing_ids`` / ``superseded_by_existing_id`` /
            ``existing_id`` all reference rows by int id).
        content: Current content text of the existing rule.
        trigger: Current trigger of the existing rule (None when unscoped).
        rationale: Current rationale text, if any.
    """

    id: int
    content: str = ""
    trigger: str | None = None
    rationale: str = ""


class CandidatePlaybook(BaseModel):
    """The single NEW candidate fragment under consideration.

    This is the freshly-extracted rule the consolidator must place: unify it
    into an existing row, reject it as redundant, differentiate it from a
    near-duplicate, or admit it as independent.

    Attributes:
        new_id: The candidate's id (a string, mirroring the live
            decision's ``new_id`` field).
        content: The candidate rule's content text.
        trigger: The candidate's trigger (None when unscoped).
        rationale: The candidate's rationale text, if any.
    """

    new_id: str
    content: str = ""
    trigger: str | None = None
    rationale: str = ""


class ConsolidationEvalCase(BaseModel):
    """One consolidation decision-eval case.

    Attributes:
        id: Stable case id (used for parametrization and reporting).
        agent_context: The agent-context prompt fragment in force.
        existing: The EXISTING playbook rows search surfaced for the
            candidate (may be empty for a clean ``independent`` case).
        candidate: The single NEW candidate fragment being placed.
        gold_kind: The eval's coarse classification of the *intended*
            decision kind. This is the harness's own intent label; it
            happens to share the literal values of the live decision's
            discriminator but is asserted by the case author, not read off
            a produced decision.
        notes: Optional free-form note for the AI judge / readers.
    """

    id: str
    agent_context: str = ""
    existing: list[ExistingPlaybook] = Field(default_factory=list)
    candidate: CandidatePlaybook
    gold_kind: GoldKind
    notes: str | None = None


def kind_for_decision(decision: ConsolidationDecision) -> str:
    """Map a produced consolidation decision to its coarse kind.

    Unlike reflection's ``label_for_decision`` — which reconstructs a label
    from field presence with a precedence rule — the consolidation kind is
    explicit: every concrete decision in the :data:`ConsolidationDecision`
    union carries a ``kind`` discriminator literal that *is* the outcome.
    This function therefore just returns that literal, with no heuristic.

    Args:
        decision: The consolidation decision produced by the service / LLM.

    Returns:
        The decision's ``kind`` literal (one of the :data:`GoldKind`
        values: ``"unify"``, ``"reject_new"``, ``"differentiate"``,
        ``"independent"``).
    """
    return decision.kind
