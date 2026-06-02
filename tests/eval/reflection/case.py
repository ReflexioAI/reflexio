"""Case schema + label mapping for the reflection decision-eval harness.

A :class:`ReflectionEvalCase` captures the inputs a single reflection
decision needs (agent context, the interaction window, and the cited
item being judged) plus the *intended* outcome as a coarse
``gold_label``.

Mapping a live decision to a label
----------------------------------
The live reflection decision (:class:`ReflectionDecision`) carries **no
mode label** — the outcome is encoded purely by which replacement fields
are set (see ``reflection_service._is_revision``). For *scoring* we
collapse that field-presence into a coarse label via
:func:`label_for_decision`:

==========================================  ===================================
Field presence on the decision              Mapped label
==========================================  ===================================
nothing set                                 ``no_change``
``new_profile_time_to_live`` set            ``ttl``
``new_content`` set (substantive)           ``rewrite``
``new_trigger`` only                        ``tighten`` / ``widen``
==========================================  ===================================

An orientation-reversing rewrite (one whose ``new_content`` restates the
rule in the opposite direction) is labeled ``rewrite`` like any other
content change. A flip is **not** mechanically distinguishable from a
plain rewrite without a polarity heuristic, and that heuristic is retired
(orientation lives only in the wording; there is no ``new_polarity``
field). Whether a rewrite *actually* reversed the rule's meaning is the
AI judge's domain, not the deterministic labeler's.

Precedence matters because a single decision may set several fields: TTL
wins, then a trigger-scope change, then a content rewrite. When a trigger
change is set but we cannot tell whether it narrows or broadens (no
cited trigger to compare against, or no length signal) we treat it as
the ambiguous ``scope`` bucket rather than guessing tighten/widen.

The narrow-vs-broaden heuristic is intentionally cheap and offline-only:
a *longer / more-qualified* new trigger is treated as ``tighten``; a
*shorter / less-qualified* one as ``widen`` (matching ``_trigger_scope``).
This is a coarse proxy, good enough to flag over-specialization
regressions, not a semantic judge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
)

# The eval's coarse classification of a reflection decision's *intent*.
GoldLabel = Literal[
    "no_change",
    "tighten",
    "widen",
    "rewrite",
    "ttl",
]

# Returned by the mapper when a trigger change cannot be classified as
# narrowing or broadening. Not a valid ``gold_label`` (cases must commit
# to tighten/widen) — surfaced so callers can score it as a mismatch.
ScopeLabel = Literal["scope"]


class CitedItem(BaseModel):
    """The cited playbook/profile row a reflection decision is judged on.

    Kept deliberately loose (a structured dict-like model rather than the
    full ``UserPlaybook`` / ``UserProfile`` entity) so fixtures stay
    small and readable. Only the fields the label mapping and the judge
    need are modelled; everything else can ride in ``extra``.

    Attributes:
        kind: ``"playbook"`` or ``"profile"``.
        target_id: Stable id of the cited row (stringified
            ``user_playbook_id`` for playbooks, ``profile_id`` for
            profiles).
        content: Current content text of the cited row.
        trigger: Current playbook trigger (None for profiles).
        profile_time_to_live: Current profile TTL (None for playbooks).
    """

    kind: Literal["playbook", "profile"]
    target_id: str
    content: str = ""
    trigger: str | None = None
    profile_time_to_live: str | None = None


class ReflectionEvalCase(BaseModel):
    """One reflection decision-eval case.

    Attributes:
        id: Stable case id (used for parametrization and reporting).
        agent_context: The agent-context prompt fragment in force.
        window: The interaction window the decision was made over.
        cited_item: The cited playbook/profile being judged.
        gold_label: The eval's coarse classification of the *intended*
            outcome. This is the harness's own label, NOT a field on the
            live decision.
        gold_new_trigger: Optional expected replacement trigger (for
            tighten/widen cases) — lets the judge compare specifics.
        notes: Optional free-form note for the AI judge / readers.
    """

    id: str
    agent_context: str = ""
    window: list[Interaction] = Field(default_factory=list)
    cited_item: CitedItem
    gold_label: GoldLabel
    gold_new_trigger: str | None = None
    notes: str | None = None


def _trigger_scope(old: str | None, new: str | None) -> ScopeLabel | GoldLabel:
    """Classify a trigger change as tighten / widen / ambiguous ``scope``.

    Heuristic, offline-only: compare the new trigger against the cited
    one. A *shorter* new trigger broadens scope (``widen``); a *longer*
    one narrows it (``tighten``). When there is no cited trigger to
    compare against, or the two are the same length, we cannot tell —
    return the ambiguous ``"scope"`` bucket.

    Args:
        old: Cited row's current trigger (may be None).
        new: Proposed replacement trigger.

    Returns:
        ``"tighten"``, ``"widen"``, or ``"scope"`` when ambiguous.
    """
    if new is None:
        return "scope"
    if not old:
        # No baseline to compare against — ambiguous.
        return "scope"
    if len(new) > len(old):
        return "tighten"
    if len(new) < len(old):
        return "widen"
    return "scope"


def label_for_decision(
    decision: ReflectionDecision,
    cited_item: CitedItem,
) -> GoldLabel | ScopeLabel:
    """Map a produced reflection decision to a coarse label.

    Field-presence is collapsed into a single label using the precedence
    documented at the module top: ttl > trigger-scope > rewrite >
    no_change. A trigger change that cannot be classified as
    narrowing/broadening yields the ambiguous ``"scope"`` label.

    An orientation-reversing rewrite is labeled ``"rewrite"`` like any
    other ``new_content`` change: a flip is not mechanically separable
    from a plain rewrite without a polarity heuristic, which is retired.
    Whether a rewrite actually reversed the rule's meaning is judged by
    the AI judge, not this deterministic mapper.

    Args:
        decision: The reflection decision produced by the service/LLM.
        cited_item: The cited row the decision targets — supplies the
            baseline trigger for comparison.

    Returns:
        One of the ``GoldLabel`` values, or ``"scope"`` for an ambiguous
        trigger change.
    """
    # 1. TTL change (profiles).
    if decision.new_profile_time_to_live is not None:
        return "ttl"

    # 2. Trigger-scope change (playbooks).
    if decision.new_trigger is not None and decision.new_trigger != cited_item.trigger:
        return _trigger_scope(cited_item.trigger, decision.new_trigger)

    # 3. Substantive content rewrite (includes orientation-reversing ones).
    if decision.new_content is not None and decision.new_content != cited_item.content:
        return "rewrite"

    # 4. Nothing changed.
    return "no_change"
