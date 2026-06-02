"""Case schema for the multi-round memory-scenario eval harness.

A :class:`MemoryScenario` describes a *chained* memory exercise: a seed
book of playbook rules followed by an ordered list of rounds. Each round
is either a ``learn`` round (drives extract -> consolidate) or a
``reflect`` round (drives the reflector against a cited rule). Between
rounds, each real component decision is mechanically reflected into an
accumulating in-memory :class:`BookRule` list (see ``book.py``), so the
next round reads the state the prior rounds settled.

This module only models the *case inputs* and the per-round *gold*
intent. It is deliberately schema-only — there is no apply logic and no
LLM here. The decision itself is the real component's judged output; the
gold here is the harness author's intent, asserted against that output by
the existing component judges.

Gold shapes
-----------
``ScenarioRound.gold`` is a small dict whose keys depend on ``kind``:

==========  ============================================================
kind        gold keys
==========  ============================================================
``learn``   ``{"consolidation_kind": "unify|reject_new|differentiate|
            independent", "extraction_signal": "<text>"}``
``reflect`` ``{"reflection": "tighten|widen|rewrite|no_change"}``
==========  ============================================================
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BookRule(BaseModel):
    """One accumulating in-memory playbook rule.

    The ``book`` is a flat list of these — the running state the scenario
    chain builds up. Mirrors the loose shape the consolidation/reflection
    eval cases use (a small readable model, not the full ``UserPlaybook``
    entity) so fixtures stay compact.

    Attributes:
        id: Stable integer id of the rule. Rules are referenced by this id
            in reflection decisions (``cited``) and by *list position* in
            unify decisions (see ``book.apply_consolidation``).
        content: Current content text of the rule.
        trigger: Current trigger (None when unscoped).
        rationale: Current rationale text, if any.
    """

    id: int
    content: str = ""
    trigger: str | None = None
    rationale: str = ""


class ScenarioRound(BaseModel):
    """One round in a multi-round memory scenario.

    A ``learn`` round feeds ``interactions`` to the extractor and routes
    each produced rule through the consolidator against the current book.
    A ``reflect`` round runs the reflector over ``interactions`` against
    the cited rule (``cited`` -> a :class:`BookRule` id).

    Attributes:
        kind: ``"learn"`` (extract -> consolidate) or ``"reflect"``.
        interactions: The interaction window for the round. Each entry is
            a loose dict (e.g. ``{"role", "content", "tools_used"}``).
        cited: For ``reflect`` rounds, the :class:`BookRule.id` the round
            reflects on (None for ``learn`` rounds).
        gold: The round's intended outcome. For ``learn``:
            ``{"consolidation_kind": ..., "extraction_signal": ...}``.
            For ``reflect``: ``{"reflection": ...}``. Keys are documented
            at the module top.
    """

    kind: Literal["learn", "reflect"]
    interactions: list[dict] = Field(default_factory=list)
    cited: int | None = None
    gold: dict = Field(default_factory=dict)


class MemoryScenario(BaseModel):
    """One multi-round memory scenario.

    Attributes:
        id: Stable scenario id (used for parametrization and reporting).
        agent_context: The agent-context prompt fragment in force across
            the scenario's rounds.
        seed_book: The initial book of :class:`BookRule` rows the scenario
            starts from (may be empty).
        rounds: The ordered rounds to run; the accumulating book is
            threaded through them.
        gold_end_state: Natural-language description of the expected final
            book, for an optional end-state judge.
    """

    id: str
    agent_context: str = ""
    seed_book: list[BookRule] = Field(default_factory=list)
    rounds: list[ScenarioRound]
    gold_end_state: str = ""
