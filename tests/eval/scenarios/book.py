"""TEST-ONLY apply shim for the multi-round memory-scenario harness.

This module mechanically reflects a *real* component decision (a
``ConsolidationDecision`` or a ``ReflectionDecision`` produced by the live
service/LLM) into the accumulating in-memory ``book`` so the next round
can read the settled state. It does **not** re-implement the services'
private ``_apply_*`` internals or any storage/embedding/timestamp
semantics — only the minimal structural change the next round needs to
observe. The decision itself is the real component's judged output; this
shim just lets state accumulate across rounds.

Position-vs-id mapping (mirrors the real consolidator's apply contract)
-----------------------------------------------------------------------
The consolidation prompt labels the EXISTING candidates by **list
position** in the order they were passed (``[EXISTING-0]``,
``[EXISTING-1]``, ...). So ``UnifyDecision.archive_existing_ids`` holds
**list positions** into the ``existing_order`` the runner passed, NOT
``BookRule.id`` values — :func:`apply_consolidation` maps each position to
the rule at that index of ``existing_order``, then removes those rules
from the book.

``DifferentiateDecision.existing_id``, by contrast, IS a ``BookRule.id``
(the consolidator references the single differentiated existing row by its
integer id), so it is matched against ``BookRule.id`` directly.

Reflection decisions reference their cited row by id (``cited_id`` ->
``BookRule.id``) and only edit playbook content/trigger; TTL/profile
fields are ignored here because the book holds playbook rules.

Everything in this module is small and pure: no I/O, no LLM.
"""

from __future__ import annotations

from reflexio.server.services.playbook.playbook_consolidator import (
    ConsolidationDecision,
    DifferentiateDecision,
    UnifyDecision,
)
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
)
from tests.eval.scenarios.case import BookRule


def _next_id(book: list[BookRule]) -> int:
    """Return the next free integer id for ``book`` (max id + 1, else 1)."""
    if not book:
        return 1
    return max(rule.id for rule in book) + 1


def apply_consolidation(
    book: list[BookRule],
    candidate: BookRule,
    decision: ConsolidationDecision,
    *,
    existing_order: list[BookRule],
) -> list[BookRule]:
    """Reflect one consolidation decision into a NEW book list.

    The four decision kinds map to structural edits as follows:

    - ``unify``: remove the rules in ``existing_order`` at the **list
      positions** in ``decision.archive_existing_ids`` (position -> rule
      via ``existing_order``; see the module docstring), then append one
      merged ``BookRule`` carrying the decision's ``content`` / ``trigger``
      / ``rationale`` with a fresh id. The candidate is NOT also appended
      (it is subsumed by the merged rule). Rules in ``book`` not named by
      ``archive_existing_ids`` are retained.
    - ``differentiate``: set the matched existing rule's ``trigger`` to
      ``decision.refined_existing_trigger`` (matched by
      ``decision.existing_id`` against ``BookRule.id``), then append the
      candidate with ``trigger = decision.refined_new_trigger`` and a fresh
      id.
    - ``reject_new``: return ``book`` unchanged (the candidate is dropped).
    - ``independent``: append the candidate with a fresh id.

    Args:
        book: The current accumulating book (not mutated).
        candidate: The freshly-extracted rule under consideration.
        decision: The real consolidator decision to reflect.
        existing_order: The EXISTING rules in the exact order they were
            presented to the consolidator — supplies the position -> rule
            mapping for ``unify``.

    Returns:
        A new ``list[BookRule]`` with the decision applied.
    """
    if isinstance(decision, UnifyDecision):
        archived_ids = {
            existing_order[pos].id
            for pos in decision.archive_existing_ids
            if 0 <= pos < len(existing_order)
        }
        new_book = [rule for rule in book if rule.id not in archived_ids]
        new_book.append(
            BookRule(
                id=_next_id(new_book),
                content=decision.content,
                trigger=decision.trigger,
                rationale=decision.rationale,
            )
        )
        return new_book

    if isinstance(decision, DifferentiateDecision):
        # NOTE: the shim refines the existing rule's trigger IN PLACE, preserving
        # its id. The real consolidator archives the existing row and emits a new
        # row (fresh id) with the refined trigger. Book content/structure match;
        # the id differs. This is harmless unless a later round re-cites a
        # post-differentiate rule by id — add that handling here if such a
        # multi-round differentiate→reflect scenario is introduced.
        new_book = [
            rule.model_copy(update={"trigger": decision.refined_existing_trigger})
            if rule.id == decision.existing_id
            else rule.model_copy()
            for rule in book
        ]
        new_book.append(
            candidate.model_copy(
                update={
                    "id": _next_id(new_book),
                    "trigger": decision.refined_new_trigger,
                }
            )
        )
        return new_book

    if decision.kind == "reject_new":
        return [rule.model_copy() for rule in book]

    # independent
    new_book = [rule.model_copy() for rule in book]
    new_book.append(candidate.model_copy(update={"id": _next_id(new_book)}))
    return new_book


def apply_reflection(
    book: list[BookRule],
    cited_id: int,
    decision: ReflectionDecision,
) -> list[BookRule]:
    """Reflect one reflection decision into a NEW book list.

    Finds the rule with ``id == cited_id`` and applies the decision's
    replacement fields: if ``decision.new_content`` is not None, update
    ``content``; if ``decision.new_trigger`` is not None, update
    ``trigger``. If both are None (a ``no_change`` decision) the book is
    returned unchanged. TTL / profile fields are ignored because the book
    holds playbook rules.

    Args:
        book: The current accumulating book (not mutated).
        cited_id: The :class:`BookRule.id` the reflection targets.
        decision: The real reflection decision to reflect.

    Returns:
        A new ``list[BookRule]`` with the decision applied.
    """
    if decision.new_content is None and decision.new_trigger is None:
        return [rule.model_copy() for rule in book]

    new_book: list[BookRule] = []
    for rule in book:
        if rule.id != cited_id:
            new_book.append(rule.model_copy())
            continue
        update: dict = {}
        if decision.new_content is not None:
            update["content"] = decision.new_content
        if decision.new_trigger is not None:
            update["trigger"] = decision.new_trigger
        new_book.append(rule.model_copy(update=update))
    return new_book
