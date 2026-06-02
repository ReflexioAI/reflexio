"""Deterministic unit tests for the thin apply shim (``book.py``).

Pure logic, no LLM and no storage: build a small book + each real
decision kind, apply it via the shim, and assert the exact resulting
book. Covers the position-vs-id mapping (unify uses ``existing_order``
list positions; differentiate uses ``existing_id`` = a ``BookRule.id``)
and the reflection content/trigger/no-change paths.
"""

from __future__ import annotations

from reflexio.server.services.playbook.playbook_consolidator import (
    DifferentiateDecision,
    IndependentDecision,
    RejectNewDecision,
    UnifyDecision,
)
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
)
from tests.eval.scenarios.book import (
    _next_id,
    apply_consolidation,
    apply_reflection,
)
from tests.eval.scenarios.case import BookRule


def _seed() -> tuple[BookRule, BookRule]:
    r1 = BookRule(id=1, content="use spaces", trigger="formatting", rationale="r1")
    r2 = BookRule(id=2, content="run tests", trigger="ci", rationale="r2")
    return r1, r2


def test_next_id() -> None:
    assert _next_id([]) == 1
    assert _next_id([BookRule(id=3), BookRule(id=7)]) == 8


def test_unify_removes_by_position_and_merges() -> None:
    r1, r2 = _seed()
    book = [r1, r2]
    candidate = BookRule(id=99, content="cand", trigger="t-cand")
    decision = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=[0],  # list position 0 -> r1
        content="merged",
        trigger="t",
        rationale="",
    )

    result = apply_consolidation(book, candidate, decision, existing_order=[r1, r2])

    # r1 (position 0) removed, r2 retained, one merged rule appended;
    # candidate is NOT appended.
    assert result == [
        BookRule(id=2, content="run tests", trigger="ci", rationale="r2"),
        BookRule(id=3, content="merged", trigger="t", rationale=""),
    ]


def test_differentiate_refines_both_triggers() -> None:
    r1, r2 = _seed()
    book = [r1, r2]
    candidate = BookRule(id=99, content="cand", trigger="t-cand")
    decision = DifferentiateDecision(
        new_id="NEW-0",
        existing_id=r1.id,  # matched by BookRule.id
        refined_new_trigger="A",
        refined_existing_trigger="B",
    )

    result = apply_consolidation(book, candidate, decision, existing_order=[r1, r2])

    # r1's trigger refined to B; r2 untouched; candidate appended with
    # trigger A and a fresh id. Both the existing and new rule present.
    assert result == [
        BookRule(id=1, content="use spaces", trigger="B", rationale="r1"),
        BookRule(id=2, content="run tests", trigger="ci", rationale="r2"),
        BookRule(id=3, content="cand", trigger="A", rationale=""),
    ]


def test_reject_new_is_noop() -> None:
    r1, r2 = _seed()
    book = [r1, r2]
    candidate = BookRule(id=99, content="cand")
    decision = RejectNewDecision(new_id="NEW-0", superseded_by_existing_id=r1.id)

    result = apply_consolidation(book, candidate, decision, existing_order=[r1, r2])

    # Book unchanged; candidate absent.
    assert result == [r1, r2]
    assert all(rule.id != 99 for rule in result)


def test_independent_appends_candidate() -> None:
    r1, r2 = _seed()
    book = [r1, r2]
    candidate = BookRule(id=99, content="cand", trigger="t-cand", rationale="rc")
    decision = IndependentDecision(new_id="NEW-0")

    result = apply_consolidation(book, candidate, decision, existing_order=[r1, r2])

    assert result == [
        r1,
        r2,
        BookRule(id=3, content="cand", trigger="t-cand", rationale="rc"),
    ]


def test_reflection_new_content_updates_cited_rule() -> None:
    r1, r2 = _seed()
    book = [r1, r2]
    decision = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_content="use tabs",
    )

    result = apply_reflection(book, cited_id=1, decision=decision)

    assert result == [
        BookRule(id=1, content="use tabs", trigger="formatting", rationale="r1"),
        r2,
    ]


def test_reflection_new_trigger_updates_cited_rule() -> None:
    r1, r2 = _seed()
    book = [r1, r2]
    decision = ReflectionDecision(
        target_kind="playbook",
        target_id="1",
        new_trigger="python formatting",
    )

    result = apply_reflection(book, cited_id=1, decision=decision)

    assert result == [
        BookRule(
            id=1,
            content="use spaces",
            trigger="python formatting",
            rationale="r1",
        ),
        r2,
    ]


def test_reflection_no_change_is_noop() -> None:
    r1, r2 = _seed()
    book = [r1, r2]
    decision = ReflectionDecision(target_kind="playbook", target_id="1")

    result = apply_reflection(book, cited_id=1, decision=decision)

    assert result == [r1, r2]
