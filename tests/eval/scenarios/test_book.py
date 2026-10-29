"""Benchmark books apply all audited outputs and preserve untouched identities."""

import pytest

from reflexio.server.services.playbook.components.reconciliation import (
    Decisions,
    Reference,
)
from tests.eval.scenarios.book import _next_id, apply_consolidation
from tests.eval.scenarios.case import BookRule

REF = {"session_id": "source", "turn": 1}


def rule(idx, content):
    return BookRule(
        id=idx,
        content=content,
        trigger="deploying",
        evidence_refs=[Reference.model_validate(REF)],
    )


def action(kind="replace", outputs=1):
    data = {"kind": kind, "new_ids": ["NEW-0"], "reason": "supported"}
    if kind == "replace":
        data["existing_ids"] = ["EXISTING-0"]
    if kind == "remove":
        data["evidence_refs"] = [REF]
    else:
        data["outputs"] = [
            {
                "content": f"authored-{i}",
                "trigger": "deploying",
                "rationale": "cited",
                "evidence_refs": [REF],
            }
            for i in range(outputs)
        ]
    return data


@pytest.mark.parametrize("kind", ["create", "replace", "remove"])
def test_apply_uses_authored_outputs_and_keeps_unconsumed_ids(kind):
    book = [rule(10, "prior"), rule(20, "unrelated")]
    candidate = rule(21, "candidate")
    original = [row.model_dump() for row in book]
    result = apply_consolidation(
        book,
        candidate,
        Decisions.model_validate({"decisions": [action(kind)]}),
        existing_order=book,
    )
    assert [row.model_dump() for row in book] == original
    assert any(row.id == 20 and row.content == "unrelated" for row in result)
    assert len({row.id for row in result}) == len(result)
    if kind == "create":
        assert [row.id for row in result[:2]] == [10, 20]
        assert result[-1].content == "authored-0"
    elif kind == "replace":
        assert all(row.id != 10 for row in result)
        assert result[-1].content == "authored-0"
    else:
        assert result == book


def test_split_keeps_every_output_with_distinct_ids_and_evidence():
    book = [rule(10, "prior")]
    result = apply_consolidation(
        book,
        rule(11, "new"),
        Decisions.model_validate({"decisions": [action(outputs=2)]}),
        existing_order=book,
    )
    assert [row.content for row in result] == ["authored-0", "authored-1"]
    assert len({row.id for row in result}) == 2
    assert all(row.evidence_refs[0].session_id == "source" for row in result)


def test_existing_only_cleanup_after_new_removal_is_not_discarded():
    replace = action()
    replace["new_ids"] = []
    batch = Decisions.model_validate({"decisions": [action("remove"), replace]})
    book = [rule(10, "prior")]
    result = apply_consolidation(book, rule(11, "new"), batch, existing_order=book)
    assert len(result) == 1 and result[0].content == "authored-0"


def test_unknown_existing_id_and_changed_order_fail_closed():
    book = [rule(10, "prior")]
    invalid = action()
    invalid["existing_ids"] = ["EXISTING-99"]
    with pytest.raises(ValueError):
        apply_consolidation(
            book,
            rule(11, "new"),
            Decisions.model_validate({"decisions": [invalid]}),
            existing_order=book,
        )
    with pytest.raises(ValueError):
        apply_consolidation(
            book,
            rule(11, "new"),
            Decisions.model_validate({"decisions": [action()]}),
            existing_order=[],
        )


def test_next_id_does_not_reuse_holes():
    assert _next_id([]) == 1
    assert _next_id([rule(10, "prior"), rule(30, "other")]) == 31
