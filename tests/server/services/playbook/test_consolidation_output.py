"""Schema tests for PlaybookConsolidationOutput as a 4-kind discriminated union.

These tests pin down the shape of the consolidation LLM output schema after the
4-kind redesign: ``unify`` (subsumes the legacy ``duplicate`` and
``prefer_new``), ``reject_new`` (renamed from ``prefer_existing``),
``differentiate``, and ``independent``. The legacy 5-kind names must no
longer parse — that structural guarantee is what the new schema buys.
"""

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.playbook_consolidator import (
    ConsolidationDecision,
    DifferentiateDecision,
    IndependentDecision,
    PlaybookConsolidationOutput,
    RejectNewDecision,
    UnifyDecision,
)


def test_decision_kinds_are_discriminated_union():
    """A bare ``UnifyDecision`` round-trips through the output schema."""
    unify = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=[1],
        content="Recommend X.",
        trigger="when Y",
        rationale="r",
    )
    out = PlaybookConsolidationOutput(decisions=[unify])
    assert out.decisions[0].kind == "unify"


def test_differentiate_requires_both_refined_triggers():
    """Empty refined triggers must fail validation."""
    with pytest.raises(ValidationError):
        DifferentiateDecision(
            new_id="NEW-0",
            existing_id=42,
            refined_new_trigger="",  # empty
            refined_existing_trigger="when narrow",
        )


def test_unify_has_no_polarity_field():
    """``UnifyDecision`` no longer declares a polarity field (wording-derived)."""
    # Parses fine without any polarity; orientation is inferred from wording at
    # apply time, not declared on the decision.
    unify = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=[1],
        content="X",
        trigger="t",
        rationale="r",
    )
    assert "polarity" not in UnifyDecision.model_fields
    assert "polarity" not in unify.model_dump()


def test_unify_accepts_empty_archive_existing_ids():
    """``unify`` with an empty archive list is a valid insert-without-archive."""
    unify = UnifyDecision(
        new_id="NEW-0",
        archive_existing_ids=[],
        content="X",
        trigger="t",
        rationale="r",
    )
    assert unify.archive_existing_ids == []


def test_reject_new_requires_superseded_existing_id():
    """``RejectNewDecision`` must name the superseding existing id."""
    with pytest.raises(ValidationError):
        RejectNewDecision(new_id="NEW-0")  # type: ignore[call-arg]


def test_all_four_kinds_round_trip_through_output():
    """All four kinds parse via the discriminated union in one batch."""
    decisions: list[ConsolidationDecision] = [
        UnifyDecision(
            new_id="NEW-0",
            archive_existing_ids=[1],
            content="X",
            trigger="t",
            rationale="r",
        ),
        RejectNewDecision(new_id="NEW-1", superseded_by_existing_id=2),
        DifferentiateDecision(
            new_id="NEW-2",
            existing_id=4,
            refined_new_trigger="when A and B",
            refined_existing_trigger="when A and not B",
        ),
        IndependentDecision(new_id="NEW-3"),
    ]
    out = PlaybookConsolidationOutput(decisions=decisions)
    kinds = [d.kind for d in out.decisions]
    assert kinds == [
        "unify",
        "reject_new",
        "differentiate",
        "independent",
    ]


@pytest.mark.parametrize(
    "legacy_payload",
    [
        # Legacy ``duplicate`` shape — subsumed by ``unify``.
        {
            "kind": "duplicate",
            "item_ids": ["NEW-0", "EXISTING-1"],
            "merged_content": "X",
            "merged_trigger": "t",
            "merged_rationale": "r",
            "merged_polarity": "positive",
        },
        # Legacy ``prefer_new`` shape — subsumed by ``unify``.
        {"kind": "prefer_new", "new_id": "NEW-0", "existing_id": 1},
        # Legacy ``prefer_existing`` shape — renamed to ``reject_new``.
        {"kind": "prefer_existing", "new_id": "NEW-0", "existing_id": 1},
    ],
)
def test_legacy_kind_literals_no_longer_parse(legacy_payload):
    """Old 5-kind discriminator values must fail to parse under the 4-kind union.

    This is the structural guarantee that the 4-kind redesign buys: callers
    that still produce the old shapes will fail loudly at decode time rather
    than silently misroute decisions.
    """
    with pytest.raises(ValidationError):
        PlaybookConsolidationOutput.model_validate({"decisions": [legacy_payload]})
