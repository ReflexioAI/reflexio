"""Schema tests for PlaybookConsolidationOutput as a 5-kind discriminated union.

These tests pin down the shape of the consolidation LLM output schema introduced
in Task E2: a list of decisions, each tagged by a ``kind`` literal that selects
between five concrete decision types (``duplicate``, ``prefer_new``,
``prefer_existing``, ``differentiate``, ``independent``).
"""

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.playbook_consolidator import (
    ConsolidationDecision,
    DifferentiateDecision,
    DuplicateDecision,
    IndependentDecision,
    PlaybookConsolidationOutput,
    PreferExistingDecision,
    PreferNewDecision,
)


def test_decision_kinds_are_discriminated_union():
    pn = PreferNewDecision(new_id="NEW-0", existing_id=42, reason="r")
    out = PlaybookConsolidationOutput(decisions=[pn])
    assert out.decisions[0].kind == "prefer_new"


def test_differentiate_requires_both_refined_triggers():
    with pytest.raises(ValidationError):
        DifferentiateDecision(
            new_id="NEW-0",
            existing_id=42,
            refined_new_trigger="",  # empty
            refined_existing_trigger="when narrow",
        )


def test_duplicate_requires_polarity():
    with pytest.raises(ValidationError):
        DuplicateDecision(
            item_ids=["NEW-0", "EXISTING-1"],
            merged_content="X",
            merged_trigger="t",
            merged_rationale="r",
            # missing merged_polarity
        )  # type: ignore[call-arg]


def test_all_five_kinds_round_trip_through_output():
    decisions: list[ConsolidationDecision] = [
        DuplicateDecision(
            item_ids=["NEW-0", "EXISTING-1"],
            merged_content="X",
            merged_trigger="t",
            merged_rationale="r",
            merged_polarity="positive",
        ),
        PreferNewDecision(new_id="NEW-1", existing_id=2),
        PreferExistingDecision(new_id="NEW-2", existing_id=3),
        DifferentiateDecision(
            new_id="NEW-3",
            existing_id=4,
            refined_new_trigger="when A and B",
            refined_existing_trigger="when A and not B",
        ),
        IndependentDecision(new_id="NEW-4"),
    ]
    out = PlaybookConsolidationOutput(decisions=decisions)
    kinds = [d.kind for d in out.decisions]
    assert kinds == [
        "duplicate",
        "prefer_new",
        "prefer_existing",
        "differentiate",
        "independent",
    ]
