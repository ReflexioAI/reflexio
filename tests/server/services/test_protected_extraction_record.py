"""Protected context never silently rewrites or invents original evidence."""

from typing import Literal

import pytest

from reflexio.server.services.extraction.input_preparation import (
    ExtractionInputBudgetExceededError,
    SourceSpan,
)
from reflexio.server.services.extraction.protected_pass import (
    _merge_candidate_evidence,
    _protected_validator,
)
from reflexio.server.services.extraction.protected_record import (
    CandidateEvidenceQuote,
    ProtectedEvidence,
    ProtectedRecord,
    ProtectedRecordUpdate,
)


def span(interaction_id, text):
    return SourceSpan(interaction_id, "r", "User", "content", 0, len(text), text)


def update(
    source,
    kind: Literal[
        "instruction", "correction", "action", "outcome", "uncertainty"
    ] = "correction",
    quote=None,
):
    return ProtectedRecordUpdate(
        evidence=[
            ProtectedEvidence(
                kind=kind,
                evidence_refs=(source.evidence_ref,),
                quote=source.text if quote is None else quote,
            )
        ]
    )


def test_distant_contradictory_corrections_keep_both_original_sources():
    first = span(1, "Use staging for the next check.")
    later = span(20, "Correction: use the isolated local database for this check.")
    record = ProtectedRecord().merge(update(first), visible_spans=(first,))
    changed = record.merge(update(later), visible_spans=(later,))
    assert [item.quote for item in changed.evidence] == [first.text, later.text]
    assert record.evidence[0].quote == first.text
    assert len(record.evidence) == 1


def test_unsupported_summary_and_invented_reference_are_rejected():
    original = span(1, "The test failed; the fix has not been verified.")
    with pytest.raises(ValueError, match="original visible"):
        ProtectedRecord().merge(
            update(original, kind="outcome", quote="The fix passed."),
            visible_spans=(original,),
        )
    with pytest.raises(ValueError, match="original visible"):
        ProtectedRecord().merge(update(original), visible_spans=())


def test_record_budget_failure_preserves_existing_record():
    first = span(1, "Never modify customer records during this investigation.")
    second = span(2, "A necessary distinct constraint. " * 100)
    record = ProtectedRecord().merge(update(first), visible_spans=(first,))
    with pytest.raises(ExtractionInputBudgetExceededError, match="protected evidence"):
        record.merge(update(second), visible_spans=(second,), max_tokens=100)
    assert [item.quote for item in record.evidence] == [first.text]


def test_same_quote_from_distinct_occurrences_keeps_both_references():
    first = span(1, "Use read-only access.")
    second = span(2, first.text)
    record = ProtectedRecord().merge(update(first), visible_spans=(first,))
    assert record.merge(update(first), visible_spans=(first,)) == record
    repeated = record.merge(update(second), visible_spans=(second,))
    assert len(repeated.evidence) == 2
    assert repeated.evidence[0].evidence_refs != repeated.evidence[1].evidence_refs


def test_boundary_quote_requires_contiguous_spans_and_preserves_author_role():
    first = SourceSpan(1, "r", "Assistant", "content", 0, 13, "The assistant")
    second = SourceSpan(
        1, "r", "Assistant", "content", 13, 36, " reported preparations."
    )
    item = ProtectedEvidence(
        kind="outcome",
        evidence_refs=(first.evidence_ref, second.evidence_ref),
        quote="assistant reported preparations.",
        role="User",
    )
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(evidence=[item]), visible_spans=(first, second)
    )
    assert record.evidence[0].role == "Assistant"
    unrelated = SourceSpan(2, "r", "User", "content", 13, 36, second.text)
    forged = item.model_copy(
        update={"evidence_refs": (first.evidence_ref, unrelated.evidence_ref)}
    )
    with pytest.raises(ValueError, match="contiguous"):
        ProtectedRecord().merge(
            ProtectedRecordUpdate(evidence=[forged]), visible_spans=(first, unrelated)
        )


def candidate_update(source, claim="C1", quote=None, **extra):
    return ProtectedRecordUpdate(
        candidate_evidence=[
            CandidateEvidenceQuote(
                candidate_ref=claim,
                kind="outcome",
                evidence_refs=(source.evidence_ref,),
                quote=source.text if quote is None else quote,
                **extra,
            )
        ]
    )


def test_candidate_evidence_keeps_adverse_sources_without_expanding_whole_turn():
    reported = SourceSpan(
        1, "r", "Assistant", "content", 0, 22, "The assistant says OK.", source_order=1
    )
    failure = SourceSpan(
        2,
        "r",
        "Tool",
        "tools_used[0]",
        0,
        25,
        "Permission denied again.",
        source_order=2,
    )
    claims = {"C1": {"content": "The operation succeeded."}}
    records = _merge_candidate_evidence(
        candidate_update(reported, role="User"), (reported,), claims, {}
    )
    expanded = _merge_candidate_evidence(
        candidate_update(failure), (failure,), claims, records
    )
    assert [item.quote for item in expanded["C1"].evidence] == [
        reported.text,
        failure.text,
    ]
    assert expanded["C1"].evidence[0].role == "Assistant"
    assert len(records["C1"].evidence) == 1


@pytest.mark.parametrize("claim,quote", [("C2", None), ("C1", "The operation passed.")])
def test_candidate_evidence_rejects_unknown_claim_or_invented_quote(claim, quote):
    source = span(1, "The operation failed.")
    validator = _protected_validator(
        ProtectedRecord(), (source,), {"C1": {"content": "A proposed lesson"}}, {}
    )
    assert validator(candidate_update(source, claim, quote))


def test_candidate_capacity_failure_cannot_evict_saved_support():
    source = span(1, "The action is approved only for local staging.")
    claims = {"C1": {"content": "Scoped approval"}}
    records = _merge_candidate_evidence(candidate_update(source), (source,), claims, {})
    large = span(2, "A distinct necessary qualification. " * 1000)
    with pytest.raises(ExtractionInputBudgetExceededError):
        _merge_candidate_evidence(candidate_update(large), (large,), claims, records)
    assert records["C1"].evidence[0].quote == source.text
