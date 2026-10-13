"""Protected context never silently rewrites or invents original evidence."""

from typing import Literal

import pytest

from reflexio.server.services.extraction.input_preparation import (
    ExtractionInputBudgetExceededError,
    PreparedChunk,
    PreparedExtractionInput,
    SourceSpan,
)
from reflexio.server.services.extraction.protected_pass import (
    _merge_candidate_evidence,
    _protected_validator,
    build_protected_record,
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
        len("Permission denied again."),
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


def test_shared_protected_quote_ambiguity_repairs_before_receipt_freeze():
    source = span(1, "First: same. Second: same.")
    prepared = PreparedExtractionInput("fixture", (PreparedChunk(0, (source,)),))
    validator = _protected_validator(ProtectedRecord(), (source,), prepared=prepared)
    assert "one physical occurrence" in validator(update(source, quote="same"))[0]
    assert validator(update(source, quote="First: same.")) == []
    exact = update(source, quote="same")
    start = source.text.index("same")
    exact.evidence[0] = exact.evidence[0].model_copy(
        update={"evidence_refs": (f"i1:content:{start}:{start + 4}",)}
    )
    assert validator(exact) == []
    # Adopted candidate carriers preserve their existing coarse attribution.
    assert validator(candidate_update(source, quote="same"))
    candidate_validator = _protected_validator(
        ProtectedRecord(), (source,), {"C1": {}}, {}, prepared=prepared
    )
    assert candidate_validator(candidate_update(source, quote="same")) == []


def test_preexisting_ambiguous_shared_record_holds_before_a_new_repair_call():
    from reflexio.server.services.extraction.overflow_presentation import (
        ProtectedQuoteAmbiguityError,
    )

    source = span(1, "same same")
    prepared = PreparedExtractionInput("fixture", (PreparedChunk(0, (source,)),))
    frozen = ProtectedRecord().merge(
        update(source, quote="same"), visible_spans=(source,)
    )
    with pytest.raises(ProtectedQuoteAmbiguityError):
        _protected_validator(frozen, (source,), prepared=prepared)
    assert frozen.evidence[0].quote == "same"


def _shared_quote_view_capacity_fixture():
    from reflexio.server.services.extraction.overflow_presentation import (
        protected_quote_view,
    )

    quotes = [f"<specific-quotation-{index:03}>" for index in range(128)]
    source = span(1, "x" * 10000 + " ".join(quotes))
    prepared = PreparedExtractionInput("fixture", (PreparedChunk(0, (source,)),))
    for count in range(1, 129):
        proposed = ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(source.evidence_ref,),
                    quote=quote,
                )
                for quote in quotes[:count]
            ]
        )
        try:
            merged = ProtectedRecord().merge(proposed, visible_spans=(source,))
        except ExtractionInputBudgetExceededError:
            break
        try:
            protected_quote_view(prepared, merged)
        except ExtractionInputBudgetExceededError:
            return source, prepared, proposed
    pytest.fail("Fixture must exceed shown capacity while the stored record fits")


def test_shared_quote_view_capacity_is_terminal_inside_structured_validation():
    from reflexio.server.llm.input_budget import InputBudgetExceededError

    source, prepared, proposed = _shared_quote_view_capacity_fixture()
    record = ProtectedRecord()
    with pytest.raises(InputBudgetExceededError, match="record capacity"):
        _protected_validator(record, (source,), prepared=prepared)(proposed)
    assert record.evidence == ()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("invalid", ["ambiguity", "capacity"])
def test_invalid_shared_quote_view_never_freezes_a_receipt_or_checkpoint(
    cached, invalid
):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from reflexio.server.services.extraction.overflow_presentation import (
        ProtectedQuoteAmbiguityError,
    )

    if invalid == "capacity":
        source, prepared, proposed = _shared_quote_view_capacity_fixture()
        expected = ExtractionInputBudgetExceededError
    else:
        source = span(1, "same same")
        prepared = PreparedExtractionInput("fixture", (PreparedChunk(0, (source,)),))
        proposed = update(source, quote="same")
        expected = ProtectedQuoteAmbiguityError
    coordinator = MagicMock()
    coordinator.source_digest = prepared.source_digest
    coordinator.provider_scope.return_value = nullcontext()
    coordinator.completed_step.return_value = (
        {"output": proposed.model_dump(mode="json")} if cached else None
    )
    client = MagicMock()
    client.generate_chat_response_with_provenance.return_value = SimpleNamespace(
        value=proposed, provenance=None
    )
    with pytest.raises(expected):
        build_protected_record(
            coordinator=coordinator,
            prepared=prepared,
            client=client,
            extraction_criteria="Keep explicit approvals.",
            agent_context="Fixture",
            minimum_following_attempts=1,
        )
    coordinator.complete_step.assert_not_called()
    coordinator.save_checkpoint.assert_not_called()
    if cached:
        client.generate_chat_response_with_provenance.assert_not_called()
        coordinator.require_remaining_work.assert_not_called()
