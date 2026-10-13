"""Protected context never silently rewrites or invents original evidence."""

from typing import Any, Literal

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
    _selection_spans,
    _selection_update,
    _selection_validator,
    build_protected_record,
)
from reflexio.server.services.extraction.protected_record import (
    CandidateEvidenceQuote,
    CandidateEvidenceSelection,
    EvidenceSelection,
    ProtectedEvidence,
    ProtectedRecord,
    ProtectedRecordSelection,
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


def test_protected_repair_identifies_rephrased_and_unescaped_entries_without_source_text():
    import json

    instruction = span(1, "Prepare only; do not deploy.")
    tool = span(2, json.dumps({"output": "first line\nsecond line"}))
    valid = update(instruction)
    valid.evidence.append(
        ProtectedEvidence(
            kind="instruction",
            evidence_refs=(instruction.evidence_ref,),
            quote="Prepare only and never deploy.",
        )
    )
    valid.candidate_evidence.append(
        CandidateEvidenceQuote(
            candidate_ref="C1",
            kind="outcome",
            evidence_refs=(tool.evidence_ref,),
            quote="first line\nsecond line",
        )
    )
    validator = _protected_validator(
        ProtectedRecord(), (instruction, tool), {"C1": {}}, {}
    )
    errors = validator(valid)
    assert len(errors) == 2
    assert errors[0].startswith("evidence[1]:")
    assert errors[1].startswith("candidate_evidence[0]:")
    assert "inner-JSON escaping" in errors[1]
    assert all(
        instruction.text not in error and tool.text not in error for error in errors
    )
    valid.evidence[1] = valid.evidence[1].model_copy(update={"quote": instruction.text})
    valid.candidate_evidence[0] = valid.candidate_evidence[0].model_copy(
        update={"quote": "first line\\nsecond line"}
    )
    assert validator(valid) == []


def test_protected_repair_feedback_is_bounded_and_unknown_candidate_remains_rejected():
    source = span(1, "Exact original evidence.")
    proposed = ProtectedRecordUpdate(
        evidence=[
            ProtectedEvidence(
                kind="instruction",
                evidence_refs=(source.evidence_ref,),
                quote="Invented",
            )
            for _ in range(12)
        ]
    )
    validator = _protected_validator(ProtectedRecord(), (source,), {"C1": {}}, {})
    errors = validator(proposed)
    assert len(errors) == 8
    assert errors[-1].startswith("evidence[7]:")
    unknown = candidate_update(source, "C2")
    assert validator(unknown) == [
        "candidate_evidence[0].candidate_ref: use a provided claim ID."
    ]


def test_reference_selection_preserves_every_character_and_copies_escaped_source():
    source = span(
        1, 'Spelling stays as-is: correciton. {"output":"first\\nsecond"} 中文 ' * 40
    )
    pieces = _selection_spans((source,))
    assert "".join(piece.text for piece in pieces) == source.text
    assert pieces[0].start == source.start and pieces[-1].end == source.end
    assert all(
        left.end == right.start
        for left, right in zip(pieces[:-1], pieces[1:], strict=True)
    )
    assert all(len(piece.text) <= 1024 for piece in pieces)
    selected = ProtectedRecordSelection(
        evidence=[
            EvidenceSelection(
                kind="correction",
                evidence_refs=(pieces[0].evidence_ref, pieces[1].evidence_ref),
            )
        ]
    )
    copied = _selection_update(selected, pieces)
    assert [item.quote for item in copied.evidence] == [
        piece.text for piece in pieces[:2]
    ]
    assert _protected_validator(ProtectedRecord(), pieces)(copied) == []
    selected.evidence[0] = selected.evidence[0].model_copy(
        update={"evidence_refs": ("i1:content:0:1025",)}
    )
    with pytest.raises(ValueError, match="visible snippet"):
        _selection_update(selected, pieces)


@pytest.mark.parametrize("invalid", ["unknown", "capacity"])
def test_reference_selection_failure_cannot_freeze_receipts_or_checkpoint(invalid):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    source = span(1, " x" * 5000)
    prepared = PreparedExtractionInput("fixture", (PreparedChunk(0, (source,)),))
    pieces = _selection_spans((source,))
    refs = (
        ("i999:content:0:10",)
        if invalid == "unknown"
        else tuple(piece.evidence_ref for piece in pieces[:8])
    )
    selection = ProtectedRecordSelection(
        evidence=[EvidenceSelection(kind="instruction", evidence_refs=refs)]
    )
    coordinator = MagicMock()
    coordinator.source_digest = prepared.source_digest
    coordinator.provider_scope.return_value = nullcontext()
    coordinator.completed_step.return_value = None
    client = MagicMock()
    client.generate_chat_response_with_provenance.return_value = SimpleNamespace(
        value=selection, provenance=None
    )
    expected = (
        ExtractionInputBudgetExceededError if invalid == "capacity" else ValueError
    )
    with pytest.raises(expected):
        build_protected_record(
            coordinator=coordinator,
            prepared=prepared,
            client=client,
            extraction_criteria="Preserve constraints",
            agent_context="",
            minimum_following_attempts=1,
        )
    coordinator.complete_step.assert_not_called()
    coordinator.save_checkpoint.assert_not_called()


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
    if not cached:
        # A quote-shaped value is no longer the live provider response contract.
        # Invalid legacy receipts still take the precise ambiguity/capacity path.
        expected = ValueError
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


@pytest.mark.parametrize("receipt", ["live", "v1", "v2"])
def test_source_selection_receipts_copy_original_and_recover_without_dispatch(receipt):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    source = span(1, 'Keep exact spelling and JSON: {"output":"a\\n b"}.')
    prepared = PreparedExtractionInput("fixture", (PreparedChunk(0, (source,)),))
    coordinator = MagicMock()
    coordinator.source_digest = prepared.source_digest
    coordinator.provider_scope.return_value = nullcontext()
    coordinator.checkpoint = {}
    saved = {"output": update(source).model_dump(mode="json")}
    coordinator.completed_step.side_effect = (
        [saved] if receipt == "v1" else [None, saved if receipt == "v2" else None]
    )
    client = MagicMock()
    client.generate_chat_response_with_provenance.return_value = SimpleNamespace(
        value=ProtectedRecordSelection(
            evidence=[
                EvidenceSelection(
                    kind="correction", evidence_refs=(source.evidence_ref,)
                )
            ]
        ),
        provenance=None,
    )
    result = build_protected_record(
        coordinator=coordinator,
        prepared=prepared,
        client=client,
        extraction_criteria="Preserve constraints",
        agent_context="",
        minimum_following_attempts=1,
    )
    assert result.evidence[0].quote == source.text
    assert result.evidence[0].evidence_refs == (source.evidence_ref,)
    coordinator.save_checkpoint.assert_called_once()
    if receipt == "live":
        coordinator.complete_step.assert_called_once()
        stored = coordinator.complete_step.call_args.args[1]
        assert stored["evidence"][0]["quote"] == source.text
        assert (
            client.generate_chat_response_with_provenance.call_args.kwargs[
                "response_format"
            ]
            is ProtectedRecordSelection
        )
    else:
        client.generate_chat_response_with_provenance.assert_not_called()
        coordinator.complete_step.assert_not_called()


def test_selected_tool_fields_never_become_a_stitched_quotation():
    first = SourceSpan(1, "r", "Assistant", "tools_used[0]", 0, 6, "failed")
    second = SourceSpan(2, "r", "Assistant", "tools_used[1]", 0, 9, "succeeded")
    selected = ProtectedRecordSelection(
        evidence=[
            EvidenceSelection(
                kind="outcome",
                evidence_refs=(first.evidence_ref, second.evidence_ref),
            )
        ]
    )
    copied = _selection_update(selected, (first, second))
    assert [(item.quote, item.evidence_refs) for item in copied.evidence] == [
        (first.text, (first.evidence_ref,)),
        (second.text, (second.evidence_ref,)),
    ]
    assert _protected_validator(ProtectedRecord(), (first, second))(copied) == []


@pytest.mark.parametrize("unshown", [False, True])
def test_reused_shared_evidence_copies_only_shown_quote_not_surrounding_source(unshown):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    quotation = "Do not deploy yet."
    source = span(1, quotation + " Unshown private surrounding text.")
    later = span(2, "Continue preparation.")
    prepared = PreparedExtractionInput(
        "fixture",
        (
            PreparedChunk(0, (source,)),
            PreparedChunk(1, (later,)),
        ),
    )
    coordinator = MagicMock()
    coordinator.source_digest = prepared.source_digest
    coordinator.provider_scope.return_value = nullcontext()
    coordinator.checkpoint = {}
    coordinator.completed_step.side_effect = [
        {"output": update(source, quote=quotation).model_dump(mode="json")},
        None,
        None,
    ]
    client = MagicMock()
    ref = source.evidence_ref if unshown else f"i1:content:0:{len(quotation)}"
    client.generate_chat_response_with_provenance.return_value = SimpleNamespace(
        value=ProtectedRecordSelection(
            candidate_evidence=[
                CandidateEvidenceSelection(
                    candidate_ref="C1", kind="correction", evidence_refs=(ref,)
                ),
            ]
        ),
        provenance=None,
    )
    kwargs: dict[str, Any] = {
        "coordinator": coordinator,
        "prepared": prepared,
        "client": client,
        "extraction_criteria": "Approval scope",
        "agent_context": "",
        "minimum_following_attempts": 1,
        "candidates": {"C1": {"content": "Prepare only"}},
    }
    if unshown:
        with pytest.raises(ValueError, match="visible snippet"):
            build_protected_record(**kwargs)
        coordinator.complete_step.assert_not_called()
        coordinator.save_checkpoint.assert_not_called()
    else:
        build_protected_record(**kwargs)
        stored = coordinator.complete_step.call_args.args[1]["candidate_evidence"][0]
        assert stored["quote"] == quotation
        assert stored["evidence_refs"] == [ref]


def test_oversized_selection_gets_repair_feedback_without_accepting_or_evicting():
    source = span(1, " x" * 5000)
    pieces = _selection_spans((source,))
    selected = ProtectedRecordSelection(
        evidence=[
            EvidenceSelection(
                kind="instruction",
                evidence_refs=tuple(s.evidence_ref for s in pieces[:8]),
            )
        ]
    )
    record = ProtectedRecord()
    validator = _selection_validator(pieces, _protected_validator(record, pieces))
    errors = validator(selected)
    assert len(errors) == 1 and "2048-token" in errors[0]
    assert record.evidence == ()
    corrected = ProtectedRecordSelection(
        evidence=[
            EvidenceSelection(
                kind="instruction",
                evidence_refs=(pieces[0].evidence_ref,),
            )
        ]
    )
    assert validator(corrected) == []


def test_reselecting_shown_shared_quote_cannot_fill_record_again():
    source = span(1, "Do not deploy yet.")
    existing = ProtectedRecord().merge(update(source), visible_spans=(source,))
    selected = ProtectedRecordSelection(
        evidence=[
            EvidenceSelection(
                kind="uncertainty",
                evidence_refs=(source.evidence_ref,),
            )
        ],
        candidate_evidence=[
            CandidateEvidenceSelection(
                candidate_ref="C1",
                kind="instruction",
                evidence_refs=(source.evidence_ref,),
            )
        ],
    )
    copied = _selection_update(selected, (source,), existing)
    assert copied.evidence == []
    assert copied.candidate_evidence[0].quote == source.text
    assert existing.evidence[0].quote == source.text


def test_expanded_selection_limit_feedback_never_copies_source_content():
    source = span(1, "PRIVATE-SOURCE-MUST-NOT-ENTER-REPAIR-FEEDBACK")
    selected = ProtectedRecordSelection(
        evidence=[
            EvidenceSelection(
                kind="instruction",
                evidence_refs=(source.evidence_ref,) * 8,
            )
        ]
        * 17
    )
    validator = _selection_validator(
        (source,), _protected_validator(ProtectedRecord(), (source,))
    )
    assert validator(selected) == [
        "Select at most 128 individual snippets per evidence field."
    ]


def test_contained_selection_copies_original_range_and_rejects_unshown_tail():
    source = SourceSpan(1, "r", "User", "content", 20, 38, "Do not deploy yet.")
    selected = ProtectedRecordSelection(
        evidence=[
            EvidenceSelection(
                kind="instruction",
                evidence_refs=("i1:content:20:33",),
            )
        ]
    )
    copied = _selection_update(selected, (source,))
    assert copied.evidence[0].quote == source.text[:13]
    assert copied.evidence[0].evidence_refs == ("i1:content:20:33",)
    selected.evidence[0] = selected.evidence[0].model_copy(
        update={"evidence_refs": ("i1:content:20:39",)},
    )
    with pytest.raises(ValueError, match="wholly contained"):
        _selection_update(selected, (source,))


def test_offsetless_field_id_requires_one_visible_snippet_and_keeps_canonical_offsets():
    source = SourceSpan(1, "r", "Assistant", "tools_used[0]", 20, 26, "failed")
    selected = ProtectedRecordSelection(
        evidence=[
            EvidenceSelection(
                kind="outcome",
                evidence_refs=("i1:tools_used[0]",),
            )
        ]
    )
    copied = _selection_update(selected, (source,))
    assert copied.evidence[0].quote == source.text
    assert copied.evidence[0].evidence_refs == (source.evidence_ref,)
    other = SourceSpan(1, "r", "Assistant", "tools_used[0]", 30, 39, "succeeded")
    with pytest.raises(ValueError, match="visible snippet"):
        _selection_update(selected, (source, other))
