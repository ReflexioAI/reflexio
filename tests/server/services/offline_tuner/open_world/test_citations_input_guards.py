"""Input guards shared by ``verify_citations`` and ``classify_citations``.

The two functions must agree on every citation (``classify_citations`` is the
diagnostic twin of the verifier), so each guard is pinned on both. The guards
are server-defect refusals -- ``TypeError`` / ``ValueError`` -- deliberately not
``CitationVerificationError``, so a caller bug is never laundered into an
abstention the analyst is blamed for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from reflexio.server.services.offline_tuner.open_world.citations import (
    CitationVerificationError,
    classify_citations,
    verify_citations,
    verify_grounded_discovery_citations,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    EvidenceCitation,
    EvidenceSpan,
)

_SPAN_ID = f"{'a' * 64}:{'b' * 64}"
_OTHER_SPAN_ID = f"{'c' * 64}:{'d' * 64}"

_Checker = Callable[..., object]
_CHECKERS: tuple[_Checker, ...] = (verify_citations, classify_citations)


def _span(
    *,
    span_id: str = _SPAN_ID,
    text: str = "the exact quote",
    speaker: str = "agent",
) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=span_id,
        tenant_id="tenant-1",
        manifest_role="discovery",
        user_id="user-1",
        session_id="session-1",
        request_id="request-1",
        interaction_id=1,
        field_path="requests[0].interactions[0].content",
        immutable_text=text,
        speaker=speaker,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("check", _CHECKERS)
def test_allowed_spans_must_be_evidence_spans(check: _Checker) -> None:
    look_alike = {"span_id": _SPAN_ID, "immutable_text": "the exact quote"}
    with pytest.raises(TypeError, match="allowed_spans must contain EvidenceSpan"):
        check(
            (EvidenceCitation(span_id=_SPAN_ID, exact_quote="exact"),),
            allowed_spans=(look_alike,),
        )


@pytest.mark.parametrize("check", _CHECKERS)
def test_allowed_spans_must_have_unique_ids(check: _Checker) -> None:
    with pytest.raises(ValueError, match="duplicate span IDs"):
        check(
            (EvidenceCitation(span_id=_SPAN_ID, exact_quote="exact"),),
            allowed_spans=(_span(), _span(text="another text")),
        )


@pytest.mark.parametrize("check", _CHECKERS)
def test_citations_must_be_evidence_citations(check: _Checker) -> None:
    with pytest.raises(TypeError, match="citations must contain EvidenceCitation"):
        check(
            ({"span_id": _SPAN_ID, "exact_quote": "exact"},),
            allowed_spans=(_span(),),
        )


def test_an_empty_quote_cannot_be_credited_at_offset_zero() -> None:
    """``str.find("")`` is 0 against any text, so an empty needle would forge a
    citation at the start of the first span. The model rejects a blank quote,
    and this pins the verifier's own door for a value that bypassed it."""
    empty = EvidenceCitation.model_construct(span_id=_SPAN_ID, exact_quote="")

    with pytest.raises(CitationVerificationError) as raised:
        verify_citations((empty,), allowed_spans=(_span(),))
    assert raised.value.category == "quote_absent"

    (verdict,) = classify_citations((empty,), allowed_spans=(_span(),))
    assert verdict.category == "quote_absent"


def test_an_empty_quote_naming_no_span_reports_unavailable_span() -> None:
    empty = EvidenceCitation.model_construct(span_id="unknown", exact_quote="")

    with pytest.raises(CitationVerificationError) as raised:
        verify_citations((empty,), allowed_spans=(_span(),))
    assert raised.value.category == "unavailable_span"


def test_a_span_id_that_is_not_a_rehydrated_span_cannot_be_verified() -> None:
    """The trajectory digest is read out of the span ID; an ID without the
    ``<digest>:<digest>`` shape is a server defect, not a model error."""
    span = _span(span_id="span-1")

    with pytest.raises(ValueError, match="not a rehydrated evidence span"):
        verify_citations(
            (EvidenceCitation(span_id="span-1", exact_quote="exact"),),
            allowed_spans=(span,),
        )


@pytest.mark.parametrize(
    "span_id",
    [
        f"{'a' * 64}",
        f"{'a' * 64}:{'b' * 63}",
        f"{'A' * 64}:{'b' * 64}",
        f"{'a' * 64}:{'b' * 64}:{'c' * 64}",
        f"{'g' * 64}:{'b' * 64}",
    ],
)
def test_malformed_span_id_shapes_are_refused(span_id: str) -> None:
    span = _span(span_id=span_id)
    with pytest.raises(ValueError, match="not a rehydrated evidence span"):
        verify_citations(
            (EvidenceCitation(span_id=span_id, exact_quote="exact"),),
            allowed_spans=(span,),
        )


def test_verified_citation_carries_the_span_trajectory_digest() -> None:
    (verified,) = verify_citations(
        (EvidenceCitation(span_id=_SPAN_ID, exact_quote="exact"),),
        allowed_spans=(_span(),),
    )
    assert verified.trajectory_digest == "a" * 64


def test_classify_reports_an_emitted_duplicate_and_still_classifies_the_rest() -> None:
    citation = EvidenceCitation(span_id=_SPAN_ID, exact_quote="exact")
    other = EvidenceCitation(span_id=_SPAN_ID, exact_quote="quote")
    absent = EvidenceCitation(span_id=_SPAN_ID, exact_quote="missing")

    verdicts = classify_citations(
        (citation, citation, other, absent), allowed_spans=(_span(),)
    )

    assert [verdict.category for verdict in verdicts] == [
        None,
        "duplicate_citation",
        None,
        "quote_absent",
    ]
    assert [verdict.index for verdict in verdicts] == [0, 1, 2, 3]
    # Every verdict echoes the MODEL's own strings, nothing from the span.
    assert verdicts[3].exact_quote == "missing"
    # The verifier agrees: the same input fails, on the first defect.
    with pytest.raises(CitationVerificationError) as raised:
        verify_citations((citation, citation), allowed_spans=(_span(),))
    assert raised.value.category == "duplicate_citation"


def test_classify_reports_a_resolved_duplicate_from_two_different_ids() -> None:
    """Two differently mistyped IDs quoting one passage are distinct as
    EMITTED, but resolve to one passage; only the first may be credited."""
    first = EvidenceCitation(span_id="mistyped-1", exact_quote="exact")
    second = EvidenceCitation(span_id="mistyped-2", exact_quote="exact")

    verdicts = classify_citations((first, second), allowed_spans=(_span(),))

    assert [verdict.category for verdict in verdicts] == [None, "duplicate_citation"]
    assert [verdict.span_id for verdict in verdicts] == ["mistyped-1", "mistyped-2"]


def test_citation_id_changes_with_any_single_resolved_field() -> None:
    def citation_id(span: EvidenceSpan, quote: str) -> str:
        (verified,) = verify_citations(
            (EvidenceCitation(span_id=span.span_id, exact_quote=quote),),
            allowed_spans=(span,),
        )
        return verified.citation_id

    base = _span(text="the exact quote")
    baseline = citation_id(base, "exact")

    assert baseline == citation_id(base, "exact")
    assert baseline != citation_id(base, "quote")
    assert baseline != citation_id(replace(base, span_id=_OTHER_SPAN_ID), "exact")
    # Same quote, different offset in the span: a different citation.
    assert baseline != citation_id(replace(base, immutable_text="exact quote"), "exact")


def test_grounded_memo_citing_only_user_turns_is_refused() -> None:
    user_span = _span(speaker="user")
    citations = (EvidenceCitation(span_id=_SPAN_ID, exact_quote="exact"),)

    with pytest.raises(CitationVerificationError) as raised:
        verify_grounded_discovery_citations(
            citations, allowed_spans=(user_span,), decision="grounded_hypothesis"
        )
    assert raised.value.category == "no_agent_side_citation"
    # An abstention makes no agent-side claim, so the same citations verify.
    assert verify_grounded_discovery_citations(
        citations, allowed_spans=(user_span,), decision="abstain"
    ) == verify_citations(citations, allowed_spans=(user_span,))
