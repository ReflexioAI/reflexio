from __future__ import annotations

import pytest

from reflexio.server.services.offline_tuner.open_world.citations import (
    CitationVerificationError,
    verify_citations,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    EvidenceCitation,
    EvidenceSpan,
)


def _span(
    *,
    span_id: str = "span-1",
    text: str = "the exact quote",
    session_id: str = "session-1",
) -> EvidenceSpan:
    return EvidenceSpan(
        span_id=("a" * 64 + ":" + "b" * 64 if span_id == "span-1" else span_id),
        tenant_id="tenant-1",
        manifest_role="held_out",
        user_id="user-1",
        session_id=session_id,
        request_id="request-1",
        interaction_id=1,
        field_path="requests[0].interactions[0].content",
        immutable_text=text,
        speaker="agent",
    )


def test_verify_citations_derives_exact_unique_quote_offsets() -> None:
    verified = verify_citations(
        (EvidenceCitation(span_id="a" * 64 + ":" + "b" * 64, exact_quote="exact"),),
        allowed_spans=(_span(),),
    )

    assert len(verified[0].citation_id) == 64
    assert verified[0].start_offset == 4
    assert verified[0].end_offset == 9
    assert verified[0].session_id == "session-1"


@pytest.mark.parametrize(
    ("citation", "spans", "category"),
    [
        # A quote present in NO allowed span: the span ID named nothing, so the
        # primary defect is still reported as `unavailable_span`.
        (
            EvidenceCitation(span_id="fabricated", exact_quote="fabricated text"),
            (_span(),),
            "unavailable_span",
        ),
        (
            EvidenceCitation(span_id="a" * 64 + ":" + "b" * 64, exact_quote="absent"),
            (_span(),),
            "quote_absent",
        ),
        (
            EvidenceCitation(span_id="span-2", exact_quote="nowhere at all"),
            (_span(),),
            "unavailable_span",
        ),
        # The quote resolves to two DIFFERENT allowed spans, so it is not a
        # pointer and must not be silently attributed to whichever came first.
        (
            EvidenceCitation(span_id="mistyped", exact_quote="shared"),
            (
                _span(text="shared text one"),
                _span(span_id="c" * 64 + ":" + "d" * 64, text="shared text two"),
            ),
            "ambiguous_quote",
        ),
    ],
)
def test_verify_citations_rejects_unavailable_or_ambiguous_evidence(
    citation: EvidenceCitation, spans: tuple[EvidenceSpan, ...], category: str
) -> None:
    with pytest.raises(CitationVerificationError) as error:
        verify_citations((citation,), allowed_spans=spans)
    assert error.value.category == category


def test_a_quote_repeated_inside_one_span_resolves_to_its_first_occurrence() -> None:
    """A repeated phrase is citable, because repetition changes no attribution.

    The discovery prompt instructs the analyst to "prefer the SHORTEST run of
    text that carries the claim", and short agent boilerplate is exactly what
    recurs inside one turn. Refusing it as `ambiguous_quote` -- which is what
    this module did -- cost the whole memo for following that instruction.

    What makes it safe is that the two occurrences are not two attributions:
    every field a `VerifiedCitation` carries is identical between them, the
    credited text is the caller's own bytes either way, and the offsets chosen
    are a pure function of the server's text.
    """
    span = _span(text="exact and exact")

    verified = verify_citations(
        (EvidenceCitation(span_id=span.span_id, exact_quote="exact"),),
        allowed_spans=(span,),
    )

    assert len(verified) == 1
    assert verified[0].span_id == span.span_id
    assert (verified[0].start_offset, verified[0].end_offset) == (0, 5)
    # The credited text is the CALLER's slice, not the model's string.
    assert (
        span.immutable_text[verified[0].start_offset : verified[0].end_offset]
        == verified[0].exact_quote
    )


def test_repeated_quote_resolution_is_deterministic_not_arbitrary() -> None:
    """The choice must be the server's, and must be the same one every time.

    An arbitrary pick would make `citation_id` unstable for one citation, which
    is the only thing that made offset-level uniqueness worth having. First
    occurrence is a pure function of `(span, quote)`.
    """
    span = _span(text="echo one echo two echo three")
    citation = EvidenceCitation(span_id=span.span_id, exact_quote="echo")

    first = verify_citations((citation,), allowed_spans=(span,))
    second = verify_citations((citation,), allowed_spans=(span,))

    assert first[0].start_offset == 0
    assert first == second


def test_a_repeated_quote_cited_twice_is_a_duplicate_not_an_ambiguity() -> None:
    """Both citations resolve to one occurrence, so the second adds nothing.

    `duplicate_citation` is the accurate category: nothing downstream could
    have told the two occurrences apart, so the second citation carries no
    evidence the first did not.
    """
    span = _span(text="exact and exact")

    with pytest.raises(CitationVerificationError) as error:
        verify_citations(
            (
                EvidenceCitation(span_id=span.span_id, exact_quote="exact"),
                EvidenceCitation(span_id="mistyped", exact_quote="exact"),
            ),
            allowed_spans=(span,),
        )

    assert error.value.category == "duplicate_citation"


def test_a_quote_in_two_spans_stays_ambiguous_however_often_it_repeats() -> None:
    """Span-level uniqueness is what was kept, and it is the load-bearing half.

    Two spans differ in speaker, user, session and trajectory digest, so
    choosing between them invents provenance. Repetition WITHIN each of them
    must not launder that into a resolution.
    """
    spans = (
        _span(text="echo and echo"),
        _span(
            span_id="c" * 64 + ":" + "d" * 64,
            text="echo again and echo",
            session_id="session-2",
        ),
    )

    with pytest.raises(CitationVerificationError) as error:
        verify_citations(
            (EvidenceCitation(span_id="mistyped", exact_quote="echo"),),
            allowed_spans=spans,
        )

    assert error.value.category == "ambiguous_quote"


def test_mistyped_span_id_resolves_on_its_quote_instead_of_abstaining() -> None:
    """Defect 1: a mistyped span ID must not cost a correct hypothesis.

    A live MiniMax run quoted the evidence character-perfectly and mistyped one
    hex character of the 129-character composite ID. ID-only verification
    rejected it, and the caller rewrote the memo to `abstain` -- reporting a
    mechanical transcription slip as an analyst judgement.
    """
    span = _span()
    mistyped = span.span_id[:-1] + ("c" if span.span_id[-1] != "c" else "d")
    assert mistyped != span.span_id
    assert len(mistyped) == len(span.span_id)

    verified = verify_citations(
        (EvidenceCitation(span_id=mistyped, exact_quote="exact"),),
        allowed_spans=(span,),
    )

    # The citation resolves, and to the REAL span -- never to the mistyped ID.
    assert len(verified) == 1
    assert verified[0].span_id == span.span_id
    assert verified[0].span_id != mistyped
    assert verified[0].start_offset == 4
    assert verified[0].end_offset == 9


def test_quote_resolution_cannot_credit_a_fabricated_quote() -> None:
    """The repair must not become a way to forge a citation.

    Resolution moves which coordinate is authoritative; it does not stop
    requiring that the cited text be text the CALLER supplied. A model that
    invents the quote as well as the ID is still rejected.
    """
    with pytest.raises(CitationVerificationError) as error:
        verify_citations(
            (
                EvidenceCitation(
                    span_id="a" * 64 + ":" + "e" * 64,
                    exact_quote="text no allowed span ever contained",
                ),
            ),
            allowed_spans=(_span(),),
        )
    assert error.value.category == "unavailable_span"


def test_two_mistyped_ids_quoting_one_span_are_still_duplicates() -> None:
    """Deduplication has to key on the RESOLVED span, not the emitted ID.

    Two differently-mistyped IDs carrying the same quote resolve to the same
    span. Keyed on the emitted ID they would look distinct and be counted
    twice against one piece of evidence.
    """
    span = _span()
    with pytest.raises(CitationVerificationError, match="duplicate") as error:
        verify_citations(
            (
                EvidenceCitation(span_id="mistyped-one", exact_quote="exact"),
                EvidenceCitation(span_id="mistyped-two", exact_quote="exact"),
            ),
            allowed_spans=(span,),
        )
    assert error.value.category == "duplicate_citation"


def test_verify_citations_rejects_duplicate_citations() -> None:
    citation = EvidenceCitation(span_id="a" * 64 + ":" + "b" * 64, exact_quote="exact")

    with pytest.raises(CitationVerificationError, match="duplicate") as error:
        verify_citations((citation, citation), allowed_spans=(_span(),))
    assert error.value.category == "duplicate_citation"
