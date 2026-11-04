"""Historical citation codec integrity; retired inference repair loops are removed."""

from __future__ import annotations

import inspect
from typing import Literal

import pytest

from reflexio.server.services.offline_tuner.open_world.citations import (
    CitationVerificationError,
    citation_repair_request,
    classify_citations,
    verify_citations,
)
from reflexio.server.services.offline_tuner.open_world.identity import (
    repair_attempt_idempotency_key,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    CitationRepairRequest,
    EvidenceCitation,
)

_TEST_PROVIDER = "deterministic-fixture"
_TEST_MODEL = "fixture-model-2026-08-17"


# --------------------------------------------------------------------------
# The prompt contract
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# The repair loop
# --------------------------------------------------------------------------


def test_the_first_attempt_key_is_the_identity_key_unchanged() -> None:
    base = "a" * 64
    assert repair_attempt_idempotency_key(base_key=base, attempt=1) == base
    assert repair_attempt_idempotency_key(base_key=base, attempt=2) != base
    assert repair_attempt_idempotency_key(
        base_key=base, attempt=2
    ) != repair_attempt_idempotency_key(base_key=base, attempt=3)


# --------------------------------------------------------------------------
# The repair payload cannot widen the evidence
# --------------------------------------------------------------------------


def test_repair_request_cannot_carry_evidence() -> None:
    """The builder has no evidence in scope, so it cannot leak any.

    This is a signature-level guarantee rather than a filtering one: there is no
    argument through which a span, its text, or its offsets could reach the
    payload, so no caller can make it happen.
    """
    parameters = set(inspect.signature(citation_repair_request).parameters)
    assert parameters == {"attempt", "verdicts"}

    verdict_fields = set(inspect.signature(classify_citations).parameters)
    assert verdict_fields == {"citations", "allowed_spans"}


def test_the_repair_echoes_only_the_models_own_strings() -> None:
    citations = (
        EvidenceCitation(span_id="fabricated-span", exact_quote="a fabricated quote"),
    )
    verdicts = classify_citations(citations, allowed_spans=())
    repair = citation_repair_request(attempt=2, verdicts=verdicts)

    assert repair is not None
    assert [quote.exact_quote for quote in repair.rejected_quotes] == [
        "a fabricated quote"
    ]
    # Every reason is one of the fixed category strings, never tenant text.
    for quote in repair.rejected_quotes:
        assert "fabricated" not in quote.reason


def test_no_repair_is_built_when_nothing_was_rejected() -> None:
    assert citation_repair_request(attempt=2, verdicts=()) is None


def test_a_repair_attempt_must_be_the_second_or_later() -> None:
    with pytest.raises(ValueError, match="attempt"):
        CitationRepairRequest(
            attempt=1,
            rejected_quotes=(),
        )


# --------------------------------------------------------------------------
# The classifier agrees with the verifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["absent", "duplicate"])
def test_classification_agrees_with_verification(
    mode: Literal["absent", "duplicate"],
) -> None:
    """Classification must never accept what verification rejects, or vice versa.

    The classifier exists only to say WHICH citation failed. If it ever
    disagreed with the verifier, the repair feedback would describe a rejection
    that did not happen -- or miss one that did.
    """
    if mode == "absent":
        citations = (
            EvidenceCitation(span_id="nope", exact_quote="not in the evidence"),
        )
    else:
        citation = EvidenceCitation(span_id="nope", exact_quote="twice over")
        citations = (citation, citation)

    verdicts = classify_citations(citations, allowed_spans=())
    all_resolved = all(verdict.category is None for verdict in verdicts)

    try:
        verify_citations(citations, allowed_spans=())
    except CitationVerificationError:
        verification_succeeded = False
    else:
        verification_succeeded = True

    assert all_resolved is verification_succeeded
    assert verification_succeeded is False


def test_classification_reports_one_verdict_per_citation_in_order() -> None:
    citations = (
        EvidenceCitation(span_id="one", exact_quote="first"),
        EvidenceCitation(span_id="two", exact_quote="second"),
    )
    verdicts = classify_citations(citations, allowed_spans=())

    assert [verdict.index for verdict in verdicts] == [0, 1]
    assert [verdict.exact_quote for verdict in verdicts] == ["first", "second"]
    assert all(verdict.category is not None for verdict in verdicts)


def test_verification_is_still_all_or_nothing() -> None:
    """The repair must not have quietly become a way to drop a bad citation."""
    citations = (EvidenceCitation(span_id="nope", exact_quote="not in the evidence"),)
    with pytest.raises(CitationVerificationError):
        verify_citations(citations, allowed_spans=())

    # And a partial success is still a total failure.
    verdicts = classify_citations(citations, allowed_spans=())
    assert any(verdict.category is not None for verdict in verdicts)
    with pytest.raises(CitationVerificationError):
        verify_citations(citations, allowed_spans=())
