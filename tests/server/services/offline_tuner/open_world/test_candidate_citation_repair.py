"""Historical citation codec integrity; retired inference repair loops are removed."""

from __future__ import annotations

import inspect

import pytest

from reflexio.server.services.offline_tuner.open_world.identity import (
    candidate_citation_repair_request,
    repair_attempt_idempotency_key,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    CandidateCitationRejectionError,
    CandidateCitationRepairRequest,
)

_TEST_PROVIDER = "deterministic-fixture"
_TEST_MODEL = "fixture-model-2026-08-17"


# --------------------------------------------------------------------------
# The prompt contract
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# The repair loop
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# ID equality stays exact
# --------------------------------------------------------------------------


def test_the_rejection_error_carries_only_the_offending_ids() -> None:
    verified = ("a" * 64, "b" * 64)
    offered = ("a" * 64, "c" * 64)
    rejected = tuple(
        citation_id for citation_id in offered if citation_id not in set(verified)
    )
    error = CandidateCitationRejectionError(rejected)

    assert error.rejected_citation_ids == ("c" * 64,)
    # A correctly copied ID is never echoed back as a failure.
    assert "a" * 64 not in error.rejected_citation_ids
    # It stays a ValueError, so an exhausted repair converges where an
    # unrepaired rejection always did.
    assert isinstance(error, ValueError)


# --------------------------------------------------------------------------
# The repair payload cannot widen the evidence
# --------------------------------------------------------------------------


def test_candidate_repair_request_cannot_carry_evidence() -> None:
    """A signature-level guarantee rather than a filtering one.

    There is no argument through which evidence, a span, or even the server's
    VERIFIED ID list could reach the payload, so no caller can make it happen.
    """
    parameters = set(inspect.signature(candidate_citation_repair_request).parameters)
    assert parameters == {"attempt", "rejected_citation_ids"}


def test_the_candidate_repair_echoes_only_the_models_own_strings() -> None:
    repair = candidate_citation_repair_request(
        attempt=2, rejected_citation_ids=("a-string-the-model-invented",)
    )

    assert repair is not None
    assert [item.citation_id for item in repair.rejected_citation_ids] == [
        "a-string-the-model-invented"
    ]
    # The reason is a fixed string, never assembled from tenant data.
    for item in repair.rejected_citation_ids:
        assert "invented" not in item.reason


# `test_the_repair_never_echoes_a_valid_id_back` was deleted here.
#
# It named a property it could not test. `candidate_citation_repair_request`
# takes `attempt` and `rejected_citation_ids` and nothing else, so no valid
# citation ID is in scope at any point during the call -- there is no input it
# could have echoed and no code path on which it could fail. It asserted that
# a rejected string round-tripped and that the tuple had one entry, which is
# what `test_the_candidate_repair_echoes_only_the_models_own_strings` above
# already covers.
#
# The property is real, but it is a property of the SIGNATURE, as that
# function's own docstring says: "Takes no evidence and no verified IDs --
# that signature is the guarantee, not a comment about one." What guards it is
# `test_candidate_repair_request_cannot_carry_evidence` above, which pins the
# parameter set to exactly `{"attempt", "rejected_citation_ids"}` and does
# fail if someone widens it.


def test_no_candidate_repair_is_built_when_nothing_was_rejected() -> None:
    assert (
        candidate_citation_repair_request(attempt=2, rejected_citation_ids=()) is None
    )


def test_a_candidate_repair_attempt_must_be_the_second_or_later() -> None:
    with pytest.raises(ValueError, match="attempt"):
        CandidateCitationRepairRequest(attempt=1, rejected_citation_ids=())


def test_duplicate_rejected_ids_collapse() -> None:
    repair = candidate_citation_repair_request(
        attempt=2, rejected_citation_ids=("q" * 63, "q" * 63)
    )

    assert repair is not None
    assert len(repair.rejected_citation_ids) == 1


# --------------------------------------------------------------------------
# The repair is not part of the question's identity
# --------------------------------------------------------------------------


def test_first_attempt_key_is_unchanged_by_the_repair_derivation() -> None:
    base = "a" * 64
    assert repair_attempt_idempotency_key(base_key=base, attempt=1) == base
    assert repair_attempt_idempotency_key(base_key=base, attempt=2) != base
