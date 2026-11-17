"""Snapshot source-read refusals are evidence faults with a closed vocabulary."""

from __future__ import annotations

import pytest

from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    EVIDENCE_FAULT_REASONS,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    EVIDENCE_REATTEMPT_PREFIX,
    EVIDENCE_SOURCE_READ_BUDGET_EXCEEDED,
    EVIDENCE_SOURCE_READ_DEADLINE_EXCEEDED,
    EVIDENCE_SOURCE_READ_LIMIT_REASONS,
    OpenWorldEvidenceSourceReadLimitError,
)
from reflexio.server.services.offline_tuner.sweep_progress import is_problem_outcome
from reflexio.server.services.storage.error import StorageError


def test_read_limit_reasons_are_exactly_the_budget_and_the_deadline() -> None:
    assert {
        "evidence_source_read_budget_exceeded",
        "evidence_source_read_deadline_exceeded",
    } == EVIDENCE_SOURCE_READ_LIMIT_REASONS


@pytest.mark.parametrize(
    "reason",
    [EVIDENCE_SOURCE_READ_BUDGET_EXCEEDED, EVIDENCE_SOURCE_READ_DEADLINE_EXCEEDED],
)
def test_a_read_limit_refusal_is_a_warning_level_evidence_fault(reason: str) -> None:
    """Recorded as ``ineligible_<reason>``, it must not sink to INFO."""
    assert reason in EVIDENCE_FAULT_REASONS
    assert is_problem_outcome(f"{EVIDENCE_REATTEMPT_PREFIX}{reason}")


@pytest.mark.parametrize(
    "reason",
    [EVIDENCE_SOURCE_READ_BUDGET_EXCEEDED, EVIDENCE_SOURCE_READ_DEADLINE_EXCEEDED],
)
def test_the_error_carries_its_reason_and_stays_a_storage_error(reason: str) -> None:
    error = OpenWorldEvidenceSourceReadLimitError(reason)
    assert isinstance(error, StorageError)
    assert error.reason == reason
    assert error.message == reason


def test_the_error_refuses_a_reason_outside_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="unknown evidence source read limit"):
        OpenWorldEvidenceSourceReadLimitError("evidence snapshot too big")
