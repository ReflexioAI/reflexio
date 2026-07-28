"""Tests for StructuredPlaybookContent optional metadata fields."""

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredExtractedPlaybookContent,
    StructuredExtractedPlaybookList,
    StructuredPlaybookContent,
    StructuredPlaybookEvidence,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("observed_failure", "observed-failure"),
        ("expert gap", "expert-gap"),
        ("failure", "observed-failure"),
        ("success", "verified-success"),
    ],
)
def test_evidence_kind_normalizes_harmless_model_variants(raw, expected):
    assert StructuredPlaybookContent(evidence_kind=raw).evidence_kind == expected


def test_structured_playbook_content_new_fields_default_to_none() -> None:
    c = StructuredPlaybookContent(trigger="t", content="c", rationale="r")
    assert c.source_span is None
    assert c.notes is None
    assert c.reader_angle is None


def test_structured_playbook_content_accepts_optional_fields() -> None:
    c = StructuredPlaybookContent(
        trigger="t",
        content="c",
        rationale="r",
        source_span="quote",
        notes="confidence=0.9",
        reader_angle="trigger",
    )
    assert c.source_span == "quote"
    assert c.reader_angle == "trigger"


def test_extracted_playbook_content_requires_small_grounding_contract() -> None:
    with pytest.raises(ValidationError):
        StructuredExtractedPlaybookContent(
            trigger="when a task starts",
            content="Do the grounded action.",
        )

    grounded = StructuredExtractedPlaybookContent(
        rationale="The correction identifies a reusable failure.",
        evidence_kind="correction",
        trigger="when a similar task starts",
        content="Do the corrected action.",
        evidence=[
            StructuredPlaybookEvidence(turn_ref="T1", source_span="correct this")
        ],
    )
    assert grounded.evidence[0].turn_ref == "T1"
    assert set(grounded.model_dump()) == {
        "rationale",
        "evidence_kind",
        "trigger",
        "content",
        "evidence",
    }


def test_extracted_contract_ignores_legacy_candidate_fields_but_requires_wrapper() -> (
    None
):
    candidate = StructuredExtractedPlaybookContent.model_validate(
        {
            "rationale": "The correction supports a future workflow rule.",
            "evidence_kind": "correction",
            "trigger": "when starting the task",
            "content": "Apply the correction before acting.",
            "evidence": [{"turn_ref": "T1", "source_span": "do this instead"}],
            "future_task_class": "legacy field",
            "reader_angle": "legacy field",
        }
    )
    assert "future_task_class" not in candidate.model_dump()
    assert "reader_angle" not in candidate.model_dump()

    with pytest.raises(ValidationError):
        StructuredExtractedPlaybookList.model_validate({})
