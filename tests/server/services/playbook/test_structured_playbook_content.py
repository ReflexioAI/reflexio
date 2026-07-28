"""Tests for StructuredPlaybookContent optional metadata fields."""

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredExtractedPlaybookList,
    StructuredPlaybookContent,
    StructuredReferencedExtractedPlaybookContent,
    StructuredReferencedExtractedPlaybookList,
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
        StructuredReferencedExtractedPlaybookContent(
            trigger="when a task starts",
            content="Do the grounded action.",
        )

    grounded = StructuredReferencedExtractedPlaybookContent(
        rationale="The correction identifies a reusable failure.",
        evidence_kind="correction",
        trigger="when a similar task starts",
        content="Do the corrected action.",
        evidence_refs=["T1"],
    )
    assert grounded.evidence_refs == ["T1"]
    assert set(grounded.model_dump()) == {
        "rationale",
        "evidence_kind",
        "trigger",
        "content",
        "evidence_refs",
    }


def test_extracted_contract_ignores_legacy_candidate_fields_but_requires_wrapper() -> (
    None
):
    candidate = StructuredReferencedExtractedPlaybookContent.model_validate(
        {
            "rationale": "The correction supports a future workflow rule.",
            "evidence_kind": "correction",
            "trigger": "when starting the task",
            "content": "Apply the correction before acting.",
            "evidence_refs": ["T1"],
            "future_task_class": "legacy field",
            "reader_angle": "legacy field",
        }
    )
    assert "future_task_class" not in candidate.model_dump()
    assert "reader_angle" not in candidate.model_dump()

    with pytest.raises(ValidationError):
        StructuredReferencedExtractedPlaybookList.model_validate({})


def test_copied_span_contract_keeps_legacy_provider_aliases() -> None:
    output = StructuredExtractedPlaybookList.model_validate(
        {
            "playbooks": [
                {
                    "Reason": "The expert correction supports a reusable rule.",
                    "Kind": "expert gap",
                    "When": "when the same expert task recurs",
                    "Action": "Apply the expert correction.",
                    "Sources": [{"Turn Ref": "T2", "Source Span": "exact text"}],
                }
            ]
        }
    )

    assert output.playbooks[0].evidence[0].turn_ref == "T2"
    assert output.playbooks[0].evidence[0].source_span == "exact text"
