"""Tests for StructuredPlaybookContent: optional metadata fields and polarity."""

from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
)


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


def test_structured_playbook_content_default_polarity_is_positive() -> None:
    p = StructuredPlaybookContent(
        content="X",
        trigger="when Y",
        rationale="because Z",
        blocking_issue=None,
    )
    assert p.polarity == "positive"


def test_structured_playbook_content_accepts_negative() -> None:
    p = StructuredPlaybookContent(
        content="Avoid X",
        trigger="when Y",
        rationale="user pushed back",
        blocking_issue=None,
        polarity="negative",
    )
    assert p.polarity == "negative"
