"""Task 2.2: optional source_span/notes/reader_angle on StructuredPlaybookContent."""

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
