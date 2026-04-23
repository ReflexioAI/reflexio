"""Task 2.1: optional source_span/notes/reader_angle on ProfileAddItem."""

from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileAddItem,
)


def test_profile_add_item_new_fields_default_to_none() -> None:
    item = ProfileAddItem(content="x", time_to_live="infinity")
    assert item.source_span is None
    assert item.notes is None
    assert item.reader_angle is None


def test_profile_add_item_accepts_optional_fields() -> None:
    item = ProfileAddItem(
        content="x",
        time_to_live="infinity",
        source_span="exact quote",
        notes="high confidence",
        reader_angle="facts",
    )
    assert item.source_span == "exact quote"
    assert item.notes == "high confidence"
    assert item.reader_angle == "facts"
