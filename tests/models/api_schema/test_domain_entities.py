"""Task 2.3: optional source_span/notes/reader_angle on UserProfile and UserPlaybook."""

from reflexio.models.api_schema.domain.entities import UserPlaybook, UserProfile


def test_user_profile_optional_new_fields_default_to_none() -> None:
    p = UserProfile(
        profile_id="p1",
        user_id="u1",
        content="x",
        last_modified_timestamp=0,
        generated_from_request_id="r1",
    )
    assert p.source_span is None
    assert p.notes is None
    assert p.reader_angle is None


def test_user_profile_accepts_optional_fields() -> None:
    p = UserProfile(
        profile_id="p2",
        user_id="u1",
        content="x",
        last_modified_timestamp=0,
        generated_from_request_id="r1",
        source_span="q",
        notes="n",
        reader_angle="facts",
    )
    assert p.reader_angle == "facts"


def test_user_playbook_optional_new_fields_default_to_none() -> None:
    pb = UserPlaybook(
        agent_version="v1",
        request_id="r1",
        trigger="t",
        content="c",
        rationale="r",
    )
    assert pb.source_span is None
    assert pb.notes is None
    assert pb.reader_angle is None


def test_user_playbook_accepts_optional_fields() -> None:
    pb = UserPlaybook(
        agent_version="v1",
        request_id="r1",
        trigger="t",
        content="c",
        rationale="r",
        source_span="q",
        notes="n",
        reader_angle="behavior",
    )
    assert pb.reader_angle == "behavior"
