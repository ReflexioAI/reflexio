from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.domain.entities import (
    UserPlaybook, AgentPlaybook, UserProfile, LineageEvent, LineageContext, RecordRef,
)


def test_status_has_tombstone_values():
    assert Status("merged") is Status.MERGED
    assert Status("superseded") is Status.SUPERSEDED


def test_playbook_pointers_default_none_and_are_int():
    pb = UserPlaybook(agent_version="v1", request_id="r1")
    assert pb.merged_into is None and pb.superseded_by is None
    pb.merged_into = 7
    assert UserPlaybook.model_validate(pb.model_dump()).merged_into == 7


def test_profile_pointers_are_str():
    p = UserProfile(profile_id="p1", user_id="u1", content="c",
                    last_modified_timestamp=0, generated_from_request_id="r1")
    assert p.merged_into is None
    p.merged_into = "p2"
    assert UserProfile.model_validate(p.model_dump()).merged_into == "p2"


def test_lineage_event_is_content_free_and_idempotency_keyed():
    e = LineageEvent(org_id="org-42", entity_type="user_playbook", entity_id="UP-1",
                     op="merge", prov_relation="wasDerivedFrom", source_ids=["UP-1"],
                     actor="consolidator", request_id="req-7", reason="dup")
    assert e.event_id == 0  # storage assigns
    assert not hasattr(e, "content")


def test_lineage_context_and_record_ref():
    ctx = LineageContext(op_kind="merge", actor="consolidator", source_ids=["UP-1"], reason="dup")
    assert ctx.request_id is None or isinstance(ctx.request_id, str)
    ref = RecordRef(id="UP-2", is_purged=False)
    assert ref.id == "UP-2" and ref.is_purged is False
