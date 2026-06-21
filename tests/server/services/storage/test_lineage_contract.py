"""Storage contract tests for the lineage mixin.

Parametrized over locally-testable backends via the shared ``storage`` fixture
in conftest.py (currently SQLite only).  Enterprise backends (postgres/supabase)
are added in Task 12 when their gated suite is built.
"""

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    LineageContext,
    LineageEvent,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import Status

pytestmark = pytest.mark.integration


def test_append_idempotent(storage) -> None:
    """Calling append_lineage_event twice with the same event returns the same id."""
    event = LineageEvent(
        org_id=storage.org_id,
        entity_type="user_playbook",
        entity_id="X",
        op="merge",
        source_ids=["Y"],
        request_id="r-idempotent",
    )
    first = storage.append_lineage_event(event)
    second = storage.append_lineage_event(event)
    assert first == second
    # F012: the duplicate must not create a second row.
    events = storage.get_lineage_events(entity_type="user_playbook", entity_id="X")
    assert len(events) == 1


def test_merge_sets_pointer_tombstone_and_event(storage) -> None:
    """merge_records sets status+merged_into on the source and appends a merge event."""
    survivor = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r-merge-survivor",
        content="survivor content",
    )
    source = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r-merge-source",
        content="source content",
    )
    storage.save_user_playbooks([survivor, source])

    storage.merge_records(
        entity_type="user_playbook",
        survivor_id=str(survivor.user_playbook_id),
        source_ids=[str(source.user_playbook_id)],
        context=LineageContext(op_kind="merge", actor="test", request_id="r-merge"),
    )

    # Source row must be tombstoned with a back-pointer to the survivor.
    tombstone = storage.get_user_playbook_by_id(
        source.user_playbook_id, include_tombstones=True
    )
    assert tombstone is not None
    assert tombstone.status is Status.MERGED
    assert str(tombstone.merged_into) == str(survivor.user_playbook_id)

    # A merge event must be recorded against the survivor's id.
    events = storage.get_lineage_events(entity_id=str(survivor.user_playbook_id))
    assert any(e.op == "merge" for e in events)


def test_merge_multi_source_skips_already_tombstoned(storage) -> None:
    """F013: merging a CURRENT + an already-MERGED source tombstones only the CURRENT one.

    The already-MERGED source keeps its original ``merged_into`` (it is skipped by
    the guard, not re-pointed), no error is raised, and exactly one merge event is
    recorded for the survivor.
    """
    survivor = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r-multi-survivor",
        content="survivor content",
    )
    current_source = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r-multi-current",
        content="current source",
    )
    # Pre-tombstoned source already merged into some OTHER id (999).
    already_merged = UserPlaybook(
        user_id="u",
        agent_version="v",
        request_id="r-multi-old",
        content="already merged",
        status=Status.MERGED,
        merged_into=999,
    )
    storage.save_user_playbooks([survivor, current_source, already_merged])

    storage.merge_records(
        entity_type="user_playbook",
        survivor_id=str(survivor.user_playbook_id),
        source_ids=[
            str(current_source.user_playbook_id),
            str(already_merged.user_playbook_id),
        ],
        context=LineageContext(op_kind="merge", actor="test", request_id="r-multi"),
    )

    # The CURRENT source is now MERGED into the survivor.
    current_tomb = storage.get_user_playbook_by_id(
        current_source.user_playbook_id, include_tombstones=True
    )
    assert current_tomb is not None
    assert current_tomb.status is Status.MERGED
    assert str(current_tomb.merged_into) == str(survivor.user_playbook_id)

    # The already-MERGED source is left intact (still points at 999, not the survivor).
    skipped = storage.get_user_playbook_by_id(
        already_merged.user_playbook_id, include_tombstones=True
    )
    assert skipped is not None
    assert skipped.status is Status.MERGED
    assert skipped.merged_into == 999

    # Exactly one merge event for the survivor.
    events = storage.get_lineage_events(entity_id=str(survivor.user_playbook_id))
    merge_events = [e for e in events if e.op == "merge"]
    assert len(merge_events) == 1


# ---------------------------------------------------------------------------
# B1 contract cases: update / hard_delete / archive / idempotency
# ---------------------------------------------------------------------------


def test_update_content_emits_revise(storage) -> None:
    """In-place update with content change emits exactly one op='revise' event."""
    pb = UserPlaybook(agent_version="v", request_id="r-update-revise", content="old")
    storage.save_user_playbooks([pb])

    storage.update_user_playbook(pb.user_playbook_id, content="new")

    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(pb.user_playbook_id)
    )
    revise_events = [e for e in events if e.op == "revise"]
    assert len(revise_events) == 1


def test_update_metadata_only_emits_status_change(storage) -> None:
    """In-place update without content (metadata-only) emits exactly one op='status_change' event."""
    pb = UserPlaybook(agent_version="v", request_id="r-update-meta", content="c")
    storage.save_user_playbooks([pb])

    storage.update_user_playbook(pb.user_playbook_id, playbook_name="new-name")

    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(pb.user_playbook_id)
    )
    status_change_events = [e for e in events if e.op == "status_change"]
    assert len(status_change_events) == 1
    # Must not emit a revise event for a metadata-only update.
    assert not any(e.op == "revise" for e in events)


def test_delete_user_playbook_emits_hard_delete(storage) -> None:
    """Physical delete of a user_playbook emits op='hard_delete' before deletion."""
    pb = UserPlaybook(agent_version="v", request_id="r-delete-up", content="c")
    storage.save_user_playbooks([pb])
    entity_id = str(pb.user_playbook_id)

    storage.delete_user_playbook(pb.user_playbook_id)

    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=entity_id
    )
    assert any(e.op == "hard_delete" for e in events)


def test_delete_profiles_by_ids_emits_hard_delete(storage) -> None:
    """Physical delete of a profile via delete_profiles_by_ids emits op='hard_delete'."""
    profile = UserProfile(
        profile_id="prof-hd-1",
        user_id="u",
        content="some content",
        last_modified_timestamp=1,
        generated_from_request_id="r-prof-delete",
    )
    storage.add_user_profile("u", [profile])

    storage.delete_profiles_by_ids([profile.profile_id])

    events = storage.get_lineage_events(
        entity_type="profile", entity_id=profile.profile_id
    )
    assert any(e.op == "hard_delete" for e in events)


def test_archive_agent_playbooks_by_ids_emits_status_change(storage) -> None:
    """archive_agent_playbooks_by_ids emits op='status_change' for each archived playbook."""
    ap = AgentPlaybook(agent_version="v", content="some content")
    storage.save_agent_playbooks([ap])
    entity_id = str(ap.agent_playbook_id)

    storage.archive_agent_playbooks_by_ids([ap.agent_playbook_id])

    events = storage.get_lineage_events(
        entity_type="agent_playbook", entity_id=entity_id
    )
    assert any(e.op == "status_change" for e in events)


def test_append_lineage_event_idempotent_exact_key(storage) -> None:
    """Duplicate append_lineage_event calls with the same 5-col key yield exactly one row."""
    event = LineageEvent(
        org_id=storage.org_id,
        entity_type="user_playbook",
        entity_id="idem-2",
        op="revise",
        request_id="r-idem-exact",
    )
    first = storage.append_lineage_event(event)
    second = storage.append_lineage_event(event)

    assert first == second
    events = storage.get_lineage_events(entity_type="user_playbook", entity_id="idem-2")
    assert len(events) == 1


def test_status_change_event_carries_structured_fields(storage) -> None:
    """status_change via archive_agent_playbooks_by_ids carries from_status/to_status/status_namespace."""
    ap = AgentPlaybook(agent_version="v", content="c for structured")
    storage.save_agent_playbooks([ap])
    entity_id = str(ap.agent_playbook_id)

    storage.archive_agent_playbooks_by_ids([ap.agent_playbook_id])

    events = storage.get_lineage_events(
        entity_type="agent_playbook", entity_id=entity_id
    )
    sc_events = [e for e in events if e.op == "status_change"]
    assert len(sc_events) == 1
    evt = sc_events[0]
    assert evt.from_status is None
    assert evt.to_status == "archived"
    assert evt.status_namespace == "lifecycle_status"


def test_status_change_null_prior_stores_real_null(storage) -> None:
    """from_status must be real NULL (None), not the string 'None', when prior status is absent."""
    ap = AgentPlaybook(agent_version="v", content="c for null-prior")
    storage.save_agent_playbooks([ap])
    storage.archive_agent_playbooks_by_ids([ap.agent_playbook_id])

    events = storage.get_lineage_events(
        entity_type="agent_playbook", entity_id=str(ap.agent_playbook_id)
    )
    sc = next(e for e in events if e.op == "status_change")
    # Must be Python None, not the string "None"
    assert sc.from_status is None
    assert sc.from_status != "None"


def test_non_status_change_events_have_null_structured_fields(storage) -> None:
    """merge/revise/hard_delete events must NOT carry status namespace fields."""
    pb = UserPlaybook(agent_version="v", request_id="r-non-sc", content="c")
    storage.save_user_playbooks([pb])
    storage.delete_user_playbook(pb.user_playbook_id)

    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(pb.user_playbook_id)
    )
    hd = next(e for e in events if e.op == "hard_delete")
    assert hd.from_status is None
    assert hd.to_status is None
    assert hd.status_namespace is None


def test_get_lineage_events_request_id_filter(storage) -> None:
    """get_lineage_events(request_id=R) returns only events tagged with R; None returns all."""
    ap1 = AgentPlaybook(agent_version="v", content="c1")
    ap2 = AgentPlaybook(agent_version="v", content="c2")
    storage.save_agent_playbooks([ap1, ap2])

    req_a = "req-filter-A"
    req_b = "req-filter-B"

    # Emit archive events to seed lineage entries; archive emits status_change
    storage.archive_agent_playbooks_by_ids([ap1.agent_playbook_id])
    # Re-fetch to get the event's request_id — use append_lineage_event directly for precision
    from reflexio.models.api_schema.domain.entities import LineageEvent

    storage.append_lineage_event(
        LineageEvent(
            org_id=storage.org_id,
            entity_type="agent_playbook",
            entity_id=str(ap1.agent_playbook_id),
            op="aggregate",
            source_ids=[],
            request_id=req_a,
        )
    )
    storage.append_lineage_event(
        LineageEvent(
            org_id=storage.org_id,
            entity_type="agent_playbook",
            entity_id=str(ap2.agent_playbook_id),
            op="aggregate",
            source_ids=[],
            request_id=req_b,
        )
    )

    # Filter to req_a: only events with request_id == req_a
    events_a = storage.get_lineage_events(
        entity_type="agent_playbook", request_id=req_a
    )
    assert all(e.request_id == req_a for e in events_a), (
        f"Expected only {req_a!r} events, got {[e.request_id for e in events_a]}"
    )
    assert any(e.entity_id == str(ap1.agent_playbook_id) for e in events_a)

    # Filter to req_b: only events with request_id == req_b
    events_b = storage.get_lineage_events(
        entity_type="agent_playbook", request_id=req_b
    )
    assert all(e.request_id == req_b for e in events_b)
    assert any(e.entity_id == str(ap2.agent_playbook_id) for e in events_b)

    # No request_id filter: returns events from both
    events_all = storage.get_lineage_events(entity_type="agent_playbook")
    req_ids_all = {e.request_id for e in events_all}
    assert req_a in req_ids_all
    assert req_b in req_ids_all
