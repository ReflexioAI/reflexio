"""Storage contract tests for the lineage mixin.

Parametrized over locally-testable backends via the shared ``storage`` fixture
in conftest.py (currently SQLite only).  Enterprise backends (postgres/supabase)
are added in Task 12 when their gated suite is built.
"""

import pytest

from reflexio.models.api_schema.domain.entities import (
    LineageContext,
    LineageEvent,
    UserPlaybook,
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
    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id="X"
    )
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
