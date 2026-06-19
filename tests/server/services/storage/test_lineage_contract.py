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
