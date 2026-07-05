"""Contract test: learning-path writes honor ``commit_scope`` atomicity (Task 5a).

Task 1 introduced ``commit_scope`` — a context manager that groups writes into a
single atomic transaction (one BEGIN IMMEDIATE / one commit) and defers FTS/vec
index ops until after commit. Writes issued inside a scope must NOT run their own
BEGIN or commit; ``_own_transaction()`` returns False while ``_scope_depth > 0``.

Before this task, the learning-path writes (profiles, user playbooks, lineage
events, operation-state bookmarks) each self-committed. That broke atomicity: a
write inside a ``commit_scope`` would durably commit even when a later write in
the same scope raised, leaving the DB in a torn, partially-committed state.

This test is the invariant guard: perform one write of each kind inside a single
``commit_scope`` that then raises, and assert every write rolled back. It fails
before the own-txn guards are added (the self-committing writes survive) and
passes once each method honors ``_own_transaction()``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import LineageEvent
from reflexio.models.api_schema.service_schemas import (
    ProfileTimeToLive,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

_ORG_ID = "learning-atomic-org"
_USER_ID = "u-atomic"


def _make_profile() -> UserProfile:
    return UserProfile(
        user_id=_USER_ID,
        profile_id="lv-test-pid",
        content="atomic profile content",
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="req-lv-test",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
    )


def _make_playbook() -> UserPlaybook:
    return UserPlaybook(
        agent_version="v1",
        request_id="req-lv-test",
        user_id=_USER_ID,
        content="atomic playbook content",
    )


def _make_lineage_event() -> LineageEvent:
    return LineageEvent(
        org_id=_ORG_ID,
        entity_type="profile",
        entity_id="lv-test-pid",
        op="create",
        request_id="req-lv-test",
        reason="test",
        actor="test",
    )


def test_learning_writes_roll_back_together_on_scope_failure(tmp_path) -> None:
    """All four learning-path writes roll back when the enclosing scope raises."""
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(
            org_id=_ORG_ID, db_path=str(tmp_path / "learning_atomic.db")
        )
        storage.migrate()

        with pytest.raises(RuntimeError, match="boom"), storage.commit_scope():
            storage.add_user_profile(_USER_ID, [_make_profile()])
            storage.save_user_playbooks([_make_playbook()])
            storage.append_lineage_event(_make_lineage_event())
            storage.upsert_operation_state("test-svc", {"k": "v"})
            raise RuntimeError("boom")

        # Every write in the failed scope must have rolled back.
        assert storage.get_user_profile(_USER_ID) == []
        assert storage.get_user_playbooks(user_id=_USER_ID) == []
        assert storage.get_lineage_events(entity_id="lv-test-pid") == []
        assert storage.get_operation_state("test-svc") is None
