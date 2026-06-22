"""Integration tests for the optimizer's atomic supersede helpers.

Tests the ``_supersede_user_playbook`` and ``_supersede_agent_playbook`` helpers
that were extracted from ``PlaybookOptimizer._commit_if_allowed`` as part of the
lineage Phase A work.  These helpers are unit-tested directly against a real
SQLite storage so no full PlaybookOptimizer construction is needed.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.playbook_optimizer.optimizer import (
    _supersede_agent_playbook,
    _supersede_user_playbook,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _storage(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(
            org_id="opt-test", db_path=str(tmp_path / "reflexio.db")
        )
    storage._get_embedding = Mock(return_value=[0.0] * 512)  # noqa: SLF001
    storage.llm_client.get_embeddings = Mock(return_value=[[0.0] * 512])
    return storage


# ---------------------------------------------------------------------------
# User-playbook supersede helper
# ---------------------------------------------------------------------------


def test_supersede_user_playbook_sets_superseded_by_and_revise_event(tmp_path):
    """Happy path: incumbent becomes SUPERSEDED with superseded_by set; a revise lineage event is recorded."""
    storage = _storage(tmp_path)
    incumbent = UserPlaybook(
        user_id="u1",
        agent_version="v1",
        request_id="req-1",
        playbook_name="support",
        content="old content",
    )
    storage.save_user_playbooks([incumbent])
    incumbent_id = incumbent.user_playbook_id

    result = _supersede_user_playbook(
        storage,
        incumbent,
        "new content",
        "playbook_optimizer",
        request_id="optjob_1",
    )

    assert result is not None, "helper should return the successor id on success"

    # Incumbent must now be SUPERSEDED
    row = storage.conn.execute(
        "SELECT status, superseded_by FROM user_playbooks WHERE user_playbook_id=?",
        (incumbent_id,),
    ).fetchone()
    assert row["status"] == Status.SUPERSEDED.value
    assert int(row["superseded_by"]) == result

    # Successor must be CURRENT (status IS NULL)
    successor_row = storage.conn.execute(
        "SELECT status, content FROM user_playbooks WHERE user_playbook_id=?",
        (result,),
    ).fetchone()
    assert successor_row["status"] is None
    assert successor_row["content"] == "new content"

    # A revise lineage event must exist for the successor
    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(result)
    )
    assert len(events) == 1
    assert events[0].op == "revise"
    assert events[0].actor == "playbook_optimizer"
    assert str(incumbent_id) in events[0].source_ids


def test_supersede_user_playbook_returns_none_for_non_current_incumbent(tmp_path):
    """If the incumbent is already SUPERSEDED (not CURRENT), the helper returns None and leaves no orphan."""
    storage = _storage(tmp_path)
    # Create an already-archived/superseded incumbent by inserting and immediately archiving
    incumbent = UserPlaybook(
        user_id="u1",
        agent_version="v1",
        request_id="req-2",
        playbook_name="support",
        content="stale content",
        status=Status.ARCHIVED,  # not CURRENT
    )
    storage.save_user_playbooks([incumbent])

    playbooks_before = storage.conn.execute(
        "SELECT COUNT(*) as cnt FROM user_playbooks"
    ).fetchone()["cnt"]

    result = _supersede_user_playbook(
        storage,
        incumbent,
        "new content",
        "playbook_optimizer",
        request_id="optjob_2",
    )

    assert result is None, "helper should return None when incumbent is not CURRENT"

    # No orphan successor should have been left behind
    playbooks_after = storage.conn.execute(
        "SELECT COUNT(*) as cnt FROM user_playbooks"
    ).fetchone()["cnt"]
    assert playbooks_after == playbooks_before, "no orphan row should remain"

    # No lineage events should exist
    events = storage.get_lineage_events(entity_type="user_playbook")
    assert events == []


# ---------------------------------------------------------------------------
# Agent-playbook supersede helper
# ---------------------------------------------------------------------------


def test_supersede_agent_playbook_sets_superseded_by_and_revise_event(tmp_path):
    """Happy path: agent incumbent becomes SUPERSEDED with superseded_by set; a revise lineage event is recorded."""
    storage = _storage(tmp_path)
    [incumbent] = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="old agent content",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )
    incumbent_id = incumbent.agent_playbook_id

    result = _supersede_agent_playbook(
        storage,
        incumbent,
        "new agent content",
        "playbook_optimizer",
        request_id="optjob_99",
    )

    assert result is not None, "helper should return the successor id on success"

    # Incumbent must now be SUPERSEDED
    row = storage.conn.execute(
        "SELECT status, superseded_by FROM agent_playbooks WHERE agent_playbook_id=?",
        (incumbent_id,),
    ).fetchone()
    assert row["status"] == Status.SUPERSEDED.value
    assert int(row["superseded_by"]) == result

    # Successor must be CURRENT (status IS NULL)
    successor_row = storage.conn.execute(
        "SELECT status, content FROM agent_playbooks WHERE agent_playbook_id=?",
        (result,),
    ).fetchone()
    assert successor_row["status"] is None
    assert successor_row["content"] == "new agent content"

    # A revise lineage event must exist for the successor
    events = storage.get_lineage_events(
        entity_type="agent_playbook", entity_id=str(result)
    )
    assert len(events) == 1
    assert events[0].op == "revise"
    assert events[0].actor == "playbook_optimizer"
    assert str(incumbent_id) in events[0].source_ids


def test_supersede_agent_playbook_returns_none_for_non_current_incumbent(tmp_path):
    """If the agent incumbent is already SUPERSEDED, the helper returns None and leaves no orphan."""
    storage = _storage(tmp_path)
    # Insert a playbook then mark it as superseded manually so it is not CURRENT
    [incumbent] = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="stale agent content",
                playbook_status=PlaybookStatus.PENDING,
                status=Status.ARCHIVED,  # not CURRENT
            )
        ]
    )

    agent_playbooks_before = storage.conn.execute(
        "SELECT COUNT(*) as cnt FROM agent_playbooks"
    ).fetchone()["cnt"]

    result = _supersede_agent_playbook(
        storage,
        incumbent,
        "new agent content",
        "playbook_optimizer",
        request_id="job-x",
    )

    assert result is None, "helper should return None when incumbent is not CURRENT"

    agent_playbooks_after = storage.conn.execute(
        "SELECT COUNT(*) as cnt FROM agent_playbooks"
    ).fetchone()["cnt"]
    assert agent_playbooks_after == agent_playbooks_before, (
        "no orphan row should remain"
    )

    events = storage.get_lineage_events(entity_type="agent_playbook")
    assert events == []


# ---------------------------------------------------------------------------
# Regression: two optimizer passes on the same agent_playbook must produce
# two distinct, non-colliding lineage events (not silently drop the second).
# Bug: _supersede_agent_playbook passed request_id=None which coerced to ""
# causing (org, "agent_playbook", id, "revise", "") to collide on the second
# call — ON CONFLICT DO NOTHING silently discarded the second event.
# ---------------------------------------------------------------------------


def test_two_optimizer_passes_produce_distinct_revise_events(tmp_path):
    """Regression: two optimizer supersede calls on the same incumbent emit two events.

    Pre-fix: both calls used request_id="" (coerced from None) so the second
    INSERT OR IGNORE hit the unique constraint and was silently dropped,
    leaving only one event in the log.

    Post-fix: each call receives a distinct job-derived request_id so both
    events are stored and both are non-empty.
    """
    storage = _storage(tmp_path)
    [incumbent] = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="original content",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )

    # First optimizer pass — simulates job 101
    result_1 = _supersede_agent_playbook(
        storage,
        incumbent,
        "improved content v1",
        "playbook_optimizer",
        request_id="optjob_101",
    )
    assert result_1 is not None, "first supersede should succeed"

    # Restore the incumbent to CURRENT so a second pass can supersede it again
    # (simulates a separate optimizer run picking up the same original row
    # before the first pass has committed, or replays on a fresh incumbent copy).
    # In practice we re-insert the original incumbent and supersede it again.
    [incumbent2] = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="original content",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )

    # Second optimizer pass — simulates job 102 (different request_id)
    result_2 = _supersede_agent_playbook(
        storage,
        incumbent2,
        "improved content v2",
        "playbook_optimizer",
        request_id="optjob_102",
    )
    assert result_2 is not None, "second supersede should succeed"

    # Both revise events must exist and carry distinct non-empty request_ids
    events = storage.get_lineage_events(entity_type="agent_playbook")
    revise_events = [e for e in events if e.op == "revise"]
    assert len(revise_events) == 2, (
        f"expected 2 distinct revise events, got {len(revise_events)}; "
        "the second may have been silently dropped by ON CONFLICT DO NOTHING "
        "(regression: request_id=None coerced to '' causing key collision)"
    )
    request_ids = {e.request_id for e in revise_events}
    assert "" not in request_ids, "no revise event should have an empty request_id"
    assert request_ids == {"optjob_101", "optjob_102"}, (
        f"expected distinct job-derived request_ids, got {request_ids}"
    )
