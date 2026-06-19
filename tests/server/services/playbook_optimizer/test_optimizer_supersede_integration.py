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
        storage, incumbent, "new content", "playbook_optimizer"
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
        storage, incumbent, "new content", "playbook_optimizer"
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
        storage, incumbent, "new agent content", "playbook_optimizer"
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
        storage, incumbent, "new agent content", "playbook_optimizer"
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
