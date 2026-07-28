"""Integration tests for the optimizer's agent supersede helper.

User playbooks now use the durable publication service. Agent playbooks retain
their separate PENDING/approval supersede behavior and are covered here.

B3 request_id contract: each supersede call must stamp a non-empty, job-derived
request_id on its revise lineage event, enabling correct run-correlation (tying
optimizer/edit events to their originating job).  An empty request_id is rejected
loudly (``ValueError``) by the guard at the top of each helper, before any write.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    PlaybookStatus,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.playbook_optimizer.optimizer import (
    _supersede_agent_playbook,
    optimizer_run_request_id,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._lineage import (
    _EMPTY_REQUEST_ID_MSG,
)

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
        request_id=optimizer_run_request_id(99),
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
    assert [event.op for event in events] == ["create", "revise"]
    assert events[1].actor == "playbook_optimizer"
    assert str(incumbent_id) in events[1].source_ids


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
    events_before = storage.get_lineage_events(entity_type="agent_playbook")

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
    assert events == events_before


# ---------------------------------------------------------------------------
# B3 contract: helpers stamp a non-empty, job-derived request_id on revise events
# ---------------------------------------------------------------------------


def test_supersede_agent_playbook_revise_event_carries_job_request_id(tmp_path):
    """_supersede_agent_playbook stamps the passed request_id on the revise event.

    Same contract as the user-playbook helper: the lineage event's request_id
    must equal the job-derived run id, enabling correct run-correlation.
    """
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

    run_id = optimizer_run_request_id(99)
    result = _supersede_agent_playbook(
        storage,
        incumbent,
        "new agent content",
        "playbook_optimizer",
        request_id=run_id,
    )

    assert result is not None
    events = storage.get_lineage_events(
        entity_type="agent_playbook", entity_id=str(result)
    )
    assert [event.op for event in events] == ["create", "revise"]
    assert events[1].request_id == run_id, (
        f"revise event must carry the job-derived run id {run_id!r}, "
        f"got {events[1].request_id!r}"
    )


def test_supersede_agent_playbook_raises_on_empty_request_id(tmp_path):
    """_supersede_agent_playbook raises ValueError on empty request_id before any write."""
    storage = _storage(tmp_path)
    [incumbent] = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="support",
                agent_version="v1",
                content="old content",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )

    with pytest.raises(ValueError, match=_EMPTY_REQUEST_ID_MSG):
        _supersede_agent_playbook(
            storage, incumbent, "new content", "playbook_optimizer", request_id=""
        )

    # No orphan successor should have been inserted
    count = storage.conn.execute("SELECT COUNT(*) FROM agent_playbooks").fetchone()[0]
    assert count == 1, "no orphan row should be inserted when request_id is empty"
