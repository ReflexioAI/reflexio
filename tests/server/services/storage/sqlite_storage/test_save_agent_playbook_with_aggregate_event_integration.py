"""TDD tests for save_agent_playbook_with_aggregate_event — SQLite atomic write side.

Three tests:
  1. Happy path: row inserted + exactly one op=aggregate event with correct
     reason / source_ids.
  2. Atomicity: if _append_event_stmt raises, the INSERT rolls back (no orphaned row).
  3. FTS: the new playbook is searchable after the call.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import reflexio.server.services.storage.sqlite_storage._playbook as _playbook_mod
from reflexio.models.api_schema.domain.entities import AgentPlaybook
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _store(tmp_path, org_id: str = "org-test") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _make_playbook(
    playbook_name: str = "my_pb",
    agent_version: str = "v1",
    content: str = "When deploying, run migrations first.",
    trigger: str = "deployment trigger",
) -> AgentPlaybook:
    return AgentPlaybook(
        playbook_name=playbook_name,
        agent_version=agent_version,
        content=content,
        trigger=trigger,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveAgentPlaybookWithAggregateEvent:
    def test_happy_path_row_and_event_both_written(self, tmp_path):
        """Row is inserted and exactly one aggregate event exists with correct fields."""
        s = _store(tmp_path)
        pb = _make_playbook()

        result = s.save_agent_playbook_with_aggregate_event(
            pb,
            source_ids=["10", "11"],
            request_id="run-x",
            run_mode="full_archive",
        )

        # Row exists and has a real ID
        assert result.agent_playbook_id > 0
        fetched = s.get_agent_playbook_by_id(result.agent_playbook_id)
        assert fetched is not None
        assert fetched.playbook_name == "my_pb"

        # Exactly one aggregate event for this entity
        events = s.get_lineage_events(
            entity_type="agent_playbook",
            entity_id=str(result.agent_playbook_id),
        )
        agg_events = [e for e in events if e.op == "aggregate"]
        assert len(agg_events) == 1
        ev = agg_events[0]
        assert ev.reason == "aggregate:full_archive"
        assert ev.source_ids == ["10", "11"]
        assert ev.request_id == "run-x"
        assert ev.actor == "aggregator"
        assert ev.prov_relation == "wasDerivedFrom"

    def test_atomicity_rollback_on_event_append_failure(self, tmp_path):
        """If _append_event_stmt raises, the INSERT is rolled back — no orphaned row."""
        s = _store(tmp_path)
        pb = _make_playbook()

        # Count rows before the call
        rows_before = len(s.get_agent_playbooks())

        with (
            patch.object(
                _playbook_mod,
                "_append_event_stmt",
                side_effect=RuntimeError("simulated event failure"),
            ),
            # handle_exceptions wraps RuntimeError into StorageError
            pytest.raises(StorageError, match="simulated event failure"),
        ):
            s.save_agent_playbook_with_aggregate_event(
                pb,
                source_ids=["1"],
                request_id="run-fail",
                run_mode="incremental",
            )

        # Row count must be unchanged — INSERT rolled back
        rows_after = len(s.get_agent_playbooks())
        assert rows_after == rows_before, (
            f"Expected {rows_before} rows but found {rows_after} "
            "— INSERT was NOT rolled back (atomicity violation)"
        )

    def test_fts_indexes_new_playbook(self, tmp_path):
        """After saving, the playbook is reachable via FTS search on its trigger text."""
        from reflexio.models.api_schema.retriever_schema import (
            SearchAgentPlaybookRequest,
        )

        s = _store(tmp_path)
        pb = _make_playbook(trigger="unique_trigger_xyz", content="some content")

        result = s.save_agent_playbook_with_aggregate_event(
            pb,
            source_ids=[],
            request_id="run-fts",
            run_mode="incremental",
        )

        hits = s.search_agent_playbooks(
            SearchAgentPlaybookRequest(query="unique_trigger_xyz", top_k=10)
        )
        assert any(h.agent_playbook_id == result.agent_playbook_id for h in hits), (
            "New playbook not found in FTS index after save_agent_playbook_with_aggregate_event"
        )
