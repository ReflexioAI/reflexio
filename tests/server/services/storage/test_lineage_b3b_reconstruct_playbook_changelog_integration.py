"""Integration tests: reconstruct_playbook_aggregation_change_log — B3b Task 2.

Verifies the read-side reconstruction of PlaybookAggregationChangeLog from
lineage events, mirroring the profile reconstruction model.

Covered scenarios:
  1. 2-add / 1-remove incremental run — correct added/removed snapshots, run_mode.
  2. Full-archive run reconstructs added/removed; run_mode="full_archive".
  3. APPROVED playbook in legacy removed is absent from reconstruction.removed
     (S3 gate still passes).
  4. Add-only run (aggregate events, no supersede) → added-only, removed=[].
  5. Remove-only run (supersede events, no aggregate) → removed-only, added=[].
  6. Empty request_id events are skipped (never merged into a group).
  7. Purged tombstone (row hard-deleted after supersede) → omitted, no crash.
  8. get_lineage_events(request_id=R) returns only R's events (Part A filter).
"""

from __future__ import annotations

import pytest

from reflexio.lib._agent_playbook import reconstruct_playbook_aggregation_change_log
from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    LineageEvent,
)
from reflexio.models.api_schema.domain.enums import PlaybookStatus, Status
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, org_id: str = "org-pb") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _make_playbook(
    agent_playbook_id: int = 1,
    playbook_name: str = "pb",
    agent_version: str = "v1",
    content: str = "content",
    playbook_status: PlaybookStatus = PlaybookStatus.PENDING,
) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=agent_playbook_id,
        playbook_name=playbook_name,
        agent_version=agent_version,
        content=content,
        playbook_status=playbook_status,
    )


def _emit_aggregate_event(
    s: SQLiteStorage,
    *,
    entity_id: str,
    request_id: str,
    run_mode: str = "incremental",
) -> None:
    """Directly emit an aggregate lineage event (as playbook_aggregator would)."""
    s.append_lineage_event(
        LineageEvent(
            org_id=s.org_id,
            entity_type="agent_playbook",
            entity_id=entity_id,
            op="aggregate",
            prov_relation="wasDerivedFrom",
            source_ids=[],
            actor="aggregator",
            request_id=request_id,
            reason=f"aggregate:{run_mode}",
        )
    )


def _emit_status_change_superseded(
    s: SQLiteStorage,
    *,
    entity_id: str,
    request_id: str,
) -> None:
    """Directly emit a status_change/superseded lineage event (as supersede helpers would)."""
    s.append_lineage_event(
        LineageEvent(
            org_id=s.org_id,
            entity_type="agent_playbook",
            entity_id=entity_id,
            op="status_change",
            prov_relation="wasInvalidatedBy",
            source_ids=[],
            actor="aggregator",
            request_id=request_id,
            reason="None->superseded",
            from_status=None,
            to_status=Status.SUPERSEDED.value,
            status_namespace="lifecycle_status",
        )
    )


def _add_playbook(s: SQLiteStorage, pb: AgentPlaybook) -> int:
    """Insert a playbook via save_agent_playbooks and return its assigned id."""
    saved = s.save_agent_playbooks([pb])
    return saved[0].agent_playbook_id


def _set_superseded(s: SQLiteStorage, agent_playbook_id: int) -> None:
    """Hard-mark a playbook as SUPERSEDED without emitting a lineage event (for purge test)."""
    s.conn.execute(
        "UPDATE agent_playbooks SET status = ? WHERE agent_playbook_id = ?",
        (Status.SUPERSEDED.value, agent_playbook_id),
    )
    s.conn.commit()


# ---------------------------------------------------------------------------
# Test 1: 2-add / 1-remove incremental run
# ---------------------------------------------------------------------------


def test_incremental_run_two_adds_one_remove(tmp_path):
    """2-add / 1-remove incremental run reconstructs correctly."""
    s = _store(tmp_path)

    # Seed the playbook to be removed
    old_pb = _make_playbook(
        playbook_name="pb", agent_version="v1", content="old content"
    )
    old_id = _add_playbook(s, old_pb)
    _set_superseded(s, old_id)

    # New playbooks
    new1 = _make_playbook(playbook_name="pb", agent_version="v1", content="new A")
    new2 = _make_playbook(playbook_name="pb", agent_version="v1", content="new B")
    new1_id = _add_playbook(s, new1)
    new2_id = _add_playbook(s, new2)

    req_id = "run-incr-1"
    _emit_aggregate_event(
        s, entity_id=str(new1_id), request_id=req_id, run_mode="incremental"
    )
    _emit_aggregate_event(
        s, entity_id=str(new2_id), request_id=req_id, run_mode="incremental"
    )
    _emit_status_change_superseded(s, entity_id=str(old_id), request_id=req_id)

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success

    # Should produce one entry
    assert len(result.change_logs) == 1
    log = result.change_logs[0]
    assert log.run_mode == "incremental"

    added_contents = {snap.content for snap in log.added_agent_playbooks}
    assert added_contents == {"new A", "new B"}

    removed_contents = {snap.content for snap in log.removed_agent_playbooks}
    assert removed_contents == {"old content"}

    assert log.updated_agent_playbooks == []


# ---------------------------------------------------------------------------
# Test 2: full-archive run
# ---------------------------------------------------------------------------


def test_full_archive_run_mode(tmp_path):
    """Full-archive run: run_mode='full_archive' is captured from event reason."""
    s = _store(tmp_path)

    new_pb = _make_playbook(
        playbook_name="pb", agent_version="v1", content="arch content"
    )
    new_id = _add_playbook(s, new_pb)

    req_id = "run-full-1"
    _emit_aggregate_event(
        s, entity_id=str(new_id), request_id=req_id, run_mode="full_archive"
    )

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert len(result.change_logs) == 1
    assert result.change_logs[0].run_mode == "full_archive"
    assert len(result.change_logs[0].added_agent_playbooks) == 1
    assert result.change_logs[0].removed_agent_playbooks == []


# ---------------------------------------------------------------------------
# Test 3: APPROVED playbook absent from reconstruction.removed (S3 gate)
# ---------------------------------------------------------------------------


def test_approved_playbook_absent_from_reconstruction_removed(tmp_path):
    """S3 gate: APPROVED playbook absent from reconstruction.removed.

    Legacy removed may contain APPROVED playbooks (full_archive snapshots all
    APPROVED into legacy.removed). Reconstruction correctly excludes them because
    supersede_agent_playbooks_by_ids skips APPROVED rows — no status_change event
    is emitted for them, so they have no removal signal.

    Gate: reconstruction.removed ⊆ legacy.removed (by content) AND
          legacy.removed \\ reconstruction.removed ⊆ {APPROVED playbooks}.
    """
    s = _store(tmp_path)

    # A PENDING playbook that gets superseded (will appear in reconstruction.removed)
    pending_pb = _make_playbook(
        playbook_name="pb",
        agent_version="v1",
        content="pending content",
        playbook_status=PlaybookStatus.PENDING,
    )
    pending_id = _add_playbook(s, pending_pb)

    # An APPROVED playbook — supersede helper skips APPROVED (no event emitted)
    approved_pb = _make_playbook(
        playbook_name="pb",
        agent_version="v1",
        content="approved content",
        playbook_status=PlaybookStatus.APPROVED,
    )
    _add_playbook(s, approved_pb)  # present in storage but no removal signal (APPROVED)
    _set_superseded(s, pending_id)  # pending is tombstoned

    # New playbook added in this run
    new_pb = _make_playbook(
        playbook_name="pb", agent_version="v1", content="new content"
    )
    new_id = _add_playbook(s, new_pb)

    req_id = "run-s3"
    _emit_aggregate_event(
        s, entity_id=str(new_id), request_id=req_id, run_mode="full_archive"
    )
    # Emit supersede event only for pending (not approved — mirrors real behavior)
    _emit_status_change_superseded(s, entity_id=str(pending_id), request_id=req_id)

    # Simulate legacy log including BOTH pending and approved in removed
    legacy_removed_contents = {"pending content", "approved content"}

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert len(result.change_logs) == 1
    log = result.change_logs[0]

    recon_removed_contents = {snap.content for snap in log.removed_agent_playbooks}

    # S3 gate: reconstruction.removed ⊆ legacy.removed
    assert recon_removed_contents <= legacy_removed_contents, (
        f"reconstruction.removed={recon_removed_contents!r} not subset of "
        f"legacy.removed={legacy_removed_contents!r}"
    )

    # S3 gate: legacy.removed \\ reconstruction.removed ⊆ {APPROVED}
    delta = legacy_removed_contents - recon_removed_contents
    assert delta <= {"approved content"}, (
        f"delta={delta!r} contains non-APPROVED entries"
    )

    # reconstruction.added == legacy.added
    assert {snap.content for snap in log.added_agent_playbooks} == {"new content"}

    # APPROVED playbook must NOT be in reconstruction.removed
    assert "approved content" not in recon_removed_contents


# ---------------------------------------------------------------------------
# Test 4: Add-only run
# ---------------------------------------------------------------------------


def test_add_only_run(tmp_path):
    """Add-only run (aggregate events, no supersede) → added non-empty, removed=[]."""
    s = _store(tmp_path)

    pb1 = _make_playbook(playbook_name="pb", agent_version="v1", content="fact A")
    pb2 = _make_playbook(playbook_name="pb", agent_version="v1", content="fact B")
    id1 = _add_playbook(s, pb1)
    id2 = _add_playbook(s, pb2)

    req_id = "run-add-only"
    _emit_aggregate_event(s, entity_id=str(id1), request_id=req_id)
    _emit_aggregate_event(s, entity_id=str(id2), request_id=req_id)

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert len(result.change_logs) == 1
    log = result.change_logs[0]
    assert {snap.content for snap in log.added_agent_playbooks} == {"fact A", "fact B"}
    assert log.removed_agent_playbooks == []


# ---------------------------------------------------------------------------
# Test 5: Remove-only run
# ---------------------------------------------------------------------------


def test_remove_only_run(tmp_path):
    """Remove-only run (supersede events, no aggregate) → removed non-empty, added=[]."""
    s = _store(tmp_path)

    old_pb = _make_playbook(playbook_name="pb", agent_version="v1", content="old")
    old_id = _add_playbook(s, old_pb)
    _set_superseded(s, old_id)

    req_id = "run-remove-only"
    _emit_status_change_superseded(s, entity_id=str(old_id), request_id=req_id)

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert len(result.change_logs) == 1
    log = result.change_logs[0]
    assert {snap.content for snap in log.removed_agent_playbooks} == {"old"}
    assert log.added_agent_playbooks == []


# ---------------------------------------------------------------------------
# Test 6: Empty request_id events are skipped
# ---------------------------------------------------------------------------


def test_empty_request_id_events_skipped(tmp_path):
    """Events with empty request_id must not produce a change-log entry."""
    s = _store(tmp_path)

    pb = _make_playbook(playbook_name="pb", agent_version="v1", content="some content")
    pid = _add_playbook(s, pb)

    # Emit event with empty request_id
    s.append_lineage_event(
        LineageEvent(
            org_id=s.org_id,
            entity_type="agent_playbook",
            entity_id=str(pid),
            op="aggregate",
            source_ids=[],
            actor="aggregator",
            request_id="",  # empty
            reason="aggregate:incremental",
        )
    )

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert result.change_logs == [], (
        "empty request_id events must be skipped — no change-log entry"
    )


# ---------------------------------------------------------------------------
# Test 7: Purged tombstone → omitted, no crash
# ---------------------------------------------------------------------------


def test_purged_tombstone_omitted_no_crash(tmp_path):
    """Purged tombstone is silently omitted from removed, no crash occurs."""
    s = _store(tmp_path)

    old_pb = _make_playbook(playbook_name="pb", agent_version="v1", content="purged")
    old_id = _add_playbook(s, old_pb)
    _set_superseded(s, old_id)

    new_pb = _make_playbook(playbook_name="pb", agent_version="v1", content="survivor")
    new_id = _add_playbook(s, new_pb)

    req_id = "run-purge"
    _emit_aggregate_event(s, entity_id=str(new_id), request_id=req_id)
    _emit_status_change_superseded(s, entity_id=str(old_id), request_id=req_id)

    # Simulate GC: physically delete the tombstone row
    s.conn.execute("DELETE FROM agent_playbooks WHERE agent_playbook_id = ?", (old_id,))
    s.conn.commit()

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert len(result.change_logs) == 1
    log = result.change_logs[0]
    # Purged tombstone → silently omitted
    assert log.removed_agent_playbooks == []
    # Added still has the survivor
    assert len(log.added_agent_playbooks) == 1
    assert log.added_agent_playbooks[0].content == "survivor"


# ---------------------------------------------------------------------------
# Test 8: get_lineage_events(request_id=...) filter
# ---------------------------------------------------------------------------


def test_get_lineage_events_request_id_filter(tmp_path):
    """get_lineage_events(request_id=R) returns only events for run R."""
    s = _store(tmp_path)

    pb1 = _make_playbook(playbook_name="pb", agent_version="v1", content="c1")
    pb2 = _make_playbook(playbook_name="pb", agent_version="v1", content="c2")
    id1 = _add_playbook(s, pb1)
    id2 = _add_playbook(s, pb2)

    req_a = "run-filter-A"
    req_b = "run-filter-B"
    _emit_aggregate_event(s, entity_id=str(id1), request_id=req_a)
    _emit_aggregate_event(s, entity_id=str(id2), request_id=req_b)

    # Filter to req_a only
    events_a = s.get_lineage_events(
        entity_type="agent_playbook", org_id=s.org_id, request_id=req_a
    )
    assert all(e.request_id == req_a for e in events_a), (
        f"expected only events for {req_a!r}, got {[e.request_id for e in events_a]}"
    )
    assert len(events_a) == 1
    assert events_a[0].entity_id == str(id1)

    # Filter to req_b only
    events_b = s.get_lineage_events(
        entity_type="agent_playbook", org_id=s.org_id, request_id=req_b
    )
    assert all(e.request_id == req_b for e in events_b)
    assert len(events_b) == 1
    assert events_b[0].entity_id == str(id2)


# ---------------------------------------------------------------------------
# Test 9: limit=0 returns empty
# ---------------------------------------------------------------------------


def test_limit_zero_returns_empty(tmp_path):
    """limit=0 returns empty change_logs list."""
    s = _store(tmp_path)
    pb = _make_playbook(playbook_name="pb", agent_version="v1", content="c")
    pid = _add_playbook(s, pb)
    _emit_aggregate_event(s, entity_id=str(pid), request_id="run-limit")

    result = reconstruct_playbook_aggregation_change_log(s, limit=0)
    assert result.success
    assert result.change_logs == []


# ---------------------------------------------------------------------------
# Test 10: run_mode default when reason doesn't match "aggregate:" prefix
# ---------------------------------------------------------------------------


def test_run_mode_defaults_to_incremental_when_reason_absent(tmp_path):
    """When event reason does not start with 'aggregate:', run_mode defaults to 'incremental'."""
    s = _store(tmp_path)

    pb = _make_playbook(playbook_name="pb", agent_version="v1", content="c")
    pid = _add_playbook(s, pb)

    # Emit aggregate event with no reason prefix
    s.append_lineage_event(
        LineageEvent(
            org_id=s.org_id,
            entity_type="agent_playbook",
            entity_id=str(pid),
            op="aggregate",
            source_ids=[],
            actor="aggregator",
            request_id="run-no-reason",
            reason="",  # blank reason
        )
    )

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert len(result.change_logs) == 1
    assert result.change_logs[0].run_mode == "incremental"
