"""Integration tests: reconstruct_playbook_aggregation_change_log — B3b Task 2.

Verifies the read-side reconstruction of PlaybookAggregationChangeLog from
lineage events, mirroring the profile reconstruction model.

Covered scenarios:
  1. 2-add / 1-remove incremental run — correct added/removed snapshots, run_mode.
  2. Full-archive run reconstructs added/removed; run_mode="full_archive".
  3. Add-only run (aggregate events, no supersede) → added-only, removed=[].
  4. Remove-only run (supersede events, no aggregate) → removed-only, added=[].
  5. Empty request_id events are skipped (never merged into a group).
  6. Purged tombstone (row hard-deleted after supersede) → omitted, no crash.
  7. get_lineage_events(request_id=R) returns only R's events (Part A filter).
  8. limit=0 returns empty change_logs.
  9. run_mode defaults to "incremental" when event reason has no "aggregate:" prefix.
  10. H2: reason="aggregate:" (empty suffix) → run_mode="incremental", no crash.
  11. H2: reason="aggregate:bogus" (unknown suffix) → run_mode="incremental", no crash.

Note: the S3 parity-gate test (formerly Test 3) was retired with the legacy
``playbook_aggregation_change_logs`` table (Track B Task 4). Parity is moot
once the legacy write path is stopped.
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
# Test 7b: added-then-superseded playbook still appears in its run's added side
# ---------------------------------------------------------------------------


def test_added_then_superseded_still_in_added(tmp_path):
    """A playbook added in run R1 and later superseded by R2 stays in R1's added side.

    Inverse of test_purged_tombstone_omitted_no_crash: while the tombstone still exists
    (not yet GC-purged), the added side is resolved with include_tombstones, so R1 is
    NOT dropped from the change log. (Pre-fix, R1's only added playbook resolved to None
    and R1 — having no removals — was dropped entirely.)
    """
    s = _store(tmp_path)

    # Run 1 adds playbook X.
    pb_x = _make_playbook(playbook_name="pb", agent_version="v1", content="X added in run1")
    x_id = _add_playbook(s, pb_x)
    _emit_aggregate_event(s, entity_id=str(x_id), request_id="run-1")

    # Run 2 supersedes X (tombstones the row + emits the superseded event).
    _set_superseded(s, x_id)
    _emit_status_change_superseded(s, entity_id=str(x_id), request_id="run-2")

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success

    run1 = next(
        (
            log
            for log in result.change_logs
            if any(
                snap.content == "X added in run1" for snap in log.added_agent_playbooks
            )
        ),
        None,
    )
    assert run1 is not None, (
        "Run 1 must remain in the change log with its added playbook, got "
        f"{[(cl.added_agent_playbooks, cl.removed_agent_playbooks) for cl in result.change_logs]}"
    )


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
# Test 9b: limit applies to the POST-filter set + short-circuits at the page
# ---------------------------------------------------------------------------


def test_limit_applies_to_post_filter_set(tmp_path):
    """``limit`` caps the FILTERED set and yields the most-recent matches.

    Seeds three matching (fb_A/v1) runs interleaved with a non-matching
    (fb_B/v2) run, then reconstructs with ``playbook_name``/``agent_version`` +
    ``limit=2``. The result must be exactly the two most-recent fb_A runs —
    proving the limit is applied AFTER filtering (not a pre-filter slice, the
    fb01ae2 contract) and that reconstruction stops once the page is full.
    """
    s = _store(tmp_path)

    def _run(name: str, version: str, content: str, request_id: str) -> None:
        pb = _make_playbook(playbook_name=name, agent_version=version, content=content)
        pid = _add_playbook(s, pb)
        _emit_aggregate_event(s, entity_id=str(pid), request_id=request_id)

    # Appended oldest -> newest; a non-matching fb_B run sits in the middle.
    _run("fb_A", "v1", "A oldest", "run-A1")
    _run("fb_A", "v1", "A middle", "run-A2")
    _run("fb_B", "v2", "B other", "run-B")
    _run("fb_A", "v1", "A newest", "run-A3")

    result = reconstruct_playbook_aggregation_change_log(
        s, limit=2, playbook_name="fb_A", agent_version="v1"
    )
    assert result.success
    assert len(result.change_logs) == 2, (
        f"limit=2 must cap the filtered set, got {len(result.change_logs)}"
    )
    # Most-recent-first: the two newest fb_A runs, oldest dropped, fb_B excluded.
    contents = [
        snap.content
        for log in result.change_logs
        for snap in log.added_agent_playbooks
    ]
    assert contents == ["A newest", "A middle"], contents
    assert all(log.playbook_name == "fb_A" for log in result.change_logs)


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


# ---------------------------------------------------------------------------
# Tests 11-12: H2 — validate run_mode suffix; empty/bogus → "incremental"
# ---------------------------------------------------------------------------


def test_run_mode_empty_suffix_falls_back_to_incremental(tmp_path):
    """H2: reason='aggregate:' (empty suffix) → run_mode='incremental', no crash.

    A future event with reason 'aggregate:' (prefix present, suffix empty)
    must not cause a ValidationError when constructing PlaybookAggregationChangeLog.
    """
    s = _store(tmp_path)

    pb = _make_playbook(
        playbook_name="pb", agent_version="v1", content="c-empty-suffix"
    )
    pid = _add_playbook(s, pb)

    s.append_lineage_event(
        LineageEvent(
            org_id=s.org_id,
            entity_type="agent_playbook",
            entity_id=str(pid),
            op="aggregate",
            source_ids=[],
            actor="aggregator",
            request_id="run-empty-suffix",
            reason="aggregate:",  # prefix present, suffix is empty string
        )
    )

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success, "must not crash on empty suffix"
    assert len(result.change_logs) == 1
    assert result.change_logs[0].run_mode == "incremental", (
        "empty suffix must fall back to 'incremental'"
    )


def test_run_mode_bogus_suffix_falls_back_to_incremental(tmp_path):
    """H2: reason='aggregate:<unknown>' → run_mode='incremental', no crash.

    An unrecognized suffix like 'aggregate:bogus' must not produce a
    ValidationError — it must fall back gracefully to 'incremental'.
    """
    s = _store(tmp_path)

    pb = _make_playbook(
        playbook_name="pb", agent_version="v1", content="c-bogus-suffix"
    )
    pid = _add_playbook(s, pb)

    s.append_lineage_event(
        LineageEvent(
            org_id=s.org_id,
            entity_type="agent_playbook",
            entity_id=str(pid),
            op="aggregate",
            source_ids=[],
            actor="aggregator",
            request_id="run-bogus-suffix",
            reason="aggregate:bogus",  # unrecognized suffix
        )
    )

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success, "must not crash on bogus suffix"
    assert len(result.change_logs) == 1
    assert result.change_logs[0].run_mode == "incremental", (
        "bogus suffix must fall back to 'incremental'"
    )


# ---------------------------------------------------------------------------
# Test 13: Multi-run grouping — core group-by-request_id invariant
# ---------------------------------------------------------------------------


def test_multi_run_grouping_distinct_request_ids(tmp_path):
    """Two runs under distinct request_ids produce exactly 2 change_logs with no cross-contamination.

    Run A: adds X, Y + supersedes Z.
    Run B: adds P + supersedes Q.
    Reconstruction must group each run's events separately.
    """
    s = _store(tmp_path)

    # --- Seed playbooks ---
    pb_z = _make_playbook(playbook_name="pb", agent_version="v1", content="Z old")
    pb_q = _make_playbook(playbook_name="pb", agent_version="v1", content="Q old")
    id_z = _add_playbook(s, pb_z)
    id_q = _add_playbook(s, pb_q)
    _set_superseded(s, id_z)
    _set_superseded(s, id_q)

    pb_x = _make_playbook(playbook_name="pb", agent_version="v1", content="X new")
    pb_y = _make_playbook(playbook_name="pb", agent_version="v1", content="Y new")
    pb_p = _make_playbook(playbook_name="pb", agent_version="v1", content="P new")
    id_x = _add_playbook(s, pb_x)
    id_y = _add_playbook(s, pb_y)
    id_p = _add_playbook(s, pb_p)

    req_a = "run-multi-A"
    req_b = "run-multi-B"

    # Run A events
    _emit_aggregate_event(s, entity_id=str(id_x), request_id=req_a)
    _emit_aggregate_event(s, entity_id=str(id_y), request_id=req_a)
    _emit_status_change_superseded(s, entity_id=str(id_z), request_id=req_a)

    # Run B events
    _emit_aggregate_event(s, entity_id=str(id_p), request_id=req_b)
    _emit_status_change_superseded(s, entity_id=str(id_q), request_id=req_b)

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert len(result.change_logs) == 2, (
        f"Expected 2 change_logs (one per run), got {len(result.change_logs)}"
    )

    logs_by_req: dict[str, object] = {}
    for log in result.change_logs:
        added_contents = {snap.content for snap in log.added_agent_playbooks}
        removed_contents = {snap.content for snap in log.removed_agent_playbooks}
        if "X new" in added_contents or "Y new" in added_contents:
            logs_by_req["A"] = (added_contents, removed_contents)
        elif "P new" in added_contents:
            logs_by_req["B"] = (added_contents, removed_contents)

    assert "A" in logs_by_req, "Run A log missing"
    assert "B" in logs_by_req, "Run B log missing"

    added_a, removed_a = logs_by_req["A"]  # type: ignore[misc]
    assert added_a == {"X new", "Y new"}, f"Run A added: {added_a}"
    assert removed_a == {"Z old"}, f"Run A removed: {removed_a}"
    # No cross-contamination from run B
    assert "P new" not in added_a
    assert "Q old" not in removed_a

    added_b, removed_b = logs_by_req["B"]  # type: ignore[misc]
    assert added_b == {"P new"}, f"Run B added: {added_b}"
    assert removed_b == {"Q old"}, f"Run B removed: {removed_b}"
    # No cross-contamination from run A
    assert "X new" not in added_b
    assert "Z old" not in removed_b


# ---------------------------------------------------------------------------
# Test 14: Non-matching status_change ops are NOT counted in removed
# ---------------------------------------------------------------------------


def test_archived_status_change_not_counted_in_removed(tmp_path):
    """status_change with to_status='archived' is NOT counted as a removal signal.

    Only status_change events with to_status='superseded' feed removed_agent_playbooks.
    An 'archived' transition (e.g. from an archive_agent_playbooks_by_ids call)
    must not appear in removed.
    """
    s = _store(tmp_path)

    pb = _make_playbook(playbook_name="pb", agent_version="v1", content="archived-pb")
    ap_id = _add_playbook(s, pb)

    # Simulate the archive step that happens before supersede in the aggregation pipeline
    s.archive_agent_playbooks_by_ids([ap_id])

    # Add an aggregate event so this request_id produces a non-empty change_log
    new_pb = _make_playbook(playbook_name="pb", agent_version="v1", content="new-pb")
    new_id = _add_playbook(s, new_pb)

    req_id = "run-archived-sc"
    _emit_aggregate_event(s, entity_id=str(new_id), request_id=req_id)

    # Emit a status_change event with to_status='archived' (NOT 'superseded')
    # under the same request_id — this must not be treated as a removal signal.
    s.append_lineage_event(
        LineageEvent(
            org_id=s.org_id,
            entity_type="agent_playbook",
            entity_id=str(ap_id),
            op="status_change",
            source_ids=[],
            actor="aggregator",
            request_id=req_id,
            from_status=None,
            to_status="archived",
            status_namespace="lifecycle_status",
        )
    )

    result = reconstruct_playbook_aggregation_change_log(s)
    assert result.success
    assert len(result.change_logs) == 1
    log = result.change_logs[0]

    # 'archived' status_change must NOT appear in removed_agent_playbooks
    assert log.removed_agent_playbooks == [], (
        f"archived status_change must not be counted as removal; "
        f"got removed={[s.content for s in log.removed_agent_playbooks]}"
    )
    # The aggregate event still shows up as added
    assert len(log.added_agent_playbooks) == 1
    assert log.added_agent_playbooks[0].content == "new-pb"
