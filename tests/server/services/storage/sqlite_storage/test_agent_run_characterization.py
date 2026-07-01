"""Characterization tests for zero-/thin-coverage agent-run storage methods.

These pin the behavior a later verbatim mixin-decomposition move MUST preserve.
There is NO storage contract suite that touches agent_run, so these are the
primary safety net for the zero-coverage methods and the Style-B transaction
atomicity invariant.

Each test is written to be NON-VACUOUS: the module docstring on each test names
the specific guard/commit/rollback/cascade whose removal makes the test fail.
"""

import tempfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)


@pytest.fixture
def storage():
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id="org_1", db_path=f"{temp_dir}/reflexio.db")


def _binding(
    *,
    org_id: str = "org_1",
    user_id: str | None = "user_1",
    request_id: str = "request_1",
    extractor_kind: str = "profile",
) -> AgentBinding:
    return AgentBinding(
        org_id=org_id,
        extractor_kind=extractor_kind,
        user_id=user_id,
        request_id=request_id,
        agent_version="v1",
        source="api",
        source_interaction_ids=[1, 2],
        window_start_interaction_id=1,
        window_end_interaction_id=2,
        extractor_config_hash="hash_1",
    )


def _run(
    run_id: str,
    status: AgentRunStatus,
    *,
    binding: AgentBinding | None = None,
    **kwargs,
) -> AgentRunRecord:
    return AgentRunRecord(
        id=run_id,
        binding=binding or _binding(),
        status=status,
        generation_request_snapshot={"request_id": "request_1"},
        **kwargs,
    )


def _pending(
    call_id: str,
    *,
    now: datetime,
    org_id: str = "org_1",
    question: str = "What is the deployment target?",
    status: PendingToolCallStatus = PendingToolCallStatus.PENDING,
    **overrides,
) -> PendingToolCallRecord:
    scope = human_feedback_scope(org_id)
    record = PendingToolCallRecord(
        id=call_id,
        org_id=org_id,
        user_id="user_1",
        scope=scope,
        scope_hash=build_scope_hash(scope),
        tool_name="ask_human",
        dedup_key=build_pending_tool_call_dedup_key(
            tool_name="ask_human",
            question_text=question,
        ),
        status=status,
        question_text=question,
        args={"question": question},
        tags=["deployment"],
        expires_at=now + timedelta(hours=1),
        cache_until=now + timedelta(minutes=5),
    )
    return replace(record, **overrides) if overrides else record


# ---------------------------------------------------------------------------
# 1. create_or_attach_pending_tool_call — both branches + idempotency
# ---------------------------------------------------------------------------


def test_create_or_attach_creates_when_no_active_call(storage):
    """CREATE branch: with no active dedup match, the record is inserted and the
    dependency is attached in ONE Style-B transaction. Non-vacuous: if the
    two-table commit is dropped/reordered, either the pending call or its
    dependency would be missing (2-table consistency).
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))

    result = storage.create_or_attach_pending_tool_call(
        record=_pending("ptc_1", now=now),
        dependency=RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1"),
        now=now,
    )

    assert result.created is True
    assert result.pending_tool_call.id == "ptc_1"
    assert storage.get_pending_tool_call("ptc_1") is not None
    deps = storage.list_run_tool_dependencies("run_1")
    assert [d.pending_tool_call_id for d in deps] == ["ptc_1"]


def test_create_or_attach_attaches_to_existing_active_call(storage):
    """ATTACH branch: a second call with the SAME dedup key attaches to the
    existing active call instead of inserting a new one. Non-vacuous: if the
    `if row is None` create/attach guard is inverted or the dedup SELECT is
    dropped, ptc_2 would be inserted and `created` would be True.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_agent_run(_run("run_2", AgentRunStatus.FINALIZED_PENDING_TOOL))

    first = storage.create_or_attach_pending_tool_call(
        record=_pending("ptc_1", now=now),
        dependency=RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1"),
        now=now,
    )
    second = storage.create_or_attach_pending_tool_call(
        record=_pending("ptc_2", now=now),
        dependency=RunToolDependencyRecord(run_id="run_2", pending_tool_call_id="ptc_2"),
        now=now,
    )

    assert first.created is True
    assert second.created is False
    # Second call attached to the pre-existing active call, NOT its own record.
    assert second.pending_tool_call.id == "ptc_1"
    assert storage.get_pending_tool_call("ptc_2") is None
    # run_2's dependency points at the existing call (ptc_1), not ptc_2.
    assert [d.pending_tool_call_id for d in storage.list_run_tool_dependencies("run_2")] == [
        "ptc_1"
    ]


def test_create_or_attach_is_idempotent_on_repeat(storage):
    """Idempotency: the same (record, dependency) applied twice yields the same
    active call and exactly ONE dependency row. Non-vacuous: if the dep insert
    were `INSERT` instead of `INSERT OR IGNORE`, the repeat would raise a PK
    violation on (run_id, pending_tool_call_id).
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    dependency = RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")

    first = storage.create_or_attach_pending_tool_call(
        record=_pending("ptc_1", now=now), dependency=dependency, now=now
    )
    second = storage.create_or_attach_pending_tool_call(
        record=_pending("ptc_1", now=now), dependency=dependency, now=now
    )

    assert first.created is True
    assert second.created is False
    assert second.pending_tool_call.id == "ptc_1"
    assert len(storage.list_run_tool_dependencies("run_1")) == 1


# ---------------------------------------------------------------------------
# 2. cancel_pending_tool_call — cascade + no-op guard
# ---------------------------------------------------------------------------


def test_cancel_pending_tool_call_backfills_dependency_and_finalizes_run(storage):
    """Cancel flips PENDING->CANCELLED, backfills the dependency `resolved_at`,
    and the AR cascade (`_finalize_runs_without_pending_dependencies_unlocked`)
    finalizes the run whose last pending dependency just cleared. Non-vacuous:
    dropping the cascade call leaves the run FINALIZED_PENDING_TOOL; dropping the
    dependency backfill leaves resolved_at NULL.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_pending_tool_call(_pending("ptc_1", now=now))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )

    cancelled = storage.cancel_pending_tool_call("ptc_1", cancelled_at=now)

    assert cancelled is not None
    assert cancelled.status == PendingToolCallStatus.CANCELLED
    deps = storage.list_run_tool_dependencies("run_1")
    assert deps[0].resolved_at == now
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZED


def test_cancel_pending_tool_call_is_noop_on_already_resolved_call(storage):
    """The `WHERE ... AND status = PENDING` guard makes cancel a no-op on a
    RESOLVED call. Non-vacuous: dropping that status predicate would flip a
    live RESOLVED answer to CANCELLED, corrupting the cache.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(
        _pending(
            "ptc_1",
            now=now,
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "AWS ECS"},
            resolved_at=now - timedelta(minutes=1),
            valid_until=now + timedelta(hours=1),
        )
    )

    returned = storage.cancel_pending_tool_call("ptc_1", cancelled_at=now)

    assert returned is not None
    assert returned.status == PendingToolCallStatus.RESOLVED
    assert returned.result == {"answer": "AWS ECS"}


# ---------------------------------------------------------------------------
# 3. fail_running_agent_runs_for_request — status guard, user scope, rowcount
# ---------------------------------------------------------------------------


def test_fail_running_only_flips_running_and_resuming(storage):
    """Only RUNNING/RESUMING runs for the request flip to FAILED; a terminal
    FINALIZED run is untouched, and the rowcount reflects only the flipped rows.
    Non-vacuous: dropping the `status IN (RUNNING, RESUMING)` guard would flip
    the FINALIZED run too and return 3.
    """
    storage.create_agent_run(_run("run_running", AgentRunStatus.RUNNING))
    storage.create_agent_run(_run("run_resuming", AgentRunStatus.RESUMING))
    storage.create_agent_run(_run("run_finalized", AgentRunStatus.FINALIZED))

    flipped = storage.fail_running_agent_runs_for_request(
        org_id="org_1",
        extractor_kind="profile",
        user_id="user_1",
        request_id="request_1",
        last_error="boom",
    )

    assert flipped == 2
    assert storage.get_agent_run("run_running").status == AgentRunStatus.FAILED
    assert storage.get_agent_run("run_resuming").status == AgentRunStatus.FAILED
    assert storage.get_agent_run("run_finalized").status == AgentRunStatus.FINALIZED
    assert storage.get_agent_run("run_running").last_error == "boom"


def test_fail_running_respects_user_id_scope(storage):
    """The `user_id IS ?` predicate scopes the bulk fail to one user — another
    user's RUNNING run in the SAME org/request is NOT failed. Non-vacuous:
    dropping the user_id predicate would flip both and return 2.
    """
    storage.create_agent_run(
        _run("run_user_1", AgentRunStatus.RUNNING, binding=_binding(user_id="user_1"))
    )
    storage.create_agent_run(
        _run("run_user_2", AgentRunStatus.RUNNING, binding=_binding(user_id="user_2"))
    )

    flipped = storage.fail_running_agent_runs_for_request(
        org_id="org_1",
        extractor_kind="profile",
        user_id="user_1",
        request_id="request_1",
        last_error="boom",
    )

    assert flipped == 1
    assert storage.get_agent_run("run_user_1").status == AgentRunStatus.FAILED
    assert storage.get_agent_run("run_user_2").status == AgentRunStatus.RUNNING


# ---------------------------------------------------------------------------
# 4. consume_run_tool_dependencies — guards + idempotent double-consume
# ---------------------------------------------------------------------------


def test_consume_run_tool_dependencies_guards_and_double_consume(storage):
    """Only resolved-AND-unconsumed dependencies are consumed; a second consume
    is a no-op returning 0. Non-vacuous: dropping `resolved_at IS NOT NULL`
    would also consume the unresolved dep (first call returns 2); dropping
    `consumed_at IS NULL` would re-consume on the second call (returns 1).
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_run("run_1", AgentRunStatus.RESUME_READY))
    storage.create_pending_tool_call(_pending("ptc_resolved", now=now))
    storage.create_pending_tool_call(
        _pending("ptc_unresolved", now=now, question="Which region?")
    )
    # Resolved, unconsumed -> eligible.
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="run_1",
            pending_tool_call_id="ptc_resolved",
            resolved_at=now,
        )
    )
    # Unresolved -> ineligible.
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_unresolved")
    )

    first = storage.consume_run_tool_dependencies("run_1")
    second = storage.consume_run_tool_dependencies("run_1")

    assert first == 1
    assert second == 0
    deps = {d.pending_tool_call_id: d for d in storage.list_run_tool_dependencies("run_1")}
    assert deps["ptc_resolved"].consumed_at is not None
    assert deps["ptc_unresolved"].consumed_at is None


# ---------------------------------------------------------------------------
# 5. delete_expired_pending_tool_calls (the 23rd method; Style-B; zero coverage)
# ---------------------------------------------------------------------------


def test_delete_expired_pending_tool_calls_only_deletes_expired_past_grace(storage):
    """Only status='expired' rows whose expires_at is older than (now - grace)
    are deleted; the row count is returned. Non-vacuous: dropping the
    `status='expired'` guard would delete the pending/resolved rows too;
    dropping the grace cutoff would delete the within-grace expired row.
    """
    now_dt = datetime(2026, 5, 28, tzinfo=UTC)
    now_epoch = int(now_dt.timestamp())
    grace = 3600  # cutoff = 2026-05-27T23:00:00+00:00

    # Expired and past the grace window -> deleted.
    storage.create_pending_tool_call(
        _pending(
            "expired_old",
            now=now_dt,
            question="q_old",
            status=PendingToolCallStatus.EXPIRED,
            expires_at=now_dt - timedelta(days=1),
        )
    )
    # Expired but still inside the grace window (expires_at == now > cutoff) -> kept.
    storage.create_pending_tool_call(
        _pending(
            "expired_recent",
            now=now_dt,
            question="q_recent",
            status=PendingToolCallStatus.EXPIRED,
            expires_at=now_dt,
        )
    )
    # Non-expired statuses are never deleted, even with a past expires_at.
    storage.create_pending_tool_call(
        _pending(
            "still_pending",
            now=now_dt,
            question="q_pending",
            expires_at=now_dt - timedelta(days=1),
        )
    )
    storage.create_pending_tool_call(
        _pending(
            "resolved_cached",
            now=now_dt,
            question="q_resolved",
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "cached"},
            resolved_at=now_dt - timedelta(days=2),
            expires_at=now_dt - timedelta(days=1),
            valid_until=now_dt + timedelta(days=30),
        )
    )

    deleted = storage.delete_expired_pending_tool_calls(now=now_epoch, grace_seconds=grace)

    assert deleted == 1
    assert storage.get_pending_tool_call("expired_old") is None
    assert storage.get_pending_tool_call("expired_recent") is not None
    assert storage.get_pending_tool_call("still_pending") is not None
    assert storage.get_pending_tool_call("resolved_cached") is not None


# ---------------------------------------------------------------------------
# 6. Style-B mid-transaction ROLLBACK atomicity (central-invariant guard)
# ---------------------------------------------------------------------------


def test_expire_pending_tool_calls_rolls_back_all_tables_on_midtxn_failure(
    storage, monkeypatch
):
    """THE central-invariant guard. `expire_pending_tool_calls` is Style-B
    (`BEGIN IMMEDIATE` + `try/except: rollback; raise`). It writes
    _pending_tool_calls (status->expired) and _run_tool_dependencies
    (resolved_at) BEFORE calling the run-finalize cascade. We monkeypatch the
    cascade to raise mid-transaction and assert ALL affected tables are
    UNCHANGED — the `except: rollback` undid both writes.

    Non-vacuous: if a future verbatim move drops or reorders the `rollback`,
    the two earlier writes stay pending on the shared connection and are visible
    to the same-connection read-back -> the call would appear EXPIRED and the
    dependency resolved. Empirically confirmed by deleting the rollback line.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_pending_tool_call(
        _pending("ptc_1", now=now, expires_at=now - timedelta(seconds=1))
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("injected mid-transaction failure")

    monkeypatch.setattr(
        storage, "_finalize_runs_without_pending_dependencies_unlocked", _boom
    )

    with pytest.raises(StorageError):
        storage.expire_pending_tool_calls(now=now)

    # All three affected tables must be unchanged (rollback fired).
    pending = storage.get_pending_tool_call("ptc_1")
    assert pending is not None
    assert pending.status == PendingToolCallStatus.PENDING
    deps = storage.list_run_tool_dependencies("run_1")
    assert deps[0].resolved_at is None
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZED_PENDING_TOOL


# ---------------------------------------------------------------------------
# 7. count_unresolved_followup_dependencies — 3-table JOIN (zero coverage)
# ---------------------------------------------------------------------------


def test_count_unresolved_followup_dependencies_joins_three_tables(storage):
    """Counts dependencies that are unresolved AND unconsumed, whose pending
    call is PENDING for the given org+extractor_kind+tool_name (a 3-table JOIN
    across runs, deps, and pending calls). Non-vacuous: a resolved dep, a
    dep whose pending call is RESOLVED, and a run of a different extractor_kind
    must all be excluded — dropping any JOIN predicate over-counts.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_run("run_1", AgentRunStatus.RUNNING))
    storage.create_agent_run(_run("run_2", AgentRunStatus.RUNNING))
    storage.create_agent_run(
        _run(
            "run_other_kind",
            AgentRunStatus.RUNNING,
            binding=_binding(extractor_kind="success"),
        )
    )

    # Counts: unresolved + pending call PENDING, across two runs.
    storage.create_pending_tool_call(_pending("ptc_1", now=now, question="q1"))
    storage.create_pending_tool_call(_pending("ptc_2", now=now, question="q2"))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_2", pending_tool_call_id="ptc_2")
    )

    # Excluded: dependency already resolved.
    storage.create_pending_tool_call(_pending("ptc_resolved_dep", now=now, question="q3"))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="run_1", pending_tool_call_id="ptc_resolved_dep", resolved_at=now
        )
    )

    # Excluded: pending call is not PENDING (RESOLVED).
    storage.create_pending_tool_call(
        _pending(
            "ptc_call_resolved",
            now=now,
            question="q4",
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "x"},
            resolved_at=now,
            valid_until=now + timedelta(hours=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_2", pending_tool_call_id="ptc_call_resolved")
    )

    # Excluded: run of a different extractor_kind.
    storage.create_pending_tool_call(_pending("ptc_other_kind", now=now, question="q5"))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="run_other_kind", pending_tool_call_id="ptc_other_kind"
        )
    )

    count = storage.count_unresolved_followup_dependencies(
        org_id="org_1", extractor_kind="profile", tool_name="ask_human"
    )

    assert count == 2


# ---------------------------------------------------------------------------
# 8. list_pending_tool_calls — filter / clamp / order (zero coverage)
# ---------------------------------------------------------------------------


def test_list_pending_tool_calls_filters_clamps_and_orders(storage):
    """Filters by org (self.org_id) + optional status, clamps limit to
    [1, 500], and orders created_at DESC, id ASC. Non-vacuous: dropping the
    status filter includes the RESOLVED call; dropping the org filter includes
    the other-org call; dropping the `max(1, ...)` clamp makes limit=0 return
    nothing.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(_pending("ptc_a", now=now, question="qa"))
    storage.create_pending_tool_call(_pending("ptc_b", now=now, question="qb"))
    storage.create_pending_tool_call(
        _pending(
            "ptc_resolved",
            now=now,
            question="qc",
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "x"},
            resolved_at=now,
            valid_until=now + timedelta(hours=1),
        )
    )
    # A call in a different org must be excluded by the self.org_id filter.
    storage.create_pending_tool_call(
        _pending("ptc_other_org", now=now, org_id="org_other", question="qd")
    )

    all_calls = storage.list_pending_tool_calls()
    pending_only = storage.list_pending_tool_calls(status=PendingToolCallStatus.PENDING)
    resolved_only = storage.list_pending_tool_calls(status=PendingToolCallStatus.RESOLVED)
    clamped_low = storage.list_pending_tool_calls(limit=0)
    clamped_high = storage.list_pending_tool_calls(limit=100_000)

    all_ids = {c.id for c in all_calls}
    assert all_ids == {"ptc_a", "ptc_b", "ptc_resolved"}
    assert "ptc_other_org" not in all_ids
    assert {c.id for c in pending_only} == {"ptc_a", "ptc_b"}
    assert {c.id for c in resolved_only} == {"ptc_resolved"}
    # limit=0 clamps up to 1 (returns a row, not zero rows).
    assert len(clamped_low) == 1
    # limit far above the cap still returns everything in the org (cap >= 3).
    assert len(clamped_high) == 3
    # Ordering contract: created_at DESC, then id ASC.
    for earlier, later in zip(all_calls, all_calls[1:], strict=False):
        assert (earlier.created_at, "") >= (later.created_at, "")
        if earlier.created_at == later.created_at:
            assert earlier.id <= later.id


# ---------------------------------------------------------------------------
# 9. claim_ready_agent_run compare-and-set (RESUMING status guard, isolated)
# ---------------------------------------------------------------------------


def test_claim_ready_agent_run_resuming_status_guard_is_compare_and_set(storage):
    """Two sequential claims on one ready run: the first flips RESUME_READY ->
    RESUMING; the second returns None SOLELY because of the RESUMING status
    guard (we never consume the dependency, so the actionable-dependency EXISTS
    clause still holds — this isolates the status CAS from the consume guard).
    A RESUMING run older than the claim TTL is re-claimable.

    Non-vacuous: if the `status = RESUMING AND claimed_at < stale` staleness
    predicate were widened to always re-admit RESUMING, the second (fresh) claim
    would succeed instead of returning None.
    """
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_run("run_1", AgentRunStatus.RESUME_READY))
    storage.create_pending_tool_call(_pending("ptc_1", now=now))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )
    storage.resolve_pending_tool_call(
        "ptc_1", result={"answer": "AWS ECS"}, resolved_at=now, valid_for_seconds=3600
    )

    first = storage.claim_ready_agent_run(org_id="org_1", worker_id="worker_1", now=now)
    # Second claim WITHOUT consuming: the only thing blocking it is the status.
    second = storage.claim_ready_agent_run(org_id="org_1", worker_id="worker_2", now=now)
    # Once the RESUMING claim is stale (past the TTL) it is re-claimable.
    stale_now = now + timedelta(seconds=700)
    third = storage.claim_ready_agent_run(
        org_id="org_1", worker_id="worker_3", now=stale_now, claim_ttl_seconds=600
    )

    assert first is not None
    assert first.status == AgentRunStatus.RESUMING
    assert first.claimed_by == "worker_1"
    assert first.resume_attempts == 1
    assert second is None
    assert third is not None
    assert third.status == AgentRunStatus.RESUMING
    assert third.claimed_by == "worker_3"
    assert third.resume_attempts == 2
