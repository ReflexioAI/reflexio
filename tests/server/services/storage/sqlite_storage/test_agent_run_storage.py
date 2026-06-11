import tempfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._agent_run import _dt
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


def test_sqlite_agent_run_parser_treats_offsetless_timestamp_as_utc():
    parsed = _dt("2026-06-10T23:47:07.50016")

    assert parsed == datetime(2026, 6, 10, 23, 47, 7, 500160, tzinfo=UTC)


def _agent_run(
    run_id: str, status: AgentRunStatus, *, org_id: str = "org_1"
) -> AgentRunRecord:
    return AgentRunRecord(
        id=run_id,
        binding=AgentBinding(
            org_id=org_id,
            extractor_kind="profile",
            user_id="user_1",
            request_id="request_1",
            agent_version="v1",
            source="api",
            source_interaction_ids=[1, 2],
            window_start_interaction_id=1,
            window_end_interaction_id=2,
            extractor_config_hash="hash_1",
        ),
        status=status,
        generation_request_snapshot={"request_id": "request_1"},
    )


def _pending_call(
    call_id: str, *, now: datetime, org_id: str = "org_1"
) -> PendingToolCallRecord:
    scope = human_feedback_scope(org_id)
    return PendingToolCallRecord(
        id=call_id,
        org_id=org_id,
        user_id="user_1",
        scope=scope,
        scope_hash=build_scope_hash(scope),
        tool_name="ask_human",
        dedup_key=build_pending_tool_call_dedup_key(
            tool_name="ask_human",
            question_text="What is the deployment target?",
        ),
        status=PendingToolCallStatus.PENDING,
        question_text="What is the deployment target?",
        args={"question": "What is the deployment target?"},
        tags=["deployment"],
        expires_at=now + timedelta(hours=1),
        cache_until=now + timedelta(minutes=5),
    )


def test_sqlite_agent_run_crud_round_trip(storage):
    created = storage.create_agent_run(
        replace(
            _agent_run("run_1", AgentRunStatus.RUNNING),
            max_steps_remaining=8,
        )
    )
    loaded = storage.get_agent_run("run_1")

    assert loaded == created
    assert loaded is not None
    assert loaded.max_steps_remaining == 8
    assert loaded.binding.source_interaction_ids == [1, 2]
    assert loaded.generation_request_snapshot == {"request_id": "request_1"}


def test_sqlite_update_agent_run_status_records_lifecycle_timestamps(storage):
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.RUNNING))

    completed = storage.update_agent_run_status(
        "run_1",
        AgentRunStatus.AGENT_COMPLETED,
        committed_output={"profiles": []},
        max_steps_remaining=6,
    )
    finalized = storage.update_agent_run_status("run_1", AgentRunStatus.FINALIZED)

    assert completed is not None
    assert completed.agent_completed_at is not None
    assert completed.max_steps_remaining == 6
    assert completed.finalized_at is None
    assert finalized is not None
    assert finalized.agent_completed_at is not None
    assert finalized.finalized_at is not None


def test_sqlite_update_agent_run_status_honors_expected_statuses(storage):
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.FAILED))

    updated = storage.update_agent_run_status(
        "run_1",
        AgentRunStatus.AGENT_COMPLETED,
        committed_output={"profiles": []},
        expected_statuses=(AgentRunStatus.RUNNING, AgentRunStatus.RESUMING),
    )

    assert updated is not None
    assert updated.status == AgentRunStatus.FAILED
    assert updated.committed_output is None
    assert updated.agent_completed_at is None


def test_sqlite_pending_tool_call_active_dedup_lookup(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    pending = storage.create_pending_tool_call(_pending_call("ptc_1", now=now))

    found = storage.find_active_pending_tool_call(
        org_id="org_1",
        scope_hash=pending.scope_hash,
        tool_name="ask_human",
        dedup_key=pending.dedup_key,
        now=now,
    )

    assert found is not None
    assert found.id == "ptc_1"
    assert found.scope == {"org_id": "org_1", "scope_kind": "org"}
    assert found.user_id == "user_1"


def test_sqlite_resolve_marks_linked_finalized_run_resume_ready(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )

    storage.resolve_pending_tool_call(
        "ptc_1",
        result={"answer": "AWS ECS"},
        resolved_at=now,
        valid_for_seconds=3600,
    )

    run = storage.get_agent_run("run_1")
    deps = storage.list_run_tool_dependencies("run_1")
    pending = storage.get_pending_tool_call("ptc_1")

    assert run is not None
    assert run.status == AgentRunStatus.RESUME_READY
    assert deps[0].resolved_at == now
    assert pending is not None
    assert pending.status == PendingToolCallStatus.RESOLVED
    assert pending.result == {"answer": "AWS ECS"}


def test_sqlite_update_resolved_answer_resets_consumed_dependency(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.FINALIZED))
    storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_1", now=now),
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "not applicable", "not_applicable": True},
            resolved_at=now - timedelta(minutes=10),
            valid_until=now + timedelta(hours=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="run_1",
            pending_tool_call_id="ptc_1",
            resolved_at=now - timedelta(minutes=10),
            consumed_at=now - timedelta(minutes=5),
        )
    )

    updated = storage.update_resolved_pending_tool_call_result(
        "ptc_1",
        result={"answer": "AWS ECS"},
        resolved_at=now,
        valid_for_seconds=3600,
    )

    run = storage.get_agent_run("run_1")
    deps = storage.list_run_tool_dependencies("run_1")
    assert updated is not None
    assert updated.result == {"answer": "AWS ECS"}
    assert run is not None
    assert run.status == AgentRunStatus.RESUME_READY
    assert deps[0].resolved_at == now
    assert deps[0].consumed_at is None


def test_sqlite_mark_not_applicable_finalizes_run_with_only_na_dependencies(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )

    updated = storage.mark_pending_tool_call_not_applicable(
        "ptc_1",
        resolved_at=now,
        valid_for_seconds=3600,
    )

    run = storage.get_agent_run("run_1")
    deps = storage.list_run_tool_dependencies("run_1")
    assert updated is not None
    assert updated.status == PendingToolCallStatus.RESOLVED
    assert updated.result == {
        "answer": "User does not have information about this question.",
        "not_applicable": True,
    }
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZED
    assert deps[0].resolved_at == now
    assert deps[0].consumed_at == now


def test_sqlite_mark_not_applicable_keeps_mixed_pending_run_waiting(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_2", now=now),
            question_text="Which region?",
            args={"question": "Which region?"},
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text="Which region?",
            ),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_2")
    )

    storage.mark_pending_tool_call_not_applicable(
        "ptc_1",
        resolved_at=now,
        valid_for_seconds=3600,
    )

    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZED_PENDING_TOOL


def test_sqlite_mark_not_applicable_keeps_other_actionable_dependency_ready(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_2", now=now),
            question_text="Which region?",
            args={"question": "Which region?"},
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text="Which region?",
            ),
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "us-west-2"},
            resolved_at=now - timedelta(minutes=1),
            valid_until=now + timedelta(hours=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="run_1",
            pending_tool_call_id="ptc_2",
            resolved_at=now - timedelta(minutes=1),
        )
    )

    storage.mark_pending_tool_call_not_applicable(
        "ptc_1",
        resolved_at=now,
        valid_for_seconds=3600,
    )

    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.RESUME_READY


def test_sqlite_resolve_supersedes_older_valid_answer_for_same_dedup_key(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_old", now=now),
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "old"},
            resolved_at=now - timedelta(days=1),
            valid_until=now + timedelta(days=30),
        )
    )
    new = storage.create_pending_tool_call(_pending_call("ptc_new", now=now))

    storage.resolve_pending_tool_call(
        new.id,
        result={"answer": "new"},
        resolved_at=now,
        valid_for_seconds=3600,
    )

    old = storage.get_pending_tool_call("ptc_old")
    resolved = storage.get_pending_tool_call("ptc_new")
    matches = storage.search_prior_tool_calls(
        org_id="org_1",
        scope_hash=new.scope_hash,
        tool_name="ask_human",
        now=now,
        limit=10,
    )

    assert old is not None
    assert old.status == PendingToolCallStatus.SUPERSEDED
    assert old.superseded_by == "ptc_new"
    assert resolved is not None
    assert resolved.status == PendingToolCallStatus.RESOLVED
    assert [match.pending_tool_call_id for match in matches] == ["ptc_new"]


def test_sqlite_expire_pending_tool_calls_finalizes_unresolved_runs(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.FINALIZED_PENDING_TOOL))
    storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_1", now=now),
            expires_at=now - timedelta(seconds=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )

    expired = storage.expire_pending_tool_calls(now=now)

    run = storage.get_agent_run("run_1")
    deps = storage.list_run_tool_dependencies("run_1")
    pending = storage.get_pending_tool_call("ptc_1")

    assert expired == 1
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZED
    assert pending is not None
    assert pending.status == PendingToolCallStatus.EXPIRED
    assert deps[0].resolved_at == now


def test_sqlite_claim_requires_resolved_unconsumed_dependency(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.RESUME_READY))
    storage.create_pending_tool_call(_pending_call("ptc_1", now=now))
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )

    assert (
        storage.claim_ready_agent_run(org_id="org_1", worker_id="worker_1", now=now)
        is None
    )

    storage.resolve_pending_tool_call(
        "ptc_1",
        result={"answer": "AWS ECS"},
        resolved_at=now,
        valid_for_seconds=3600,
    )
    # A worker for a different org must NOT claim org_1's ready run.
    assert (
        storage.claim_ready_agent_run(org_id="other_org", worker_id="worker_x", now=now)
        is None
    )
    claimed = storage.claim_ready_agent_run(
        org_id="org_1", worker_id="worker_1", now=now
    )

    assert claimed is not None
    assert claimed.id == "run_1"
    assert claimed.status == AgentRunStatus.RESUMING
    assert claimed.claimed_by == "worker_1"
    assert claimed.resume_attempts == 1

    assert storage.consume_run_tool_dependencies("run_1") == 1
    assert (
        storage.claim_ready_agent_run(org_id="org_1", worker_id="worker_2", now=now)
        is None
    )


def test_sqlite_claim_ignores_not_applicable_dependencies(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", AgentRunStatus.RESUME_READY))
    storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_1", now=now),
            status=PendingToolCallStatus.RESOLVED,
            result={
                "answer": "User does not have information about this question.",
                "not_applicable": True,
            },
            resolved_at=now,
            valid_until=now + timedelta(hours=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="run_1",
            pending_tool_call_id="ptc_1",
            resolved_at=now,
        )
    )

    assert (
        storage.claim_ready_agent_run(org_id="org_1", worker_id="worker_1", now=now)
        is None
    )


def test_sqlite_claim_finalization_failed_requires_committed_output(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(
        _agent_run("run_without_output", AgentRunStatus.FINALIZATION_FAILED)
    )
    storage.create_agent_run(
        replace(
            _agent_run("run_with_output", AgentRunStatus.FINALIZATION_FAILED),
            committed_output={"profiles": []},
        )
    )

    claimed = storage.claim_finalization_failed_agent_run(
        org_id="org_1",
        worker_id="worker_1",
        now=now,
    )

    assert claimed is not None
    assert claimed.id == "run_with_output"
    assert claimed.status == AgentRunStatus.FINALIZING
    assert claimed.claimed_by == "worker_1"
    assert (
        storage.claim_finalization_failed_agent_run(
            org_id="org_1", worker_id="worker_2", now=now
        )
        is None
    )


def test_sqlite_finalization_claim_reclaims_only_stale_agent_completed(storage):
    """Orphaned AGENT_COMPLETED runs (publish-time finalize crashed) are
    reclaimed for finalization retry once stale, but never while fresh."""
    storage.create_agent_run(
        replace(
            _agent_run("orphan_run", AgentRunStatus.AGENT_COMPLETED),
            committed_output={"profiles": []},
        )
    )

    # A freshly AGENT_COMPLETED run (in the publish path's finalize window) must
    # NOT be reclaimed — otherwise the worker would race the in-flight publish.
    fresh = datetime.now(UTC)
    assert (
        storage.claim_finalization_failed_agent_run(
            org_id="org_1", worker_id="worker_1", now=fresh, claim_ttl_seconds=600
        )
        is None
    )

    # Once older than the claim TTL, the orphan is reclaimed for finalization.
    stale = fresh + timedelta(seconds=700)
    claimed = storage.claim_finalization_failed_agent_run(
        org_id="org_1", worker_id="worker_1", now=stale, claim_ttl_seconds=600
    )
    assert claimed is not None
    assert claimed.id == "orphan_run"
    assert claimed.status == AgentRunStatus.FINALIZING


def test_sqlite_list_resumable_work_org_ids_scans_all_orgs(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    # org_a: a run ready to resume.
    storage.create_agent_run(
        _agent_run("ready_run", AgentRunStatus.RESUME_READY, org_id="org_a")
    )
    storage.create_pending_tool_call(
        replace(
            _pending_call("ready_call", now=now, org_id="org_a"),
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "AWS ECS"},
            resolved_at=now,
            valid_until=now + timedelta(hours=1),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id="ready_run",
            pending_tool_call_id="ready_call",
            resolved_at=now,
        )
    )
    # org_b: a run awaiting finalization retry.
    storage.create_agent_run(
        replace(
            _agent_run(
                "fin_failed_run", AgentRunStatus.FINALIZATION_FAILED, org_id="org_b"
            ),
            committed_output={"profiles": []},
        )
    )
    # org_c: a pending tool call already past its expiry.
    storage.create_pending_tool_call(
        replace(
            _pending_call("expired_call", now=now, org_id="org_c"),
            expires_at=now - timedelta(minutes=1),
            cache_until=now - timedelta(minutes=2),
        )
    )
    # org_d: a terminal FINALIZED run with no follow-up work — must NOT surface.
    storage.create_agent_run(
        _agent_run("done_run", AgentRunStatus.FINALIZED, org_id="org_d")
    )

    org_ids = storage.list_resumable_work_org_ids(now=now)

    assert set(org_ids) == {"org_a", "org_b", "org_c"}
    assert "org_d" not in org_ids


def test_sqlite_search_prior_tool_calls_filters_by_org_scope_and_validity(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    valid = storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_valid", now=now),
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "AWS ECS"},
            resolved_at=now - timedelta(days=1),
            valid_until=now + timedelta(days=30),
            embedding=[1.0, 0.0],
        )
    )
    storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_older_same_question", now=now),
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "Old target"},
            resolved_at=now - timedelta(days=10),
            valid_until=now + timedelta(days=30),
            embedding=[1.0, 0.0],
        )
    )
    storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_expired", now=now),
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "Old answer"},
            resolved_at=now - timedelta(days=2),
            valid_until=now - timedelta(seconds=1),
            embedding=[0.0, 1.0],
        )
    )
    pending = storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_pending", now=now),
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text="Which compliance target applies?",
            ),
            question_text="Which compliance target applies?",
        )
    )

    matches = storage.search_prior_tool_calls(
        org_id="org_1",
        scope_hash=valid.scope_hash,
        tool_name="ask_human",
        query_embedding=[1.0, 0.0],
        now=now,
        limit=10,
    )

    assert {match.pending_tool_call_id for match in matches} == {
        valid.id,
        pending.id,
    }
    assert matches[0].pending_tool_call_id == valid.id
    assert matches[0].similarity == 1.0


def test_sqlite_search_prior_tool_calls_scores_before_limit(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    scope = human_feedback_scope("org_1")
    for index in range(12):
        question = f"Recent low-similarity question {index}"
        storage.create_pending_tool_call(
            replace(
                _pending_call(f"ptc_recent_{index}", now=now),
                dedup_key=build_pending_tool_call_dedup_key(
                    tool_name="ask_human",
                    question_text=question,
                ),
                question_text=question,
                status=PendingToolCallStatus.RESOLVED,
                result={"answer": "low relevance"},
                resolved_at=now - timedelta(minutes=index),
                valid_until=now + timedelta(days=30),
                embedding=[0.0, 1.0],
            )
        )
    old_relevant = storage.create_pending_tool_call(
        replace(
            _pending_call("ptc_old_relevant", now=now),
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text="Older high-similarity question",
            ),
            question_text="Older high-similarity question",
            status=PendingToolCallStatus.RESOLVED,
            result={"answer": "high relevance"},
            resolved_at=now - timedelta(days=14),
            valid_until=now + timedelta(days=30),
            embedding=[1.0, 0.0],
        )
    )

    matches = storage.search_prior_tool_calls(
        org_id="org_1",
        scope_hash=build_scope_hash(scope),
        tool_name="ask_human",
        query_embedding=[1.0, 0.0],
        now=now,
        limit=1,
    )

    assert [match.pending_tool_call_id for match in matches] == [old_relevant.id]
    assert matches[0].similarity == 1.0
