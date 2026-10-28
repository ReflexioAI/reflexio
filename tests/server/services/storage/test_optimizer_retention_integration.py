"""Capacity eviction must not invalidate a live optimizer's durable evidence."""

import sqlite3
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Literal
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain import (
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

TARGETS = (
    "playbook_optimization_jobs",
    "playbook_optimization_candidates",
    "playbook_optimization_evaluations",
    "playbook_optimization_events",
)


@pytest.fixture
def store(tmp_path, monkeypatch) -> Iterator[SQLiteStorage]:
    monkeypatch.setenv("OPENAI_API_KEY", "owned-unused-retention-key")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(
            org_id="optimizer-retention", db_path=str(tmp_path / "owned.db")
        )
    try:
        yield storage
    finally:
        storage.conn.close()


def seed(
    store: SQLiteStorage,
    status: Literal["pending", "running", "completed", "failed", "skipped"],
    created_at: int,
) -> PlaybookOptimizationJob:
    job = store.create_playbook_optimization_job(
        PlaybookOptimizationJob(
            optimizer_kind="gepa",
            target_kind="agent_playbook",
            target_id=1,
            status=status,
            created_at=created_at,
        )
    )
    candidate = store.insert_playbook_optimization_candidate(
        PlaybookOptimizationCandidate(
            job_id=job.job_id, content="owned evidence", created_at=created_at
        )
    )
    store.insert_playbook_optimization_evaluation(
        PlaybookOptimizationEvaluation(
            job_id=job.job_id,
            candidate_id=candidate.candidate_id,
            target_kind="agent_playbook",
            target_id=1,
            created_at=created_at,
        )
    )
    store.insert_playbook_optimization_event(
        PlaybookOptimizationEvent(
            job_id=job.job_id, event_type="evaluation", created_at=created_at
        )
    )
    return job


def snapshot(store: SQLiteStorage, job_id: int) -> list[list[tuple]]:
    return [
        [
            tuple(row)
            for row in store.conn.execute(
                f"SELECT * FROM {target} WHERE job_id = ? ORDER BY 1",  # noqa: S608 - literal fixture tables.
                (job_id,),
            ).fetchall()
        ]
        for target in TARGETS
    ]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("status", ["pending", "running"])
def test_live_history_survives_and_does_not_starve_terminal_cleanup(
    store, target, status
):
    live = seed(store, status, 1)
    terminal = seed(store, "completed", 2)
    before = snapshot(store, live.job_id)
    assert store.delete_oldest_retention_target_rows(target, 1) == 1
    assert snapshot(store, live.job_id) == before
    assert store.get_playbook_optimization_job(live.job_id) is not None
    if target == "playbook_optimization_jobs":
        assert all(not rows for rows in snapshot(store, terminal.job_id))
    else:
        assert store.count_retention_target_rows(target) == 1


@pytest.mark.parametrize("target", TARGETS)
def test_live_lease_protects_even_a_terminal_job(store, target):
    job = seed(store, "completed", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_owner = ?, lease_fence = 1, "
        "lease_expires_at = ? WHERE job_id = ?",
        ("owned-worker", int(time.time()) + 3600, job.job_id),
    )
    store.conn.commit()
    before = snapshot(store, job.job_id)
    assert store.delete_oldest_retention_target_rows(target, 1) == 0
    assert snapshot(store, job.job_id) == before


@pytest.mark.parametrize("target", TARGETS)
def test_expired_terminal_history_can_be_reclaimed(store, target):
    job = seed(store, "failed", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (int(time.time()) - 1, job.job_id),
    )
    store.conn.commit()
    assert store.delete_oldest_retention_target_rows(target, 1) == 1


def test_optimizer_cascade_rolls_back_if_parent_delete_fails(store, monkeypatch):
    job = seed(store, "completed", 1)
    before = snapshot(store, job.job_id)

    def fail(*_args):
        raise RuntimeError("owned target deletion failure")

    monkeypatch.setattr(store, "_retention_delete_target_rows", fail)
    with pytest.raises(StorageError, match="owned target deletion failure"):
        store.delete_oldest_retention_target_rows("playbook_optimization_jobs", 1)
    assert snapshot(store, job.job_id) == before


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("hold", ["publishing", "claim", "open_world"])
def test_publication_evidence_is_not_capacity_evictable(store, target, hold):
    job = seed(store, "completed", 1)
    if hold == "claim":
        store.conn.execute(
            "INSERT INTO user_playbook_publication_claims "
            "(optimizer_kind, job_id, owner, publication_fence, worker_fence, updated_at) "
            "VALUES ('gepa', ?, 'owned-worker', 1, 1, 1)",
            (job.job_id,),
        )
    elif hold == "publishing":
        store.conn.execute(
            "UPDATE playbook_optimization_jobs SET stage = 'publishing' WHERE job_id = ?",
            (job.job_id,),
        )
    else:
        store.conn.execute(
            "UPDATE playbook_optimization_jobs SET optimizer_kind = 'offline_tuner_open_world' WHERE job_id = ?",
            (job.job_id,),
        )
    store.conn.commit()
    before = snapshot(store, job.job_id)
    assert store.delete_oldest_retention_target_rows(target, 1) == 0
    assert snapshot(store, job.job_id) == before


def test_job_activation_commits_before_waiting_capacity_cleanup(store):
    job = seed(store, "completed", 1)
    filename = store.conn.execute("PRAGMA database_list").fetchone()[2]
    attempted = Event()
    store.conn.set_trace_callback(
        lambda statement: attempted.set() if statement == "BEGIN IMMEDIATE" else None
    )
    with sqlite3.connect(filename) as writer, ThreadPoolExecutor(max_workers=1) as pool:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "UPDATE playbook_optimization_jobs SET status = 'running' WHERE job_id = ?",
            (job.job_id,),
        )
        cleanup = pool.submit(
            store.delete_oldest_retention_target_rows, "playbook_optimization_jobs", 1
        )
        try:
            assert attempted.wait(timeout=5), (
                "cleanup never attempted its write transaction"
            )
        finally:
            writer.commit()
        assert cleanup.result(timeout=5) == 0
    assert store.get_playbook_optimization_job(job.job_id).status == "running"
