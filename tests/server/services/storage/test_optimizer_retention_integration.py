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
    PlaybookOptimizationJob,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

GRAPH = "playbook_optimization_history"

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
    candidate_id = store.conn.execute(
        """INSERT INTO playbook_optimization_candidates
           (job_id, candidate_index, content, parent_candidate_ids,
            aggregate_score, is_winner, metadata_json, created_at)
           VALUES (?, 0, 'owned evidence', '[]', NULL, 0, '{}', ?)""",
        (job.job_id, created_at),
    ).lastrowid
    store.conn.execute(
        """INSERT INTO playbook_optimization_evaluations
           (job_id, candidate_id, target_kind, target_id,
            scenario_user_playbook_id, source_interaction_ids, score, verdict,
            likert, rationale, asi_json, incumbent_rollout_json,
            candidate_rollout_json, created_at)
           VALUES (?, ?, 'agent_playbook', 1, NULL, '[]', 0.0, 'tie', 0, '',
                   '{}', '[]', '[]', ?)""",
        (job.job_id, candidate_id, created_at),
    )
    store.conn.commit()
    insert_event(store, job.job_id, "evaluation", created_at)
    if status in ("pending", "running"):
        # Live work holds a lease; an unleased GEPA job is abandoned and retired.
        store.conn.execute(
            "UPDATE playbook_optimization_jobs SET lease_owner = 'owned-worker', "
            "lease_fence = 1, lease_expires_at = ? WHERE job_id = ?",
            (int(time.time()) + 3600, job.job_id),
        )
        store.conn.commit()
    return job


def insert_event(
    store: SQLiteStorage, job_id: int, event_type: str, created_at: int
) -> None:
    store.conn.execute(
        """INSERT INTO playbook_optimization_events
           (job_id, event_type, payload_json, created_at) VALUES (?, ?, '{}', ?)""",
        (job_id, event_type, created_at),
    )
    store.conn.commit()


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


@pytest.mark.parametrize("status", ["pending", "running"])
def test_live_history_survives_and_does_not_starve_terminal_cleanup(store, status):
    live = seed(store, status, 1)
    terminal = seed(store, "completed", 2)
    before = snapshot(store, live.job_id)
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 4
    assert snapshot(store, live.job_id) == before
    assert store.get_playbook_optimization_job(live.job_id) is not None
    assert all(not rows for rows in snapshot(store, terminal.job_id))


def test_live_lease_protects_even_a_terminal_job(store):
    job = seed(store, "completed", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_owner = ?, lease_fence = 1, "
        "lease_expires_at = ? WHERE job_id = ?",
        ("owned-worker", int(time.time()) + 3600, job.job_id),
    )
    store.conn.commit()
    before = snapshot(store, job.job_id)
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 0
    assert snapshot(store, job.job_id) == before


def test_expired_terminal_history_can_be_reclaimed(store):
    job = seed(store, "failed", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (int(time.time()) - 1, job.job_id),
    )
    store.conn.commit()
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 4


def test_optimizer_cascade_rolls_back_if_parent_delete_fails(store, monkeypatch):
    job = seed(store, "completed", 1)
    before = snapshot(store, job.job_id)

    def fail(*_args):
        raise RuntimeError("owned target deletion failure")

    monkeypatch.setattr(store, "_retention_delete_target_rows", fail)
    with pytest.raises(StorageError, match="owned target deletion failure"):
        store.delete_oldest_retention_target_rows(GRAPH, 1)
    assert snapshot(store, job.job_id) == before


@pytest.mark.parametrize("hold", ["publishing", "claim", "open_world"])
def test_publication_evidence_is_not_capacity_evictable(store, hold):
    job = seed(store, "completed", 1)
    if hold != "open_world":
        store.conn.execute(
            "UPDATE playbook_optimization_jobs SET lease_expires_at = ? "
            "WHERE job_id = ?",
            (int(time.time()) + 3600, job.job_id),
        )
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
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 0
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
            "UPDATE playbook_optimization_jobs SET status = 'running', "
            "lease_expires_at = ? WHERE job_id = ?",
            (int(time.time()) + 3600, job.job_id),
        )
        cleanup = pool.submit(store.delete_oldest_retention_target_rows, GRAPH, 1)
        try:
            assert attempted.wait(timeout=5), (
                "cleanup never attempted its write transaction"
            )
        finally:
            writer.commit()
        assert cleanup.result(timeout=5) == 0
    assert store.get_playbook_optimization_job(job.job_id).status == "running"


@pytest.mark.parametrize("hold", ["running_publishing", "claim"])
def test_migrate_retires_interrupted_gepa_jobs_so_retention_can_reclaim(store, hold):
    job = seed(store, "running", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ?, "
        "stage = ?, lease_owner = 'dead-worker' WHERE job_id = ?",
        (int(time.time()) - 60, "publishing" if hold != "claim" else None, job.job_id),
    )
    if hold == "claim":
        store.conn.execute(
            "UPDATE playbook_optimization_jobs SET status = 'completed' "
            "WHERE job_id = ?",
            (job.job_id,),
        )
        store.conn.execute(
            "INSERT INTO user_playbook_publication_claims "
            "(optimizer_kind, job_id, owner, publication_fence, worker_fence, "
            "updated_at) VALUES ('gepa', ?, 'dead-worker', 1, 1, 1)",
            (job.job_id,),
        )
    store.conn.commit()

    store.migrate()

    row = store.conn.execute(
        "SELECT status, stage, decision_reason, lease_owner "
        "FROM playbook_optimization_jobs WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert row["stage"] == "failed" and row["decision_reason"] == "retired_gepa"
    assert row["status"] in {"failed", "completed"} and row["lease_owner"] is None
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 4


def test_migrate_never_retires_a_live_gepa_lease(store):
    job = seed(store, "running", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ?, "
        "stage = 'publishing' WHERE job_id = ?",
        (int(time.time()) + 3600, job.job_id),
    )
    store.conn.commit()
    store.migrate()
    row = store.conn.execute(
        "SELECT status, stage FROM playbook_optimization_jobs WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert (row["status"], row["stage"]) == ("running", "publishing")


def test_future_lease_survives_migrate_then_is_retired_by_retention_without_reopen(
    store,
):
    job = seed(store, "running", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ?, "
        "stage = 'publishing', lease_owner = 'worker' WHERE job_id = ?",
        (int(time.time()) + 3600, job.job_id),
    )
    store.conn.commit()
    store.migrate()
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 0
    row = store.conn.execute(
        "SELECT status, stage FROM playbook_optimization_jobs WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert (row["status"], row["stage"]) == ("running", "publishing")

    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (int(time.time()) - 1, job.job_id),
    )
    store.conn.commit()
    assert store.delete_oldest_retention_target_rows(GRAPH, 1) == 4


def test_retirement_holds_the_writer_lock_so_no_claim_can_interleave(store, tmp_path):
    job = seed(store, "running", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (int(time.time()) - 60, job.job_id),
    )
    store.conn.commit()
    other = sqlite3.connect(str(tmp_path / "owned.db"), timeout=0)
    outcomes: list[str] = []
    real_time = time.time

    def probe() -> float:
        try:
            other.execute(
                "UPDATE playbook_optimization_jobs SET lease_owner = 'thief', "
                "lease_expires_at = ? WHERE job_id = ?",
                (int(real_time()) + 3600, job.job_id),
            )
            outcomes.append("claimed")
        except sqlite3.OperationalError:
            outcomes.append("locked")
        return real_time()

    with patch("reflexio.server.services.storage.sqlite_storage._base.time") as clock:
        clock.time.side_effect = probe
        store._retire_unfinished_gepa_jobs()
    other.close()
    assert outcomes == ["locked"]
    row = store.conn.execute(
        "SELECT lease_owner, decision_reason FROM playbook_optimization_jobs "
        "WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert row["lease_owner"] is None and row["decision_reason"] == "retired_gepa"


def test_retention_never_commits_another_threads_open_transaction(store):
    job = seed(store, "completed", 1)
    store.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (int(time.time()) - 1, job.job_id),
    )
    store.conn.commit()
    live = seed(store, "running", 2)
    started, release = Event(), Event()
    result: list[int] = []

    def writer() -> None:
        with store._lock:
            store.conn.execute("BEGIN IMMEDIATE")
            store.conn.execute(
                "UPDATE playbook_optimization_jobs SET decision_reason = 'uncommitted' "
                "WHERE job_id = ?",
                (live.job_id,),
            )
            started.set()
            assert release.wait(timeout=10)
            store.conn.rollback()

    def cleanup() -> None:
        store._retire_unfinished_gepa_jobs()
        result.append(store.delete_oldest_retention_target_rows(GRAPH, 1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(writer)
        assert started.wait(timeout=5)
        b = pool.submit(cleanup)
        time.sleep(0.5)
        release.set()
        a.result(timeout=10)
        b.result(timeout=10)
    # The writer's rolled-back change never became durable, and the cleanup ran
    # only after the writer released the connection.
    assert result == [4]
    with sqlite3.connect(
        store.conn.execute("PRAGMA database_list").fetchone()[2]
    ) as fresh:
        assert (
            fresh.execute(
                "SELECT COUNT(*) FROM playbook_optimization_jobs "
                "WHERE decision_reason = 'uncommitted'"
            ).fetchone()[0]
            == 0
        )
