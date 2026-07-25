"""SQLite contracts for durable replay optimizer jobs and artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema import service_schemas as schemas
from reflexio.server.services.storage.error import (
    OptimizationJobLeaseLiveError,
    StorageError,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage(tmp_path: Path) -> Generator[BaseStorage]:
    store = SQLiteStorage(
        org_id="optimization-replay-contract",
        db_path=str(tmp_path / "reflexio.db"),
    )
    try:
        yield store
    finally:
        store.conn.close()


def _replay_job(
    discovery_key: str,
    attempt_key: str,
) -> schemas.PlaybookOptimizationJob:
    return schemas.PlaybookOptimizationJob(
        optimizer_kind="offline_tuner_replay",
        target_kind="user_playbook",
        target_id=41,
        discovery_key=discovery_key,
        attempt_key=attempt_key,
        stage="evidence_frozen",
        expected_population_manifest_digest="a" * 64,
        generation_selection_manifest_digest="b" * 64,
        replay_manifest_digest="c" * 64,
        candidate_content_digest="d" * 64,
        search_projection_digest="e" * 64,
        publication_scope_digest="f" * 64,
    )


def _gepa_user_publication_job(target_id: int = 41) -> schemas.PlaybookOptimizationJob:
    return schemas.PlaybookOptimizationJob(
        optimizer_kind="gepa",
        target_kind="user_playbook",
        target_id=target_id,
        status="running",
        stage="publishing",
        best_candidate_id=17,
        metadata_json="{}",
        attempt_key="gepa-user-attempt",
        lease_owner="worker-a",
        lease_fence=1,
        lease_expires_at=2_000,
    )


def _artifact(
    *,
    job_id: int,
    digest: str | None = None,
    content_json: str = '{"eligible_ids":[1,2]}',
):
    canonical = json.dumps(
        json.loads(content_json),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return schemas.PlaybookOptimizationArtifact(
        job_id=job_id,
        artifact_kind="expected_population_manifest",
        content_json=content_json,
        content_digest=digest or sha256(canonical.encode()).hexdigest(),
    )


def test_replay_job_model_exposes_typed_durable_fields() -> None:
    job = _replay_job("d1", "a1")

    assert job.optimizer_kind == "offline_tuner_replay"
    assert job.stage == "evidence_frozen"
    assert job.lease_fence == 0
    assert job.terminal_outcome is None
    assert job.expected_population_manifest_digest == "a" * 64


def test_same_discovery_key_returns_one_active_job(storage: BaseStorage) -> None:
    first = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    second = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))

    assert second.job_id == first.job_id


def test_same_attempt_key_returns_one_active_job(storage: BaseStorage) -> None:
    first = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    second = storage.create_or_get_playbook_optimization_job(_replay_job("d2", "a1"))

    assert second.job_id == first.job_id


def test_conflicting_active_identity_is_rejected(storage: BaseStorage) -> None:
    storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))

    with pytest.raises(StorageError, match="immutable optimizer job identity"):
        storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a2"))


def test_gepa_publication_reclaim_contract_has_none_live_and_reclaimed_outcomes(
    storage: BaseStorage,
) -> None:
    assert (
        storage.reclaim_gepa_user_playbook_publishing_job(
            41, "worker-b", lease_seconds=60, now=2_001
        )
        is None
    )

    job = storage.create_playbook_optimization_job(_gepa_user_publication_job())
    with pytest.raises(OptimizationJobLeaseLiveError):
        storage.reclaim_gepa_user_playbook_publishing_job(
            41, "worker-b", lease_seconds=60, now=1_999
        )

    reclaimed = storage.reclaim_gepa_user_playbook_publishing_job(
        41, "worker-b", lease_seconds=60, now=2_001
    )
    assert reclaimed is not None
    assert reclaimed.job_id == job.job_id
    assert reclaimed.lease_owner == "worker-b"
    assert reclaimed.lease_fence == 2
    assert reclaimed.lease_expires_at == 2_061


def test_stale_lease_fence_cannot_advance_stage(storage: BaseStorage) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=1_000,
    )
    reclaimed = storage.reclaim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-b",
        now=claim.expires_at + 1,
        lease_seconds=60,
    )

    assert reclaimed.fence > claim.fence
    assert (
        storage.advance_playbook_optimization_stage(
            job_id=job.job_id,
            fence=claim.fence,
            stage="candidate_generated",
            now=claim.expires_at + 1,
        )
        is False
    )
    assert (
        storage.advance_playbook_optimization_stage(
            job_id=job.job_id,
            fence=reclaimed.fence,
            stage="candidate_generated",
            now=claim.expires_at + 1,
        )
        is True
    )


def test_current_owner_can_renew_but_stale_owner_cannot(storage: BaseStorage) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=2_000,
    )
    renewed = storage.renew_playbook_optimization_job_lease(
        job_id=job.job_id,
        owner="worker-a",
        fence=claim.fence,
        lease_seconds=120,
        now=2_010,
    )

    assert renewed.expires_at == 2_130
    with pytest.raises(StorageError, match="lease is no longer current"):
        storage.renew_playbook_optimization_job_lease(
            job_id=job.job_id,
            owner="worker-b",
            fence=claim.fence,
            lease_seconds=120,
            now=2_020,
        )


def test_stage_advancement_is_linear(storage: BaseStorage) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=3_000,
    )

    assert (
        storage.advance_playbook_optimization_stage(
            job_id=job.job_id,
            fence=claim.fence,
            stage="replay_running",
            now=3_001,
        )
        is False
    )
    assert (
        storage.advance_playbook_optimization_stage(
            job_id=job.job_id,
            fence=claim.fence,
            stage="candidate_generated",
            now=3_001,
        )
        is True
    )


@pytest.mark.parametrize(
    ("stage", "outcome", "expected_status"),
    [
        ("abstained", "candidate_did_not_improve", "skipped"),
        ("failed", "generation_failed", "failed"),
    ],
)
def test_terminal_stage_records_outcome_and_releases_lease(
    storage: BaseStorage,
    stage: schemas.OptimizationJobStage,
    outcome: schemas.OptimizationTerminalOutcome,
    expected_status: str,
) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=3_000,
    )

    assert storage.advance_playbook_optimization_stage(
        job_id=job.job_id,
        fence=claim.fence,
        stage=stage,
        terminal_outcome=outcome,
        now=3_001,
    )
    assert isinstance(storage, SQLiteStorage)
    row = storage.conn.execute(
        "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?", (job.job_id,)
    ).fetchone()
    assert row["stage"] == stage
    assert row["terminal_outcome"] == outcome
    assert row["status"] == expected_status
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None


def test_stale_lease_fence_cannot_write_singleton_artifact(
    storage: BaseStorage,
) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=4_000,
    )
    reclaimed = storage.reclaim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-b",
        lease_seconds=60,
        now=claim.expires_at + 1,
    )

    with pytest.raises(StorageError, match="lease is no longer current"):
        storage.upsert_playbook_optimization_artifact(
            _artifact(job_id=job.job_id),
            fence=claim.fence,
            now=claim.expires_at + 1,
        )

    saved = storage.upsert_playbook_optimization_artifact(
        _artifact(job_id=job.job_id),
        fence=reclaimed.fence,
        now=claim.expires_at + 1,
    )
    assert saved.job_id == job.job_id


def test_artifact_upsert_canonicalizes_equivalent_json_and_requires_digest_and_content(
    storage: BaseStorage,
) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=5_000,
    )
    first = storage.upsert_playbook_optimization_artifact(
        _artifact(
            job_id=job.job_id,
            content_json='{"eligible_ids":[1,2],"meta":{"b":2,"a":1}}',
        ),
        fence=claim.fence,
        now=5_001,
    )
    second = storage.upsert_playbook_optimization_artifact(
        _artifact(
            job_id=job.job_id,
            content_json='{ "meta" : { "a" : 1, "b" : 2 }, "eligible_ids" : [1,2] }',
        ),
        fence=claim.fence,
        now=5_001,
    )

    assert second.artifact_id == first.artifact_id
    assert second.content_json == '{"eligible_ids":[1,2],"meta":{"a":1,"b":2}}'

    with pytest.raises(StorageError, match="artifact digest conflict"):
        storage.upsert_playbook_optimization_artifact(
            _artifact(
                job_id=job.job_id,
                digest="b" * 64,
                content_json='{"eligible_ids":[1,2],"meta":{"b":2,"a":1}}',
            ),
            fence=claim.fence,
            now=5_001,
        )
    with pytest.raises(StorageError, match="artifact content conflict"):
        storage.upsert_playbook_optimization_artifact(
            _artifact(
                job_id=job.job_id,
                digest=first.content_digest,
                content_json='{"eligible_ids":[3]}',
            ),
            fence=claim.fence,
            now=5_001,
        )
    with pytest.raises(StorageError, match="artifact digest conflict"):
        storage.upsert_playbook_optimization_artifact(
            _artifact(
                job_id=job.job_id,
                digest="b" * 64,
                content_json='{"eligible_ids":[3]}',
            ),
            fence=claim.fence,
            now=5_001,
        )


def test_artifact_model_rejects_malformed_json() -> None:
    with pytest.raises(
        ValidationError, match="artifact content_json must be valid JSON"
    ):
        schemas.PlaybookOptimizationArtifact(
            job_id=1,
            artifact_kind="expected_population_manifest",
            content_json="{",
            content_digest="a" * 64,
        )


def _create_legacy_optimizer_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE playbook_optimization_jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_kind TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            best_candidate_id INTEGER,
            successor_target_id INTEGER,
            decision_reason TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE playbook_optimization_candidates (
            candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            parent_candidate_ids TEXT NOT NULL DEFAULT '[]',
            aggregate_score REAL,
            is_winner INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at INTEGER NOT NULL
        );
        CREATE TABLE playbook_optimization_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at INTEGER NOT NULL
        );
        """
    )
    jobs = [
        (1, "running", {"offline_tuner": {}}),
        (
            2,
            "pending",
            {
                "source_window_count": 3,
                "train_window_count": 2,
                "validation_window_count": 1,
            },
        ),
        (
            3,
            "running",
            {
                "offline_tuner": {},
                "source_window_count": 3,
                "train_window_count": 2,
                "validation_window_count": 1,
            },
        ),
        (4, "pending", {}),
        (5, "completed", {}),
        (6, "pending", {}),
    ]
    conn.executemany(
        """INSERT INTO playbook_optimization_jobs
           (job_id, target_kind, target_id, status, metadata_json, created_at, updated_at)
           VALUES (?, 'user_playbook', ?, ?, ?, 1, 1)""",
        [
            (job_id, job_id, status, json.dumps(metadata))
            for job_id, status, metadata in jobs
        ],
    )
    conn.execute(
        """INSERT INTO playbook_optimization_events
           (job_id, event_type, created_at) VALUES (5, 'offline_tuner_selected', 1)"""
    )
    conn.execute(
        """INSERT INTO playbook_optimization_candidates
           (job_id, content, metadata_json, created_at)
           VALUES (6, 'candidate', '{"proposed_edit": {}}', 1)"""
    )
    conn.commit()
    conn.close()


def test_legacy_optimizer_rows_are_classified_mutually_exclusively(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.db"
    _create_legacy_optimizer_schema(db_path)

    store = SQLiteStorage(org_id="legacy-classification", db_path=str(db_path))
    try:
        rows = store.conn.execute(
            """SELECT job_id, optimizer_kind, status, decision_reason
               FROM playbook_optimization_jobs ORDER BY job_id"""
        ).fetchall()
    finally:
        store.conn.close()

    assert [row["optimizer_kind"] for row in rows] == [
        "offline_tuner_legacy",
        "gepa",
        "optimizer_legacy_unknown",
        "optimizer_legacy_unknown",
        "offline_tuner_legacy",
        "offline_tuner_legacy",
    ]
    assert (rows[0]["status"], rows[0]["decision_reason"]) == (
        "skipped",
        "retired_by_replay_redesign",
    )
    assert rows[1]["status"] == "pending"
    assert rows[2]["status"] == "skipped"
    assert rows[3]["status"] == "skipped"
    assert rows[4]["status"] == "completed"
    assert rows[5]["status"] == "skipped"


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("optimizer_kind", "not-an-optimizer"),
        ("stage", "not-a-stage"),
        ("terminal_outcome", "not-an-outcome"),
    ],
)
def test_upgraded_legacy_optimizer_schema_rejects_invalid_durable_values(
    tmp_path: Path,
    column: str,
    invalid_value: str,
) -> None:
    db_path = tmp_path / "legacy.db"
    _create_legacy_optimizer_schema(db_path)
    store = SQLiteStorage(org_id="legacy-constraints", db_path=str(db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                f"UPDATE playbook_optimization_jobs SET {column} = ? WHERE job_id = 2",  # noqa: S608
                (invalid_value,),
            )
    finally:
        store.conn.close()
