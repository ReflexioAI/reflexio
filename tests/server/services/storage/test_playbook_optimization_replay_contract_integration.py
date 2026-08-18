"""SQLite contracts for durable replay optimizer jobs and artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema import service_schemas as schemas
from reflexio.server.services.playbook.publication import canonical_json_bytes
from reflexio.server.services.storage.error import (
    OptimizationArtifactIntegrityError,
    OptimizationJobIdentityConflictError,
    OptimizationJobLeaseLiveError,
    StorageError,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


_STAGE_PATHS_BY_OPTIMIZER: dict[str, tuple[str, ...]] = {
    "offline_tuner_replay": (
        "evidence_frozen",
        "candidate_generated",
        "replay_running",
        "replay_evaluated",
        "publishing",
    ),
    "offline_tuner_open_world": (
        "evidence_frozen",
        "discovery_analyzed",
        "candidate_generated",
        "held_out_analyzed",
    ),
}
_ORDINARY_STAGES = (
    "discovery_analyzed",
    "candidate_generated",
    "replay_running",
    "replay_evaluated",
    "held_out_analyzed",
    "publishing",
)
_INVALID_STAGE_REQUESTS = (
    ("evidence_frozen", None),
    ("failed", None),
    ("abstained", None),
    ("unknown_stage", None),
    ("unknown_stage", "unknown_terminal_outcome"),
    ("failed", "unknown_terminal_outcome"),
    ("abstained", "unknown_terminal_outcome"),
    *(
        (stage, "infrastructure_failure")
        for stage in ("evidence_frozen", *_ORDINARY_STAGES)
    ),
)
_ALL_TERMINAL_OUTCOMES = (
    "applied",
    "insufficient_negative_evidence",
    "insufficient_positive_evidence",
    "insufficient_coverage",
    "replay_unsupported",
    "deployment_unsupported",
    "incomplete_replay_scope",
    "insufficient_replay_cases",
    "replay_inconclusive",
    "candidate_regressed",
    "candidate_did_not_improve",
    "incumbent_changed",
    "generation_failed",
    "replay_failed",
    "publication_failed",
    "governance_erased",
    "no_grounded_hypothesis",
    "analyst_unqualified",
    "heldout_evidence_failed",
    "stale_incumbent",
    "governance_invalidated",
    "infrastructure_failure",
)
_TERMINAL_OUTCOMES_BY_OPTIMIZER: dict[str, dict[str, frozenset[str]]] = {
    "offline_tuner_replay": {
        "failed": frozenset(
            {
                "generation_failed",
                "replay_failed",
                "publication_failed",
                "infrastructure_failure",
            }
        ),
        "abstained": frozenset(
            {
                "insufficient_negative_evidence",
                "insufficient_positive_evidence",
                "insufficient_coverage",
                "replay_unsupported",
                "deployment_unsupported",
                "incomplete_replay_scope",
                "insufficient_replay_cases",
                "replay_inconclusive",
                "candidate_regressed",
                "candidate_did_not_improve",
                "incumbent_changed",
            }
        ),
    },
    "offline_tuner_open_world": {
        "failed": frozenset({"infrastructure_failure"}),
        "abstained": frozenset(
            {
                "no_grounded_hypothesis",
                "analyst_unqualified",
                "heldout_evidence_failed",
                "stale_incumbent",
                "governance_invalidated",
            }
        ),
    },
}


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
    canonical = canonical_json_bytes(json.loads(content_json)).decode()
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

    with pytest.raises(
        OptimizationJobIdentityConflictError,
        match="immutable optimizer job identity",
    ):
        storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a2"))


def test_sqlite_persists_open_world_optimizer_jobs(storage: BaseStorage) -> None:
    open_world_job = _replay_job("open-world-discovery", "open-world-attempt")
    open_world_job.optimizer_kind = "offline_tuner_open_world"

    saved = storage.create_or_get_playbook_optimization_job(open_world_job)

    assert saved.optimizer_kind == "offline_tuner_open_world"


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
        ("failed", "replay_failed", "failed"),
        ("failed", "publication_failed", "failed"),
        ("failed", "infrastructure_failure", "failed"),
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

    with pytest.raises(StorageError, match="digest does not match content"):
        storage.upsert_playbook_optimization_artifact(
            first.model_copy(update={"content_digest": "b" * 64}),
            fence=claim.fence,
            now=5_001,
        )
    with pytest.raises(StorageError, match="digest does not match content"):
        storage.upsert_playbook_optimization_artifact(
            first.model_copy(update={"content_json": '{"eligible_ids":[3]}'}),
            fence=claim.fence,
            now=5_001,
        )
    with pytest.raises(
        OptimizationArtifactIntegrityError,
        match="artifact digest conflict",
    ):
        storage.upsert_playbook_optimization_artifact(
            _artifact(
                job_id=job.job_id,
                content_json='{"eligible_ids":[3]}',
            ),
            fence=claim.fence,
            now=5_001,
        )


def test_malformed_persisted_artifact_raises_typed_integrity_error(
    storage: BaseStorage,
) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=5_000,
    )
    saved = storage.upsert_playbook_optimization_artifact(
        _artifact(job_id=job.job_id),
        fence=claim.fence,
        now=5_001,
    )
    assert isinstance(storage, SQLiteStorage)
    storage.conn.execute(
        "UPDATE playbook_optimization_artifacts SET content_digest = ? "
        "WHERE artifact_id = ?",
        ("f" * 64, saved.artifact_id),
    )
    storage.conn.commit()

    with pytest.raises(
        OptimizationArtifactIntegrityError,
        match="optimizer artifact row is malformed",
    ):
        storage.get_playbook_optimization_artifact(
            job.job_id,
            "expected_population_manifest",
        )


def test_artifact_storage_failures_remain_generic(storage: BaseStorage) -> None:
    assert isinstance(storage, SQLiteStorage)
    storage.conn.execute("DROP TABLE playbook_optimization_artifacts")

    with pytest.raises(StorageError) as raised:
        storage.get_playbook_optimization_artifact(
            1,
            "expected_population_manifest",
        )

    assert type(raised.value) is StorageError


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


def test_artifact_model_rejects_digest_for_different_canonical_content() -> None:
    with pytest.raises(ValidationError, match="artifact digest must match"):
        schemas.PlaybookOptimizationArtifact(
            job_id=1,
            artifact_kind="expected_population_manifest",
            content_json='{"eligible_ids":[1,2]}',
            content_digest=sha256(b'{"eligible_ids":[2,1]}').hexdigest(),
        )


def test_artifact_model_binds_digest_to_canonical_equivalent_json() -> None:
    digest = sha256(b'{"eligible_ids":[1,2],"meta":{"a":1,"b":2}}').hexdigest()

    artifact = schemas.PlaybookOptimizationArtifact(
        job_id=1,
        artifact_kind="expected_population_manifest",
        content_json='{ "meta": {"b": 2, "a": 1}, "eligible_ids": [1, 2] }',
        content_digest=digest,
    )

    assert artifact.content_json == '{"eligible_ids":[1,2],"meta":{"a":1,"b":2}}'
    assert artifact.content_digest == digest


def test_artifact_model_uses_rfc8785_utf16_key_order() -> None:
    canonical = '{"\U00010000":"astral","\ue000":"bmp"}'

    artifact = schemas.PlaybookOptimizationArtifact(
        job_id=1,
        artifact_kind="expected_population_manifest",
        content_json='{"\ue000":"bmp","\U00010000":"astral"}',
        content_digest=sha256(canonical.encode()).hexdigest(),
    )

    assert artifact.content_json == canonical


def test_artifact_model_uses_publication_numeric_contract() -> None:
    canonical = '{"maximum":9007199254740991,"minimum":-9007199254740991}'
    artifact = schemas.PlaybookOptimizationArtifact(
        job_id=1,
        artifact_kind="expected_population_manifest",
        content_json='{ "minimum": -9007199254740991, "maximum": 9007199254740991 }',
        content_digest=sha256(canonical.encode()).hexdigest(),
    )

    assert artifact.content_json == canonical
    with pytest.raises(
        ValidationError, match="artifact content_json must be valid JSON"
    ):
        schemas.PlaybookOptimizationArtifact(
            job_id=1,
            artifact_kind="expected_population_manifest",
            content_json='{"value":1.0}',
            content_digest="a" * 64,
        )


def test_previous_artifact_schema_is_upgraded_without_losing_constraints(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "previous-artifact-schema.db"
    initial_store = SQLiteStorage(
        org_id="previous-artifact-schema", db_path=str(db_path)
    )
    initial_store.conn.close()

    legacy_content = '{"source":"legacy"}'
    legacy_digest = sha256(legacy_content.encode()).hexdigest()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("DROP INDEX idx_poa_job")
    conn.execute("DROP TABLE playbook_optimization_artifacts")
    conn.executescript(
        """
        CREATE TABLE playbook_optimization_artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            artifact_kind TEXT NOT NULL CHECK (artifact_kind IN (
                'expected_population_manifest',
                'generation_selection',
                'replay_manifest',
                'candidate',
                'candidate_search_projection'
            )),
            content_json TEXT NOT NULL,
            content_digest TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            UNIQUE (job_id, artifact_kind),
            FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
                ON DELETE CASCADE
        );
        CREATE INDEX idx_poa_job ON playbook_optimization_artifacts(job_id);
        """
    )
    conn.execute(
        """INSERT INTO playbook_optimization_jobs (
               job_id, optimizer_kind, target_kind, target_id, status,
               lease_owner, lease_fence, lease_expires_at, created_at, updated_at
           ) VALUES (
               41, 'offline_tuner_replay', 'user_playbook', 9, 'running',
               'worker-a', 3, 1000, 101, 102
           )"""
    )
    conn.execute(
        """INSERT INTO playbook_optimization_artifacts (
               artifact_id, job_id, artifact_kind, content_json, content_digest,
               created_at, updated_at
           ) VALUES (77, 41, 'candidate', ?, ?, 103, 104)""",
        (legacy_content, legacy_digest),
    )
    conn.commit()
    conn.close()

    store = SQLiteStorage(org_id="previous-artifact-schema", db_path=str(db_path))
    try:
        legacy_row = store.conn.execute(
            "SELECT * FROM playbook_optimization_artifacts WHERE artifact_id = 77"
        ).fetchone()
        assert dict(legacy_row) == {
            "artifact_id": 77,
            "job_id": 41,
            "artifact_kind": "candidate",
            "content_json": legacy_content,
            "content_digest": legacy_digest,
            "created_at": 103,
            "updated_at": 104,
        }
        assert store.conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                """INSERT INTO playbook_optimization_artifacts VALUES
                   (78, 41, 'candidate', '{}', ?, 105, 106)""",
                (sha256(b"{}").hexdigest(),),
            )
        store.conn.rollback()
        assert (
            store.conn.execute(
                """SELECT 1 FROM sqlite_master
               WHERE type = 'index' AND name = 'idx_poa_job'"""
            ).fetchone()
            is not None
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                """INSERT INTO playbook_optimization_artifacts VALUES
                   (79, 41, 'unknown_kind', '{}', ?, 105, 106)""",
                (sha256(b"{}").hexdigest(),),
            )
        store.conn.rollback()

        for artifact_kind in (
            "open_world_evidence_bundle",
            "open_world_discovery_memo",
            "open_world_candidate",
            "open_world_attempt_decision",
        ):
            content_json = f'{{"artifact_kind":"{artifact_kind}"}}'
            artifact = schemas.PlaybookOptimizationArtifact(
                job_id=41,
                artifact_kind=artifact_kind,
                content_json=content_json,
                content_digest=sha256(content_json.encode()).hexdigest(),
                created_at=107,
                updated_at=108,
            )
            saved = store.upsert_playbook_optimization_artifact(
                artifact,
                fence=3,
                now=500,
            )
            assert store.get_playbook_optimization_artifact(41, artifact_kind) == saved

        assert store.migrate() is True
        store.conn.execute("DELETE FROM playbook_optimization_jobs WHERE job_id = 41")
        store.conn.commit()
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM playbook_optimization_artifacts WHERE job_id = 41"
            ).fetchone()[0]
            == 0
        )
    finally:
        store.conn.close()

    assert not hasattr(BaseStorage, "load_open_world_evidence_snapshot")
    assert not hasattr(SQLiteStorage, "load_open_world_evidence_snapshot")


def test_optimizer_job_rebuild_preserves_deleted_id_high_water_and_repeats(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-job-sequence.db"
    _create_legacy_optimizer_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM playbook_optimization_jobs WHERE job_id IN (5, 6)")
    conn.commit()
    conn.close()

    first_store = SQLiteStorage(org_id="job-sequence-first", db_path=str(db_path))
    first = first_store.create_playbook_optimization_job(_replay_job("d1", "a1"))
    first_store.conn.close()
    second_store = SQLiteStorage(org_id="job-sequence-second", db_path=str(db_path))
    second = second_store.create_playbook_optimization_job(
        _replay_job("d2", "a2").model_copy(update={"target_id": 42})
    )
    second_store.conn.close()

    assert first.job_id == 7
    assert second.job_id == 8


def test_empty_optimizer_job_rebuild_preserves_deleted_id_high_water(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "empty-legacy-job-sequence.db"
    _create_legacy_optimizer_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM playbook_optimization_jobs")
    conn.commit()
    conn.close()

    store = SQLiteStorage(org_id="empty-job-sequence", db_path=str(db_path))
    job = store.create_playbook_optimization_job(_replay_job("d1", "a1"))
    store.conn.close()

    assert job.job_id == 7


def test_artifact_rebuild_preserves_deleted_id_high_water_and_repeats(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-artifact-sequence.db"
    store = SQLiteStorage(org_id="artifact-sequence-setup", db_path=str(db_path))
    parent = store.create_playbook_optimization_job(_replay_job("d1", "a1"))
    store.conn.execute("DROP INDEX idx_poa_job")
    store.conn.execute("DROP TABLE playbook_optimization_artifacts")
    store.conn.executescript(
        """
        CREATE TABLE playbook_optimization_artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            artifact_kind TEXT NOT NULL CHECK (artifact_kind IN ('candidate')),
            content_json TEXT NOT NULL,
            content_digest TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            UNIQUE (job_id, artifact_kind),
            FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
                ON DELETE CASCADE
        );
        CREATE INDEX idx_poa_job ON playbook_optimization_artifacts(job_id);
        """
    )
    digest = sha256(b"{}").hexdigest()
    store.conn.execute(
        "INSERT INTO playbook_optimization_artifacts VALUES "
        "(9, ?, 'candidate', '{}', ?, 1, 1)",
        (parent.job_id, digest),
    )
    store.conn.execute("DELETE FROM playbook_optimization_artifacts")
    store.conn.commit()
    store.conn.close()

    first_store = SQLiteStorage(org_id="artifact-sequence-first", db_path=str(db_path))
    first_store.conn.execute(
        "INSERT INTO playbook_optimization_artifacts "
        "(job_id, artifact_kind, content_json, content_digest, created_at, updated_at) "
        "VALUES (?, 'candidate', '{}', ?, 1, 1)",
        (parent.job_id, digest),
    )
    first_id = first_store.conn.execute(
        "SELECT artifact_id FROM playbook_optimization_artifacts"
    ).fetchone()[0]
    first_store.conn.execute("DELETE FROM playbook_optimization_artifacts")
    first_store.conn.commit()
    first_store.conn.close()

    second_store = SQLiteStorage(
        org_id="artifact-sequence-second", db_path=str(db_path)
    )
    second_store.conn.execute(
        "INSERT INTO playbook_optimization_artifacts "
        "(job_id, artifact_kind, content_json, content_digest, created_at, updated_at) "
        "VALUES (?, 'candidate', '{}', ?, 1, 1)",
        (parent.job_id, digest),
    )
    second_id = second_store.conn.execute(
        "SELECT artifact_id FROM playbook_optimization_artifacts"
    ).fetchone()[0]
    second_store.conn.close()

    assert first_id == 10
    assert second_id == 11


@pytest.mark.parametrize(
    "current_stage",
    [
        "evidence_frozen",
        "candidate_generated",
        "replay_running",
        "replay_evaluated",
    ],
)
def test_applied_terminal_stage_requires_publishing(
    storage: BaseStorage,
    current_stage: schemas.OptimizationJobStage,
) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=6_000,
    )
    assert isinstance(storage, SQLiteStorage)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET stage = ? WHERE job_id = ?",
        (current_stage, job.job_id),
    )
    storage.conn.commit()

    assert (
        storage.advance_playbook_optimization_stage(
            job_id=job.job_id,
            fence=claim.fence,
            stage="applied",
            terminal_outcome="applied",
            now=6_001,
        )
        is False
    )
    persisted = storage.get_playbook_optimization_job(job.job_id)
    assert persisted is not None
    assert persisted.stage == current_stage
    assert persisted.status == "running"
    assert persisted.terminal_outcome is None


def test_applied_terminal_stage_advances_from_publishing(storage: BaseStorage) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=6_000,
    )
    assert isinstance(storage, SQLiteStorage)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET stage = 'publishing' WHERE job_id = ?",
        (job.job_id,),
    )
    storage.conn.commit()

    assert storage.advance_playbook_optimization_stage(
        job_id=job.job_id,
        fence=claim.fence,
        stage="applied",
        terminal_outcome="applied",
        now=6_001,
    )
    persisted = storage.get_playbook_optimization_job(job.job_id)
    assert persisted is not None
    assert persisted.stage == "applied"
    assert persisted.status == "completed"
    assert persisted.terminal_outcome == "applied"


def test_governance_erased_terminal_outcome_round_trips(
    storage: BaseStorage,
) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    assert isinstance(storage, SQLiteStorage)
    storage.conn.execute(
        """UPDATE playbook_optimization_jobs
           SET status = 'skipped', stage = 'failed',
               terminal_outcome = 'governance_erased'
           WHERE job_id = ?""",
        (job.job_id,),
    )
    storage.conn.commit()

    persisted = storage.get_playbook_optimization_job(job.job_id)

    assert persisted is not None
    assert persisted.terminal_outcome == "governance_erased"


def test_ordinary_stage_advance_rejects_governance_erased(
    storage: BaseStorage,
) -> None:
    job = storage.create_or_get_playbook_optimization_job(_replay_job("d1", "a1"))
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=6_000,
    )

    assert (
        storage.advance_playbook_optimization_stage(
            job_id=job.job_id,
            fence=claim.fence,
            stage="abstained",
            terminal_outcome="governance_erased",
            now=6_001,
        )
        is False
    )
    persisted = storage.get_playbook_optimization_job(job.job_id)
    assert persisted is not None
    assert persisted.status == "running"
    assert persisted.stage == "evidence_frozen"
    assert persisted.terminal_outcome is None


def _claimed_job_at_stage(
    storage: BaseStorage,
    stage: schemas.OptimizationJobStage,
    *,
    now: int,
    optimizer_kind: schemas.OptimizerKind = "offline_tuner_replay",
    target_id: int = 41,
) -> tuple[int, int]:
    job = _replay_job(f"d-{target_id}", f"a-{target_id}")
    job.optimizer_kind = optimizer_kind
    job.target_id = target_id
    job = storage.create_or_get_playbook_optimization_job(job)
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=now,
    )
    assert isinstance(storage, SQLiteStorage)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET stage = ? WHERE job_id = ?",
        (stage, job.job_id),
    )
    storage.conn.commit()
    return job.job_id, claim.fence


def _optimization_job_row(storage: BaseStorage, job_id: int) -> dict[str, object]:
    assert isinstance(storage, SQLiteStorage)
    row = storage.conn.execute(
        "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    assert row is not None
    return dict(row)


def test_invalid_stage_inputs_leave_sqlite_job_unchanged(
    storage: BaseStorage,
) -> None:
    case = 0
    for optimizer_kind in _STAGE_PATHS_BY_OPTIMIZER:
        for stage, terminal_outcome in _INVALID_STAGE_REQUESTS:
            case += 1
            job_id, fence = _claimed_job_at_stage(
                storage,
                "evidence_frozen",
                now=7_000,
                optimizer_kind=cast(schemas.OptimizerKind, optimizer_kind),
                target_id=50_000 + case,
            )
            before = _optimization_job_row(storage, job_id)

            assert (
                storage.advance_playbook_optimization_stage(
                    job_id=job_id,
                    fence=fence,
                    stage=cast(schemas.OptimizationJobStage, stage),
                    terminal_outcome=cast(
                        schemas.OptimizationTerminalOutcome | None, terminal_outcome
                    ),
                    now=7_001,
                )
                is False
            )
            assert _optimization_job_row(storage, job_id) == before

    for optimizer_kind in (
        "gepa",
        "offline_tuner_legacy",
        "optimizer_legacy_unknown",
    ):
        for stage, terminal_outcome in (
            ("candidate_generated", None),
            ("failed", "infrastructure_failure"),
            ("abstained", "candidate_did_not_improve"),
        ):
            case += 1
            job_id, fence = _claimed_job_at_stage(
                storage,
                "evidence_frozen",
                now=7_000,
                optimizer_kind=cast(schemas.OptimizerKind, optimizer_kind),
                target_id=60_000 + case,
            )
            before = _optimization_job_row(storage, job_id)

            assert (
                storage.advance_playbook_optimization_stage(
                    job_id=job_id,
                    fence=fence,
                    stage=cast(schemas.OptimizationJobStage, stage),
                    terminal_outcome=cast(
                        schemas.OptimizationTerminalOutcome | None, terminal_outcome
                    ),
                    now=7_001,
                )
                is False
            )
            assert _optimization_job_row(storage, job_id) == before


def test_optimizer_kind_stage_matrix_is_exact(storage: BaseStorage) -> None:
    """Reject every non-edge, including every cross-family stage."""
    case = 0
    for optimizer_kind, stages in _STAGE_PATHS_BY_OPTIMIZER.items():
        for current_stage in stages:
            for target_stage in _ORDINARY_STAGES:
                case += 1
                job_id, fence = _claimed_job_at_stage(
                    storage,
                    cast(schemas.OptimizationJobStage, current_stage),
                    now=7_000,
                    optimizer_kind=cast(schemas.OptimizerKind, optimizer_kind),
                    target_id=10_000 + case,
                )
                expected = (
                    stages.index(target_stage) == stages.index(current_stage) + 1
                    if target_stage in stages
                    else False
                )

                assert (
                    storage.advance_playbook_optimization_stage(
                        job_id=job_id,
                        fence=fence,
                        stage=cast(schemas.OptimizationJobStage, target_stage),
                        now=7_001,
                    )
                    is expected
                )
                persisted = storage.get_playbook_optimization_job(job_id)
                assert persisted is not None
                assert persisted.stage == (target_stage if expected else current_stage)
                assert persisted.status == "running"


def test_optimizer_kind_terminal_outcome_matrix_is_exact(
    storage: BaseStorage,
) -> None:
    """Reject every terminal outcome assigned to the other family."""
    case = 0
    for optimizer_kind, stages in _STAGE_PATHS_BY_OPTIMIZER.items():
        for current_stage in stages:
            for terminal_stage in ("failed", "abstained"):
                for outcome in _ALL_TERMINAL_OUTCOMES:
                    case += 1
                    job_id, fence = _claimed_job_at_stage(
                        storage,
                        cast(schemas.OptimizationJobStage, current_stage),
                        now=7_000,
                        optimizer_kind=cast(schemas.OptimizerKind, optimizer_kind),
                        target_id=20_000 + case,
                    )
                    expected = (
                        outcome
                        in _TERMINAL_OUTCOMES_BY_OPTIMIZER[optimizer_kind][
                            terminal_stage
                        ]
                    )

                    assert (
                        storage.advance_playbook_optimization_stage(
                            job_id=job_id,
                            fence=fence,
                            stage=cast(schemas.OptimizationJobStage, terminal_stage),
                            terminal_outcome=cast(
                                schemas.OptimizationTerminalOutcome, outcome
                            ),
                            now=7_001,
                        )
                        is expected
                    )
                    persisted = storage.get_playbook_optimization_job(job_id)
                    assert persisted is not None
                    assert persisted.stage == (
                        terminal_stage if expected else current_stage
                    )
                    assert persisted.terminal_outcome == (outcome if expected else None)
                    assert persisted.status == (
                        ("failed" if terminal_stage == "failed" else "skipped")
                        if expected
                        else "running"
                    )

            for outcome in (None, *_ALL_TERMINAL_OUTCOMES):
                case += 1
                job_id, fence = _claimed_job_at_stage(
                    storage,
                    cast(schemas.OptimizationJobStage, current_stage),
                    now=7_000,
                    optimizer_kind=cast(schemas.OptimizerKind, optimizer_kind),
                    target_id=30_000 + case,
                )
                expected = (
                    optimizer_kind == "offline_tuner_replay"
                    and current_stage == "publishing"
                    and outcome in (None, "applied")
                )

                assert (
                    storage.advance_playbook_optimization_stage(
                        job_id=job_id,
                        fence=fence,
                        stage="applied",
                        terminal_outcome=cast(
                            schemas.OptimizationTerminalOutcome | None, outcome
                        ),
                        now=7_001,
                    )
                    is expected
                )
                persisted = storage.get_playbook_optimization_job(job_id)
                assert persisted is not None
                assert persisted.stage == ("applied" if expected else current_stage)
                assert persisted.terminal_outcome == ("applied" if expected else None)
                assert persisted.status == ("completed" if expected else "running")


def test_widened_terminal_outcome_rejects_stale_lease_and_settled_job(
    storage: BaseStorage,
) -> None:
    job_id, fence = _claimed_job_at_stage(
        storage,
        "held_out_analyzed",
        now=7_000,
        optimizer_kind="offline_tuner_open_world",
    )
    assert isinstance(storage, SQLiteStorage)

    def _abstain(*, at_fence: int, now: int) -> bool:
        return storage.advance_playbook_optimization_stage(
            job_id=job_id,
            fence=at_fence,
            stage="abstained",
            terminal_outcome="stale_incumbent",
            now=now,
        )

    assert _abstain(at_fence=fence + 1, now=7_001) is False
    assert _abstain(at_fence=fence, now=7_060) is False

    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET stage = 'failed' WHERE job_id = ?",
        (job_id,),
    )
    storage.conn.commit()
    assert _abstain(at_fence=fence, now=7_001) is False

    storage.conn.execute(
        "UPDATE playbook_optimization_jobs "
        "SET stage = 'held_out_analyzed', status = 'completed' WHERE job_id = ?",
        (job_id,),
    )
    storage.conn.commit()
    assert _abstain(at_fence=fence, now=7_001) is False


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


def test_legacy_duplicate_active_gepa_jobs_are_deterministically_retired(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-duplicates.db"
    _create_legacy_optimizer_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO playbook_optimization_jobs
           (job_id, target_kind, target_id, status, metadata_json, created_at, updated_at)
           VALUES (7, 'user_playbook', 2, 'running', ?, 2, 2)""",
        (
            json.dumps(
                {
                    "source_window_count": 3,
                    "train_window_count": 2,
                    "validation_window_count": 1,
                }
            ),
        ),
    )
    conn.commit()
    conn.close()

    store = SQLiteStorage(org_id="legacy-duplicates", db_path=str(db_path))
    try:
        rows = store.conn.execute(
            """SELECT job_id, optimizer_kind, status, decision_reason
               FROM playbook_optimization_jobs
               WHERE target_kind = 'user_playbook' AND target_id = 2
               ORDER BY job_id"""
        ).fetchall()
    finally:
        store.conn.close()

    assert [(row["job_id"], row["status"]) for row in rows] == [
        (2, "pending"),
        (7, "skipped"),
    ]
    assert rows[1]["optimizer_kind"] == "gepa"
    assert rows[1]["decision_reason"] == "retired_duplicate_legacy_active_job"


@pytest.mark.parametrize("key_column", ["discovery_key", "attempt_key"])
def test_legacy_duplicate_active_gepa_job_keys_are_deterministically_retired(
    tmp_path: Path,
    key_column: str,
) -> None:
    db_path = tmp_path / f"legacy-duplicate-{key_column}.db"
    _create_legacy_optimizer_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE playbook_optimization_jobs ADD COLUMN discovery_key TEXT")
    conn.execute("ALTER TABLE playbook_optimization_jobs ADD COLUMN attempt_key TEXT")
    metadata_json = json.dumps(
        {
            "source_window_count": 3,
            "train_window_count": 2,
            "validation_window_count": 1,
        }
    )
    conn.execute(
        """INSERT INTO playbook_optimization_jobs
           (job_id, target_kind, target_id, status, metadata_json,
            discovery_key, attempt_key, created_at, updated_at)
           VALUES (7, 'user_playbook', 7, 'running', ?, 'discovery-7',
                   'attempt-7', 2, 2)""",
        (metadata_json,),
    )
    conn.execute(
        f"UPDATE playbook_optimization_jobs SET {key_column} = ? "  # noqa: S608
        "WHERE job_id IN (2, 7)",
        (f"duplicate-{key_column}",),
    )
    conn.commit()
    conn.close()

    store = SQLiteStorage(org_id=f"legacy-duplicate-{key_column}", db_path=str(db_path))
    try:
        rows = store.conn.execute(
            """SELECT job_id, status, decision_reason
               FROM playbook_optimization_jobs
               WHERE job_id IN (2, 7) ORDER BY job_id"""
        ).fetchall()
    finally:
        store.conn.close()

    assert [(row["job_id"], row["status"]) for row in rows] == [
        (2, "pending"),
        (7, "skipped"),
    ]
    assert rows[1]["decision_reason"] == "retired_duplicate_legacy_active_job"


def _create_legacy_optimizer_schema_with_child(
    db_path: Path,
    *,
    orphan_child: bool = False,
    stale_rebuild_table: bool = False,
) -> None:
    _create_legacy_optimizer_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """CREATE TABLE optimizer_job_child (
               child_id INTEGER PRIMARY KEY,
               job_id INTEGER NOT NULL,
               FOREIGN KEY (job_id) REFERENCES playbook_optimization_jobs(job_id)
                   ON DELETE CASCADE
           )"""
    )
    conn.execute("INSERT INTO optimizer_job_child VALUES (1, 2)")
    conn.commit()
    if orphan_child:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("INSERT INTO optimizer_job_child VALUES (2, 999)")
    if stale_rebuild_table:
        conn.execute(
            "CREATE TABLE playbook_optimization_jobs_new (job_id INTEGER PRIMARY KEY)"
        )
    conn.commit()
    conn.close()


def test_legacy_optimizer_rebuild_preserves_fk_children_and_restores_enforcement(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-child.db"
    _create_legacy_optimizer_schema_with_child(db_path)

    store = SQLiteStorage(org_id="legacy-child", db_path=str(db_path))
    try:
        child = store.conn.execute("SELECT * FROM optimizer_job_child").fetchone()
        foreign_keys = store.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        violations = store.conn.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        store.conn.close()

    assert child is not None
    assert child["job_id"] == 2
    assert foreign_keys == 1
    assert violations == []


def test_legacy_optimizer_rebuild_recovers_stale_temporary_table(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-stale-rebuild.db"
    _create_legacy_optimizer_schema_with_child(db_path, stale_rebuild_table=True)

    store = SQLiteStorage(org_id="legacy-stale-rebuild", db_path=str(db_path))
    try:
        stale_table = store.conn.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type = 'table' AND name = 'playbook_optimization_jobs_new'"""
        ).fetchone()
        child_count = store.conn.execute(
            "SELECT COUNT(*) FROM optimizer_job_child"
        ).fetchone()[0]
    finally:
        store.conn.close()

    assert stale_table is None
    assert child_count == 1


def test_legacy_optimizer_rebuild_rolls_back_on_foreign_key_violation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-invalid-child.db"
    _create_legacy_optimizer_schema_with_child(db_path, orphan_child=True)

    with pytest.raises(sqlite3.IntegrityError, match="foreign key check"):
        SQLiteStorage(org_id="legacy-invalid-child", db_path=str(db_path))

    conn = sqlite3.connect(db_path)
    try:
        parent_table = conn.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type = 'table' AND name = 'playbook_optimization_jobs'"""
        ).fetchone()
        stale_table = conn.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type = 'table' AND name = 'playbook_optimization_jobs_new'"""
        ).fetchone()
        conn.execute("DELETE FROM optimizer_job_child WHERE job_id = 999")
        conn.commit()
    finally:
        conn.close()

    assert parent_table is not None
    assert stale_table is None

    store = SQLiteStorage(org_id="legacy-invalid-child-retry", db_path=str(db_path))
    store.conn.close()


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
