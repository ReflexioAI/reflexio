"""Generic SQLite identity coverage for the open-world analysis vocabulary."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest

from reflexio.models.api_schema import service_schemas as schemas
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage(tmp_path: Path) -> Generator[SQLiteStorage]:
    store = SQLiteStorage(
        org_id="open-world-analysis-identity",
        db_path=str(tmp_path / "reflexio.db"),
    )
    try:
        yield store
    finally:
        store.conn.close()


def _job() -> schemas.PlaybookOptimizationJob:
    return schemas.PlaybookOptimizationJob(
        optimizer_kind="offline_tuner_open_world",
        target_kind="user_playbook",
        target_id=41,
        discovery_key="discovery-key",
        attempt_key="attempt-key",
        stage="evidence_frozen",
    )


def test_open_world_analysis_stage_path_round_trips(storage: SQLiteStorage) -> None:
    job = storage.create_playbook_optimization_job(_job())
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=1_000,
    )

    assert storage.advance_playbook_optimization_stage(
        job_id=job.job_id,
        fence=claim.fence,
        stage="discovery_analyzed",
        now=1_001,
    )
    assert storage.advance_playbook_optimization_stage(
        job_id=job.job_id,
        fence=claim.fence,
        stage="candidate_generated",
        now=1_002,
    )
    assert storage.advance_playbook_optimization_stage(
        job_id=job.job_id,
        fence=claim.fence,
        stage="held_out_analyzed",
        now=1_003,
    )

    persisted = storage.get_playbook_optimization_job(job.job_id)
    assert persisted is not None
    assert persisted.stage == "held_out_analyzed"


@pytest.mark.parametrize(
    ("stage", "outcome", "expected_status"),
    [
        ("abstained", "no_grounded_hypothesis", "skipped"),
        ("abstained", "analyst_unqualified", "skipped"),
        ("abstained", "heldout_evidence_failed", "skipped"),
        ("abstained", "stale_incumbent", "skipped"),
        ("abstained", "governance_invalidated", "skipped"),
        ("failed", "infrastructure_failure", "failed"),
    ],
)
def test_open_world_terminal_outcomes_are_durable(
    storage: SQLiteStorage,
    stage: schemas.OptimizationJobStage,
    outcome: schemas.OptimizationTerminalOutcome,
    expected_status: str,
) -> None:
    job = storage.create_playbook_optimization_job(_job())
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=2_000,
    )

    assert storage.advance_playbook_optimization_stage(
        job_id=job.job_id,
        fence=claim.fence,
        stage=stage,
        terminal_outcome=outcome,
        now=2_001,
    )
    persisted = storage.get_playbook_optimization_job(job.job_id)
    assert persisted is not None
    assert persisted.status == expected_status
    assert persisted.terminal_outcome == outcome


@pytest.mark.parametrize(
    "outcome",
    [
        "infrastructure_failure",
        "analyst_unqualified",
        "stale_incumbent",
        "governance_invalidated",
    ],
)
def test_open_world_failed_terminal_outcomes_are_durable(
    storage: SQLiteStorage,
    outcome: schemas.OptimizationTerminalOutcome,
) -> None:
    job = storage.create_playbook_optimization_job(_job())
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=2_000,
    )

    assert storage.advance_playbook_optimization_stage(
        job_id=job.job_id,
        fence=claim.fence,
        stage="failed",
        terminal_outcome=outcome,
        now=2_001,
    )
    persisted = storage.get_playbook_optimization_job(job.job_id)
    assert persisted is not None
    assert persisted.stage == "failed"
    assert persisted.status == "failed"
    assert persisted.terminal_outcome == outcome


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param("no_grounded_hypothesis", id="open-world-abstention"),
        pytest.param("generation_failed", id="replay-family-failure"),
    ],
)
def test_open_world_failed_stage_rejects_unrelated_outcomes(
    storage: SQLiteStorage,
    outcome: schemas.OptimizationTerminalOutcome,
) -> None:
    job = storage.create_playbook_optimization_job(_job())
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=2_000,
    )

    assert not storage.advance_playbook_optimization_stage(
        job_id=job.job_id,
        fence=claim.fence,
        stage="failed",
        terminal_outcome=outcome,
        now=2_001,
    )
    persisted = storage.get_playbook_optimization_job(job.job_id)
    assert persisted is not None
    assert persisted.stage == "evidence_frozen"
    assert persisted.status == "running"
    assert persisted.terminal_outcome is None


def test_legacy_optimizer_kind_allowlist_is_rebuilt_for_open_world_job(
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "legacy-optimizer-kind.db")
    initial = SQLiteStorage(org_id="legacy-optimizer-kind", db_path=db_path)
    table_sql = initial.conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name = 'playbook_optimization_jobs'"
    ).fetchone()[0]
    initial.conn.close()

    legacy_table_sql = table_sql.replace("'offline_tuner_open_world',", "")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE playbook_optimization_jobs")
        conn.execute(legacy_table_sql)
        conn.commit()
    finally:
        conn.close()

    upgraded = SQLiteStorage(org_id="legacy-optimizer-kind", db_path=db_path)
    try:
        job = upgraded.create_playbook_optimization_job(_job())
    finally:
        upgraded.conn.close()

    assert job.optimizer_kind == "offline_tuner_open_world"


@pytest.mark.parametrize(
    "artifact_kind",
    [
        "open_world_discovery_memo",
        "open_world_candidate",
        "open_world_attempt_decision",
    ],
)
def test_open_world_analysis_artifact_round_trips(
    storage: SQLiteStorage,
    artifact_kind: schemas.OptimizationArtifactKind,
) -> None:
    job = storage.create_playbook_optimization_job(_job())
    claim = storage.claim_playbook_optimization_job(
        job_id=job.job_id,
        owner="worker-a",
        lease_seconds=60,
        now=2_000,
    )
    content_json = '{"schema_version":"offline-tuner-open-world-analysis-v1"}'
    artifact = schemas.PlaybookOptimizationArtifact(
        job_id=job.job_id,
        artifact_kind=artifact_kind,
        content_json=content_json,
        content_digest=sha256(content_json.encode()).hexdigest(),
    )

    saved = storage.upsert_playbook_optimization_artifact(
        artifact,
        fence=claim.fence,
        now=2_001,
    )

    assert (
        storage.get_playbook_optimization_artifact(job.job_id, artifact_kind) == saved
    )


def test_sqlite_does_not_expose_open_world_invocations(
    storage: SQLiteStorage,
) -> None:
    table = storage.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'offline_tuner_open_world_invocations'"
    ).fetchone()

    assert table is None
    for method_name in (
        "prepare_open_world_invocation",
        "complete_open_world_invocation",
        "load_open_world_invocation",
    ):
        assert not hasattr(BaseStorage, method_name)
        assert not hasattr(SQLiteStorage, method_name)
