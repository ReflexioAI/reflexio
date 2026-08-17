"""Generic SQLite identity coverage for the open-world analysis vocabulary."""

from __future__ import annotations

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
        optimizer_kind="offline_tuner_replay",
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
