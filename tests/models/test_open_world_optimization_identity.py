from hashlib import sha256

from reflexio.models.api_schema.domain.entities import (
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
)


def test_open_world_optimization_identity_is_accepted() -> None:
    content_json = '{"schema_version":"offline-tuner-open-world-evidence-v1"}'
    job = PlaybookOptimizationJob(
        optimizer_kind="offline_tuner_open_world",
        target_kind="user_playbook",
        target_id=7,
    )
    artifact = PlaybookOptimizationArtifact(
        job_id=1,
        artifact_kind="open_world_evidence_bundle",
        content_json=content_json,
        content_digest=sha256(content_json.encode()).hexdigest(),
    )

    assert job.optimizer_kind == "offline_tuner_open_world"
    assert artifact.artifact_kind == "open_world_evidence_bundle"
