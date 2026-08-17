from hashlib import sha256

import pytest

from reflexio.models.api_schema.domain.entities import (
    OptimizationArtifactKind,
    OptimizationJobStage,
    OptimizationTerminalOutcome,
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


@pytest.mark.parametrize("stage", ["discovery_analyzed", "held_out_analyzed"])
def test_open_world_analysis_stage_is_accepted(stage: OptimizationJobStage) -> None:
    job = PlaybookOptimizationJob(
        optimizer_kind="offline_tuner_open_world",
        target_kind="user_playbook",
        target_id=7,
        stage=stage,
    )

    assert job.stage == stage


@pytest.mark.parametrize(
    "terminal_outcome",
    [
        "no_grounded_hypothesis",
        "analyst_unqualified",
        "heldout_evidence_failed",
        "stale_incumbent",
        "governance_invalidated",
        "infrastructure_failure",
    ],
)
def test_open_world_analysis_terminal_outcome_is_accepted(
    terminal_outcome: OptimizationTerminalOutcome,
) -> None:
    job = PlaybookOptimizationJob(
        optimizer_kind="offline_tuner_open_world",
        target_kind="user_playbook",
        target_id=7,
        terminal_outcome=terminal_outcome,
    )

    assert job.terminal_outcome == terminal_outcome


@pytest.mark.parametrize(
    "artifact_kind",
    [
        "open_world_discovery_memo",
        "open_world_candidate",
        "open_world_attempt_decision",
    ],
)
def test_open_world_analysis_artifact_kind_is_accepted(
    artifact_kind: OptimizationArtifactKind,
) -> None:
    content_json = '{"schema_version":"offline-tuner-open-world-analysis-v1"}'
    artifact = PlaybookOptimizationArtifact(
        job_id=1,
        artifact_kind=artifact_kind,
        content_json=content_json,
        content_digest=sha256(content_json.encode()).hexdigest(),
    )

    assert artifact.artifact_kind == artifact_kind
