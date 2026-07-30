import pytest

from reflexio.models.api_schema.domain import AgentSuccessEvaluationResult
from reflexio.models.config_schema import (
    Config,
    RetrievalExperimentConfig,
    RetrievalExperimentRecord,
    StorageConfigSQLite,
)
from reflexio.server.services.retrieval_experiment import (
    RetrievalExperimentArm,
    active_retrieval_experiment_assignment,
    assign_retrieval_experiment_arm,
    build_retrieval_experiment_results,
    validate_retrieval_experiment_attribution,
)


def _config() -> Config:
    record = RetrievalExperimentRecord(
        experiment_id="exp-1",
        holdout_percentage=25,
        started_at=100,
    )
    return Config(
        storage_config=StorageConfigSQLite(),
        retrieval_experiment_config=RetrievalExperimentConfig(
            experiment_id=record.experiment_id,
            holdout_percentage=record.holdout_percentage,
        ),
        retrieval_experiment_history=[record],
    )


def test_assignment_is_stable_and_namespaced_by_experiment() -> None:
    first = [
        assign_retrieval_experiment_arm(
            org_id="org-1",
            experiment_id="exp-1",
            user_id=f"user-{index}",
            holdout_percentage=25,
        )
        for index in range(100)
    ]
    repeated = [
        assign_retrieval_experiment_arm(
            org_id="org-1",
            experiment_id="exp-1",
            user_id=f"user-{index}",
            holdout_percentage=25,
        )
        for index in range(100)
    ]
    second_experiment = [
        assign_retrieval_experiment_arm(
            org_id="org-1",
            experiment_id="exp-2",
            user_id=f"user-{index}",
            holdout_percentage=25,
        )
        for index in range(100)
    ]

    assert first == repeated
    assert first != second_experiment
    assert 10 <= first.count("holdout") <= 40


def test_publish_attribution_must_match_the_user_assignment() -> None:
    config = _config()
    assignment = active_retrieval_experiment_assignment(
        config=config, org_id="org-1", user_id="user-1"
    )
    assert assignment is not None

    validate_retrieval_experiment_attribution(
        config=config,
        org_id="org-1",
        user_id="user-1",
        experiment_id=assignment.experiment_id,
        arm=assignment.arm,
    )

    wrong_arm = "treatment" if assignment.arm == "holdout" else "holdout"
    try:
        validate_retrieval_experiment_attribution(
            config=config,
            org_id="org-1",
            user_id="user-1",
            experiment_id=assignment.experiment_id,
            arm=wrong_arm,
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched attribution was accepted")


def test_results_join_session_outcomes_and_report_clustered_interval() -> None:
    assignments: dict[tuple[str, str], RetrievalExperimentArm] = {
        ("t1", "s1"): "treatment",
        ("t1", "s2"): "treatment",
        ("t2", "s3"): "treatment",
        ("h1", "s4"): "holdout",
        ("h1", "s5"): "holdout",
        ("h2", "s6"): "holdout",
    }
    outcomes = [
        AgentSuccessEvaluationResult(
            user_id=user_id,
            agent_version="v1",
            session_id=session_id,
            is_success=is_success,
        )
        for user_id, session_id, is_success in (
            ("t1", "s1", True),
            ("t1", "s2", True),
            ("t2", "s3", False),
            ("h1", "s4", False),
            ("h1", "s5", False),
            ("h2", "s6", True),
            ("untagged", "s7", True),
        )
    ]

    response = build_retrieval_experiment_results(
        experiment=RetrievalExperimentRecord(
            experiment_id="exp-1",
            holdout_percentage=50,
            started_at=100,
        ),
        assignments=assignments,
        evaluation_results=outcomes,
        output_token_counts={
            ("t1", "s1"): 90,
            ("t2", "s3"): 30,
            ("h1", "s4"): 120,
            ("h1", "s5"): 0,
            ("h2", "s6"): 60,
        },
    )

    assert response.treatment.evaluated_session_count == 3
    assert response.treatment.evaluated_user_count == 2
    assert response.holdout.evaluated_session_count == 3
    assert response.success_rate_lift_percentage_points == pytest.approx(100 / 3)
    assert response.confidence_interval_95_percentage_points is not None
    assert response.evaluated_session_coverage == 6 / 7
    assert response.unattributed_evaluated_session_count == 1
    assert response.treatment.average_output_tokens == 60
    assert response.treatment.output_token_session_count == 2
    assert response.holdout.average_output_tokens == 60
    assert response.holdout.output_token_session_count == 3
