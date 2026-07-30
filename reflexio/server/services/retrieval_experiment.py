"""Deterministic user assignment and session-level retrieval experiment metrics."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal

from reflexio.models.api_schema.domain import AgentSuccessEvaluationResult
from reflexio.models.api_schema.retriever_schema import (
    RetrievalExperimentArmMetrics,
    RetrievalExperimentAssignment,
    RetrievalExperimentResultsResponse,
)
from reflexio.models.config_schema import Config, RetrievalExperimentRecord

RetrievalExperimentArm = Literal["treatment", "holdout"]
_ASSIGNMENT_NAMESPACE = "retrieval-ab:v1"


def assign_retrieval_experiment_arm(
    *,
    org_id: str,
    experiment_id: str,
    user_id: str,
    holdout_percentage: float,
) -> RetrievalExperimentArm:
    """Return a stable user-level assignment for one organization experiment."""
    key = f"{_ASSIGNMENT_NAMESPACE}\0{org_id}\0{experiment_id}\0{user_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest, "big") / 2 ** (8 * len(digest))
    return "holdout" if fraction < holdout_percentage / 100 else "treatment"


def active_retrieval_experiment_assignment(
    *, config: Config, org_id: str, user_id: str
) -> RetrievalExperimentAssignment | None:
    experiment = config.retrieval_experiment_config
    if experiment is None:
        return None
    return RetrievalExperimentAssignment(
        experiment_id=experiment.experiment_id,
        arm=assign_retrieval_experiment_arm(
            org_id=org_id,
            experiment_id=experiment.experiment_id,
            user_id=user_id,
            holdout_percentage=experiment.holdout_percentage,
        ),
    )


def find_retrieval_experiment(
    config: Config, experiment_id: str
) -> RetrievalExperimentRecord | None:
    return next(
        (
            experiment
            for experiment in config.retrieval_experiment_history
            if experiment.experiment_id == experiment_id
        ),
        None,
    )


def validate_retrieval_experiment_attribution(
    *,
    config: Config,
    org_id: str,
    user_id: str,
    experiment_id: str | None,
    arm: RetrievalExperimentArm | None,
) -> None:
    """Validate client-echoed search attribution before persisting a request."""
    if experiment_id is None and arm is None:
        return
    if experiment_id is None or arm is None:
        raise ValueError(
            "retrieval_experiment_id and retrieval_experiment_arm must be provided together"
        )
    experiment = find_retrieval_experiment(config, experiment_id)
    if experiment is None:
        raise ValueError("retrieval experiment does not exist for this organization")
    expected = assign_retrieval_experiment_arm(
        org_id=org_id,
        experiment_id=experiment.experiment_id,
        user_id=user_id,
        holdout_percentage=experiment.holdout_percentage,
    )
    if arm != expected:
        raise ValueError("retrieval experiment arm does not match the user assignment")


def build_retrieval_experiment_results(
    *,
    experiment: RetrievalExperimentRecord,
    assignments: dict[tuple[str, str], RetrievalExperimentArm],
    evaluation_results: list[AgentSuccessEvaluationResult],
    output_token_counts: dict[tuple[str, str], int] | None = None,
) -> RetrievalExperimentResultsResponse:
    """Join request attribution to session evaluations and aggregate both arms."""
    attributed: dict[RetrievalExperimentArm, list[AgentSuccessEvaluationResult]] = {
        "treatment": [],
        "holdout": [],
    }
    unattributed = 0
    for result in evaluation_results:
        arm = assignments.get((result.user_id, result.session_id))
        if arm is None:
            unattributed += 1
            continue
        attributed[arm].append(result)

    measured_output_tokens = output_token_counts or {}
    treatment = _arm_metrics(
        "treatment", assignments, attributed["treatment"], measured_output_tokens
    )
    holdout = _arm_metrics(
        "holdout", assignments, attributed["holdout"], measured_output_tokens
    )
    lift: float | None = None
    relative_lift: float | None = None
    confidence_interval: tuple[float, float] | None = None
    if treatment.success_rate is not None and holdout.success_rate is not None:
        lift = treatment.success_rate - holdout.success_rate
        if holdout.success_rate > 0:
            relative_lift = lift / holdout.success_rate
        standard_error = _clustered_difference_standard_error(
            attributed["treatment"], attributed["holdout"]
        )
        if standard_error is not None:
            confidence_interval = (
                max(-100.0, (lift - 1.96 * standard_error) * 100),
                min(100.0, (lift + 1.96 * standard_error) * 100),
            )

    total_evaluated = len(evaluation_results)
    attributed_evaluated = total_evaluated - unattributed
    return RetrievalExperimentResultsResponse(
        experiment=experiment,
        treatment=treatment,
        holdout=holdout,
        success_rate_lift_percentage_points=(lift * 100 if lift is not None else None),
        relative_success_lift=relative_lift,
        confidence_interval_95_percentage_points=confidence_interval,
        evaluated_session_coverage=(
            attributed_evaluated / total_evaluated if total_evaluated else None
        ),
        unattributed_evaluated_session_count=unattributed,
    )


def _arm_metrics(
    arm: RetrievalExperimentArm,
    assignments: dict[tuple[str, str], RetrievalExperimentArm],
    results: list[AgentSuccessEvaluationResult],
    output_token_counts: dict[tuple[str, str], int],
) -> RetrievalExperimentArmMetrics:
    assigned_pairs = [
        pair for pair, assigned_arm in assignments.items() if assigned_arm == arm
    ]
    success_rate = _mean(float(result.is_success) for result in results)
    corrections = _mean(
        float(result.number_of_correction_per_session) for result in results
    )
    turns = _mean(
        float(result.user_turns_to_resolution)
        for result in results
        if result.user_turns_to_resolution is not None
    )
    escalation_rate = _mean(float(result.is_escalated) for result in results)
    measured_output_token_counts = [
        float(output_token_counts[pair])
        for pair in assigned_pairs
        if pair in output_token_counts
    ]
    return RetrievalExperimentArmMetrics(
        arm=arm,
        assigned_user_count=len({user_id for user_id, _ in assigned_pairs}),
        published_session_count=len(assigned_pairs),
        evaluated_user_count=len({result.user_id for result in results}),
        evaluated_session_count=len(results),
        success_rate=success_rate,
        average_corrections=corrections,
        average_turns_to_resolution=turns,
        escalation_rate=escalation_rate,
        average_output_tokens=_mean(measured_output_token_counts),
        output_token_session_count=len(measured_output_token_counts),
    )


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _clustered_mean_variance(
    results: list[AgentSuccessEvaluationResult],
) -> float | None:
    users: dict[str, list[float]] = defaultdict(list)
    for result in results:
        users[result.user_id].append(float(result.is_success))
    if len(users) < 2 or not results:
        return None
    mean = sum(value for values in users.values() for value in values) / len(results)
    cluster_scores = [
        sum(value - mean for value in values) for values in users.values()
    ]
    finite_sample = len(users) / (len(users) - 1)
    return (
        finite_sample
        * sum(score * score for score in cluster_scores)
        / (len(results) ** 2)
    )


def _clustered_difference_standard_error(
    treatment: list[AgentSuccessEvaluationResult],
    holdout: list[AgentSuccessEvaluationResult],
) -> float | None:
    treatment_variance = _clustered_mean_variance(treatment)
    holdout_variance = _clustered_mean_variance(holdout)
    if treatment_variance is None or holdout_variance is None:
        return None
    return math.sqrt(treatment_variance + holdout_variance)
