"""Retrieval experiment lifecycle and session-outcome reporting routes."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status

from reflexio.models.api_schema.retriever_schema import (
    RetrievalExperimentListResponse,
    RetrievalExperimentResultsResponse,
    StartRetrievalExperimentRequest,
    StopRetrievalExperimentRequest,
)
from reflexio.models.config_schema import (
    RetrievalExperimentConfig,
    RetrievalExperimentRecord,
)
from reflexio.server.auth import default_get_org_id
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter
from reflexio.server.services.configurator.config_storage import (
    ConfigWriteConflictError,
)
from reflexio.server.services.retrieval_experiment import (
    build_retrieval_experiment_results,
    find_retrieval_experiment,
)

router = APIRouter()


def _now() -> int:
    return int(datetime.now(UTC).timestamp())


def _list_response(org_id: str) -> RetrievalExperimentListResponse:
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    config = reflexio.request_context.configurator.get_config()
    active = None
    if config.retrieval_experiment_config is not None:
        active = find_retrieval_experiment(
            config, config.retrieval_experiment_config.experiment_id
        )
    return RetrievalExperimentListResponse(
        active_experiment=active,
        experiments=sorted(
            config.retrieval_experiment_history,
            key=lambda experiment: experiment.started_at,
            reverse=True,
        ),
    )


def _save_experiment_patch(org_id: str, partial: dict[str, object]) -> None:
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    configurator = reflexio.request_context.configurator
    try:
        reflexio.set_config(configurator.prepare_config_patch(partial))
    except ConfigWriteConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "config_write_conflict", "message": str(exc)},
        ) from exc
    reflexio_cache.invalidate_reflexio_cache(org_id=org_id)


@router.get(
    "/api/retrieval_experiments",
    response_model=RetrievalExperimentListResponse,
    response_model_exclude_none=True,
)
def list_retrieval_experiments(
    org_id: str = Depends(default_get_org_id),
) -> RetrievalExperimentListResponse:
    return _list_response(org_id)


@router.post(
    "/api/retrieval_experiments",
    response_model=RetrievalExperimentListResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def start_retrieval_experiment(
    request: Request,
    payload: StartRetrievalExperimentRequest,
    org_id: str = Depends(default_get_org_id),
) -> RetrievalExperimentListResponse:
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    config = reflexio.request_context.configurator.get_config()
    if config.retrieval_experiment_config is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stop the active retrieval experiment before starting another",
        )
    if find_retrieval_experiment(config, payload.experiment_id) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Retrieval experiment IDs cannot be reused",
        )
    record = RetrievalExperimentRecord(
        experiment_id=payload.experiment_id,
        holdout_percentage=payload.holdout_percentage,
        started_at=_now(),
    )
    _save_experiment_patch(
        org_id,
        {
            "retrieval_experiment_config": RetrievalExperimentConfig(
                experiment_id=payload.experiment_id,
                holdout_percentage=payload.holdout_percentage,
            ),
            "retrieval_experiment_history": [
                *config.retrieval_experiment_history,
                record,
            ],
        },
    )
    return _list_response(org_id)


@router.post(
    "/api/retrieval_experiments/stop",
    response_model=RetrievalExperimentListResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def stop_retrieval_experiment(
    request: Request,
    payload: StopRetrievalExperimentRequest,
    org_id: str = Depends(default_get_org_id),
) -> RetrievalExperimentListResponse:
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    config = reflexio.request_context.configurator.get_config()
    active = config.retrieval_experiment_config
    if active is None or active.experiment_id != payload.experiment_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The requested retrieval experiment is not active",
        )
    stopped_at = _now()
    history = [
        experiment.model_copy(
            update={"ended_at": max(stopped_at, experiment.started_at)}
        )
        if experiment.experiment_id == payload.experiment_id
        else experiment
        for experiment in config.retrieval_experiment_history
    ]
    _save_experiment_patch(
        org_id,
        {
            "retrieval_experiment_config": None,
            "retrieval_experiment_history": history,
        },
    )
    return _list_response(org_id)


@router.get(
    "/api/retrieval_experiments/{experiment_id}/results",
    response_model=RetrievalExperimentResultsResponse,
    response_model_exclude_none=True,
)
def get_retrieval_experiment_results(
    experiment_id: str,
    org_id: str = Depends(default_get_org_id),
) -> RetrievalExperimentResultsResponse:
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    config = reflexio.request_context.configurator.get_config()
    experiment = find_retrieval_experiment(config, experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrieval experiment not found",
        )
    storage = reflexio.request_context.storage
    assignments = (
        storage.get_retrieval_experiment_assignments(experiment_id)
        if storage is not None
        else {}
    )
    evaluation_results = (
        storage.get_agent_success_evaluation_results_in_window(
            from_ts=experiment.started_at,
            to_ts=experiment.ended_at or _now(),
            limit=None,
        )
        if storage is not None
        else []
    )
    output_token_counts = (
        storage.get_retrieval_experiment_output_token_counts(experiment_id)
        if storage is not None
        else {}
    )
    return build_retrieval_experiment_results(
        experiment=experiment,
        assignments=assignments,
        evaluation_results=evaluation_results,
        output_token_counts=output_token_counts,
    )
