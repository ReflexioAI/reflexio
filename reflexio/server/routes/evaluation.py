"""Evaluation route handlers (extracted from api.py, Tier3 A2)."""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from reflexio.models.api_schema.eval_overview_schema import (
    GetEvaluationOverviewRequest,
    GetEvaluationOverviewResponse,
    GetRecentShadowComparisonsResponse,
    GradeOnDemandRequest,
    GradeOnDemandResponse,
    RegenerateFailure,
    RegenerateRequest,
    RegenerateStartResponse,
    RegenerateStatusResponse,
)
from reflexio.models.api_schema.retriever_schema import (
    GetAgentSuccessEvaluationResultsRequest,
    GetEvaluationResultsViewResponse,
)
from reflexio.models.api_schema.ui.converters import (
    to_evaluation_result_view,
)
from reflexio.models.config_schema import (
    SINGLETON_AGENT_SUCCESS_EVALUATION_NAME,
)
from reflexio.server.auth import (
    default_get_org_id,
)
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter
from reflexio.server.services.agent_success_evaluation.regen_jobs import (
    REGEN_JOBS,
    run_regen,
)
from reflexio.server.services.agent_success_evaluation.runner import (
    run_group_evaluation,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_GRADE_ON_DEMAND_CACHE_TTL_SECONDS = 24 * 60 * 60

_GRADE_ON_DEMAND_CACHE_KEY_PREFIX = "grade_on_demand"

_RECENT_SHADOW_COMPARISONS_LOOKBACK_SECONDS = 30 * 24 * 60 * 60

_RECENT_SHADOW_COMPARISONS_MAX_LIMIT = 100

REGENERATE_MAX_WORKERS = 2

_regen_executor = ThreadPoolExecutor(
    max_workers=REGENERATE_MAX_WORKERS,
    thread_name_prefix="reflexio-regen",
)


def _grade_on_demand_cache_key(
    org_id: str, session_id: str, agent_version: str, evaluation_name: str
) -> str:
    """Build the operation_state key for the on-demand grading cache.

    The key embeds every active singleton dimension that could change the
    verdict: org_id (multi-tenant scope), session_id (the unit of work),
    agent_version (eval results are versioned), and evaluation_name (kept as a
    compatibility/readback discriminator for historical multi-evaluator rows).

    Args:
        org_id (str): Tenant identifier from the auth context.
        session_id (str): Target session.
        agent_version (str): Agent version filter.
        evaluation_name (str): Evaluator/result namespace to isolate cache rows.
    Returns:
        str: A namespaced key suitable for ``storage.upsert_operation_state``.
    """
    # Length-prefix each free-form component (``f"{len(s)}:{s}"``) so the join is
    # injective: distinct component tuples can never collapse to the same key even
    # when a component itself contains the ``::`` delimiter. Keeps the prefix intact
    # for prefix-based filtering and the key human-readable for inspection.
    parts = "::".join(
        f"{len(s)}:{s}" for s in (org_id, session_id, agent_version, evaluation_name)
    )
    return f"{_GRADE_ON_DEMAND_CACHE_KEY_PREFIX}::{parts}"


def _read_grade_on_demand_cache(
    storage: Any, cache_key: str, *, now: int
) -> int | None:
    """Return the cached ``result_id`` if a valid entry exists, else None.

    Returns None on three conditions: no entry, malformed entry, or entry
    whose ``last_graded_at`` is older than the 24h TTL. Keeps the handler
    body focused on the happy path.

    Args:
        storage: The request's storage backend.
        cache_key (str): Key produced by ``_grade_on_demand_cache_key``.
        now (int): Current Unix-seconds wall-clock timestamp.

    Returns:
        int | None: Cached result_id when fresh, else None.
    """
    cached_state = storage.get_operation_state(cache_key)
    if not cached_state:
        return None
    state = cached_state.get("operation_state")
    if not isinstance(state, dict):
        return None
    last_graded_at = state.get("last_graded_at")
    if not isinstance(last_graded_at, int):
        return None
    if (now - last_graded_at) >= _GRADE_ON_DEMAND_CACHE_TTL_SECONDS:
        return None
    cached_result_id = state.get("result_id")
    return cached_result_id if isinstance(cached_result_id, int) else None


def _resolve_session_user_id(storage: Any, session_id: str) -> str | None:
    """Look up the user_id that owns a session_id without requiring it as input.

    Uses the first-request bulk helper even for this single-session path so the
    lookup can use the same indexed query shape as evaluation overview.

    Args:
        storage: The request's storage backend.
        session_id (str): The target session whose owner to resolve.

    Returns:
        str | None: The user_id of the earliest request in the session,
        or None when no requests exist.
    """
    first_requests = storage.get_first_requests_by_session_ids([session_id])
    first = first_requests.get(session_id)
    if first is None:
        return None
    return first.user_id


def _find_fresh_result_id(
    storage: Any,
    *,
    user_id: str,
    session_id: str,
    agent_version: str,
    evaluation_name: str,
    previous_result_ids: set[int],
) -> int | None:
    """Locate the result_id written by the most-recent grade for this session.

    The runner writes rows but doesn't return the id. Use the targeted result-id
    lookup so this path does not scan broad evaluation windows.

    Args:
        storage: The request's storage backend.
        user_id (str): The user whose session slice was graded.
        session_id (str): The graded session.
        agent_version (str): The version dimension.
        evaluation_name (str): Evaluator/result namespace to isolate readback.
        previous_result_ids (set[int]): Matching rows observed before grading.

    Returns:
        int | None: result_id of the latest matching row, or None if the
        runner wrote nothing.
    """
    result_ids = storage.get_agent_success_evaluation_result_ids(
        user_id=user_id,
        session_id=session_id,
        evaluation_name=evaluation_name,
        agent_version=agent_version,
    )
    fresh_result_ids = [rid for rid in result_ids if rid not in previous_result_ids]
    if not fresh_result_ids:
        return None
    return max(fresh_result_ids)


@router.post(
    "/api/get_agent_success_evaluation_results",
    response_model=GetEvaluationResultsViewResponse,
    response_model_exclude_none=True,
)
def get_agent_success_evaluation_results(
    request: GetAgentSuccessEvaluationResultsRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetEvaluationResultsViewResponse:
    """Get agent success evaluation results.

    Args:
        request (GetAgentSuccessEvaluationResultsRequest): The get request
        org_id (str): Organization ID

    Returns:
        GetEvaluationResultsViewResponse: Response containing agent success evaluation results
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    response = reflexio.get_agent_success_evaluation_results(request)
    return GetEvaluationResultsViewResponse(
        success=response.success,
        agent_success_evaluation_results=[
            to_evaluation_result_view(r)
            for r in response.agent_success_evaluation_results
        ],
        msg=response.msg,
    )


@router.post(
    "/api/get_evaluation_overview",
    response_model=GetEvaluationOverviewResponse,
    response_model_exclude_none=True,
)
def get_evaluation_overview(
    request: GetEvaluationOverviewRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetEvaluationOverviewResponse:
    """Return the redesigned /evaluations page payload.

    Aggregates hero state, four context tiles with deltas, top rule
    attribution, and a corrections-per-session distribution into a single
    response shaped exactly as the frontend renders it.

    Args:
        request (GetEvaluationOverviewRequest): Window + bucket granularity.
        org_id (str): Organization ID resolved by the auth dependency.

    Returns:
        GetEvaluationOverviewResponse: Full overview payload.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    return reflexio.get_evaluation_overview(request)


@router.post(
    "/api/evaluations/regenerate",
    response_model=RegenerateStartResponse,
    response_model_exclude_none=True,
)
@limiter.limit("5/minute")
def start_regenerate(
    request: Request,
    payload: RegenerateRequest,
    org_id: str = Depends(default_get_org_id),
) -> RegenerateStartResponse:
    """Start a singleton regenerate job over a time window.

    Args:
        payload (RegenerateRequest): Window bounds plus optional legacy evaluator name.
        org_id (str): Organization ID resolved by the auth dependency.

    Returns:
        RegenerateStartResponse: ``job_id`` to poll/cancel and ``total``
            tuples queued.

    Raises:
        HTTPException: 409 when a regenerate for the same org is already
            running. 503 when storage is not configured.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage not configured")
    descriptors = storage.get_session_ids_in_window(
        from_ts=payload.from_ts, to_ts=payload.to_ts
    )
    try:
        job = REGEN_JOBS.create(
            org_id=org_id,
            from_ts=payload.from_ts,
            to_ts=payload.to_ts,
            total=len(descriptors),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    _regen_executor.submit(
        run_regen,
        job=job,
        request_context=reflexio.request_context,
        llm_client=reflexio.llm_client,
    )
    return RegenerateStartResponse(job_id=job.job_id, total=job.total)


@router.get(
    "/api/evaluations/regenerate/{job_id}",
    response_model=RegenerateStatusResponse,
    response_model_exclude_none=True,
)
def get_regenerate_status(
    job_id: str,
    org_id: str = Depends(default_get_org_id),
) -> RegenerateStatusResponse:
    """Poll the status of a regenerate job.

    Args:
        job_id (str): Opaque handle returned by POST /api/evaluations/regenerate.
        org_id (str): Organization ID resolved by the auth dependency.

    Returns:
        RegenerateStatusResponse: Counters, status, and failure list.

    Raises:
        HTTPException: 404 when ``job_id`` is unknown or belongs to a
            different org.
    """
    job = REGEN_JOBS.get(job_id)
    if job is None or job.org_id != org_id:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return RegenerateStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total=job.total,
        completed=job.completed,
        failed=job.failed,
        failures=[
            RegenerateFailure(session_id=f.session_id, reason=f.reason)
            for f in job.failures
        ],
        started_at=job.started_at,
        finished_at=job.finished_at,
        # F3 informational counters — surface sampler + concurrency facts
        # so the dashboard can render "n_sampled / total_candidates" and
        # the configured worker cap without a second round-trip.
        total_candidates=job.total_candidates,
        sampled_count=job.sampled_count,
        concurrency_limit=job.concurrency_limit,
    )


@router.delete("/api/evaluations/regenerate/{job_id}")
def cancel_regenerate(
    job_id: str,
    org_id: str = Depends(default_get_org_id),
) -> dict[str, str]:
    """Request cancellation of a running regenerate job.

    Sets the worker's cancel event; the worker checks the flag between
    sessions and transitions to ``"cancelled"`` on its next iteration.

    Args:
        job_id (str): Opaque handle returned by POST /api/evaluations/regenerate.
        org_id (str): Organization ID resolved by the auth dependency.

    Returns:
        dict[str, str]: ``{"status": "cancelled"}`` on successful flag set.

    Raises:
        HTTPException: 404 when ``job_id`` is unknown or belongs to a
            different org.
    """
    job = REGEN_JOBS.get(job_id)
    if job is None or job.org_id != org_id:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    REGEN_JOBS.cancel(job_id)
    return {"status": "cancelled"}


@router.post(
    "/api/evaluations/grade_on_demand",
    response_model=GradeOnDemandResponse,
    response_model_exclude_none=False,
)
def grade_on_demand(
    payload: GradeOnDemandRequest,
    org_id: str = Depends(default_get_org_id),
) -> GradeOnDemandResponse:
    """Grade a single session synchronously; serve cached results within 24h.

    Flow:
      1. Read the operation_state cache; if a fresh entry exists, return it
         with ``cached=True``.
      2. Resolve the session's user_id from storage (skip with ``NO_REQUESTS``
         when the session is unknown — surfaced as 200 + ``skipped_reason``
         so the frontend's bounded-list click-through can handle stale ids
         locally without polluting 5xx telemetry).
      3. Invoke ``run_group_evaluation(force_regenerate=True)`` so the
         "already evaluated" short-circuit doesn't suppress a customer's
         explicit click.
      4. Find the freshly-written result_id and persist it in the cache
         with ``last_graded_at`` so future calls within 24h short-circuit.

    Args:
        payload (GradeOnDemandRequest): Session + version plus optional legacy evaluator name.
        org_id (str): Tenant identifier resolved by the auth dependency.

    Returns:
        GradeOnDemandResponse: Echoes ``session_id`` and carries either
            a fresh ``result_id`` (``cached=False``), a cached one
            (``cached=True``), or a ``skipped_reason`` (NO_REQUESTS).

    Raises:
        HTTPException: 503 when storage is not configured.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage not configured")

    evaluation_name = payload.evaluation_name or SINGLETON_AGENT_SUCCESS_EVALUATION_NAME
    cache_key = _grade_on_demand_cache_key(
        org_id,
        payload.session_id,
        payload.agent_version,
        evaluation_name,
    )
    now = int(datetime.now(UTC).timestamp())

    cached_result_id = _read_grade_on_demand_cache(storage, cache_key, now=now)
    if cached_result_id is not None:
        return GradeOnDemandResponse(
            session_id=payload.session_id,
            result_id=cached_result_id,
            cached=True,
            skipped_reason=None,
        )

    user_id = _resolve_session_user_id(storage, payload.session_id)
    if user_id is None:
        return GradeOnDemandResponse(
            session_id=payload.session_id,
            result_id=None,
            cached=False,
            skipped_reason="NO_REQUESTS",
        )

    previous_result_ids = set(
        storage.get_agent_success_evaluation_result_ids(
            user_id=user_id,
            session_id=payload.session_id,
            evaluation_name=evaluation_name,
            agent_version=payload.agent_version,
        )
    )

    # Two operation_state rows are intentionally written for this session:
    #   1) `grade_on_demand::{len:org}::{len:session}::{len:version}::{len:eval}`
    #      (each component length-prefixed by `_grade_on_demand_cache_key` so the
    #      key is injective) — our 24h cache, set below after result_id resolves.
    #   2) `agent_success_group_eval::{org_id}::{user_id}::{session_id}`
    #      — the runner's own "evaluated" marker, written by
    #      run_group_evaluation. Future background runs without
    #      force_regenerate will skip this session as a result.
    # The cache key namespaces are distinct so the two markers do not
    # interfere; the explicit force_regenerate=True here is what makes
    # an on-demand grade always do real work on a cache miss.
    run_group_evaluation(
        org_id=org_id,
        user_id=user_id,
        session_id=payload.session_id,
        agent_version=payload.agent_version,
        source=None,
        request_context=reflexio.request_context,
        llm_client=reflexio.llm_client,
        force_regenerate=True,
    )

    result_id = _find_fresh_result_id(
        storage,
        user_id=user_id,
        session_id=payload.session_id,
        agent_version=payload.agent_version,
        evaluation_name=evaluation_name,
        previous_result_ids=previous_result_ids,
    )

    storage.upsert_operation_state(
        cache_key,
        {"last_graded_at": now, "result_id": result_id},
    )

    return GradeOnDemandResponse(
        session_id=payload.session_id,
        result_id=result_id,
        cached=False,
        skipped_reason=None,
    )


@router.get(
    "/api/evaluations/shadow_comparisons/recent",
    response_model=GetRecentShadowComparisonsResponse,
)
def get_recent_shadow_comparisons(
    limit: int = 10,
    org_id: str = Depends(default_get_org_id),
) -> GetRecentShadowComparisonsResponse:
    """Return the N most recent shadow comparison verdicts for the pinned rubric.

    Filters to the org's currently pinned
    ``Config.shadow_comparison_judge_prompt_version`` so verdicts produced
    under an older rubric do not mix into the drawer or the Top 10
    disagreements widget. Storage returns verdicts newest-first and caps the
    read at ``limit``.

    Args:
        limit (int): Max verdicts to return. Clamped to ``[1, 100]``.
            Default 10 — matches the size of the drawer and Top 10 widget.
        org_id (str): Tenant identifier resolved by the auth dependency.

    Returns:
        GetRecentShadowComparisonsResponse: Verdicts in newest-first order.
            Empty list when the backend does not support the
            ``shadow_comparison_verdicts`` storage feature, when no verdicts
            exist in the 30-day window, or when no verdicts match the pinned
            prompt version.

    Raises:
        HTTPException: 503 when storage is not configured.
    """
    clamped_limit = max(1, min(int(limit), _RECENT_SHADOW_COMPARISONS_MAX_LIMIT))
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage not configured")

    config = reflexio.request_context.configurator.get_config()
    pinned_version = (
        config.shadow_comparison_judge_prompt_version
        if config is not None
        else "v1.0.0"
    )

    now = int(datetime.now(UTC).timestamp())
    try:
        verdicts = storage.get_recent_shadow_comparison_verdicts(
            from_ts=now - _RECENT_SHADOW_COMPARISONS_LOOKBACK_SECONDS,
            to_ts=now,
            judge_prompt_version=pinned_version,
            limit=clamped_limit,
        )
    except NotImplementedError:
        # Backends that don't support shadow verdicts (e.g., Disk) should
        # quietly return empty rather than 5xx — the surface degrades to
        # "no data yet" in the UI.
        return GetRecentShadowComparisonsResponse(verdicts=[])

    return GetRecentShadowComparisonsResponse(verdicts=verdicts)
