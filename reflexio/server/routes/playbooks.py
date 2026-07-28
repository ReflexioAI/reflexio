"""Playbook route handlers (extracted from api.py, Tier3 A2)."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Request,
)

from reflexio.models.api_schema.retriever_schema import (
    GetAgentPlaybooksRequest,
    GetAgentPlaybooksViewResponse,
    GetPlaybookApplicationStatsRequest,
    GetPlaybookApplicationStatsResponse,
    GetUserPlaybooksRequest,
    GetUserPlaybooksViewResponse,
    UpdateAgentPlaybookRequest,
    UpdateAgentPlaybookResponse,
    UpdatePlaybookStatusRequest,
    UpdatePlaybookStatusResponse,
    UpdateUserPlaybookRequest,
    UpdateUserPlaybookResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AddAgentPlaybookRequest,
    AddAgentPlaybookResponse,
    AddUserPlaybookRequest,
    AddUserPlaybookResponse,
    BulkDeleteResponse,
    DeleteAgentPlaybookRequest,
    DeleteAgentPlaybookResponse,
    DeleteAgentPlaybooksByIdsRequest,
    DeleteUserPlaybookRequest,
    DeleteUserPlaybookResponse,
    DeleteUserPlaybooksByIdsRequest,
    DowngradeUserPlaybooksRequest,
    DowngradeUserPlaybooksResponse,
    ManualPlaybookGenerationRequest,
    ManualPlaybookGenerationResponse,
    PlaybookAggregationChangeLogResponse,
    RerunPlaybookGenerationRequest,
    RerunPlaybookGenerationResponse,
    ReviewUserPlaybooksRequest,
    ReviewUserPlaybooksResponse,
    RunPlaybookAggregationRequest,
    RunPlaybookAggregationResponse,
    UpgradeUserPlaybooksRequest,
    UpgradeUserPlaybooksResponse,
)
from reflexio.models.api_schema.ui.converters import (
    to_agent_playbook_view,
    to_user_playbook_view,
)
from reflexio.server.api_endpoints import (
    publisher_api,
)
from reflexio.server.auth import (
    default_billing_gate,
    default_get_caller_type,
    default_get_org_id,
)
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter
from reflexio.server.routes._common import _run_limited_api
from reflexio.server.routes._metering import (
    _meter_applied_learnings,
)
from reflexio.server.services.playbook.review_service import new_review_run_id

logger = logging.getLogger(__name__)
router = APIRouter()


def _review_user_playbooks_in_background(
    reflexio: Any,
    org_id: str,
    payload: ReviewUserPlaybooksRequest,
    run_id: str,
) -> None:
    """Run an apply-mode review off the request, keeping the operation limit."""
    try:
        _run_limited_api(
            org_id,
            "aggregation",
            lambda: reflexio.review_user_playbooks(payload, run_id=run_id),
        )
    except Exception:
        # Nothing is left to raise to: the response was already sent.
        logger.exception(
            "event=review_user_playbooks_background_failed org_id=%s run_id=%s",
            org_id,
            run_id,
        )


@router.post(
    "/api/review_user_playbooks",
    response_model=ReviewUserPlaybooksResponse,
    response_model_exclude_none=True,
)
@limiter.limit("5/minute")
def review_user_playbooks_endpoint(
    request: Request,
    payload: ReviewUserPlaybooksRequest,
    background_tasks: BackgroundTasks,
    org_id: str = Depends(default_get_org_id),
) -> ReviewUserPlaybooksResponse:
    """Re-review a bounded creation-time window of current user playbooks.

    Report mode answers inline because its report exists only in the response.
    Apply mode issues one LLM call and one transaction per selected playbook, so
    it runs in the background to avoid a proxy timeout that would strand the
    client with committed writes and no response; each applied decision is
    recorded on the replacement's lineage under the returned ``run_id``.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    if payload.report_only:
        return _run_limited_api(
            org_id,
            "aggregation",
            lambda: reflexio.review_user_playbooks(payload),
        )

    run_id = new_review_run_id()
    background_tasks.add_task(
        _review_user_playbooks_in_background, reflexio, org_id, payload, run_id
    )
    return ReviewUserPlaybooksResponse(
        success=True,
        report_only=False,
        run_id=run_id,
        msg="User playbook review started",
    )


@router.post(
    "/api/add_user_playbook",
    response_model=AddUserPlaybookResponse,
    response_model_exclude_none=True,
)
@limiter.limit("60/minute")  # Rate limit for write operations
def add_user_playbook_endpoint(
    request: Request,
    payload: AddUserPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> AddUserPlaybookResponse:
    """Add user playbook directly to storage.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (AddUserPlaybookRequest): The request containing user playbooks
        org_id (str): Organization ID

    Returns:
        AddUserPlaybookResponse: Response containing success status, message, and added count
    """
    return publisher_api.add_user_playbook(org_id=org_id, request=payload)


@router.post(
    "/api/add_agent_playbook",
    response_model=AddAgentPlaybookResponse,
    response_model_exclude_none=True,
)
@limiter.limit("60/minute")  # Rate limit for write operations
def add_agent_playbook_endpoint(
    request: Request,
    payload: AddAgentPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> AddAgentPlaybookResponse:
    """Add agent playbook directly to storage.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (AddAgentPlaybookRequest): The request containing agent playbooks
        org_id (str): Organization ID

    Returns:
        AddAgentPlaybookResponse: Response containing success status, message, and added count
    """
    return publisher_api.add_agent_playbook(org_id=org_id, request=payload)


@router.get(
    "/api/playbook_aggregation_change_logs",
    response_model=PlaybookAggregationChangeLogResponse,
)
def get_playbook_aggregation_change_logs(
    playbook_name: str,
    agent_version: str,
    org_id: str = Depends(default_get_org_id),
) -> PlaybookAggregationChangeLogResponse:
    return reflexio_cache.get_reflexio(
        org_id=org_id
    ).get_playbook_aggregation_change_logs(
        playbook_name=playbook_name,
        agent_version=agent_version,
    )


@router.delete(
    "/api/delete_agent_playbook",
    response_model=DeleteAgentPlaybookResponse,
    response_model_exclude_none=True,
)
def delete_agent_playbook(
    request: DeleteAgentPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteAgentPlaybookResponse:
    return publisher_api.delete_agent_playbook(org_id=org_id, request=request)


@router.delete(
    "/api/delete_user_playbook",
    response_model=DeleteUserPlaybookResponse,
    response_model_exclude_none=True,
)
def delete_user_playbook(
    request: DeleteUserPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteUserPlaybookResponse:
    return publisher_api.delete_user_playbook(org_id=org_id, request=request)


@router.delete(
    "/api/delete_agent_playbooks_by_ids",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_agent_playbooks_by_ids(
    request: DeleteAgentPlaybooksByIdsRequest,
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete multiple agent playbooks by their IDs.

    Args:
        request (DeleteAgentPlaybooksByIdsRequest): Request containing list of agent playbook IDs to delete
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_agent_playbooks_by_ids_bulk(
        org_id=org_id, request=request
    )


@router.delete(
    "/api/delete_user_playbooks_by_ids",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_user_playbooks_by_ids(
    request: DeleteUserPlaybooksByIdsRequest,
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete multiple user playbooks by their IDs.

    Args:
        request (DeleteUserPlaybooksByIdsRequest): Request containing list of user playbook IDs to delete
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_user_playbooks_by_ids_bulk(
        org_id=org_id, request=request
    )


@router.delete(
    "/api/delete_all_playbooks",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def delete_all_playbooks(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all playbooks (both user and agent).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_playbooks_bulk(org_id=org_id)


@router.delete(
    "/api/delete_all_user_playbooks",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def delete_all_user_playbooks(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all user playbooks (user only, not agent).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_user_playbooks_bulk(org_id=org_id)


@router.delete(
    "/api/delete_all_agent_playbooks",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def delete_all_agent_playbooks(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all agent playbooks (agent only, not user).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_agent_playbooks_bulk(org_id=org_id)


@router.post(
    "/api/run_playbook_aggregation",
    response_model=RunPlaybookAggregationResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")  # Strict limit for expensive operations
def run_playbook_aggregation(
    request: Request,
    payload: RunPlaybookAggregationRequest,
    org_id: str = Depends(default_get_org_id),
) -> RunPlaybookAggregationResponse:
    return _run_limited_api(
        org_id,
        "aggregation",
        lambda: publisher_api.run_playbook_aggregation(org_id=org_id, request=payload),
    )


@router.post(
    "/api/get_user_playbooks",
    response_model=GetUserPlaybooksViewResponse,
    response_model_exclude_none=True,
)
def get_user_playbooks(
    request: GetUserPlaybooksRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetUserPlaybooksViewResponse:
    """Get user playbooks with internal fields filtered out.

    Args:
        request (GetUserPlaybooksRequest): The get request
        org_id (str): Organization ID

    Returns:
        GetUserPlaybooksViewResponse: Response containing user playbooks without internal fields
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    response = reflexio.get_user_playbooks(request)
    return GetUserPlaybooksViewResponse(
        success=response.success,
        user_playbooks=[to_user_playbook_view(rf) for rf in response.user_playbooks],
        msg=response.msg,
    )


@router.post(
    "/api/get_agent_playbooks",
    response_model=GetAgentPlaybooksViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")  # Rate limit for read operations
def get_agent_playbooks(
    request: Request,
    payload: GetAgentPlaybooksRequest,
    org_id: str = Depends(default_get_org_id),
    caller_type: str = Depends(default_get_caller_type),
    _gate: None = Depends(default_billing_gate("application")),  # noqa: B008
) -> GetAgentPlaybooksViewResponse:
    """Get agent playbooks with internal fields filtered out.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (GetAgentPlaybooksRequest): The get request
        org_id (str): Organization ID
        caller_type (str): Billing caller classification (injected via dependency).

    Returns:
        GetAgentPlaybooksViewResponse: Response containing agent playbooks without internal fields
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    response = reflexio.get_agent_playbooks(payload)
    resp = GetAgentPlaybooksViewResponse(
        success=response.success,
        agent_playbooks=[to_agent_playbook_view(fb) for fb in response.agent_playbooks],
        msg=response.msg,
    )
    _meter_applied_learnings(
        org_id=org_id,
        caller_type=caller_type,
        surfaced_count=len(resp.agent_playbooks),
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    return resp


@router.put(
    "/api/update_agent_playbook_status",
    response_model=UpdatePlaybookStatusResponse,
    response_model_exclude_none=True,
)
def update_agent_playbook_status_endpoint(
    request: UpdatePlaybookStatusRequest,
    org_id: str = Depends(default_get_org_id),
) -> UpdatePlaybookStatusResponse:
    """Update the status of a specific playbook.

    Args:
        request (UpdatePlaybookStatusRequest): The update request
        org_id (str): Organization ID

    Returns:
        UpdatePlaybookStatusResponse: Response containing success status and message
    """
    return publisher_api.update_agent_playbook_status(org_id=org_id, request=request)


@router.put(
    "/api/update_agent_playbook",
    response_model=UpdateAgentPlaybookResponse,
    response_model_exclude_none=True,
)
def update_agent_playbook_endpoint(
    request: UpdateAgentPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> UpdateAgentPlaybookResponse:
    """Update editable fields of a specific agent playbook.

    Args:
        request (UpdateAgentPlaybookRequest): The update request
        org_id (str): Organization ID

    Returns:
        UpdateAgentPlaybookResponse: Response containing success status and message
    """
    return publisher_api.update_agent_playbook(org_id=org_id, request=request)


@router.put(
    "/api/update_user_playbook",
    response_model=UpdateUserPlaybookResponse,
    response_model_exclude_none=True,
)
def update_user_playbook_endpoint(
    request: UpdateUserPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> UpdateUserPlaybookResponse:
    """Update editable fields of a specific user playbook.

    Args:
        request (UpdateUserPlaybookRequest): The update request
        org_id (str): Organization ID

    Returns:
        UpdateUserPlaybookResponse: Response containing success status and message
    """
    return publisher_api.update_user_playbook(org_id=org_id, request=request)


@router.post(
    "/api/get_playbook_application_stats",
    response_model=GetPlaybookApplicationStatsResponse,
    response_model_exclude_none=True,
)
def get_playbook_application_stats(
    request: GetPlaybookApplicationStatsRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetPlaybookApplicationStatsResponse:
    """Get per-rule citation counts aggregated from interactions.

    Returns one row per cited (kind, real_id) over the look-back window,
    sorted by applied_count descending. Lets the dashboard show users a
    per-rule "track record" — how often each playbook or profile has been
    applied and when it last fired.

    Args:
        request (GetPlaybookApplicationStatsRequest): Request containing
            days_back.
        org_id (str): Organization ID.

    Returns:
        GetPlaybookApplicationStatsResponse: Response containing aggregated
            stats.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    return reflexio.get_playbook_application_stats(request)


@router.post(
    "/api/rerun_playbook_generation",
    response_model=RerunPlaybookGenerationResponse,
    response_model_exclude_none=True,
)
@limiter.limit("5/minute")  # Strict limit for expensive operations
def rerun_playbook_generation_endpoint(
    request: Request,
    payload: RerunPlaybookGenerationRequest,
    background_tasks: BackgroundTasks,
    org_id: str = Depends(default_get_org_id),
) -> RerunPlaybookGenerationResponse:
    """Rerun playbook generation with filtered interactions.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (RerunPlaybookGenerationRequest): Request containing agent_version, time filters, and optional playbook_name
        background_tasks (BackgroundTasks): Background task runner
        org_id (str): Organization ID

    Returns:
        RerunPlaybookGenerationResponse: Response containing success status and playbooks generated count
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Run the long-running task in the background to avoid proxy timeout
    # Client polls get_operation_status for progress
    background_tasks.add_task(reflexio.rerun_playbook_generation, payload)

    return RerunPlaybookGenerationResponse(
        success=True, msg="Playbook generation started"
    )


@router.post(
    "/api/manual_playbook_generation",
    response_model=ManualPlaybookGenerationResponse,
    response_model_exclude_none=True,
)
@limiter.limit("5/minute")  # Strict limit for expensive operations
def manual_playbook_generation_endpoint(
    request: Request,
    payload: ManualPlaybookGenerationRequest,
    org_id: str = Depends(default_get_org_id),
) -> ManualPlaybookGenerationResponse:
    """Manually trigger playbook generation with window-sized interactions and CURRENT output.

    Runs with auto_run=False, which bypasses the regular stride/should_run
    gates. Only playbook extraction is triggered. Each extractor uses its own
    window_size_override when present, falling back to the global window_size.
    Output is CURRENT playbooks only.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (ManualPlaybookGenerationRequest): Request containing agent_version, source, and playbook_name
        org_id (str): Organization ID

    Returns:
        ManualPlaybookGenerationResponse: Response containing success status and playbooks generated count
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Call manual_playbook_generation
    return reflexio.manual_playbook_generation(payload)


@router.post(
    "/api/upgrade_all_user_playbooks",
    response_model=UpgradeUserPlaybooksResponse,
    response_model_exclude_none=True,
)
def upgrade_all_user_playbooks_endpoint(
    request: UpgradeUserPlaybooksRequest,
    org_id: str = Depends(default_get_org_id),
) -> UpgradeUserPlaybooksResponse:
    """Upgrade all user playbooks by deleting old ARCHIVED, archiving CURRENT, and promoting PENDING.

    This operation performs three atomic steps:
    1. Delete all ARCHIVED user playbooks (old archived from previous upgrades)
    2. Archive all CURRENT user playbooks → ARCHIVED (save current state for potential rollback)
    3. Promote all PENDING user playbooks → CURRENT (activate new user playbooks)

    Args:
        request (UpgradeUserPlaybooksRequest): The upgrade request with optional agent_version and playbook_name filters
        org_id (str): Organization ID

    Returns:
        UpgradeUserPlaybooksResponse: Response containing success status and counts
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Call upgrade_all_user_playbooks with request
    return reflexio.upgrade_all_user_playbooks(request=request)


@router.post(
    "/api/downgrade_all_user_playbooks",
    response_model=DowngradeUserPlaybooksResponse,
    response_model_exclude_none=True,
)
def downgrade_all_user_playbooks_endpoint(
    request: DowngradeUserPlaybooksRequest,
    org_id: str = Depends(default_get_org_id),
) -> DowngradeUserPlaybooksResponse:
    """Downgrade all user playbooks by archiving CURRENT and restoring ARCHIVED.

    This operation performs three atomic steps:
    1. Mark all CURRENT user playbooks → ARCHIVE_IN_PROGRESS (temporary status)
    2. Restore all ARCHIVED user playbooks → CURRENT
    3. Move all ARCHIVE_IN_PROGRESS user playbooks → ARCHIVED

    Args:
        request (DowngradeUserPlaybooksRequest): The downgrade request with optional agent_version and playbook_name filters
        org_id (str): Organization ID

    Returns:
        DowngradeUserPlaybooksResponse: Response containing success status and counts
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Call downgrade_all_user_playbooks with request
    return reflexio.downgrade_all_user_playbooks(request=request)
