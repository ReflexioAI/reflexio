"""System/meta/stats/operations route handlers (extracted from api.py, Tier3 A2)."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)

from reflexio.models.api_schema.retriever_schema import (
    GetDashboardStatsRequest,
    GetDashboardStatsResponse,
    StorageStatsRequest,
    StorageStatsResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AdminInvalidateCacheRequest,
    AdminInvalidateCacheResponse,
    CancelOperationRequest,
    CancelOperationResponse,
    ClearUserDataRequest,
    ClearUserDataResponse,
    GetOperationStatusRequest,
    GetOperationStatusResponse,
    WhoamiResponse,
)
from reflexio.server.api_endpoints import (
    account_api,
    publisher_api,
)
from reflexio.server.auth import (
    default_get_org_id,
)
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter
from reflexio.server.routes._common import _run_limited_api

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Reflexio API",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=None)
async def health_check() -> dict[str, str]:
    """Health check endpoint for ECS/container orchestration."""
    return {"status": "healthy"}


@router.get(
    "/api/whoami",
    response_model=WhoamiResponse,
    response_model_exclude_none=True,
)
def whoami_endpoint(
    org_id: str = Depends(default_get_org_id),
) -> WhoamiResponse:
    """Return the caller's org and masked storage routing.

    Powers ``reflexio status``. Safe to call unauthenticated in
    self-host mode; the enterprise server wraps this in Bearer auth.
    """
    return account_api.whoami(org_id=org_id)


@router.get(
    "/api/storage_stats",
    response_model=StorageStatsResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")  # Rate limit for read operations
def storage_stats(
    request: Request,
    user_id: str,
    org_id: str = Depends(default_get_org_id),
) -> StorageStatsResponse:
    """Return lightweight metadata about a user's profiles and playbooks.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        user_id (str): Target user id, passed as a query parameter so this is
            a cacheable, idempotent GET.
        org_id (str): Organization ID

    Returns:
        StorageStatsResponse: Counts and timestamp range for the user.
    """
    return _run_limited_api(
        org_id,
        "search",
        lambda: reflexio_cache.get_reflexio(org_id=org_id).storage_stats(
            StorageStatsRequest(user_id=user_id)
        ),
    )


@router.post(
    "/api/clear_user_data",
    response_model=ClearUserDataResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def clear_user_data(
    request: Request,
    payload: ClearUserDataRequest,
    org_id: str = Depends(default_get_org_id),
) -> ClearUserDataResponse:
    """Delete all rows scoped to a single ``user_id``.

    Removes the user's interactions, session outcomes, user playbooks,
    profiles, and requests. Does NOT touch ``agent_playbooks`` — they are
    intentionally shared cross-project. Used by paired-protocol
    harnesses (e.g. SWE-bench) to isolate per-task data on a shared
    backend without one task's clear-all nuking another in-flight
    task's rows.

    Args:
        request (ClearUserDataRequest): Request containing the target user_id
        org_id (str): Organization ID

    Returns:
        ClearUserDataResponse: Response with per-entity deletion counts
    """
    return publisher_api.clear_user_data(org_id=org_id, request=payload)


@router.post("/api/admin/cache/invalidate")
def admin_invalidate_cache(
    payload: AdminInvalidateCacheRequest,
    org_id: str = Depends(default_get_org_id),
) -> AdminInvalidateCacheResponse:
    """Explicitly evict the per-org Reflexio cache entry.

    Necessary when the running config has been mutated through a
    channel the server can't observe — e.g. another replica wrote to
    the shared DB, or an operator hand-edited a self-host config file
    on a backend that doesn't support cheap version probing. The
    file-mtime check (Phase 1) and DB version check (Phase 3) cover
    most cases automatically; this endpoint is the manual escape hatch.

    Auth uses the same dependency as ``/api/set_config`` — callers
    can only invalidate their own org's cache. If the request body
    supplies ``org_id`` it must match the dep-resolved value;
    cross-org invalidation is intentionally NOT exposed here.

    Args:
        payload: Optional ``org_id`` (verification only — must match
            the caller's authenticated org if provided).
        org_id: Organization ID resolved by the auth layer.

    Returns:
        AdminInvalidateCacheResponse: ``invalidated`` is True iff an
        entry was evicted (False is a successful no-op when nothing
        was cached).

    Raises:
        HTTPException: 403 when the body's ``org_id`` differs from the
            caller's authenticated org.
    """
    if payload.org_id is not None and payload.org_id != org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Cross-org cache invalidation is not supported; "
                "omit org_id or pass your own."
            ),
        )
    invalidated = reflexio_cache.invalidate_reflexio_cache(org_id=org_id)
    return AdminInvalidateCacheResponse(invalidated=invalidated, org_id=org_id)


@router.post(
    "/api/get_dashboard_stats",
    response_model=GetDashboardStatsResponse,
    response_model_exclude_none=True,
)
@limiter.limit("30/minute")
def get_dashboard_stats(
    request: Request,
    payload: GetDashboardStatsRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetDashboardStatsResponse:
    """Get comprehensive dashboard statistics including counts and time-series data.

    Args:
        request (GetDashboardStatsRequest): Request containing days_back and granularity
        org_id (str): Organization ID

    Returns:
        GetDashboardStatsResponse: Response containing dashboard statistics
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Get dashboard stats using Reflexio's method
    return reflexio.get_dashboard_stats(payload)


@router.get(
    "/api/get_operation_status",
    response_model=GetOperationStatusResponse,
    response_model_exclude_none=True,
)
def get_operation_status_endpoint(
    service_name: str = "profile_generation",
    org_id: str = Depends(default_get_org_id),
) -> GetOperationStatusResponse:
    """Get the status of an operation (e.g., profile generation rerun or manual).

    Args:
        service_name (str): The service name to query. Defaults to "profile_generation"
        org_id (str): Organization ID

    Returns:
        GetOperationStatusResponse: Response containing operation status info
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Get operation status
    request = GetOperationStatusRequest(service_name=service_name)
    return reflexio.get_operation_status(request)


@router.post(
    "/api/cancel_operation",
    response_model=CancelOperationResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def cancel_operation_endpoint(
    request: Request,
    payload: CancelOperationRequest,
    org_id: str = Depends(default_get_org_id),
) -> CancelOperationResponse:
    """Cancel an in-progress operation (rerun or manual generation).

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (CancelOperationRequest): Request containing optional service_name
        org_id (str): Organization ID

    Returns:
        CancelOperationResponse: Response with list of services that were cancelled
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    return reflexio.cancel_operation(payload)
