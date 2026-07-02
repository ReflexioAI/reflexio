"""Profile route handlers (extracted from api.py, Tier3 A2)."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Request,
)

from reflexio.models.api_schema.retriever_schema import (
    GetProfileStatisticsResponse,
    GetProfilesViewResponse,
    GetUserProfilesRequest,
    ProfileChangeLogViewResponse,
    UpdateUserProfileRequest,
    UpdateUserProfileResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AddUserProfileRequest,
    AddUserProfileResponse,
    BulkDeleteResponse,
    DeleteProfilesByIdsRequest,
    DeleteUserProfileRequest,
    DeleteUserProfileResponse,
    DowngradeProfilesRequest,
    DowngradeProfilesResponse,
    ManualProfileGenerationRequest,
    ManualProfileGenerationResponse,
    RerunProfileGenerationRequest,
    RerunProfileGenerationResponse,
    Status,
    UpgradeProfilesRequest,
    UpgradeProfilesResponse,
)
from reflexio.models.api_schema.ui.converters import (
    to_profile_change_log_view,
    to_profile_view,
)
from reflexio.server.api_endpoints import (
    publisher_api,
)
from reflexio.server.auth import (
    default_get_org_id,
)
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/api/add_user_profile",
    response_model=AddUserProfileResponse,
    response_model_exclude_none=True,
)
@limiter.limit("60/minute")  # Rate limit for write operations
def add_user_profile_endpoint(
    request: Request,
    payload: AddUserProfileRequest,
    org_id: str = Depends(default_get_org_id),
) -> AddUserProfileResponse:
    """Add user profile directly to storage, bypassing inference.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (AddUserProfileRequest): The request containing user profiles
        org_id (str): Organization ID

    Returns:
        AddUserProfileResponse: Response containing success status, message, and added count
    """
    return publisher_api.add_user_profile(org_id=org_id, request=payload)


@router.get("/api/profile_change_log", response_model=ProfileChangeLogViewResponse)
def get_profile_change_log(
    org_id: str = Depends(default_get_org_id),
) -> ProfileChangeLogViewResponse:
    # Serves the reconstructed profile change log (rebuilt from lineage events). The
    # legacy `profile_change_logs` table is no longer written; see
    # reconstruct_profile_change_log in lib/_profiles.py.
    response = reflexio_cache.get_reflexio(org_id=org_id).get_profile_change_logs()
    return ProfileChangeLogViewResponse(
        success=response.success,
        profile_change_logs=[
            to_profile_change_log_view(log) for log in response.profile_change_logs
        ],
    )


@router.delete(
    "/api/delete_profile",
    response_model=DeleteUserProfileResponse,
    response_model_exclude_none=True,
)
def delete_profile(
    request: DeleteUserProfileRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteUserProfileResponse:
    return publisher_api.delete_user_profile(org_id=org_id, request=request)


@router.delete(
    "/api/delete_profiles_by_ids",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_profiles_by_ids(
    request: DeleteProfilesByIdsRequest,
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete multiple profiles by their IDs.

    Args:
        request (DeleteProfilesByIdsRequest): Request containing list of profile IDs to delete
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_profiles_by_ids(org_id=org_id, request=request)


@router.delete(
    "/api/delete_all_profiles",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def delete_all_profiles(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all profiles.

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_profiles_bulk(org_id=org_id)


@router.post(
    "/api/get_profiles",
    response_model=GetProfilesViewResponse,
    response_model_exclude_none=True,
)
def get_profiles(
    request: GetUserProfilesRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetProfilesViewResponse:
    response = reflexio_cache.get_reflexio(org_id=org_id).get_profiles(request)
    return GetProfilesViewResponse(
        success=response.success,
        user_profiles=[to_profile_view(p) for p in response.user_profiles],
        msg=response.msg,
    )


@router.get(
    "/api/get_all_profiles",
    response_model=GetProfilesViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("30/minute")
def get_all_profiles(
    request: Request,
    limit: int = 100,
    status_filter: str | None = None,
    profile_id: str | None = None,
    user_id: str | None = None,
    query: str | None = None,
    source: str | None = None,
    profile_time_to_live: str | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
    include_tombstones: bool = False,
    org_id: str = Depends(default_get_org_id),
) -> GetProfilesViewResponse:
    """Get all user profiles across all users.

    Args:
        limit (int, optional): Maximum number of profiles to return. Defaults to 100.
        status_filter (str, optional): Filter by profile status. Can be "current", "pending", or "archived".
        profile_id (str, optional): Exact profile ID to retrieve.
        user_id (str, optional): Exact user ID to filter by.
        query (str, optional): Case-insensitive text filter across visible fields.
        source (str, optional): Exact profile source to filter by.
        profile_time_to_live (str, optional): Exact TTL value to filter by.
        start_time (int, optional): Minimum last-modified epoch seconds.
        end_time (int, optional): Maximum last-modified epoch seconds.
        include_tombstones (bool, optional): Include merged/superseded rows when
            looking up a specific profile_id.
        org_id (str): Organization ID

    Returns:
        GetProfilesViewResponse: Response containing all user profiles
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    if profile_id and include_tombstones:
        storage = reflexio.request_context.storage
        if storage is None:
            return GetProfilesViewResponse(
                success=True,
                user_profiles=[],
                msg="Storage is not configured",
            )
        profile = storage.get_profile_by_id(
            profile_id, include_tombstones=include_tombstones
        )
        profiles = [profile] if profile else []
        return GetProfilesViewResponse(
            success=True,
            user_profiles=[to_profile_view(p) for p in profiles],
            msg=f"Found {len(profiles)} profile(s)",
        )

    # Map status_filter string to Status list
    status_filter_list = None
    if status_filter == "current":
        status_filter_list = [None]
    elif status_filter == "pending":
        status_filter_list = [Status.PENDING]
    elif status_filter == "archived":
        status_filter_list = [Status.ARCHIVED]

    response = reflexio.get_all_profiles(
        limit=limit,
        status_filter=status_filter_list,  # type: ignore[reportArgumentType]
        user_id=user_id,
        profile_id=profile_id,
        query=query,
        source=source,
        profile_time_to_live=profile_time_to_live,
        start_time=start_time,
        end_time=end_time,
    )
    return GetProfilesViewResponse(
        success=response.success,
        user_profiles=[to_profile_view(p) for p in response.user_profiles],
        msg=response.msg,
    )


@router.get(
    "/api/get_profile_statistics",
    response_model=GetProfileStatisticsResponse,
    response_model_exclude_none=True,
)
def get_profile_statistics(
    org_id: str = Depends(default_get_org_id),
) -> GetProfileStatisticsResponse:
    """Get efficient profile statistics using storage layer queries.

    Args:
        org_id (str): Organization ID

    Returns:
        GetProfileStatisticsResponse: Response containing profile counts by status
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Get profile statistics using Reflexio's method
    return reflexio.get_profile_statistics()


@router.put(
    "/api/update_user_profile",
    response_model=UpdateUserProfileResponse,
    response_model_exclude_none=True,
)
def update_user_profile_endpoint(
    request: UpdateUserProfileRequest,
    org_id: str = Depends(default_get_org_id),
) -> UpdateUserProfileResponse:
    """Apply a partial update to an existing user profile.

    Args:
        request (UpdateUserProfileRequest): The update request
        org_id (str): Organization ID

    Returns:
        UpdateUserProfileResponse: Response containing success status and message
    """
    return publisher_api.update_user_profile(org_id=org_id, request=request)


@router.post(
    "/api/rerun_profile_generation",
    response_model=RerunProfileGenerationResponse,
    response_model_exclude_none=True,
)
@limiter.limit("5/minute")  # Strict limit for expensive operations
def rerun_profile_generation_endpoint(
    request: Request,
    payload: RerunProfileGenerationRequest,
    background_tasks: BackgroundTasks,
    org_id: str = Depends(default_get_org_id),
) -> RerunProfileGenerationResponse:
    """Rerun profile generation for a user with filtered interactions.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (RerunProfileGenerationRequest): Request containing user_id, time filters, and source
        background_tasks (BackgroundTasks): Background task runner
        org_id (str): Organization ID

    Returns:
        RerunProfileGenerationResponse: Response containing success status and profiles generated count
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Run the long-running task in the background to avoid proxy timeout
    # Client polls get_operation_status for progress
    background_tasks.add_task(reflexio.rerun_profile_generation, payload)

    return RerunProfileGenerationResponse(
        success=True, msg="Profile generation started"
    )


@router.post(
    "/api/manual_profile_generation",
    response_model=ManualProfileGenerationResponse,
    response_model_exclude_none=True,
)
@limiter.limit("5/minute")  # Strict limit for expensive operations
def manual_profile_generation_endpoint(
    request: Request,
    payload: ManualProfileGenerationRequest,
    org_id: str = Depends(default_get_org_id),
) -> ManualProfileGenerationResponse:
    """Manually trigger profile generation with window-sized interactions and CURRENT output.

    Runs with auto_run=False, which bypasses the regular stride/should_run
    gates. Only profile extraction is triggered. Each extractor uses its own
    window_size_override when present, falling back to the global window_size.
    Output is CURRENT profiles only.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (ManualProfileGenerationRequest): Request containing user_id, source, and extractor_names
        org_id (str): Organization ID

    Returns:
        ManualProfileGenerationResponse: Response containing success status and profiles generated count
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Call manual_profile_generation
    return reflexio.manual_profile_generation(payload)


@router.post(
    "/api/upgrade_all_profiles",
    response_model=UpgradeProfilesResponse,
    response_model_exclude_none=True,
)
def upgrade_all_profiles_endpoint(
    request: UpgradeProfilesRequest,
    org_id: str = Depends(default_get_org_id),
) -> UpgradeProfilesResponse:
    """Upgrade all profiles by deleting old ARCHIVED, archiving CURRENT, and promoting PENDING.

    This operation performs three atomic steps:
    1. Delete all ARCHIVED profiles (old archived profiles from previous upgrades)
    2. Archive all CURRENT profiles → ARCHIVED (save current state for potential rollback)
    3. Promote all PENDING profiles → CURRENT (activate new profiles)

    Args:
        request (UpgradeProfilesRequest): The upgrade request with only_affected_users parameter
        org_id (str): Organization ID

    Returns:
        UpgradeProfilesResponse: Response containing success status and counts
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Call upgrade_all_profiles with request
    return reflexio.upgrade_all_profiles(request=request)


@router.post(
    "/api/downgrade_all_profiles",
    response_model=DowngradeProfilesResponse,
    response_model_exclude_none=True,
)
def downgrade_all_profiles_endpoint(
    request: DowngradeProfilesRequest,
    org_id: str = Depends(default_get_org_id),
) -> DowngradeProfilesResponse:
    """Downgrade all profiles by demoting CURRENT to PENDING and restoring ARCHIVED.

    This operation performs two atomic steps:
    1. Demote all CURRENT profiles → PENDING
    2. Restore all ARCHIVED profiles → CURRENT

    Args:
        request (DowngradeProfilesRequest): The downgrade request with only_affected_users parameter
        org_id (str): Organization ID

    Returns:
        DowngradeProfilesResponse: Response containing success status and counts
    """
    # Create Reflexio instance
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    # Call downgrade_all_profiles with request
    return reflexio.downgrade_all_profiles(request=request)
