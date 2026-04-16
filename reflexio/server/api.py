import asyncio
import logging
from collections.abc import Callable

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from reflexio.models.api_schema.retriever_schema import (
    GetAgentPlaybooksRequest,
    GetAgentPlaybooksViewResponse,
    GetAgentSuccessEvaluationResultsRequest,
    GetDashboardStatsRequest,
    GetDashboardStatsResponse,
    GetEvaluationResultsViewResponse,
    GetInteractionsRequest,
    GetInteractionsViewResponse,
    GetProfileStatisticsResponse,
    GetProfilesViewResponse,
    GetRequestsRequest,
    GetRequestsViewResponse,
    GetUserPlaybooksRequest,
    GetUserPlaybooksViewResponse,
    GetUserProfilesRequest,
    ProfileChangeLogViewResponse,
    RequestDataView,
    SearchAgentPlaybookRequest,
    SearchAgentPlaybooksViewResponse,
    SearchInteractionRequest,
    SearchInteractionsViewResponse,
    SearchProfilesViewResponse,
    SearchUserPlaybookRequest,
    SearchUserPlaybooksViewResponse,
    SearchUserProfileRequest,
    SessionView,
    SetConfigResponse,
    UnifiedSearchRequest,
    UnifiedSearchViewResponse,
    UpdateAgentPlaybookRequest,
    UpdateAgentPlaybookResponse,
    UpdatePlaybookStatusRequest,
    UpdatePlaybookStatusResponse,
    UpdateUserPlaybookRequest,
    UpdateUserPlaybookResponse,
    UpdateUserProfileRequest,
    UpdateUserProfileResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AddAgentPlaybookRequest,
    AddAgentPlaybookResponse,
    AddUserPlaybookRequest,
    AddUserPlaybookResponse,
    AddUserProfileRequest,
    AddUserProfileResponse,
    BulkDeleteResponse,
    CancelOperationRequest,
    CancelOperationResponse,
    DeleteAgentPlaybookRequest,
    DeleteAgentPlaybookResponse,
    DeleteAgentPlaybooksByIdsRequest,
    DeleteProfilesByIdsRequest,
    DeleteRequestRequest,
    DeleteRequestResponse,
    DeleteRequestsByIdsRequest,
    DeleteSessionRequest,
    DeleteSessionResponse,
    DeleteUserInteractionRequest,
    DeleteUserInteractionResponse,
    DeleteUserPlaybookRequest,
    DeleteUserPlaybookResponse,
    DeleteUserPlaybooksByIdsRequest,
    DeleteUserProfileRequest,
    DeleteUserProfileResponse,
    DowngradeProfilesRequest,
    DowngradeProfilesResponse,
    DowngradeUserPlaybooksRequest,
    DowngradeUserPlaybooksResponse,
    GetOperationStatusRequest,
    GetOperationStatusResponse,
    ManualPlaybookGenerationRequest,
    ManualPlaybookGenerationResponse,
    ManualProfileGenerationRequest,
    ManualProfileGenerationResponse,
    MyConfigResponse,
    PlaybookAggregationChangeLogResponse,
    PublishUserInteractionRequest,
    PublishUserInteractionResponse,
    RerunPlaybookGenerationRequest,
    RerunPlaybookGenerationResponse,
    RerunProfileGenerationRequest,
    RerunProfileGenerationResponse,
    RunPlaybookAggregationRequest,
    RunPlaybookAggregationResponse,
    Status,
    UpgradeProfilesRequest,
    UpgradeProfilesResponse,
    UpgradeUserPlaybooksRequest,
    UpgradeUserPlaybooksResponse,
    WhoamiResponse,
)
from reflexio.models.api_schema.ui.converters import (
    to_agent_playbook_view,
    to_evaluation_result_view,
    to_interaction_view,
    to_profile_change_log_view,
    to_profile_view,
    to_user_playbook_view,
)
from reflexio.models.config_schema import Config
from reflexio.server.api_endpoints import account_api, publisher_api, retriever_api
from reflexio.server.cache.reflexio_cache import (
    get_reflexio,
    invalidate_reflexio_cache,
)
from reflexio.server.correlation import correlation_id_var, generate_correlation_id

logger = logging.getLogger(__name__)

# Bot protection configuration
REQUEST_TIMEOUT_SECONDS = 60
SYNC_REQUEST_TIMEOUT_SECONDS = (
    600  # Longer timeout for synchronous processing (wait_for_response=true)
)
SUSPICIOUS_USER_AGENTS = ["bot", "crawler", "spider", "scraper", "curl", "wget"]
ALLOWED_EMPTY_UA_PATHS = ["/health", "/"]  # Paths that allow empty user agents


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on IP address.

    Args:
        request (Request): The incoming request

    Returns:
        str: Rate limit key (IP address)
    """
    return get_remote_address(request)


# Initialize rate limiter
limiter = Limiter(key_func=get_rate_limit_key)


def configure_rate_limiter(key_func: Callable[..., str]) -> None:
    """
    Replace the rate limiter's key function.

    This is the supported way to override the default IP-based key function
    (e.g. with an org-scoped or token-scoped variant in the enterprise layer).

    Args:
        key_func: A callable that accepts a Request and returns a string key.
    """
    limiter._key_func = key_func  # type: ignore[reportAttributeAccessIssue]


class BotProtectionMiddleware(BaseHTTPMiddleware):
    """Middleware to detect and block suspicious bot-like requests."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and block suspicious patterns.

        Args:
            request (Request): The incoming request
            call_next (RequestResponseEndpoint): Next middleware/handler in chain

        Returns:
            Response: The response from the next handler or a 403 JSON response
        """
        from starlette.responses import JSONResponse

        user_agent = request.headers.get("user-agent", "").lower()
        path = request.url.path

        # Allow health check and root without user agent
        if path not in ALLOWED_EMPTY_UA_PATHS:
            # Block requests with no user agent
            if not user_agent:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Forbidden: Missing user agent"},
                )

            # Block requests with suspicious user agents
            for suspicious in SUSPICIOUS_USER_AGENTS:
                if suspicious in user_agent:
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"detail": "Forbidden: Suspicious user agent"},
                    )

        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeout."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with timeout enforcement.

        Args:
            request (Request): The incoming request
            call_next (RequestResponseEndpoint): Next middleware/handler in chain

        Returns:
            Response: The response from the next handler or a 504 JSON response
        """
        from starlette.responses import JSONResponse

        # Use longer timeout for synchronous processing requests
        timeout = REQUEST_TIMEOUT_SECONDS
        if request.query_params.get("wait_for_response", "").lower() == "true":
            timeout = SYNC_REQUEST_TIMEOUT_SECONDS

        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": "Request timeout"},
            )


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique correlation ID to each request.

    The ID is stored in a ContextVar so it propagates to log records
    (via CorrelationIdFilter) and to ThreadPoolExecutor workers when
    ``contextvars.copy_context()`` is used.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        cid = generate_correlation_id()
        correlation_id_var.set(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


DEFAULT_ORG_ID = "self-host-org"

core_router = APIRouter()


def default_get_org_id() -> str:
    """Return the default organization ID for local hosting."""
    return DEFAULT_ORG_ID


@core_router.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Reflexio API",
        "docs": "/docs",
        "health": "/health",
    }


@core_router.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint for ECS/container orchestration."""
    return {"status": "healthy"}


@core_router.get(
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


@core_router.get(
    "/api/my_config",
    response_model=MyConfigResponse,
    response_model_exclude_none=True,
)
def my_config_endpoint(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> MyConfigResponse:
    """Return raw storage credentials for the caller's org.

    Enablement is controlled by two independent opt-ins so the endpoint
    is closed by default on unauthenticated self-host deployments:

    - ``request.app.state.my_config_enabled`` — set to True by
      :func:`create_app` when the host wires in a Bearer-auth
      ``get_org_id`` dependency, so enterprise callers are always
      authenticated before they reach this route.
    - ``REFLEXIO_ALLOW_MY_CONFIG=true`` — OS self-host escape hatch.

    If neither is set we return a closed response instead of a 404 so
    the CLI can display an actionable hint.
    """
    app_state_enabled = bool(getattr(request.app.state, "my_config_enabled", False))
    if not (app_state_enabled or account_api.my_config_allowed()):
        return MyConfigResponse(
            success=False,
            message=(
                "GET /api/my_config is disabled. Set "
                "REFLEXIO_ALLOW_MY_CONFIG=true to enable."
            ),
        )
    return account_api.my_config(org_id=org_id)


@core_router.post(
    "/api/publish_interaction",
    response_model=PublishUserInteractionResponse,
    response_model_exclude_none=True,
)
@limiter.limit("60/minute")  # Rate limit for write operations
def publish_user_interaction(
    request: Request,
    payload: PublishUserInteractionRequest,
    background_tasks: BackgroundTasks,
    org_id: str = Depends(default_get_org_id),
    wait_for_response: bool = False,
) -> PublishUserInteractionResponse:
    if wait_for_response:
        # Process synchronously so the caller gets the real result
        return publisher_api.add_user_interaction(org_id=org_id, request=payload)
    # Run in background — caller gets immediate acknowledgement
    background_tasks.add_task(
        publisher_api.add_user_interaction, org_id=org_id, request=payload
    )
    return PublishUserInteractionResponse(
        success=True, message="Interaction queued for processing"
    )


@core_router.post(
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


@core_router.post(
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


@core_router.post(
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


@core_router.post(
    "/api/search_profiles",
    response_model=SearchProfilesViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")  # Rate limit for read operations
def search_profiles(
    request: Request,
    payload: SearchUserProfileRequest,
    org_id: str = Depends(default_get_org_id),
) -> SearchProfilesViewResponse:
    response = retriever_api.search_user_profiles(org_id=org_id, request=payload)
    return SearchProfilesViewResponse(
        success=response.success,
        user_profiles=[to_profile_view(p) for p in response.user_profiles],
        msg=response.msg,
    )


@core_router.post(
    "/api/search_interactions",
    response_model=SearchInteractionsViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")  # Rate limit for read operations
def search_interactions(
    request: Request,
    payload: SearchInteractionRequest,
    org_id: str = Depends(default_get_org_id),
) -> SearchInteractionsViewResponse:
    response = retriever_api.search_interactions(org_id=org_id, request=payload)
    return SearchInteractionsViewResponse(
        success=response.success,
        interactions=[to_interaction_view(i) for i in response.interactions],
        msg=response.msg,
    )


@core_router.post(
    "/api/search_user_playbooks",
    response_model=SearchUserPlaybooksViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")  # Rate limit for read operations
def search_user_playbooks_endpoint(
    request: Request,
    payload: SearchUserPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> SearchUserPlaybooksViewResponse:
    """Search user playbooks with semantic search and advanced filtering.

    Supports filtering by user_id (via request_id linkage), agent_version,
    playbook_name, datetime range, and status.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (SearchUserPlaybookRequest): The search request
        org_id (str): Organization ID

    Returns:
        SearchUserPlaybooksViewResponse: Response containing matching user playbooks
    """
    response = retriever_api.search_user_playbooks(org_id=org_id, request=payload)
    return SearchUserPlaybooksViewResponse(
        success=response.success,
        user_playbooks=[to_user_playbook_view(rf) for rf in response.user_playbooks],
        msg=response.msg,
    )


@core_router.post(
    "/api/search_agent_playbooks",
    response_model=SearchAgentPlaybooksViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")  # Rate limit for read operations
def search_agent_playbooks_endpoint(
    request: Request,
    payload: SearchAgentPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> SearchAgentPlaybooksViewResponse:
    """Search agent playbooks with semantic search and advanced filtering.

    Supports filtering by agent_version, playbook_name, datetime range,
    status_filter, and playbook_status_filter.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (SearchAgentPlaybookRequest): The search request
        org_id (str): Organization ID

    Returns:
        SearchAgentPlaybooksViewResponse: Response containing matching agent playbooks
    """
    response = retriever_api.search_agent_playbooks(org_id=org_id, request=payload)
    return SearchAgentPlaybooksViewResponse(
        success=response.success,
        agent_playbooks=[to_agent_playbook_view(fb) for fb in response.agent_playbooks],
        msg=response.msg,
    )


@core_router.post(
    "/api/search",
    response_model=UnifiedSearchViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")
def unified_search_endpoint(
    request: Request,
    payload: UnifiedSearchRequest,
    org_id: str = Depends(default_get_org_id),
) -> UnifiedSearchViewResponse:
    """Search across all entity types (profiles, agent playbooks, user playbooks).

    Runs query rewriting and embedding generation in parallel, then searches
    all entity types in parallel. Query rewriting is gated behind the
    enable_reformulation request param.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (UnifiedSearchRequest): The unified search request
        org_id (str): Organization ID

    Returns:
        UnifiedSearchViewResponse: Combined search results
    """
    response = retriever_api.unified_search(org_id=org_id, request=payload)
    return UnifiedSearchViewResponse(
        success=response.success,
        profiles=[to_profile_view(p) for p in response.profiles],
        agent_playbooks=[to_agent_playbook_view(fb) for fb in response.agent_playbooks],
        user_playbooks=[to_user_playbook_view(rf) for rf in response.user_playbooks],
        reformulated_query=response.reformulated_query,
        msg=response.msg,
    )


@core_router.get("/api/profile_change_log", response_model=ProfileChangeLogViewResponse)
def get_profile_change_log(
    org_id: str = Depends(default_get_org_id),
) -> ProfileChangeLogViewResponse:
    response = retriever_api.get_profile_change_logs(org_id=org_id)
    return ProfileChangeLogViewResponse(
        success=response.success,
        profile_change_logs=[
            to_profile_change_log_view(log) for log in response.profile_change_logs
        ],
    )


@core_router.get(
    "/api/playbook_aggregation_change_logs",
    response_model=PlaybookAggregationChangeLogResponse,
)
def get_playbook_aggregation_change_logs(
    playbook_name: str,
    agent_version: str,
    org_id: str = Depends(default_get_org_id),
) -> PlaybookAggregationChangeLogResponse:
    return retriever_api.get_playbook_aggregation_change_logs(
        org_id=org_id,
        playbook_name=playbook_name,
        agent_version=agent_version,
    )


@core_router.delete(
    "/api/delete_profile",
    response_model=DeleteUserProfileResponse,
    response_model_exclude_none=True,
)
def delete_profile(
    request: DeleteUserProfileRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteUserProfileResponse:
    return publisher_api.delete_user_profile(org_id=org_id, request=request)


@core_router.delete(
    "/api/delete_interaction",
    response_model=DeleteUserInteractionResponse,
    response_model_exclude_none=True,
)
def delete_interaction(
    request: DeleteUserInteractionRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteUserInteractionResponse:
    return publisher_api.delete_user_interaction(org_id=org_id, request=request)


@core_router.delete(
    "/api/delete_request",
    response_model=DeleteRequestResponse,
    response_model_exclude_none=True,
)
def delete_request(
    request: DeleteRequestRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteRequestResponse:
    return publisher_api.delete_request(org_id=org_id, request=request)


@core_router.delete(
    "/api/delete_session",
    response_model=DeleteSessionResponse,
    response_model_exclude_none=True,
)
def delete_session(
    request: DeleteSessionRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteSessionResponse:
    return publisher_api.delete_session(org_id=org_id, request=request)


@core_router.delete(
    "/api/delete_agent_playbook",
    response_model=DeleteAgentPlaybookResponse,
    response_model_exclude_none=True,
)
def delete_agent_playbook(
    request: DeleteAgentPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteAgentPlaybookResponse:
    return publisher_api.delete_agent_playbook(org_id=org_id, request=request)


@core_router.delete(
    "/api/delete_user_playbook",
    response_model=DeleteUserPlaybookResponse,
    response_model_exclude_none=True,
)
def delete_user_playbook(
    request: DeleteUserPlaybookRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteUserPlaybookResponse:
    return publisher_api.delete_user_playbook(org_id=org_id, request=request)


@core_router.delete(
    "/api/delete_requests_by_ids",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_requests_by_ids(
    request: DeleteRequestsByIdsRequest,
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete multiple requests by their IDs.

    Args:
        request (DeleteRequestsByIdsRequest): Request containing list of request IDs to delete
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_requests_by_ids(org_id=org_id, request=request)


@core_router.delete(
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


@core_router.delete(
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


@core_router.delete(
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


@core_router.delete(
    "/api/delete_all_interactions",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_all_interactions(
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all requests and their associated interactions.

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_interactions_bulk(org_id=org_id)


@core_router.delete(
    "/api/delete_all_profiles",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_all_profiles(
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all profiles.

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_profiles_bulk(org_id=org_id)


@core_router.delete(
    "/api/delete_all_playbooks",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_all_playbooks(
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all playbooks (both user and agent).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_playbooks_bulk(org_id=org_id)


@core_router.delete(
    "/api/delete_all_user_playbooks",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_all_user_playbooks(
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all user playbooks (user only, not agent).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_user_playbooks_bulk(org_id=org_id)


@core_router.delete(
    "/api/delete_all_agent_playbooks",
    response_model=BulkDeleteResponse,
    response_model_exclude_none=True,
)
def delete_all_agent_playbooks(
    org_id: str = Depends(default_get_org_id),
) -> BulkDeleteResponse:
    """Delete all agent playbooks (agent only, not user).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    return publisher_api.delete_all_agent_playbooks_bulk(org_id=org_id)


@core_router.post(
    "/api/get_interactions",
    response_model=GetInteractionsViewResponse,
    response_model_exclude_none=True,
)
def get_interactions(
    request: GetInteractionsRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetInteractionsViewResponse:
    response = retriever_api.get_user_interactions(org_id=org_id, request=request)
    return GetInteractionsViewResponse(
        success=response.success,
        interactions=[to_interaction_view(i) for i in response.interactions],
        msg=response.msg,
    )


@core_router.get(
    "/api/get_all_interactions",
    response_model=GetInteractionsViewResponse,
    response_model_exclude_none=True,
)
def get_all_interactions(
    limit: int = 100,
    org_id: str = Depends(default_get_org_id),
) -> GetInteractionsViewResponse:
    """Get all user interactions across all users.

    Args:
        limit (int, optional): Maximum number of interactions to return. Defaults to 100.
        org_id (str): Organization ID

    Returns:
        GetInteractionsViewResponse: Response containing all user interactions
    """
    reflexio = get_reflexio(org_id=org_id)
    response = reflexio.get_all_interactions(limit=limit)
    return GetInteractionsViewResponse(
        success=response.success,
        interactions=[to_interaction_view(i) for i in response.interactions],
        msg=response.msg,
    )


@core_router.post(
    "/api/get_requests",
    response_model=GetRequestsViewResponse,
    response_model_exclude_none=True,
)
def get_requests_endpoint(
    request: GetRequestsRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetRequestsViewResponse:
    """Get requests with their associated interactions.

    Args:
        request (GetRequestsRequest): The get request
        org_id (str): Organization ID

    Returns:
        GetRequestsViewResponse: Response containing requests with their interactions
    """
    internal_response = retriever_api.get_requests(org_id=org_id, request=request)
    return GetRequestsViewResponse(
        success=internal_response.success,
        sessions=[
            SessionView(
                session_id=s.session_id,
                requests=[
                    RequestDataView(
                        request=rd.request,
                        interactions=[to_interaction_view(i) for i in rd.interactions],
                    )
                    for rd in s.requests
                ],
            )
            for s in internal_response.sessions
        ],
        has_more=internal_response.has_more,
        msg=internal_response.msg,
    )


@core_router.post(
    "/api/get_profiles",
    response_model=GetProfilesViewResponse,
    response_model_exclude_none=True,
)
def get_profiles(
    request: GetUserProfilesRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetProfilesViewResponse:
    response = retriever_api.get_user_profiles(org_id=org_id, request=request)
    return GetProfilesViewResponse(
        success=response.success,
        user_profiles=[to_profile_view(p) for p in response.user_profiles],
        msg=response.msg,
    )


@core_router.get(
    "/api/get_all_profiles",
    response_model=GetProfilesViewResponse,
    response_model_exclude_none=True,
)
def get_all_profiles(
    limit: int = 100,
    status_filter: str | None = None,
    org_id: str = Depends(default_get_org_id),
) -> GetProfilesViewResponse:
    """Get all user profiles across all users.

    Args:
        limit (int, optional): Maximum number of profiles to return. Defaults to 100.
        status_filter (str, optional): Filter by profile status. Can be "current", "pending", or "archived".
        org_id (str): Organization ID

    Returns:
        GetProfilesViewResponse: Response containing all user profiles
    """
    reflexio = get_reflexio(org_id=org_id)

    # Map status_filter string to Status list
    status_filter_list = None
    if status_filter == "current":
        status_filter_list = [None]
    elif status_filter == "pending":
        status_filter_list = [Status.PENDING]
    elif status_filter == "archived":
        status_filter_list = [Status.ARCHIVED]

    response = reflexio.get_all_profiles(limit=limit, status_filter=status_filter_list)  # type: ignore[reportArgumentType]
    return GetProfilesViewResponse(
        success=response.success,
        user_profiles=[to_profile_view(p) for p in response.user_profiles],
        msg=response.msg,
    )


@core_router.get(
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
    reflexio = get_reflexio(org_id=org_id)

    # Get profile statistics using Reflexio's method
    return reflexio.get_profile_statistics()


@core_router.post(
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
    return publisher_api.run_playbook_aggregation(org_id=org_id, request=payload)


@core_router.post("/api/set_config")
def set_config(
    config: Config,
    org_id: str = Depends(default_get_org_id),
) -> SetConfigResponse:
    """Set configuration for the organization.

    Args:
        config (Config): The configuration to set
        org_id (str): Organization ID

    Returns:
        dict: Response containing success status and message
    """
    # Create Reflexio instance to access the configurator through request_context
    reflexio = get_reflexio(org_id=org_id)

    # Set the config using Reflexio's set_config method
    response = reflexio.set_config(config)

    # Invalidate cache on successful config change to ensure fresh instance next request
    if response.success:
        invalidate_reflexio_cache(org_id=org_id)

    return response


@core_router.get("/api/get_config", response_model=Config)
def get_config(
    org_id: str = Depends(default_get_org_id),
) -> Config:
    """Get configuration for the organization.

    Args:
        org_id (str): Organization ID

    Returns:
        Config: The current configuration
    """
    # Create Reflexio instance to access the configurator through request_context
    reflexio = get_reflexio(org_id=org_id)

    # Get the config using Reflexio's get_config method
    return reflexio.get_config()


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)
    response = reflexio.get_user_playbooks(request)
    return GetUserPlaybooksViewResponse(
        success=response.success,
        user_playbooks=[to_user_playbook_view(rf) for rf in response.user_playbooks],
        msg=response.msg,
    )


@core_router.post(
    "/api/get_agent_playbooks",
    response_model=GetAgentPlaybooksViewResponse,
    response_model_exclude_none=True,
)
def get_agent_playbooks(
    request: GetAgentPlaybooksRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetAgentPlaybooksViewResponse:
    """Get agent playbooks with internal fields filtered out.

    Args:
        request (GetAgentPlaybooksRequest): The get request
        org_id (str): Organization ID

    Returns:
        GetAgentPlaybooksViewResponse: Response containing agent playbooks without internal fields
    """
    reflexio = get_reflexio(org_id=org_id)
    response = reflexio.get_agent_playbooks(request)
    return GetAgentPlaybooksViewResponse(
        success=response.success,
        agent_playbooks=[to_agent_playbook_view(fb) for fb in response.agent_playbooks],
        msg=response.msg,
    )


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)
    response = reflexio.get_agent_success_evaluation_results(request)
    return GetEvaluationResultsViewResponse(
        success=response.success,
        agent_success_evaluation_results=[
            to_evaluation_result_view(r)
            for r in response.agent_success_evaluation_results
        ],
        msg=response.msg,
    )


@core_router.put(
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


@core_router.put(
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


@core_router.put(
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


@core_router.put(
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


@core_router.post(
    "/api/get_dashboard_stats",
    response_model=GetDashboardStatsResponse,
    response_model_exclude_none=True,
)
def get_dashboard_stats(
    request: GetDashboardStatsRequest,
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
    reflexio = get_reflexio(org_id=org_id)

    # Get dashboard stats using Reflexio's method
    return reflexio.get_dashboard_stats(request)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)

    # Run the long-running task in the background to avoid proxy timeout
    # Client polls get_operation_status for progress
    background_tasks.add_task(reflexio.rerun_profile_generation, payload)

    return RerunProfileGenerationResponse(
        success=True, msg="Profile generation started"
    )


@core_router.post(
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

    This behaves like regular generation (uses batch_size from config,
    outputs CURRENT profiles) but only runs profile extraction.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (ManualProfileGenerationRequest): Request containing user_id, source, and extractor_names
        org_id (str): Organization ID

    Returns:
        ManualProfileGenerationResponse: Response containing success status and profiles generated count
    """
    # Create Reflexio instance
    reflexio = get_reflexio(org_id=org_id)

    # Call manual_profile_generation
    return reflexio.manual_profile_generation(payload)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)

    # Run the long-running task in the background to avoid proxy timeout
    # Client polls get_operation_status for progress
    background_tasks.add_task(reflexio.rerun_playbook_generation, payload)

    return RerunPlaybookGenerationResponse(
        success=True, msg="Playbook generation started"
    )


@core_router.post(
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

    This behaves like regular generation (uses batch_size from config,
    outputs CURRENT playbooks) but only runs playbook extraction.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (ManualPlaybookGenerationRequest): Request containing agent_version, source, and playbook_name
        org_id (str): Organization ID

    Returns:
        ManualPlaybookGenerationResponse: Response containing success status and playbooks generated count
    """
    # Create Reflexio instance
    reflexio = get_reflexio(org_id=org_id)

    # Call manual_playbook_generation
    return reflexio.manual_playbook_generation(payload)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)

    # Call upgrade_all_profiles with request
    return reflexio.upgrade_all_profiles(request=request)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)

    # Call downgrade_all_profiles with request
    return reflexio.downgrade_all_profiles(request=request)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)

    # Call upgrade_all_user_playbooks with request
    return reflexio.upgrade_all_user_playbooks(request=request)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)

    # Call downgrade_all_user_playbooks with request
    return reflexio.downgrade_all_user_playbooks(request=request)


@core_router.get(
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
    reflexio = get_reflexio(org_id=org_id)

    # Get operation status
    request = GetOperationStatusRequest(service_name=service_name)
    return reflexio.get_operation_status(request)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.cancel_operation(payload)


# Paths that should remain publicly accessible (no lock icon in Swagger)
_PUBLIC_PATHS = frozenset(
    {"/", "/health", "/meta/version", "/token", "/docs", "/openapi.json"}
)
_PUBLIC_PATH_PREFIXES = ("/api/register", "/api/registration-config", "/api/auth/")


def _add_openapi_security(app: FastAPI) -> None:
    """Inject Bearer auth security scheme into the OpenAPI spec.

    Overrides the default openapi() method to add a global HTTPBearer security
    requirement while exempting public endpoints (login, register, health, etc.).
    """
    original_openapi = app.openapi

    def custom_openapi() -> dict:  # type: ignore[type-arg]
        if app.openapi_schema:
            return app.openapi_schema

        schema = original_openapi()

        # Add security scheme
        schema.setdefault("components", {}).setdefault("securitySchemes", {})
        schema["components"]["securitySchemes"]["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "description": "API key or JWT token. Pass as: Authorization: Bearer <token>",
        }

        # Apply security globally, then remove from public endpoints
        for path, methods in schema.get("paths", {}).items():
            is_public = path in _PUBLIC_PATHS or any(
                path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES
            )
            for method_detail in methods.values():
                if isinstance(method_detail, dict):
                    if is_public:
                        method_detail["security"] = []
                    else:
                        method_detail.setdefault("security", [{"BearerAuth": []}])

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


def create_app(
    get_org_id: Callable[..., str] | None = None,
    additional_routers: list[APIRouter] | None = None,
    middleware_config: dict | None = None,
    require_auth: bool = False,
) -> FastAPI:
    """Factory to create a FastAPI app.

    Args:
        get_org_id: Custom dependency for resolving org_id (e.g., from JWT auth).
            When provided, overrides the default_get_org_id dependency globally.
        additional_routers: Extra APIRouter instances (e.g., enterprise login/oauth).
        middleware_config: Optional middleware overrides (not used yet, reserved for future).
        require_auth: When True, declares a Bearer security scheme in the OpenAPI spec
            so Swagger UI shows lock icons and the Authorize button works.

    Returns:
        Configured FastAPI application.
    """
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from reflexio.server.llm.model_defaults import validate_llm_availability

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
        validate_llm_availability()
        yield

    app = FastAPI(docs_url="/docs", lifespan=lifespan)

    if require_auth:
        _add_openapi_security(app)

    @app.get("/meta/version")
    def get_version_info() -> dict[str, str]:
        from importlib.metadata import PackageNotFoundError, version

        try:
            server_version = version("reflexio")
        except PackageNotFoundError:
            server_version = "0.0.0-dev"
        return {
            "server_version": server_version,
            "api_version": "v1",
            "min_client_version": "0.1.0",
        }

    # Configure rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[reportArgumentType]

    # CORS
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Timeout middleware
    app.add_middleware(TimeoutMiddleware)

    # Bot protection
    app.add_middleware(BotProtectionMiddleware)

    # Correlation ID — added last so it runs outermost (Starlette reverses order)
    app.add_middleware(CorrelationIdMiddleware)

    # Override get_org_id dependency if custom one provided
    if get_org_id is not None:
        app.dependency_overrides[default_get_org_id] = get_org_id

    # When a custom get_org_id is provided together with require_auth,
    # auth is enforced on every route — mark this app instance so the
    # token-gated my_config endpoint becomes reachable. Using
    # ``app.state`` instead of a module-level global keeps the gate
    # scoped to this FastAPI instance, so multiple apps (e.g. tests,
    # multi-tenant embeddings) can coexist without leaking state.
    app.state.my_config_enabled = bool(get_org_id is not None and require_auth)

    # Include core routes
    app.include_router(core_router)

    # Include additional routers
    for router in additional_routers or []:
        app.include_router(router)

    return app


# Default standalone app (no auth)
app = create_app()
