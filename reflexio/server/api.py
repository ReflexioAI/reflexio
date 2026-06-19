import asyncio
import inspect
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from anyio.to_thread import current_default_thread_limiter
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from reflexio.models.api_schema.braintrust_schema import (
    BraintrustStatusResponse,
    ConnectBraintrustRequest,
    ConnectBraintrustResponse,
    SelectProjectsRequest,
    SelectProjectsResponse,
    SyncBraintrustResponse,
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
    GetAgentPlaybooksRequest,
    GetAgentPlaybooksViewResponse,
    GetAgentSuccessEvaluationResultsRequest,
    GetDashboardStatsRequest,
    GetDashboardStatsResponse,
    GetEvaluationResultsViewResponse,
    GetInteractionsRequest,
    GetInteractionsViewResponse,
    GetPlaybookApplicationStatsRequest,
    GetPlaybookApplicationStatsResponse,
    GetProfileStatisticsResponse,
    GetProfilesViewResponse,
    GetRequestsRequest,
    GetRequestsViewResponse,
    GetUserPlaybooksRequest,
    GetUserPlaybooksViewResponse,
    GetUserProfilesRequest,
    ProfileChangeLogViewResponse,
    RequestDataView,
    RerankUserProfilesRequest,
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
    StorageStatsRequest,
    StorageStatsResponse,
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
    AdminInvalidateCacheRequest,
    AdminInvalidateCacheResponse,
    BulkDeleteResponse,
    CancelOperationRequest,
    CancelOperationResponse,
    ClearUserDataRequest,
    ClearUserDataResponse,
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
from reflexio.models.config_schema import (
    SINGLETON_AGENT_SUCCESS_EVALUATION_NAME,
    Config,
)
from reflexio.server._auth import (
    DEFAULT_ORG_ID,
    default_billing_gate,
    default_get_caller_type,
    default_get_org_id,
)
from reflexio.server.api_endpoints import (
    account_api,
    health_api,
    pending_tool_call_api,
    publisher_api,
    stall_state_api,
)
from reflexio.server.cache.reflexio_cache import (
    get_reflexio,
    invalidate_reflexio_cache,
)
from reflexio.server.correlation import correlation_id_var, generate_correlation_id
from reflexio.server.operation_limiter import (
    OperationName,
    limiter_http_exception,
    run_with_operation_limit,
)
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)
from reflexio.server.services.agent_success_evaluation.regen_jobs import (
    REGEN_JOBS,
    run_regen,
)
from reflexio.server.tracing import profile_step

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility — callers that did
# ``from reflexio.server.api import default_get_org_id`` or ``DEFAULT_ORG_ID``
# continue to work.
__all__ = [
    "DEFAULT_ORG_ID",
    "create_app",
    "default_billing_gate",
    "default_get_caller_type",
    "default_get_org_id",
]

# Bot protection configuration
REQUEST_TIMEOUT_SECONDS = 60
SYNC_REQUEST_TIMEOUT_SECONDS = (
    600  # Longer timeout for synchronous processing (wait_for_response=true)
)
SUSPICIOUS_USER_AGENTS = ["bot", "crawler", "spider", "scraper", "curl", "wget"]
ALLOWED_EMPTY_UA_PATHS = ["/health", "/"]  # Paths that allow empty user agents
DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024
REGENERATE_MAX_WORKERS = 2
_regen_executor = ThreadPoolExecutor(
    max_workers=REGENERATE_MAX_WORKERS,
    thread_name_prefix="reflexio-regen",
)


def _resolve_cors_origins() -> list[str]:
    """Resolve browser origins allowed to make credentialed CORS requests."""
    configured_origins = os.getenv("REFLEXIO_ALLOWED_ORIGINS", "").strip()
    if configured_origins:
        origins = [
            origin.strip().rstrip("/")
            for origin in configured_origins.split(",")
            if origin.strip()
        ]
        return origins or ["http://localhost:8080"]

    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url:
        return [frontend_url.rstrip("/")]

    return ["http://localhost:8080"]


def _max_body_bytes_from_env() -> int:
    raw_value = os.getenv("REFLEXIO_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))
    try:
        max_bytes = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid REFLEXIO_MAX_BODY_BYTES=%r; using %s",
            raw_value,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    if max_bytes <= 0:
        logger.warning(
            "Ignoring non-positive REFLEXIO_MAX_BODY_BYTES=%r; using %s",
            raw_value,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    return max_bytes


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on IP address.

    Args:
        request (Request): The incoming request

    Returns:
        str: Rate limit key (IP address)
    """
    return get_remote_address(request)


def _storage_backend_name(limiter_obj: Limiter) -> str:
    storage = getattr(limiter_obj, "_storage", None)
    if storage is None:
        return "unknown"
    return storage.__class__.__name__


def _trace_external_rate_limit_backend(limiter_obj: Limiter) -> None:
    """Trace rate-limit storage hits when the backend is an external service."""
    backend = _storage_backend_name(limiter_obj)
    if backend == "MemoryStorage":
        return

    strategy = getattr(limiter_obj, "limiter", None)
    if strategy is None or getattr(strategy, "_reflexio_traced", False):
        return

    original_hit = strategy.hit

    def traced_hit(item: Any, *identifiers: str, cost: int = 1) -> bool:
        with profile_step(
            "rate_limit.backend_hit",
            storage_backend=backend,
            strategy=strategy.__class__.__name__,
            cost=cost,
        ):
            return original_hit(item, *identifiers, cost=cost)

    strategy.hit = traced_hit
    strategy._reflexio_traced = True


# Initialize rate limiter
limiter = Limiter(key_func=get_rate_limit_key)
_trace_external_rate_limit_backend(limiter)


def _run_limited_api[T](
    org_id: str,
    operation: OperationName,
    fn: Callable[[], T],
) -> T:
    try:
        return run_with_operation_limit(
            org_id=org_id,
            operation=operation,
            fn=fn,
        )
    except TimeoutError as exc:
        http_exc = limiter_http_exception(operation)
        raise http_exc from exc


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


class _RequestBodyTooLargeError(Exception):
    """Raised when the streamed request body exceeds the configured limit."""


class BodySizeLimitMiddleware:
    """Reject requests whose declared or streamed body size exceeds the limit."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from starlette.responses import JSONResponse

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_body_bytes = _max_body_bytes_from_env()
        content_length = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                content_length = value.decode("latin-1")
                break

        if content_length is not None:
            try:
                body_bytes = int(content_length)
            except ValueError:
                body_bytes = 0
            if body_bytes > max_body_bytes:
                await JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": "Request body too large"},
                )(scope, receive, send)
                return

        consumed_bytes = 0

        async def limited_receive() -> Message:
            nonlocal consumed_bytes
            message = await receive()
            if message["type"] == "http.request":
                consumed_bytes += len(message.get("body", b""))
                if consumed_bytes > max_body_bytes:
                    raise _RequestBodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLargeError:
            await JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"detail": "Request body too large"},
            )(scope, receive, send)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach conservative browser security headers to every response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https"
        ):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


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
        try:
            stats = current_default_thread_limiter().statistics()
            request.state.tp_borrowed = stats.borrowed_tokens
            request.state.tp_total = stats.total_tokens
            request.state.tp_waiting = stats.tasks_waiting
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to snapshot threadpool limiter stats: %s", exc)
            request.state.tp_borrowed = None
            request.state.tp_total = None
            request.state.tp_waiting = None
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


core_router = APIRouter()


def _stamp_search_dependencies_done(request: Request) -> None:
    """Stamp when dependency resolution reaches its final search dependency."""
    request.state.search_deps_done_monotonic = time.monotonic()


def _meter_applied_learnings(
    *,
    org_id: str,
    caller_type: str,
    surfaced_count: int,
    request_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Emit the ③ Application event via the OSS emission helper.

    No-op unless a production-agent caller surfaced >= 1 result.  The cheap
    caller-type and count guard runs first so the get_reflexio / config lookup
    is skipped on the free paths (dashboard / empty result).

    Args:
        org_id: Organization ID for the requesting caller.
        caller_type: Resolved caller classification (e.g. ``"production_agent"``).
        surfaced_count: Total number of learnings returned to the caller.
        request_id: Optional request correlation ID from the payload.
        session_id: Optional session ID from the payload.
    """
    if caller_type != "production_agent" or surfaced_count <= 0:
        return
    try:
        from reflexio.server.billing_meter import record_applied_learnings
        from reflexio.server.billing_signals import platform_llm_from_config

        config = get_reflexio(org_id=org_id).request_context.configurator.get_config()
        record_applied_learnings(
            org_id=org_id,
            surfaced_count=surfaced_count,
            caller_type=caller_type,
            platform_llm=platform_llm_from_config(config),
            platform_storage=None,  # resolved enterprise-side at rollup (Phase 1)
            request_id=request_id,
            session_id=session_id,
        )
    except Exception:
        logger.warning(
            "applied-learnings metering failed for org %s", org_id, exc_info=True
        )


@core_router.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Reflexio API",
        "docs": "/docs",
        "health": "/health",
    }


@core_router.get("/health")
async def health_check() -> dict[str, str]:
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
    _gate: None = Depends(default_billing_gate("learnings_generated")),  # noqa: B008
) -> PublishUserInteractionResponse:
    if wait_for_response:
        # Process synchronously so the caller gets the real result
        return _run_limited_api(
            org_id,
            "publish",
            lambda: publisher_api.add_user_interaction(org_id=org_id, request=payload),
        )

    def _limited_publish_task() -> None:
        try:
            run_with_operation_limit(
                org_id=org_id,
                operation="publish",
                fn=lambda: publisher_api.add_user_interaction(
                    org_id=org_id, request=payload
                ),
                wait_forever=False,
            )
        except TimeoutError:
            logger.warning(
                "Dropped queued publish for org %s because publish limiter is saturated",
                org_id,
            )

    # Run in background — caller gets immediate acknowledgement
    background_tasks.add_task(_limited_publish_task)
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
def search_user_profiles(
    request: Request,
    payload: SearchUserProfileRequest,
    org_id: str = Depends(default_get_org_id),
    caller_type: str = Depends(default_get_caller_type),
    _gate: None = Depends(default_billing_gate("application")),  # noqa: B008
) -> SearchProfilesViewResponse:
    response = _run_limited_api(
        org_id,
        "search",
        lambda: get_reflexio(org_id=org_id).search_user_profiles(payload),
    )
    resp = SearchProfilesViewResponse(
        success=response.success,
        user_profiles=[to_profile_view(p) for p in response.user_profiles],
        msg=response.msg,
    )
    _meter_applied_learnings(
        org_id=org_id,
        caller_type=caller_type,
        surfaced_count=len(resp.user_profiles),
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    return resp


@core_router.post(
    "/api/rerank_user_profiles",
    response_model=SearchProfilesViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")  # Rate limit for read operations
def rerank_user_profiles(
    request: Request,
    payload: RerankUserProfilesRequest,
    org_id: str = Depends(default_get_org_id),
) -> SearchProfilesViewResponse:
    """Rerank a list of profile ids by query relevance using a cross-encoder.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (RerankUserProfilesRequest): The rerank request
        org_id (str): Organization ID

    Returns:
        SearchProfilesViewResponse: Reranked profiles, top_k entries.
    """
    response = _run_limited_api(
        org_id,
        "search",
        lambda: get_reflexio(org_id=org_id).rerank_user_profiles(payload),
    )
    return SearchProfilesViewResponse(
        success=response.success,
        user_profiles=[to_profile_view(p) for p in response.user_profiles],
        msg=response.msg,
    )


@core_router.get(
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
        lambda: get_reflexio(org_id=org_id).storage_stats(
            StorageStatsRequest(user_id=user_id)
        ),
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
    response = _run_limited_api(
        org_id,
        "search",
        lambda: get_reflexio(org_id=org_id).search_interactions(payload),
    )
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
    caller_type: str = Depends(default_get_caller_type),
    _gate: None = Depends(default_billing_gate("application")),  # noqa: B008
) -> SearchUserPlaybooksViewResponse:
    """Search user playbooks with semantic search and advanced filtering.

    Supports filtering by user_id (via request_id linkage), agent_version,
    playbook_name, datetime range, and status.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (SearchUserPlaybookRequest): The search request
        org_id (str): Organization ID
        caller_type (str): Billing caller classification (injected via dependency).

    Returns:
        SearchUserPlaybooksViewResponse: Response containing matching user playbooks
    """
    response = _run_limited_api(
        org_id,
        "search",
        lambda: get_reflexio(org_id=org_id).search_user_playbooks(payload),
    )
    resp = SearchUserPlaybooksViewResponse(
        success=response.success,
        user_playbooks=[to_user_playbook_view(rf) for rf in response.user_playbooks],
        msg=response.msg,
    )
    _meter_applied_learnings(
        org_id=org_id,
        caller_type=caller_type,
        surfaced_count=len(resp.user_playbooks),
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    return resp


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
    caller_type: str = Depends(default_get_caller_type),
    _gate: None = Depends(default_billing_gate("application")),  # noqa: B008
) -> SearchAgentPlaybooksViewResponse:
    """Search agent playbooks with semantic search and advanced filtering.

    Supports filtering by agent_version, playbook_name, datetime range,
    status_filter, and playbook_status_filter.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (SearchAgentPlaybookRequest): The search request
        org_id (str): Organization ID
        caller_type (str): Billing caller classification (injected via dependency).

    Returns:
        SearchAgentPlaybooksViewResponse: Response containing matching agent playbooks
    """
    response = _run_limited_api(
        org_id,
        "search",
        lambda: get_reflexio(org_id=org_id).search_agent_playbooks(payload),
    )
    resp = SearchAgentPlaybooksViewResponse(
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


@core_router.post(
    "/api/search",
    response_model=UnifiedSearchViewResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")
def unified_search_endpoint(
    request: Request,
    payload: UnifiedSearchRequest,
    background_tasks: BackgroundTasks,
    org_id: str = Depends(default_get_org_id),
    caller_type: str = Depends(default_get_caller_type),
    _gate: None = Depends(default_billing_gate("application")),  # noqa: B008
    _deps_done: None = Depends(_stamp_search_dependencies_done),
) -> UnifiedSearchViewResponse:
    """Search across all entity types (profiles, agent playbooks, user playbooks).

    Runs query rewriting and embedding generation in parallel, then searches
    all entity types in parallel. Query rewriting is gated behind the
    enable_reformulation request param.

    Args:
        request (Request): The HTTP request object (for rate limiting)
        payload (UnifiedSearchRequest): The unified search request
        org_id (str): Organization ID
        caller_type (str): Billing caller classification (injected via dependency).

    Returns:
        UnifiedSearchViewResponse: Combined search results
    """
    deps_done = getattr(request.state, "search_deps_done_monotonic", None)
    deps_to_body_ms = (
        int((time.monotonic() - deps_done) * 1000) if deps_done is not None else None
    )
    with profile_step(
        "search.endpoint",
        enabled=bool(payload.enable_reformulation),
        has_conversation_history=bool(payload.conversation_history),
        search_mode=payload.search_mode,
    ) as endpoint_span:
        endpoint_span.set_data("deps_to_body_ms", deps_to_body_ms)
        endpoint_span.set_data(
            "tp_borrowed", getattr(request.state, "tp_borrowed", None)
        )
        endpoint_span.set_data("tp_total", getattr(request.state, "tp_total", None))
        endpoint_span.set_data("tp_waiting", getattr(request.state, "tp_waiting", None))

        def run_search() -> Any:
            with profile_step("search.reflexio_cache"):
                reflexio = get_reflexio(org_id=org_id)
            return reflexio.unified_search(payload, org_id=org_id)

        response = _run_limited_api(org_id, "search", run_search)
        with profile_step("search.response_view"):
            resp = UnifiedSearchViewResponse(
                success=response.success,
                profiles=[to_profile_view(p) for p in response.profiles],
                agent_playbooks=[
                    to_agent_playbook_view(fb) for fb in response.agent_playbooks
                ],
                user_playbooks=[
                    to_user_playbook_view(rf) for rf in response.user_playbooks
                ],
                reformulated_query=response.reformulated_query,
                msg=response.msg,
                agent_trace=response.agent_trace,
                rehydrated_text=response.rehydrated_text,
            )
        background_tasks.add_task(
            _meter_applied_learnings,
            org_id=org_id,
            caller_type=caller_type,
            surfaced_count=len(resp.profiles)
            + len(resp.agent_playbooks)
            + len(resp.user_playbooks),
            request_id=getattr(payload, "request_id", None),
            session_id=getattr(payload, "session_id", None),
        )
    return resp


@core_router.get("/api/profile_change_log", response_model=ProfileChangeLogViewResponse)
def get_profile_change_log(
    org_id: str = Depends(default_get_org_id),
) -> ProfileChangeLogViewResponse:
    response = get_reflexio(org_id=org_id).get_profile_change_logs()
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
    return get_reflexio(org_id=org_id).get_playbook_aggregation_change_logs(
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
@limiter.limit("10/minute")
def delete_all_interactions(
    request: Request,
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


@core_router.delete(
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


@core_router.delete(
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


@core_router.delete(
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


@core_router.post(
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

    Removes the user's interactions, user playbooks, profiles, and
    requests. Does NOT touch ``agent_playbooks`` — they are
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


@core_router.post(
    "/api/get_interactions",
    response_model=GetInteractionsViewResponse,
    response_model_exclude_none=True,
)
def get_interactions(
    request: GetInteractionsRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetInteractionsViewResponse:
    response = get_reflexio(org_id=org_id).get_interactions(request)
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
@limiter.limit("30/minute")
def get_all_interactions(
    request: Request,
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
    internal_response = get_reflexio(org_id=org_id).get_requests(request)
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
    response = get_reflexio(org_id=org_id).get_profiles(request)
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
@limiter.limit("30/minute")
def get_all_profiles(
    request: Request,
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
    return _run_limited_api(
        org_id,
        "aggregation",
        lambda: publisher_api.run_playbook_aggregation(org_id=org_id, request=payload),
    )


@core_router.post("/api/set_config")
@limiter.limit("10/minute")
def set_config(
    request: Request,
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


@core_router.post("/api/update_config")
@limiter.limit("10/minute")
def update_config(
    request: Request,
    partial: dict[str, Any],
    org_id: str = Depends(default_get_org_id),
) -> SetConfigResponse:
    """Apply a partial update to the org's config (PATCH semantics).

    Performs a **top-level shallow merge** of *partial* over the existing
    config and round-trips through ``Config(**merged)`` so Pydantic
    validates the result and rejects bogus top-level fields.

    .. warning::
       Nested objects (e.g. ``storage_config``, ``profile_extractor_config``,
       ``user_playbook_extractor_config``) are **replaced wholesale**.
       Deep merging is intentionally not supported -- the discriminator on
       ``storage_config`` would be lost on partial updates, and merging nested
       dicts has ambiguous semantics.

       To update a field inside an extractor config you must resend that
       extractor object fully populated (including ``extractor_name``,
       ``extraction_definition_prompt``, etc.). For one-off mutations prefer
       ``GET /api/get_config`` followed by ``POST /api/set_config`` with the
       modified full config.

    Unlike :func:`set_config`, callers do not need to re-send the full
    config (including required fields like ``storage_config``) just to
    flip a single top-level boolean. The merge happens server-side
    atomically within the request, eliminating the read-modify-write
    race a client would otherwise hit.

    Args:
        partial: Top-level fields to overlay on the existing config.
        org_id: Organization ID resolved by the auth layer.

    Returns:
        SetConfigResponse: Success status and message from
        :meth:`Reflexio.set_config`.
    """
    from pydantic import ValidationError

    reflexio = get_reflexio(org_id=org_id)
    existing = reflexio.request_context.configurator.get_config().model_dump(
        mode="python"
    )
    merged = {**existing, **partial}
    # Pydantic validates the merged shape and rejects unknown / malformed
    # fields here, before storage validation in reflexio.set_config.
    # Convert ValidationError into 422 so callers passing a partial that
    # would replace a nested extractor object with an incomplete dict (e.g.
    # {"user_playbook_extractor_config": {"aggregation_config": {...}}})
    # get a clean client-error response instead of a 500.
    try:
        merged_config = Config(**merged)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "Invalid partial config (top-level shallow merge)",
                "hint": (
                    "Nested objects (e.g. user_playbook_extractor_config) "
                    "are replaced wholesale, not deep-merged. To mutate a "
                    "single nested field, fetch the full config via "
                    "/api/get_config, edit, and POST it back via "
                    "/api/set_config."
                ),
                "validation_errors": exc.errors(),
            },
        ) from exc
    response = reflexio.set_config(merged_config)
    if response.success:
        invalidate_reflexio_cache(org_id=org_id)
    return response


@core_router.post("/api/admin/cache/invalidate")
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
    invalidated = invalidate_reflexio_cache(org_id=org_id)
    return AdminInvalidateCacheResponse(invalidated=invalidated, org_id=org_id)


@core_router.get("/api/get_config")
def get_config(
    org_id: str = Depends(default_get_org_id),
) -> dict[str, Any]:
    """Get configuration for the organization.

    Args:
        org_id (str): Organization ID

    Returns:
        Config: The current configuration
    """
    # Create Reflexio instance to access the configurator through request_context
    reflexio = get_reflexio(org_id=org_id)

    return reflexio.request_context.configurator.get_config_for_response()


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
    reflexio = get_reflexio(org_id=org_id)
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
    reflexio = get_reflexio(org_id=org_id)

    # Get dashboard stats using Reflexio's method
    return reflexio.get_dashboard_stats(payload)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.get_playbook_application_stats(request)


# ============================================================================
# Braintrust connector (Plan C-backend)
# ============================================================================


@core_router.post(
    "/api/braintrust/connect",
    response_model=ConnectBraintrustResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def braintrust_connect(
    request: Request,
    payload: ConnectBraintrustRequest,
    org_id: str = Depends(default_get_org_id),
) -> ConnectBraintrustResponse:
    """Step 1: validate the Braintrust API key and list workspaces/projects.

    Persists nothing — call `/api/braintrust/select_projects` to commit.

    Args:
        request (Request): The HTTP request object for rate limiting.
        payload (ConnectBraintrustRequest): Customer's Braintrust API key.
        org_id (str): Resolved by auth dependency.

    Returns:
        ConnectBraintrustResponse: Workspaces tree on success; `success=False`
            with a message when the key is rejected.
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.braintrust_connect(payload)


@core_router.post(
    "/api/braintrust/select_projects",
    response_model=SelectProjectsResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def braintrust_select_projects(
    request: Request,
    payload: SelectProjectsRequest,
    org_id: str = Depends(default_get_org_id),
) -> SelectProjectsResponse:
    """Step 2: commit the Braintrust connection with selected projects.

    The API key is encrypted at rest. Subsequent syncs use the persisted
    connection until the customer calls DELETE /api/braintrust/connection.
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.braintrust_select_projects(payload)


@core_router.get(
    "/api/braintrust/status",
    response_model=BraintrustStatusResponse,
    response_model_exclude_none=True,
)
def braintrust_status(
    org_id: str = Depends(default_get_org_id),
) -> BraintrustStatusResponse:
    """Return Braintrust connection state. Never echoes the API key."""
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.braintrust_status()


@core_router.delete("/api/braintrust/connection")
@limiter.limit("10/minute")
def braintrust_disconnect(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> dict:
    """Delete the persisted Braintrust connection for the org.

    Args:
        org_id (str): Resolved by auth dependency.

    Returns:
        dict: ``{"success": True}`` on completion.
    """
    reflexio = get_reflexio(org_id=org_id)
    reflexio.braintrust_disconnect()
    return {"success": True}


@core_router.post(
    "/api/braintrust/sync",
    response_model=SyncBraintrustResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def braintrust_sync(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> SyncBraintrustResponse:
    """Trigger a one-shot sync of Braintrust scorer outputs.

    Scheduled (cron) sync is a follow-up; for now the endpoint exists so
    operators can drive a manual import.
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.braintrust_sync()


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.get_evaluation_overview(request)


# ---------------------------------------------------------------------------
# /api/evaluations/regenerate — replay the LLM judge over a window
# ---------------------------------------------------------------------------


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)
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


@core_router.get(
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


@core_router.delete("/api/evaluations/regenerate/{job_id}")
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


# ---------------------------------------------------------------------------
# /api/evaluations/grade_on_demand — single-session click-through grading
# ---------------------------------------------------------------------------
#
# F3 sampling means most sessions in a regen window are NEVER graded —
# the sampler keeps cost predictable by capping each (day, group) stratum
# at ``Config.eval_sample_n_per_stratum``. Plan 3 (F1) surfaces a
# bounded list of sessions in the UI; when a customer clicks one that
# wasn't in the sampled subset, the frontend hits this endpoint to grade
# it on demand. The 24h cache prevents repeated clicks from triggering
# redundant LLM calls.

_GRADE_ON_DEMAND_CACHE_TTL_SECONDS = 24 * 60 * 60
_GRADE_ON_DEMAND_CACHE_KEY_PREFIX = "grade_on_demand"


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
    return (
        f"{_GRADE_ON_DEMAND_CACHE_KEY_PREFIX}::{org_id}::{session_id}"
        f"::{agent_version}::{evaluation_name}"
    )


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
        session_id (str): The graded session.
        agent_version (str): The version dimension.
        evaluation_name (str): Evaluator/result namespace to isolate readback.
        previous_result_ids (set[int]): Matching rows observed before grading.

    Returns:
        int | None: result_id of the latest matching row, or None if the
        runner wrote nothing.
    """
    result_ids = storage.get_agent_success_evaluation_result_ids(
        session_id=session_id,
        evaluation_name=evaluation_name,
        agent_version=agent_version,
    )
    fresh_result_ids = [rid for rid in result_ids if rid not in previous_result_ids]
    if not fresh_result_ids:
        return None
    return max(fresh_result_ids)


@core_router.post(
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
    reflexio = get_reflexio(org_id=org_id)
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
            session_id=payload.session_id,
            evaluation_name=evaluation_name,
            agent_version=payload.agent_version,
        )
    )

    # Two operation_state rows are intentionally written for this session:
    #   1) `grade_on_demand::{org_id}::{session_id}::{agent_version}::{evaluation_name}`
    #      — our 24h cache, set below after the result_id is resolved.
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


# ---------------------------------------------------------------------------
# /api/evaluations/shadow_comparisons/recent — F1 drawer + Top 10 widget
# ---------------------------------------------------------------------------
#
# Powers two surfaces on /evaluations:
#   1. The drawer triggered from the per-turn comparison tile — shows the
#      N most recent verdicts so customers can spot-check the judge.
#   2. The "Top 10 disagreements" widget — fetches a wider pool and the
#      frontend filters to ``is_significantly_better=True`` losses to surface
#      actionable rule-correction candidates.
#
# Filtering is restricted to the org's currently pinned
# ``shadow_comparison_judge_prompt_version`` so verdicts from an older rubric
# never mix into the drawer. The 30-day lookback is a defensive cap that lets
# the storage layer use an index range scan instead of a full table read.

_RECENT_SHADOW_COMPARISONS_LOOKBACK_SECONDS = 30 * 24 * 60 * 60
_RECENT_SHADOW_COMPARISONS_MAX_LIMIT = 100


@core_router.get(
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
    reflexio = get_reflexio(org_id=org_id)
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
    get_caller_type: Callable[..., str] | None = None,
    get_billing_gate: Callable[[str], Callable[..., None]] | None = None,
    mount_data_plane: bool = True,
) -> FastAPI:
    """Factory to create a FastAPI app.

    Args:
        get_org_id: Custom dependency for resolving org_id (e.g., from JWT auth).
            When provided, overrides the default_get_org_id dependency globally.
        additional_routers: Extra APIRouter instances (e.g., enterprise login/oauth).
        middleware_config: Optional middleware overrides (not used yet, reserved for future).
        require_auth: When True, declares a Bearer security scheme in the OpenAPI spec
            so Swagger UI shows lock icons and the Authorize button works.
        get_caller_type: Custom dependency for classifying the caller (e.g., production
            agent vs dashboard).  When provided, overrides the default_get_caller_type
            dependency globally, exactly mirroring the get_org_id override.
        get_billing_gate: Optional factory ``(line: str) -> FastAPI dependency`` that
            replaces the default no-op gate for each billable billing line.  When
            provided, for every line used in the app (``"application"`` and
            ``"learnings_generated"``) the returned dependency overrides the
            ``default_billing_gate(line)`` sentinel in ``dependency_overrides``,
            exactly mirroring the ``get_caller_type`` override pattern.
        mount_data_plane: When True (default), include the data-plane routers
            (core, stall-state, pending-tool-call) and run the data-plane
            lifespan work (LLM availability check, cross-encoder prewarm,
            resume scheduler). When False, skip both so a control-plane host
            can build an app without requiring LLM/storage or starting the
            scheduler, while keeping all other scaffolding (middleware, CORS,
            auth overrides, OpenAPI security, health, ``/meta/version``,
            ``additional_routers``).

    Returns:
        Configured FastAPI application.
    """
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager

    from reflexio.server._auth import (
        default_billing_gate,
        default_get_caller_type,
        default_get_org_id,
    )
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.model_defaults import validate_llm_availability
    from reflexio.server.services.extraction.resume_scheduler import (
        maybe_start_resume_scheduler,
    )

    def _lifespan_org_id() -> str:
        if get_org_id is None:
            return default_get_org_id()
        try:
            signature = inspect.signature(get_org_id)
        except (TypeError, ValueError):
            return default_get_org_id()
        if signature.parameters:
            return default_get_org_id()
        try:
            return str(get_org_id())
        except Exception:
            logger.exception("Failed to resolve lifespan org_id; using default org")
            return default_get_org_id()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
        scheduler = None
        if mount_data_plane:
            validate_llm_availability()
            from reflexio.server.llm.rerank import prewarm as _prewarm_cross_encoder

            _prewarm_cross_encoder()
            # The scheduler discovers every org with resumable work each tick and
            # drives a per-org worker with org-scoped claims, so it is not limited
            # to the bootstrap org. The bootstrap org is only used to read config
            # and to seed cross-org discovery.
            scheduler = maybe_start_resume_scheduler(
                lambda org_id: RequestContext(org_id=org_id),
                bootstrap_org_id=_lifespan_org_id(),
            )
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.stop()

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
    # The locked-down, credentialed allowlist is an enterprise concern: only
    # hosts that wire in auth (``require_auth=True``) restrict browser origins.
    # OSS/local runs have no auth and bundle their own docs playground on a
    # separate port, so they allow any origin (no credentials needed).
    if require_auth:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_resolve_cors_origins(),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Reject oversized requests before they reach endpoint handlers.
    app.add_middleware(BodySizeLimitMiddleware)

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Timeout middleware
    app.add_middleware(TimeoutMiddleware)

    # Bot protection
    app.add_middleware(BotProtectionMiddleware)

    # Correlation ID — added last so it runs outermost (Starlette reverses order)
    app.add_middleware(CorrelationIdMiddleware)

    # Override get_org_id dependency if custom one provided
    if get_org_id is not None:
        app.dependency_overrides[default_get_org_id] = get_org_id

    # Override get_caller_type dependency if custom one provided
    if get_caller_type is not None:
        app.dependency_overrides[default_get_caller_type] = get_caller_type

    # Override billing gate dependencies if a custom gate factory is provided.
    # Each billing line needs its own override because dependency_overrides is
    # keyed by callable identity.  ``default_billing_gate`` uses lru_cache so
    # the same sentinel object is returned for the same line — which is why
    # the overrides reliably fire at request time.
    if get_billing_gate is not None:
        for line in ("application", "learnings_generated"):
            app.dependency_overrides[default_billing_gate(line)] = get_billing_gate(
                line
            )

    # When a custom get_org_id is provided together with require_auth,
    # auth is enforced on every route — mark this app instance so the
    # token-gated my_config endpoint becomes reachable. Using
    # ``app.state`` instead of a module-level global keeps the gate
    # scoped to this FastAPI instance, so multiple apps (e.g. tests,
    # multi-tenant embeddings) can coexist without leaking state.
    app.state.my_config_enabled = bool(get_org_id is not None and require_auth)

    # Include data-plane routes (core, stall-state, pending-tool-call). A
    # control-plane host sets mount_data_plane=False to skip these while
    # keeping every other piece of scaffolding below.
    if mount_data_plane:
        # Include core routes
        app.include_router(core_router)

        # Include stall_state routes
        app.include_router(stall_state_api.router)

        # Include pending tool call routes
        app.include_router(pending_tool_call_api.router)

    # Include additional routers
    for router in additional_routers or []:
        app.include_router(router)

    # Health/observability endpoint (per-worker metrics for recycling)
    health_api.install(app)

    return app


# Default standalone app (no auth)
app = create_app()
