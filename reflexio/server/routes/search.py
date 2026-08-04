"""Search/retrieval route handlers (extracted from api.py, Tier3 A2)."""

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    status,
)

from reflexio.models.api_schema.retriever_schema import (
    RerankUserProfilesRequest,
    RetrievalExperimentAssignment,
    SearchAgentPlaybookRequest,
    SearchAgentPlaybooksViewResponse,
    SearchInteractionRequest,
    SearchInteractionsViewResponse,
    SearchProfilesViewResponse,
    SearchUserPlaybookRequest,
    SearchUserPlaybooksViewResponse,
    SearchUserProfileRequest,
    UnifiedSearchRequest,
    UnifiedSearchViewResponse,
)
from reflexio.models.api_schema.ui.converters import (
    to_agent_playbook_view,
    to_interaction_view,
    to_profile_view,
    to_user_playbook_view,
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
    _meter_search_request,
    _stamp_search_dependencies_done,
)
from reflexio.server.services.retrieval_experiment import (
    active_retrieval_experiment_assignment,
)
from reflexio.server.tracing import profile_step

logger = logging.getLogger(__name__)
router = APIRouter()


def _retrieval_experiment_assignment(
    *, org_id: str, caller_type: str, user_id: str | None
) -> RetrievalExperimentAssignment | None:
    """Resolve agent traffic assignment while leaving dashboard inspection untouched."""
    if caller_type == "dashboard":
        return None
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    config = reflexio.request_context.configurator.get_config()
    if config.retrieval_experiment_config is None:
        return None
    if user_id is None or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="user_id is required while a retrieval experiment is active",
        )
    return active_retrieval_experiment_assignment(
        config=config,
        org_id=org_id,
        user_id=user_id,
    )


@router.post(
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
    assignment = _retrieval_experiment_assignment(
        org_id=org_id, caller_type=caller_type, user_id=payload.user_id
    )
    if assignment is not None and assignment.arm == "holdout":
        resp = SearchProfilesViewResponse(
            success=True,
            user_profiles=[],
            msg="Retrieval withheld by experiment assignment",
            experiment=assignment,
        )
    else:
        response = _run_limited_api(
            org_id,
            "search",
            lambda: reflexio_cache.get_reflexio(org_id=org_id).search_user_profiles(
                payload
            ),
        )
        resp = SearchProfilesViewResponse(
            success=response.success,
            user_profiles=[to_profile_view(p) for p in response.user_profiles],
            msg=response.msg,
            experiment=assignment,
        )
    _meter_search_request(
        org_id=org_id,
        caller_type=caller_type,
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    _meter_applied_learnings(
        org_id=org_id,
        caller_type=caller_type,
        surfaced_count=len(resp.user_profiles),
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    return resp


@router.post(
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
        lambda: reflexio_cache.get_reflexio(org_id=org_id).rerank_user_profiles(
            payload
        ),
    )
    return SearchProfilesViewResponse(
        success=response.success,
        user_profiles=[to_profile_view(p) for p in response.user_profiles],
        msg=response.msg,
    )


@router.post(
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
        lambda: reflexio_cache.get_reflexio(org_id=org_id).search_interactions(payload),
    )
    return SearchInteractionsViewResponse(
        success=response.success,
        interactions=[to_interaction_view(i) for i in response.interactions],
        msg=response.msg,
    )


@router.post(
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
    assignment = _retrieval_experiment_assignment(
        org_id=org_id, caller_type=caller_type, user_id=payload.user_id
    )
    if assignment is not None and assignment.arm == "holdout":
        resp = SearchUserPlaybooksViewResponse(
            success=True,
            user_playbooks=[],
            msg="Retrieval withheld by experiment assignment",
            experiment=assignment,
        )
    else:
        response = _run_limited_api(
            org_id,
            "search",
            lambda: reflexio_cache.get_reflexio(org_id=org_id).search_user_playbooks(
                payload
            ),
        )
        resp = SearchUserPlaybooksViewResponse(
            success=response.success,
            user_playbooks=[
                to_user_playbook_view(rf) for rf in response.user_playbooks
            ],
            msg=response.msg,
            experiment=assignment,
        )
    _meter_search_request(
        org_id=org_id,
        caller_type=caller_type,
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    _meter_applied_learnings(
        org_id=org_id,
        caller_type=caller_type,
        surfaced_count=len(resp.user_playbooks),
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    return resp


@router.post(
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
    assignment = _retrieval_experiment_assignment(
        org_id=org_id, caller_type=caller_type, user_id=payload.user_id
    )
    if assignment is not None and assignment.arm == "holdout":
        resp = SearchAgentPlaybooksViewResponse(
            success=True,
            agent_playbooks=[],
            msg="Retrieval withheld by experiment assignment",
            experiment=assignment,
        )
    else:
        response = _run_limited_api(
            org_id,
            "search",
            lambda: reflexio_cache.get_reflexio(org_id=org_id).search_agent_playbooks(
                payload
            ),
        )
        resp = SearchAgentPlaybooksViewResponse(
            success=response.success,
            agent_playbooks=[
                to_agent_playbook_view(fb) for fb in response.agent_playbooks
            ],
            msg=response.msg,
            experiment=assignment,
        )
    _meter_search_request(
        org_id=org_id,
        caller_type=caller_type,
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    _meter_applied_learnings(
        org_id=org_id,
        caller_type=caller_type,
        surfaced_count=len(resp.agent_playbooks),
        request_id=getattr(payload, "request_id", None),
        session_id=getattr(payload, "session_id", None),
    )
    return resp


@router.post(
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
    assignment = _retrieval_experiment_assignment(
        org_id=org_id, caller_type=caller_type, user_id=payload.user_id
    )
    if assignment is not None and assignment.arm == "holdout":
        resp = UnifiedSearchViewResponse(
            success=True,
            profiles=[],
            agent_playbooks=[],
            user_playbooks=[],
            msg="Retrieval withheld by experiment assignment",
            experiment=assignment,
        )
        background_tasks.add_task(
            _meter_search_request,
            org_id=org_id,
            caller_type=caller_type,
            request_id=payload.request_id,
            session_id=payload.session_id,
        )
        return resp

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
                reflexio = reflexio_cache.get_reflexio(org_id=org_id)
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
                experiment=assignment,
            )
        background_tasks.add_task(
            _meter_search_request,
            org_id=org_id,
            caller_type=caller_type,
            request_id=getattr(payload, "request_id", None),
            session_id=getattr(payload, "session_id", None),
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
