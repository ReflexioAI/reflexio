"""Interaction route handlers (extracted from api.py, Tier3 A2)."""

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from reflexio.models.api_schema.retriever_schema import (
    GetInteractionsRequest,
    GetInteractionsViewResponse,
    GetRequestsRequest,
    GetRequestsViewResponse,
    RequestDataView,
    SessionView,
)
from reflexio.models.api_schema.service_schemas import (
    BulkDeleteResponse,
    DeleteRequestRequest,
    DeleteRequestResponse,
    DeleteRequestsByIdsRequest,
    DeleteSessionRequest,
    DeleteSessionResponse,
    DeleteUserInteractionRequest,
    DeleteUserInteractionResponse,
    GetSessionOutcomesRequest,
    GetSessionOutcomesResponse,
    LearningStatusResponse,
    PublishUserInteractionRequest,
    PublishUserInteractionResponse,
    SetSessionOutcomeRequest,
    SetSessionOutcomeResponse,
)
from reflexio.models.api_schema.ui.converters import (
    to_interaction_view,
)
from reflexio.server.api_endpoints import (
    publisher_api,
)
from reflexio.server.auth import (
    default_billing_gate,
    default_get_org_id,
)
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter

logger = logging.getLogger(__name__)
router = APIRouter()
PUBLISH_REQUEST_TIMEOUT_SECONDS = 240.0


@router.post(
    "/api/session_outcome",
    response_model=SetSessionOutcomeResponse,
    response_model_exclude_none=True,
)
@limiter.limit("60/minute")
def set_session_outcome(
    request: Request,
    payload: SetSessionOutcomeRequest,
    org_id: str = Depends(default_get_org_id),
) -> SetSessionOutcomeResponse:
    return publisher_api.set_session_outcome(org_id=org_id, request=payload)


@router.post(
    "/api/get_session_outcomes",
    response_model=GetSessionOutcomesResponse,
    response_model_exclude_none=True,
)
@limiter.limit("60/minute")
def get_session_outcomes(
    request: Request,
    payload: GetSessionOutcomesRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetSessionOutcomesResponse:
    return publisher_api.get_session_outcomes(org_id=org_id, request=payload)


@router.post(
    "/api/publish_interaction",
    response_model=PublishUserInteractionResponse,
    response_model_exclude_none=True,
)
@limiter.limit("60/minute")  # Rate limit for write operations
async def publish_user_interaction(
    request: Request,
    payload: PublishUserInteractionRequest,
    org_id: str = Depends(default_get_org_id),
    wait_for_response: bool = False,
    _gate: None = Depends(default_billing_gate("learnings_generated")),  # noqa: B008
) -> PublishUserInteractionResponse:
    from reflexio.server.services.durable_learning.waiting import (
        acquire_ingestion,
        acquire_waiter,
        admission_deadline,
        coverage_stalled,
        release_ingestion,
        release_waiter,
    )

    deadline = time.monotonic() + PUBLISH_REQUEST_TIMEOUT_SECONDS
    payload.request_id = payload.request_id or str(uuid.uuid4())
    if not await acquire_ingestion(org_id, deadline):
        raise HTTPException(
            status_code=503, detail="Publish capacity deadline exceeded"
        )
    token = admission_deadline.set(deadline)
    try:
        # Success is returned only after the atomic admission transaction.
        # Cancellation must not release the slot while ingestion still runs.
        operation = asyncio.create_task(
            asyncio.to_thread(
                publisher_api.add_user_interaction,
                org_id=org_id,
                request=payload,
                use_publish_limiter=False,
                defer_learning=True,
            )
        )
        try:
            response = await asyncio.wait_for(
                asyncio.shield(operation), timeout=max(0, deadline - time.monotonic())
            )
        except TimeoutError:
            operation.add_done_callback(lambda _: release_ingestion(org_id))
            raise HTTPException(
                status_code=504,
                detail={
                    "reason": "admission_timeout",
                    "request_id": payload.request_id,
                    "message": "Admission was not confirmed before the deadline; check this request ID before retrying.",
                },
            ) from None
        except asyncio.CancelledError:
            operation.add_done_callback(lambda _: release_ingestion(org_id))
            raise
        except Exception:
            release_ingestion(org_id)
            raise
        else:
            release_ingestion(org_id)
    finally:
        admission_deadline.reset(token)
    response.warnings = [*response.warnings, *payload.payload_warnings()]
    if not response.success or not wait_for_response:
        return response
    if not acquire_waiter(org_id):
        response.learning_status = "deferred"
        response.learning_reason = "waiter_capacity"
        return response
    try:
        storage = reflexio_cache.get_reflexio(org_id=org_id).get_storage()
        stalled_reason: str | None = None
        while time.monotonic() < deadline:
            status = await asyncio.to_thread(
                storage.extraction_status, payload.user_id, payload.request_id
            )
            if status["status"] == "done":
                counts = await asyncio.to_thread(
                    storage.extraction_counts, payload.user_id, payload.request_id
                )
                response.learning_status = "done"
                response.learning_reason = status["reason"]
                response.profiles_added = counts["profile"]
                response.playbooks_added = counts["playbook"]
                return response
            # Holding the connection open for a window that only new input can
            # close wastes the caller's deadline and a waiter slot, and reports
            # `wait_timeout` for a healthy stream. Same break as the library
            # waiter in generation_service.run -- one condition, both paths.
            if coverage_stalled(status):
                stalled_reason = status["reason"]
                break
            if await request.is_disconnected():
                break
            await asyncio.sleep(min(0.25, max(0, deadline - time.monotonic())))
        response.learning_status = "deferred"
        response.learning_reason = stalled_reason or "wait_timeout"
        return response
    finally:
        release_waiter(org_id)


@router.delete(
    "/api/delete_interaction",
    response_model=DeleteUserInteractionResponse,
    response_model_exclude_none=True,
)
def delete_interaction(
    request: DeleteUserInteractionRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteUserInteractionResponse:
    return publisher_api.delete_user_interaction(org_id=org_id, request=request)


@router.delete(
    "/api/delete_request",
    response_model=DeleteRequestResponse,
    response_model_exclude_none=True,
)
def delete_request(
    request: DeleteRequestRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteRequestResponse:
    return publisher_api.delete_request(org_id=org_id, request=request)


@router.delete(
    "/api/delete_session",
    response_model=DeleteSessionResponse,
    response_model_exclude_none=True,
)
def delete_session(
    request: DeleteSessionRequest,
    org_id: str = Depends(default_get_org_id),
) -> DeleteSessionResponse:
    return publisher_api.delete_session(org_id=org_id, request=request)


@router.delete(
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


@router.delete(
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


@router.post(
    "/api/get_interactions",
    response_model=GetInteractionsViewResponse,
    response_model_exclude_none=True,
)
def get_interactions(
    request: GetInteractionsRequest,
    org_id: str = Depends(default_get_org_id),
) -> GetInteractionsViewResponse:
    response = reflexio_cache.get_reflexio(org_id=org_id).get_interactions(request)
    return GetInteractionsViewResponse(
        success=response.success,
        interactions=[to_interaction_view(i) for i in response.interactions],
        msg=response.msg,
    )


@router.get(
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
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    response = reflexio.get_all_interactions(limit=limit)
    return GetInteractionsViewResponse(
        success=response.success,
        interactions=[to_interaction_view(i) for i in response.interactions],
        msg=response.msg,
    )


@router.post(
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
    internal_response = reflexio_cache.get_reflexio(org_id=org_id).get_requests(request)
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


@router.get(
    "/api/learning_status",
    response_model=LearningStatusResponse,
    response_model_exclude_none=True,
)
@limiter.limit("120/minute")
def get_learning_status(
    request: Request,
    request_id: str,
    org_id: str = Depends(default_get_org_id),
) -> LearningStatusResponse:
    """Report each required cursor's coverage of this request's arrival range.

    Pending work may await a full window, capacity, or a retry. Historical
    requests without admission metadata are not_tracked. Unknown IDs are 404.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage not configured")
    req = storage.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="request not found")
    status = storage.extraction_status(req.user_id, request_id)
    return LearningStatusResponse(**status)
