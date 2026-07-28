"""Interaction route handlers (extracted from api.py, Tier3 A2)."""

import logging
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
)

from reflexio.models.api_schema.common import sanitise_for_log
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
    LearningStatusResponse,
    PublishUserInteractionRequest,
    PublishUserInteractionResponse,
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
from reflexio.server.routes._common import _run_limited_api

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
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
    # Anything the request validation quietly altered: unrecognised keys that
    # were stripped, and empty interactions that were dropped. Returned to the
    # caller AND recorded server-side, because a mis-keyed field is otherwise
    # silent data loss. Names only, bounded, control characters stripped -- the
    # values are caller payload and the names are equally caller-controlled.
    #
    # INFO, not WARNING: an empty placeholder turn is normal plugin behaviour,
    # and warning on the routine case trains operators to ignore the channel.
    payload_warnings = payload.payload_warnings()
    if payload_warnings:
        logger.info(
            "Publish for org %s altered the payload: %s",
            org_id,
            "; ".join(payload_warnings),
        )

    if wait_for_response:
        # Sync callers wait for the real result, so preserve bounded backpressure
        # before any storage side effects. The inner service limiter is disabled
        # because this route already owns the publish slot.
        sync_response = _run_limited_api(
            org_id,
            "publish",
            lambda: publisher_api.add_user_interaction(
                org_id=org_id,
                request=payload,
                use_publish_limiter=False,
            ),
        )
        # Appended, not assigned: `warnings` already carries extraction-stall
        # warnings from the generation service, which the CLI renders.
        sync_response.warnings = [*sync_response.warnings, *payload_warnings]
        return sync_response

    # Resolve the request_id BEFORE backgrounding so we can return it to the
    # caller for polling. GenerationService uses a caller-supplied request_id
    # verbatim (and generates one only when absent), so pinning it on the
    # payload guarantees the background task stores its status under the same
    # id we hand back here.
    request_id = payload.request_id or str(uuid.uuid4())
    payload.request_id = request_id
    # request_id is caller-supplied (NonEmptyStr: no length cap, no character
    # restrictions) and is logged below at ERROR, which Sentry ingests as an
    # event body. A newline in it would forge a line in a shared multi-tenant
    # log stream -- the same hazard as the unknown field names.
    safe_request_id = sanitise_for_log(request_id)

    def _publish_task() -> None:
        try:
            response = publisher_api.add_user_interaction(
                org_id=org_id,
                request=payload,
                defer_learning=True,
            )
            # The caller was already handed 200 "queued" below, so a
            # ``success=False`` return here is otherwise invisible -- it is not
            # an exception, so the handler underneath never sees it. Log it, or
            # a rejected publish looks identical to an accepted one from both
            # ends. Per-interaction shape problems are already caught as a 422
            # by PublishUserInteractionRequest's validators; this covers
            # whatever else add_user_interaction can refuse.
            #
            # Deliberately NOT logging ``response.message``: on the storage
            # path it is ``str(e)`` from a catch-all (lib/_interactions.py), an
            # unbounded exception string with no content-freeness guarantee,
            # and Sentry ingests ERROR records as event bodies without
            # scrubbing them. #377 established that such messages must be
            # content-free by construction, so log a bounded shape instead.
            if not response.success:
                logger.error(
                    "Background publish rejected for org %s (request_id=%s):"
                    " %d-char reason withheld (may contain Customer Content)",
                    org_id,
                    safe_request_id,
                    len(response.message or ""),
                )
        except Exception as exc:  # noqa: BLE001 - a background task must never raise past add_task
            # Same content-freeness problem as response.message above: an
            # exception string (and its traceback locals) has no guarantee of
            # being free of Customer Content, and Sentry ingests this as an
            # event body. Log the type and the raising location -- file:line is
            # content-free and is the only way to find a background failure --
            # but never the message.
            tb = exc.__traceback__
            while tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
            origin = (
                f"{tb.tb_frame.f_code.co_filename}:{tb.tb_lineno}"
                if tb is not None
                else "unknown"
            )
            logger.error(
                "Background publish failed for org %s (request_id=%s): %s at %s"
                " (message withheld, may contain Customer Content)",
                org_id,
                safe_request_id,
                type(exc).__name__,
                origin,
            )

    # Run in background — caller gets immediate acknowledgement.
    # learning_status="deferred" tells the caller that extraction has not yet
    # run; they can poll GET /api/learning_status?request_id=... (using the
    # request_id returned here) to track it.
    background_tasks.add_task(_publish_task)
    return PublishUserInteractionResponse(
        success=True,
        message="Interaction queued for processing",
        request_id=request_id,
        learning_status="deferred",
        warnings=payload_warnings,
    )


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
    """Return the coverage-based learning status for a published request.

    The status reflects whether a durable learning job has processed through
    the request's creation timestamp:

    - ``pending``: not yet picked up.
    - ``processing``: a worker currently holds the job.
    - ``done``: at least one completed job covers this request.
    - ``failed``: a dead job covers this request and no done job does.

    Note: this endpoint reads ``learning_jobs`` rows written by the durable
    queue. When the durable queue is OFF (in-memory deferred path) it returns
    absence-based status (``pending`` for recent requests, ``done`` for old
    ones) — acceptable for v1; the poll contract is tied to the durable queue.

    Raises:
        HTTPException: 404 when ``request_id`` is not found for this org.
            Never reports ``done`` for a request that never existed.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage not configured")
    req = storage.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="request not found")
    status = storage.get_learning_status_for_request(
        user_id=req.user_id,
        request_created_at=float(req.created_at),
    )
    return LearningStatusResponse(status=status)
