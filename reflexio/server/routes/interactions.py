"""Interaction route handlers (extracted from api.py, Tier3 A2)."""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
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
    BackstopTimeoutResponse,
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
    PublishCapacityRefusedDetail,
    PublishCapacityRefusedResponse,
    PublishTimeoutDetail,
    PublishTimeoutResponse,
    PublishUserInteractionRequest,
    PublishUserInteractionResponse,
    SetSessionOutcomeRequest,
    SetSessionOutcomeResponse,
)
from reflexio.models.api_schema.ui.converters import (
    to_interaction_view,
)
from reflexio.server import publish_timing
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
# The RESPONSE clock: when the route stops waiting and answers 202.
PUBLISH_REQUEST_TIMEOUT_SECONDS = 240.0
# The WORKER clock is this much LATER, and the gap is the whole point.
#
# `asyncio.to_thread` copies the context, so whatever `admission_deadline`
# holds when the worker starts is the clock `check_admission_deadline()` runs
# against for the rest of the publish -- and the route cannot reach it
# afterwards, because the worker holds a COPY (`admission_deadline.reset()`
# below restores the ROUTE's context only). Handing the worker the response
# deadline therefore made the 202 false at the instant it was sent: `wait_for`
# gave up and the worker's next checkpoint aborted, so the route reported an
# in-flight publish about work that was being discarded.
#
# 30s is sized against the LARGEST gap between two consecutive
# `check_admission_deadline()` calls in `generation_service.run` -- the
# `prepare_interaction_embeddings` network call, then `lock_extraction_stream`
# inside `commit_scope`. Production p99 for an ENTIRE publish is 22.5s
# (34,944 spans, 30d), so 30s clears any single phase of it. Past the third
# checkpoint there is no further check, so a commit already under way is never
# interrupted; the grace only has to cover the run-up.
#
# Too small and the 202 goes back to being briefly true and then false. Too
# large and a wedged publish holds an anyio worker thread -- and its ingestion
# slot, which is released from the done-callback -- for longer. 30s also keeps
# the whole chain monotone and inside the middleware backstop:
# response 240s < worker 270s < ROUTE_BACKSTOP_SECONDS 300s.
PUBLISH_WORKER_GRACE_SECONDS = 30.0

# The 202 body's message. It promises exactly one thing, and that thing is
# verifiable: a retry under the SAME id is safe.
#
# It deliberately claims NEITHER outcome. `operation.done() is False` says only
# that the task has not returned -- a publish can commit and then sit in
# post-commit work (`ensure_local_extraction`, success metering, the
# `_safe_coverage` reads), and if that tail crosses the response deadline this
# same 202 is sent for work that IS committed. "Still running" and "not
# committed" are both false in that case, so neither is stated.
#
# It also stops directing the caller to `GET /api/learning_status` as the
# answer. That endpoint reads `storage.get_request`, and the request row is
# written INSIDE `commit_scope`, so it 404s for every publish that has not
# committed yet -- i.e. for the common case this message is sent in. A 404
# there means "not yet visible", never "lost", and a contract that cannot
# distinguish those is not one to hand a caller as its primary path.
#
# The retry instruction is checked, not assumed: `request_id` is a PRIMARY KEY
# in both backends (`sqlite_storage/_base.py` "request_id TEXT PRIMARY KEY";
# the Supabase tenant baseline's `requests_pkey PRIMARY KEY (request_id)`), and
# `generation_service.run` reads `get_request` twice -- once up front, once
# under `lock_extraction_stream` INSIDE `commit_scope` -- raising rather than
# writing a second row. So the retry publishes if and only if the first attempt
# did not.
PAST_DEADLINE_202_MESSAGE = (
    "Accepted; the outcome is not yet known. The server stopped waiting "
    "before this publish confirmed, so it may or may not have committed. "
    "Retry with this same request_id: the server rejects a duplicate, so the "
    "retry is safe whether or not the first attempt landed. A NEW request_id "
    "would publish it a second time."
)


def _release_and_drain(org_id: str) -> "Callable[[asyncio.Task[Any]], None]":
    """Build the done-callback for a publish the route has stopped awaiting.

    Two jobs, and the second one is not optional. It releases the ingestion
    slot, and it RETRIEVES the task's exception.

    Without the retrieve, a worker that aborts after the 202 -- the ordinary
    end of the grace window, where ``check_admission_deadline`` raises
    ``PublishDeadlineExceededError`` -- leaves an un-consumed exception on a task
    nobody awaits, and asyncio logs "Task exception was never retrieved" at
    collection time. That is a normal, already-reported outcome, not an
    unhandled error, and logging it as one trains readers to ignore the
    message. It became reachable the moment the deadline abort stopped being
    swallowed at the library boundary.
    """
    from reflexio.server.services.durable_learning.waiting import release_ingestion

    def _done(task: "asyncio.Task[Any]") -> None:
        release_ingestion(org_id)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.info(
                "Publish for org %s ended with %s after the route had already "
                "answered; the caller was told the outcome was unknown.",
                org_id,
                type(exc).__name__,
                exc_info=exc,
            )

    return _done


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
    responses={
        202: {
            "model": PublishUserInteractionResponse,
            "description": (
                "Accepted, outcome not yet known: the server's RESPONSE "
                "deadline was reached before the publish confirmed. This "
                "asserts NEITHER that the work committed nor that it did "
                "not -- the worker runs on a deliberately later deadline and "
                "is usually still going, but the same 202 is sent when the "
                "durable write has already committed and only post-commit "
                "work is outstanding. success is true and learning_reason is "
                "'server_deadline'. Retry with the SAME request_id: it is a "
                "primary key and the server rejects a duplicate, so the retry "
                "is safe either way. Never retry with a NEW request_id. "
                "GET /api/learning_status?request_id=<id> may 404 while the "
                "write is still in flight; that means 'not yet visible', not "
                "'lost'."
            ),
        },
        # Modeled, not just described: a contract a generated client cannot
        # discover from the schema is one first-party callers will not adopt,
        # and a description alone leaves reason/request_id/message -- which
        # HTTPException additionally nests under `detail` -- untyped. The route
        # constructs both bodies from these same models, so the declaration
        # cannot drift from what is emitted.
        503: {
            "model": PublishCapacityRefusedResponse,
            "description": (
                "Not admitted: publish capacity was not available before the "
                "deadline, so nothing started and nothing will commit. "
                "reason is 'capacity_deadline_exceeded'. Retry with the SAME "
                "request_id."
            ),
        },
        504: {
            # TWO producers, two shapes, so the declaration is a union. The
            # route's own 504 carries the structured detail; the middleware
            # backstop's fires when this handler never returned and carries
            # only a string detail plus a correlation_id, because it sits
            # outside the handler and never parsed the body. Declaring just
            # the first would make a generated client fail to decode exactly
            # the fallback the backstop exists to produce.
            "model": PublishTimeoutResponse | BackstopTimeoutResponse,
            "description": (
                "The publish did not confirm and may or may not have "
                "committed. It differs from 202 only in that nothing is "
                "still running. Two shapes: the route's own timeout has "
                "reason 'publish_timeout' inside a structured detail with "
                "the request_id; the middleware backstop's has a string "
                "detail and a correlation_id but NO request_id, which is why "
                "you should send your own. Retry with the SAME request_id "
                "either way: the server rejects a duplicate, so the retry is "
                "safe."
            ),
        },
    },
)
@limiter.limit("60/minute")  # Rate limit for write operations
async def publish_user_interaction(
    request: Request,
    payload: PublishUserInteractionRequest,
    http_response: Response,
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

    # Two clocks. The route answers on `response_deadline`; the worker keeps
    # running until `worker_deadline`, which is STRICTLY LATER -- see
    # PUBLISH_WORKER_GRACE_SECONDS. Collapsing these into one deadline is the
    # defect the 202 exit was shipped with.
    #
    # Both are measured from the moment `TimeoutMiddleware` started ITS clock,
    # not from the moment this handler began. Those differ by however long
    # body parsing and the sync auth/billing dependencies took, and the
    # backstop has been counting throughout. Starting an independent clock
    # here made the guarded ordering (240 < 270 < 300) true of the CONSTANTS
    # while false in wall-clock terms: burn the 60s gap in dependency
    # resolution and the middleware expires first, answering its generic 504
    # with no request_id for a publish this route was still shepherding.
    #
    # The fallback keeps the route working when the middleware is absent --
    # which is every TestClient app built without it, and any embedding of
    # this router elsewhere.
    arrival = getattr(request.state, "backstop_started", None)
    if arrival is None:
        arrival = time.monotonic()
    response_deadline = arrival + PUBLISH_REQUEST_TIMEOUT_SECONDS
    worker_deadline = response_deadline + PUBLISH_WORKER_GRACE_SECONDS
    payload.request_id = payload.request_id or str(uuid.uuid4())
    # The accumulator opens BEFORE `acquire_ingestion` -- pure queueing behind
    # other publishes for this org -- so that wait is both a field on the line
    # and part of the duration the threshold is applied to. `collect` starts
    # the whole-request clock here; `GenerationService.run` starts its own only
    # after admission, and a threshold applied to that one alone is blind to an
    # 8s queue wait followed by 50ms of fast work.
    with publish_timing.collect():
        try:
            # Admission waits on the RESPONSE clock: there is no point
            # holding a slot past the moment the route would have to answer
            # anyway.
            with publish_timing.phase("admission"):
                # Sampled BEFORE the call, because `acquire_ingestion` cannot
                # tell us afterwards: its loop condition is checked first, so
                # an already-spent budget returns False without ever reading
                # the ingestion ledger. Reporting that as capacity exhaustion
                # would claim a saturation nobody observed -- a slot may have
                # been free the whole time.
                budget_already_spent = time.monotonic() >= response_deadline
                admitted = await acquire_ingestion(org_id, response_deadline)
            if not admitted:
                # Nothing was admitted and nothing will commit, so this is safe
                # to retry -- and safe to retry WITH THIS ID, which is why it is
                # here.
                raise HTTPException(
                    status_code=503,
                    detail=PublishCapacityRefusedDetail(
                        reason=(
                            "budget_exhausted_before_admission"
                            if budget_already_spent
                            else "capacity_deadline_exceeded"
                        ),
                        request_id=payload.request_id,
                        message=(
                            (
                                "This request's time budget was already spent "
                                "before admission was attempted, so nothing "
                                "was admitted; publish capacity was never "
                                "checked. Retry with this same request_id."
                            )
                            if budget_already_spent
                            else (
                                "Publish capacity deadline exceeded; nothing "
                                "was admitted. Retry with this same "
                                "request_id."
                            )
                        ),
                    ).model_dump(),
                )
        except BaseException:
            # The ONLY exits that never reach `GenerationService.run`, and so
            # the only ones nothing else will report: the 503 above, and a
            # client disconnect while this request is still queued.
            #
            # That second one is not rare. 97.4% of production publishes end as
            # an ELB 460 -- the client's own ~8s timeout fires first -- so a
            # request that spends its whole life in this queue and is then
            # abandoned is exactly the shape the instrument exists to catch,
            # and it would otherwise be the silent majority.
            #
            # Once the worker HAS started this is unnecessary, and a `finally`
            # spanning the whole handler would be actively wrong: the work is
            # shielded and runs to completion in a thread that keeps its own
            # copy of this context, so it emits the full duration itself.
            # Reported here too, the route would latch its shorter
            # client-visible time first and suppress the real one. Measured,
            # not assumed -- see the cancellation test.
            publish_timing.emit(org_id=org_id, request_id=payload.request_id)
            raise
        token = admission_deadline.set(worker_deadline)
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
                    asyncio.shield(operation),
                    timeout=max(0, response_deadline - time.monotonic()),
                )
            except TimeoutError as exc:
                # TWO different events arrive here, and only one of them leaves
                # the outcome open. `asyncio.wait_for` re-raises whatever the
                # awaited task raised, so a TimeoutError raised BY THE WORKER
                # lands on the same handler as `wait_for` giving up on a task
                # that is still going. Answering 202 for both would leave the
                # outcome open about an operation that is already dead, and
                # withhold the 504's retry instruction.
                #
                # Which worker failures actually arrive here is decided one
                # layer down, by `InteractionsMixin.publish_interaction`: it
                # re-raises `PublishDeadlineExceededError` and flattens
                # everything else -- including a plain `socket.timeout`, which
                # IS a TimeoutError since 3.10 -- into `success=False`. That
                # narrowing is deliberate; see the mixin.
                #
                # `operation.done()` separates the two exactly. `wait_for`
                # cancels the SHIELD, never the operation, so a timed-out wait
                # leaves it False; a task that finished by raising leaves it
                # True.
                if operation.done():
                    # Nothing is running, so release inline rather than from a
                    # done-callback, and do NOT also register the callback
                    # below -- that would be a second release, which raises
                    # KeyError.
                    release_ingestion(org_id)
                    raise HTTPException(
                        status_code=504,
                        # Deliberately does not claim either outcome: the
                        # timeout may have fired before or after the durable
                        # write. `from exc` keeps the real error for Sentry
                        # while the caller gets a body it can act on. The
                        # retry instruction, not a poll, is the actionable
                        # half -- see PAST_DEADLINE_202_MESSAGE for why.
                        detail=PublishTimeoutDetail(
                            request_id=payload.request_id,
                            message=(
                                "The publish did not confirm, and may or may "
                                "not have committed. Retry with this same "
                                "request_id: the server rejects a duplicate, "
                                "so the retry is safe whether or not the "
                                "first attempt landed."
                            ),
                        ).model_dump(),
                    ) from exc
                # The operation is SHIELDED *and* holds a worker deadline that
                # is still in the future, so it is genuinely still going --
                # the shield alone never made that true, because the
                # cooperative deadline check is not asyncio cancellation. The
                # response still does not SAY so; see
                # PAST_DEADLINE_202_MESSAGE.
                operation.add_done_callback(_release_and_drain(org_id))
                http_response.status_code = status.HTTP_202_ACCEPTED
                return PublishUserInteractionResponse(
                    success=True,
                    request_id=payload.request_id,
                    learning_status="deferred",
                    learning_reason="server_deadline",
                    # Computed from the REQUEST, not the result, so it is
                    # available here. Omitting it left `warnings` serialising
                    # as [], which a caller cannot tell from "nothing was
                    # altered".
                    warnings=payload.payload_warnings(),
                    message=PAST_DEADLINE_202_MESSAGE,
                )
            except asyncio.CancelledError:
                operation.add_done_callback(_release_and_drain(org_id))
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
            while time.monotonic() < response_deadline:
                # Named to avoid shadowing the module-level `fastapi.status`
                # import: assigning to `status` anywhere in this function makes
                # Python treat it as local for the WHOLE function body, which
                # breaks the `status.HTTP_202_ACCEPTED` reference above.
                extraction_status = await asyncio.to_thread(
                    storage.extraction_status, payload.user_id, payload.request_id
                )
                if extraction_status["status"] == "done":
                    counts = await asyncio.to_thread(
                        storage.extraction_counts, payload.user_id, payload.request_id
                    )
                    response.learning_status = "done"
                    response.learning_reason = extraction_status["reason"]
                    response.profiles_added = counts["profile"]
                    response.playbooks_added = counts["playbook"]
                    return response
                # Holding the connection open for a window that only new input can
                # close wastes the caller's deadline and a waiter slot, and reports
                # `wait_timeout` for a healthy stream. Same break as the library
                # waiter in generation_service.run -- one condition, both paths.
                if coverage_stalled(extraction_status):
                    stalled_reason = extraction_status["reason"]
                    break
                if await request.is_disconnected():
                    break
                await asyncio.sleep(
                    min(0.25, max(0, response_deadline - time.monotonic()))
                )
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
    # Named to avoid shadowing the module-level `fastapi.status` import for
    # this function's whole body -- see the identical note in
    # publish_user_interaction above.
    extraction_status = storage.extraction_status(req.user_id, request_id)
    return LearningStatusResponse(**extraction_status)
