"""HTTP endpoints exposing the stall_state row."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from reflexio.models.api_schema.stall_state_schema import (
    MarkNotifiedResponse,
    StallStateResponse,
)
from reflexio.server.api_endpoints.request_context import RequestContext, get_request_context

router = APIRouter(tags=["stall_state"])


@router.get("/stall_state", response_model=StallStateResponse)
def read_stall_state(
    ctx: RequestContext = Depends(get_request_context),
) -> StallStateResponse:
    """Return the current singleton stall_state row.

    Args:
        ctx (RequestContext): Injected request context with storage attached.

    Returns:
        StallStateResponse: ``stalled=False`` with null fields when clean.
    """
    state = ctx.storage.get_stall_state()
    return StallStateResponse(
        stalled=state.stalled,
        reason=state.reason,
        stalled_at=state.stalled_at,
        reset_estimate=state.reset_estimate,
        notified_in_cc=state.notified_in_cc,
        error_message=state.error_message,
    )


@router.post("/stall_state/notified", response_model=MarkNotifiedResponse)
def post_notified(
    ctx: RequestContext = Depends(get_request_context),
) -> MarkNotifiedResponse:
    """Idempotently flip ``notified_in_cc=1`` for the current stall.

    No-op (and still returns 200) when there's no active stall — the SessionStart
    hook may race with auto-clear.

    Args:
        ctx (RequestContext): Injected request context with storage attached.

    Returns:
        MarkNotifiedResponse: ``notified_in_cc=True`` after the call (when stalled),
            ``False`` when no stall is active.
    """
    ctx.storage.mark_stall_notified()
    state = ctx.storage.get_stall_state()
    return MarkNotifiedResponse(notified_in_cc=state.notified_in_cc)
