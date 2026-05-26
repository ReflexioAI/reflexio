"""HTTP endpoints exposing the stall_state row."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from reflexio.models.api_schema.stall_state_schema import (
    MarkNotifiedResponse,
    StallStateResponse,
)
from reflexio.server.api_endpoints.request_context import (
    RequestContext,
    get_request_context,
)

router = APIRouter(tags=["stall_state"])

_UNSUPPORTED_DETAIL = "Stall state is not supported by the configured storage backend"


def _require_storage(ctx: RequestContext):
    """Return ``ctx.storage`` or raise 503 when storage isn't configured.

    Cloud-mode deployments may run with ``ctx.storage is None`` until the
    caller provisions a backend; stall_state has no meaningful response in
    that state, so we surface 503 instead of an ``AttributeError``.

    Args:
        ctx (RequestContext): The injected request context.

    Returns:
        BaseStorage: The configured storage backend.

    Raises:
        HTTPException: 503 when storage is not configured.
    """
    if not ctx.is_storage_configured():
        raise HTTPException(status_code=503, detail="Storage not configured")
    assert ctx.storage is not None  # narrows BaseStorage | None -> BaseStorage
    return ctx.storage


@router.get("/stall_state", response_model=StallStateResponse)
def read_stall_state(
    ctx: RequestContext = Depends(get_request_context),
) -> StallStateResponse:
    """Return the current singleton stall_state row.

    Args:
        ctx (RequestContext): Injected request context with storage attached.

    Returns:
        StallStateResponse: ``stalled=False`` with null fields when clean.

    Raises:
        HTTPException: 503 when storage is not configured, or when the
            backend does not implement stall_state (e.g. disk storage).
    """
    storage = _require_storage(ctx)
    try:
        state = storage.get_stall_state()
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=_UNSUPPORTED_DETAIL) from exc
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

    Raises:
        HTTPException: 503 when storage is not configured, or when the
            backend does not implement stall_state (e.g. disk storage).
    """
    storage = _require_storage(ctx)
    try:
        storage.mark_stall_notified()
        state = storage.get_stall_state()
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=_UNSUPPORTED_DETAIL) from exc
    return MarkNotifiedResponse(notified_in_cc=state.notified_in_cc)
