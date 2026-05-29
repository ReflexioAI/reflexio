"""HTTP endpoints for resolving pending tool calls."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
)

from reflexio.models.api_schema.pending_tool_call_schema import (
    PendingToolCallListResponse,
    PendingToolCallResponse,
    ResolvePendingToolCallRequest,
)
from reflexio.server.api_endpoints.request_context import (
    RequestContext,
    get_request_context,
)
from reflexio.server.services.extraction.resumable_agent import (
    is_resumable_extraction_enabled,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.storage_base import (
    BaseStorage,
    PendingToolCallRecord,
    PendingToolCallStatus,
)
from reflexio.server.usage_metrics import record_usage_event

router = APIRouter(prefix="/api/pending_tool_calls", tags=["pending_tool_calls"])
logger = logging.getLogger(__name__)

_UNSUPPORTED_DETAIL = (
    "Pending tool calls are not supported by the configured storage backend"
)
_HMAC_TIMESTAMP_TOLERANCE_SECONDS = 300


def _require_storage(ctx: RequestContext) -> BaseStorage:
    storage = ctx.storage
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage not configured")
    return storage


def _require_org_record(
    record: PendingToolCallRecord | None,
    *,
    org_id: str,
) -> PendingToolCallRecord:
    if record is None or record.org_id != org_id:
        raise HTTPException(status_code=404, detail="Pending tool call not found")
    return record


def _drain_resumable_followups(ctx: RequestContext) -> None:
    if not is_resumable_extraction_enabled(ctx):
        return
    worker = ExtractionResumeWorker(request_context=ctx)
    worker.drain()


def _is_missing_pending_tool_call_storage_error(exc: StorageError) -> bool:
    message = exc.message if hasattr(exc, "message") else str(exc)
    return "_pending_tool_calls" in message and (
        "schema cache" in message or "does not exist" in message
    )


def _run_with_pending_tool_call_migration_retry[T](
    *,
    storage: BaseStorage,
    operation: Callable[[], T],
) -> T:
    try:
        return operation()
    except StorageError as exc:
        if not _is_missing_pending_tool_call_storage_error(exc):
            raise
        migrate = getattr(storage, "migrate", None)
        if not callable(migrate):
            raise
        logger.warning(
            "event=pending_tool_call_storage_schema_missing action=migrate_and_retry"
        )
        migrate()
        return operation()


def _verify_hmac_signature(
    *,
    body: bytes,
    timestamp: str | None,
    signature: str | None,
    secrets: list[str],
) -> None:
    if not secrets:
        return
    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="Missing HMAC signature")
    try:
        timestamp_value = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid HMAC timestamp") from exc
    if abs(time.time() - timestamp_value) > _HMAC_TIMESTAMP_TOLERANCE_SECONDS:
        raise HTTPException(status_code=401, detail="Expired HMAC timestamp")

    expected_prefix = "sha256="
    provided = (
        signature
        if signature.startswith(expected_prefix)
        else f"{expected_prefix}{signature}"
    )
    signed_payload = timestamp.encode("utf-8") + b"." + body
    for secret in secrets:
        digest = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()
        if hmac.compare_digest(provided, f"{expected_prefix}{digest}"):
            return
    raise HTTPException(status_code=401, detail="Invalid HMAC signature")


@router.get("", response_model=PendingToolCallListResponse)
def list_pending_tool_calls(
    status: PendingToolCallStatus | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    ctx: RequestContext = Depends(get_request_context),
) -> PendingToolCallListResponse:
    storage = _require_storage(ctx)
    try:
        records = _run_with_pending_tool_call_migration_retry(
            storage=storage,
            operation=lambda: storage.list_pending_tool_calls(
                status=status,
                limit=limit,
            ),
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=_UNSUPPORTED_DETAIL) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    return PendingToolCallListResponse(
        pending_tool_calls=[
            PendingToolCallResponse.from_record(record)
            for record in records
            if record.org_id == ctx.org_id
        ]
    )


@router.get("/{pending_tool_call_id}", response_model=PendingToolCallResponse)
def get_pending_tool_call(
    pending_tool_call_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> PendingToolCallResponse:
    storage = _require_storage(ctx)
    try:
        record = _run_with_pending_tool_call_migration_retry(
            storage=storage,
            operation=lambda: storage.get_pending_tool_call(pending_tool_call_id),
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=_UNSUPPORTED_DETAIL) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    return PendingToolCallResponse.from_record(
        _require_org_record(record, org_id=ctx.org_id)
    )


@router.post("/{pending_tool_call_id}/resolve", response_model=PendingToolCallResponse)
async def resolve_pending_tool_call(
    pending_tool_call_id: str,
    request: Request,
    payload: ResolvePendingToolCallRequest,
    background_tasks: BackgroundTasks,
    x_reflexio_signature: str | None = Header(default=None),
    x_reflexio_timestamp: str | None = Header(default=None),
    ctx: RequestContext = Depends(get_request_context),
) -> PendingToolCallResponse:
    storage = _require_storage(ctx)
    pending_config = ctx.configurator.get_config().pending_tool_call_config
    _verify_hmac_signature(
        body=await request.body(),
        timestamp=x_reflexio_timestamp,
        signature=x_reflexio_signature,
        secrets=pending_config.hmac_secrets,
    )
    try:
        existing = _run_with_pending_tool_call_migration_retry(
            storage=storage,
            operation=lambda: storage.get_pending_tool_call(pending_tool_call_id),
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=_UNSUPPORTED_DETAIL) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    record = _require_org_record(existing, org_id=ctx.org_id)

    if record.status == PendingToolCallStatus.RESOLVED:
        if record.result == payload.result:
            return PendingToolCallResponse.from_record(record)
        raise HTTPException(
            status_code=409,
            detail="Pending tool call already resolved with a different result",
        )
    if record.status != PendingToolCallStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"Pending tool call is {record.status.value}",
        )

    valid_for_seconds = (
        payload.valid_for_seconds
        or pending_config.for_tool(record.tool_name).prior_answer_valid_seconds
    )
    try:
        resolved = storage.resolve_pending_tool_call(
            pending_tool_call_id,
            result=payload.result,
            resolved_at=datetime.now(UTC),
            valid_for_seconds=valid_for_seconds,
        )
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    if resolved is None:
        latest = _run_with_pending_tool_call_migration_retry(
            storage=storage,
            operation=lambda: storage.get_pending_tool_call(pending_tool_call_id),
        )
        latest = _require_org_record(latest, org_id=ctx.org_id)
        if latest.status == PendingToolCallStatus.RESOLVED:
            if latest.result == payload.result:
                return PendingToolCallResponse.from_record(latest)
            raise HTTPException(
                status_code=409,
                detail="Pending tool call already resolved with a different result",
            )
        raise HTTPException(
            status_code=409,
            detail=f"Pending tool call is {latest.status.value}",
        )
    if resolved.status != PendingToolCallStatus.RESOLVED:
        raise HTTPException(
            status_code=409,
            detail=f"Pending tool call is {resolved.status.value}",
        )
    if resolved.result != payload.result:
        raise HTTPException(
            status_code=409,
            detail="Pending tool call already resolved with a different result",
        )
    logger.info(
        "event=pending_tool_call_resolved org_id=%s user_id=%s "
        "pending_tool_call_id=%s tool_name=%s",
        ctx.org_id,
        record.user_id,
        pending_tool_call_id,
        record.tool_name,
    )
    record_usage_event(
        org_id=ctx.org_id,
        event_name="pending_tool_call_resolved",
        event_category="extraction_agent",
        user_id=record.user_id,
        outcome="resolved",
        metadata={
            "pending_tool_call_id": pending_tool_call_id,
            "tool_name": record.tool_name,
        },
    )
    background_tasks.add_task(_drain_resumable_followups, ctx)
    return PendingToolCallResponse.from_record(resolved)


@router.post("/{pending_tool_call_id}/cancel", response_model=PendingToolCallResponse)
def cancel_pending_tool_call(
    pending_tool_call_id: str,
    ctx: RequestContext = Depends(get_request_context),
) -> PendingToolCallResponse:
    storage = _require_storage(ctx)
    try:
        existing = _run_with_pending_tool_call_migration_retry(
            storage=storage,
            operation=lambda: storage.get_pending_tool_call(pending_tool_call_id),
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=_UNSUPPORTED_DETAIL) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    record = _require_org_record(existing, org_id=ctx.org_id)

    if record.status == PendingToolCallStatus.CANCELLED:
        return PendingToolCallResponse.from_record(record)
    if record.status != PendingToolCallStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"Pending tool call is {record.status.value}",
        )

    try:
        cancelled = storage.cancel_pending_tool_call(pending_tool_call_id)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    record_usage_event(
        org_id=ctx.org_id,
        event_name="pending_tool_call_cancelled",
        event_category="extraction_agent",
        user_id=record.user_id,
        outcome="cancelled",
        metadata={
            "pending_tool_call_id": pending_tool_call_id,
            "tool_name": record.tool_name,
        },
    )
    return PendingToolCallResponse.from_record(
        _require_org_record(cancelled, org_id=ctx.org_id)
    )
