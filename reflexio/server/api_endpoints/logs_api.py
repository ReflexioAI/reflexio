"""HTTP endpoint for querying OSS structured log events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from reflexio.models.api_schema.logs_schema import GetLogsResponse, LogEventResponse
from reflexio.server.auth import default_get_org_id

router = APIRouter(prefix="/api", tags=["logs"])

_VALID_LEVELS = frozenset({"warning", "error", "critical"})
_DEFAULT_LEVELS = {"error", "critical"}


@router.get("/logs", response_model=GetLogsResponse)
def get_logs(
    request: Request,
    _org_id: Annotated[str, Depends(default_get_org_id)],
    levels: Annotated[str | None, Query()] = None,
    since: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query()] = None,
    limit: Annotated[str | None, Query()] = None,
) -> GetLogsResponse:
    """Return structured log events newest-first."""
    handle = getattr(request.app.state, "structured_logging_handle", None)
    if handle is None:
        raise HTTPException(
            status_code=503,
            detail="Structured logs are unavailable for this server",
        )

    parsed_levels = _parse_levels(levels)
    parsed_since = _parse_since(since)
    parsed_limit = _parse_limit(limit)
    events = handle.query(
        levels=parsed_levels,
        since=parsed_since,
        q=q.strip() if q and q.strip() else None,
        limit=parsed_limit,
    )
    return GetLogsResponse(
        items=[
            LogEventResponse(
                timestamp=event.timestamp,
                level=event.level,
                logger_name=event.logger_name,
                message=event.message,
                exception_text=event.exception_text,
            )
            for event in events
        ],
        limit=parsed_limit,
    )


def _parse_levels(raw: str | None) -> set[str]:
    if raw is None:
        return set(_DEFAULT_LEVELS)
    parts = [part.strip().lower() for part in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise HTTPException(status_code=400, detail="levels must not be empty")
    unknown = sorted(set(parts) - _VALID_LEVELS)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported log level(s): {', '.join(unknown)}",
        )
    return set(parts)


def _parse_since(raw: str | None) -> datetime | None:
    if raw is None or not raw.strip():
        return None
    value = raw.strip()
    if value[-1:] in {"h", "d"} and value[:-1].isdigit():
        amount = int(value[:-1])
        if amount <= 0:
            raise HTTPException(status_code=400, detail="since must be positive")
        delta = (
            timedelta(hours=amount) if value.endswith("h") else timedelta(days=amount)
        )
        return datetime.now(UTC) - delta
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid since value") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_limit(raw: str | None) -> int:
    if raw is None or not raw.strip():
        return 200
    try:
        limit = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="limit must be an integer") from exc
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")
    return limit
