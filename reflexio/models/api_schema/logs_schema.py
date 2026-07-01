"""Request/response models for structured log API endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LogLevel = Literal["warning", "error", "critical"]


class LogEventResponse(BaseModel):
    timestamp: str
    level: LogLevel
    logger_name: str
    message: str
    exception_text: str | None = None


class GetLogsResponse(BaseModel):
    items: list[LogEventResponse] = Field(default_factory=list)
    limit: int
