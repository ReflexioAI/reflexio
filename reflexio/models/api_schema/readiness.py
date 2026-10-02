"""Cached operational readiness response schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ReadinessCheck(BaseModel):
    status: Literal["passed", "failed", "pending", "not_checked"]
    required: bool = True
    reason: str | None = None
    checked_at: datetime | None = None


class ReadinessResponse(BaseModel):
    status: Literal["healthy", "unhealthy"] = "healthy"
    checks: dict[str, ReadinessCheck] = Field(default_factory=dict)
