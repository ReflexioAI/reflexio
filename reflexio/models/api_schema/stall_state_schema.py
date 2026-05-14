"""Pydantic models for the stall_state HTTP endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class StallStateResponse(BaseModel):
    """Returned by GET /stall_state."""

    stalled: bool = Field(..., description="True when learning is currently paused.")
    reason: Literal["billing_error", "auth_error"] | None = None
    stalled_at: datetime | None = None
    reset_estimate: datetime | None = None
    notified_in_cc: bool = False
    error_message: str | None = None


class MarkNotifiedResponse(BaseModel):
    """Returned by POST /stall_state/notified."""

    notified_in_cc: bool
