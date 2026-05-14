"""Pydantic models for the stall_state HTTP endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

StallReason = Literal["billing_error", "auth_error"]
"""Canonical stall-reason discriminator.

Imported by the storage layer (``sqlite_storage._stall_state``) and the
LiteLLM provider's stream parser so all three layers share one source of
truth for the allowed values."""


class StallStateResponse(BaseModel):
    """Returned by GET /stall_state."""

    stalled: bool = Field(..., description="True when learning is currently paused.")
    reason: StallReason | None = None
    stalled_at: datetime | None = None
    reset_estimate: datetime | None = None
    notified_in_cc: bool = False
    error_message: str | None = None


class MarkNotifiedResponse(BaseModel):
    """Returned by POST /stall_state/notified."""

    notified_in_cc: bool
