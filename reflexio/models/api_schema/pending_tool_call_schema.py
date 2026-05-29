"""Pydantic models for pending tool call review endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
)


class PendingToolCallResponse(BaseModel):
    id: str
    org_id: str
    scope: dict[str, Any]
    scope_hash: str
    tool_name: str
    dedup_key: str
    status: PendingToolCallStatus
    question_text: str
    args: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    user_id: str | None = None
    answer_format: str | None = None
    result: dict[str, Any] | None = None
    superseded_by: str | None = None
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    expires_at: datetime | None = None
    cache_until: datetime | None = None
    valid_until: datetime | None = None

    @classmethod
    def from_record(cls, record: PendingToolCallRecord) -> PendingToolCallResponse:
        return cls(
            id=record.id,
            org_id=record.org_id,
            scope=record.scope,
            scope_hash=record.scope_hash,
            tool_name=record.tool_name,
            dedup_key=record.dedup_key,
            status=record.status,
            question_text=record.question_text,
            args=record.args,
            tags=record.tags,
            user_id=record.user_id,
            answer_format=record.answer_format,
            result=record.result,
            superseded_by=record.superseded_by,
            created_at=record.created_at,
            resolved_at=record.resolved_at,
            expires_at=record.expires_at,
            cache_until=record.cache_until,
            valid_until=record.valid_until,
        )


class PendingToolCallListResponse(BaseModel):
    pending_tool_calls: list[PendingToolCallResponse]


class ResolvePendingToolCallRequest(BaseModel):
    result: dict[str, Any]
    valid_for_seconds: int | None = Field(default=None, gt=0)
