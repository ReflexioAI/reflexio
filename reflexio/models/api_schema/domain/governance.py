from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

AuditActorType = Literal["api_token", "jwt", "system"]
AuditOperation = Literal["READ", "EXPORT", "ERASE", "CREATE", "UPDATE", "DELETE"]
AuditEntityType = Literal[
    "profile",
    "user_playbook",
    "agent_playbook",
    "interaction",
    "request",
    "session",
    "agent_success_evaluation_result",
    "playbook_retrieval_log",
    "org",
]
AuditStatus = Literal["ok", "error"]
PurgeOperationType = Literal["user_erasure", "org_purge"]
PurgeScopeType = Literal["user", "org"]
PurgeStatus = Literal["pending", "running", "failed", "complete"]
PurgeTargetStatus = Literal["pending", "running", "failed", "complete"]

__all__ = [
    "AuditActorType",
    "AuditOperation",
    "AuditEntityType",
    "AuditStatus",
    "PurgeOperationType",
    "PurgeScopeType",
    "PurgeStatus",
    "PurgeTargetStatus",
    "AuditEvent",
    "PurgeOperation",
    "PurgeOperationTarget",
    "UserExportResult",
    "UserEraseResult",
]


def _now_epoch() -> int:
    return int(datetime.now(UTC).timestamp())


class AuditEvent(BaseModel):
    org_id: str
    actor_type: AuditActorType = "system"
    actor_ref: str | None = None
    operation: AuditOperation
    entity_type: AuditEntityType
    entity_id: str | None = None
    subject_ref: str | None = None
    request_ref: str
    idempotency_key: str | None = None
    status: AuditStatus = "ok"
    detail: dict[str, Any] | None = None
    created_at: int = Field(default_factory=_now_epoch)


class PurgeOperation(BaseModel):
    purge_id: str
    org_id: str
    operation_type: PurgeOperationType
    scope_type: PurgeScopeType
    subject_ref: str | None = None
    request_ref: str
    idempotency_key: str
    status: PurgeStatus = "pending"
    error_code: str | None = None
    error_detail: str | None = None
    created_at: int = Field(default_factory=_now_epoch)
    updated_at: int = Field(default_factory=_now_epoch)
    completed_at: int | None = None


class PurgeOperationTarget(BaseModel):
    purge_id: str
    target_name: str
    target_ref: str = ""
    phase: str
    status: PurgeTargetStatus = "pending"
    detail: dict[str, Any] | None = None
    deleted_count: int = 0
    error_detail: str | None = None
    started_at: int | None = None
    completed_at: int | None = None


class UserExportResult(BaseModel):
    subject_ref: str
    export_id: str
    bundle: dict[str, Any]


class UserEraseResult(BaseModel):
    subject_ref: str
    purge_id: str
    status: PurgeStatus
    deleted_counts: dict[str, int] = Field(default_factory=dict)
    rebuilt_agent_playbook_ids: list[int] = Field(default_factory=list)
