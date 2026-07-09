"""Shared storage types and helpers for resumable extraction agent runs.

Leaf module (imports NO mixin, so it never participates in an import cycle).
The three ``agent_run`` ABC sub-mixins and the residual ``AgentRunMixin``
composite import their records/enums/helpers from here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from reflexio.models.api_schema.pending_tool_call_schema import PendingToolCallStatus

_WHITESPACE_RE = re.compile(r"\s+")


class AgentRunStatus(StrEnum):
    RUNNING = "running"
    AGENT_COMPLETED = "agent_completed"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    FINALIZED_PENDING_TOOL = "finalized_pending_tool"
    RESUME_READY = "resume_ready"
    RESUMING = "resuming"
    FINALIZATION_FAILED = "finalization_failed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


NOT_APPLICABLE_ANSWER = "User does not have information about this question."


class RunToolDependencyKind(StrEnum):
    FOLLOWUP = "followup"


@dataclass(frozen=True)
class AgentBinding:
    """Logical run binding flattened into `_agent_runs` storage columns.

    ``request_id`` remains the persisted provenance field name at this storage
    boundary even when higher-level helpers refer to the same value as a
    generation request id.
    """

    org_id: str
    extractor_kind: str
    user_id: str | None
    request_id: str
    agent_version: str | None
    source: str | None
    source_interaction_ids: list[int] = field(default_factory=list)
    window_start_interaction_id: int | None = None
    window_end_interaction_id: int | None = None
    extractor_config_hash: str | None = None


@dataclass(frozen=True)
class AgentRunRecord:
    """Durable agent run row plus snapshots used for resume/finalization.

    ``generation_request_snapshot`` intentionally keeps a legacy ``request_id``
    key for stored provenance.
    """

    id: str
    binding: AgentBinding
    status: AgentRunStatus
    generation_request_snapshot: dict[str, Any]
    service_config_snapshot: dict[str, Any] | None = None
    agent_context_snapshot: str | None = None
    committed_output: dict[str, Any] | None = None
    pending_tool_call_ids: list[str] = field(default_factory=list)
    max_steps_remaining: int | None = None
    resume_attempts: int = 0
    finalization_attempts: int = 0
    next_resume_at: datetime | None = None
    claimed_by: str | None = None
    claimed_at: datetime | None = None
    agent_completed_at: datetime | None = None
    finalized_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class PendingToolCallRecord:
    id: str
    org_id: str
    scope: dict[str, Any]
    scope_hash: str
    tool_name: str
    dedup_key: str
    status: PendingToolCallStatus
    question_text: str
    args: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    user_id: str | None = None
    answer_format: str | None = None
    result: dict[str, Any] | None = None
    embedding: list[float] | None = None
    superseded_by: str | None = None
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    expires_at: datetime | None = None
    cache_until: datetime | None = None
    valid_until: datetime | None = None


@dataclass(frozen=True)
class RunToolDependencyRecord:
    run_id: str
    pending_tool_call_id: str
    dependency_kind: RunToolDependencyKind = RunToolDependencyKind.FOLLOWUP
    resolved_at: datetime | None = None
    consumed_at: datetime | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class PendingToolCallUpsertResult:
    pending_tool_call: PendingToolCallRecord
    created: bool


@dataclass(frozen=True)
class PriorAnswerMatch:
    pending_tool_call_id: str
    status: PendingToolCallStatus
    question_text: str
    result: dict[str, Any] | None
    valid_until: datetime | None
    answer_format: str | None = None
    created_at: datetime | None = None
    resolved_at: datetime | None = None
    expires_at: datetime | None = None
    similarity: float | None = None


def canonical_json(value: Any) -> str:
    """Return deterministic compact JSON for storage hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_scope_hash(scope: dict[str, Any]) -> str:
    """Stable hash for a tool scope dictionary."""
    return hashlib.sha256(canonical_json(scope).encode("utf-8")).hexdigest()


def human_feedback_scope(org_id: str) -> dict[str, str]:
    """Human feedback is always org-scoped, never user-scoped."""
    return {"org_id": org_id, "scope_kind": "org"}


def normalize_dedup_text(value: str | None) -> str:
    """Normalize text before pending-tool-call dedup hashing."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE_RE.sub(" ", normalized.strip())
    return normalized.casefold()


def build_pending_tool_call_dedup_key(
    *,
    tool_name: str,
    question_text: str,
    answer_format: str | None = None,
) -> str:
    """Stable dedup hash for a normalized tool question."""
    parts = (
        normalize_dedup_text(tool_name),
        normalize_dedup_text(question_text),
        normalize_dedup_text(answer_format),
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def not_applicable_tool_result() -> dict[str, Any]:
    return {"answer": NOT_APPLICABLE_ANSWER, "not_applicable": True}


def is_not_applicable_tool_result(result: dict[str, Any] | None) -> bool:
    return isinstance(result, dict) and result.get("not_applicable") is True


def embedding_similarity(a: list[float] | None, b: list[float] | None) -> float | None:
    """Cosine similarity for optional embedding vectors."""
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a_sq = sum(x * x for x in a)
    mag_b_sq = sum(y * y for y in b)
    if mag_a_sq == 0.0 or mag_b_sq == 0.0:
        return None
    return dot / ((mag_a_sq**0.5) * (mag_b_sq**0.5))
