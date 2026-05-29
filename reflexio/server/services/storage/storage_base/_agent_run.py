"""Shared storage types and helpers for resumable extraction agent runs."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

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


class PendingToolCallStatus(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    CANCELLED = "cancelled"


NOT_APPLICABLE_ANSWER = "User does not have information about this question."


class RunToolDependencyKind(StrEnum):
    FOLLOWUP = "followup"


@dataclass(frozen=True)
class AgentBinding:
    """Logical run binding flattened into `_agent_runs` storage columns."""

    org_id: str
    extractor_kind: str
    extractor_name: str
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


class AgentRunMixin:
    """Backend-neutral helpers shared by resumable extraction storage backends."""

    build_scope_hash = staticmethod(build_scope_hash)
    human_feedback_scope = staticmethod(human_feedback_scope)
    build_pending_tool_call_dedup_key = staticmethod(build_pending_tool_call_dedup_key)

    def create_agent_run(self, record: AgentRunRecord) -> AgentRunRecord:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def get_agent_run(self, run_id: str) -> AgentRunRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def update_agent_run_status(
        self,
        run_id: str,
        status: AgentRunStatus,
        *,
        committed_output: dict[str, Any] | None = None,
        pending_tool_call_ids: list[str] | None = None,
        max_steps_remaining: int | None = None,
        next_resume_at: datetime | None = None,
        last_error: str | None = None,
        increment_finalization_attempts: bool = False,
    ) -> AgentRunRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def create_pending_tool_call(
        self, record: PendingToolCallRecord
    ) -> PendingToolCallRecord:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def create_or_attach_pending_tool_call(
        self,
        *,
        record: PendingToolCallRecord,
        dependency: RunToolDependencyRecord,
        now: datetime | None = None,
    ) -> PendingToolCallUpsertResult:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def get_pending_tool_call(self, call_id: str) -> PendingToolCallRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def list_pending_tool_calls(
        self,
        *,
        status: PendingToolCallStatus | None = None,
        limit: int = 100,
    ) -> list[PendingToolCallRecord]:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def cancel_pending_tool_call(
        self,
        call_id: str,
        *,
        cancelled_at: datetime | None = None,
    ) -> PendingToolCallRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def expire_pending_tool_calls(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def find_active_pending_tool_call(
        self,
        *,
        org_id: str,
        scope_hash: str,
        tool_name: str,
        dedup_key: str,
        now: datetime | None = None,
    ) -> PendingToolCallRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def search_prior_tool_calls(
        self,
        *,
        org_id: str,
        scope_hash: str,
        tool_name: str,
        query_embedding: list[float] | None = None,
        now: datetime | None = None,
        limit: int = 8,
    ) -> list[PriorAnswerMatch]:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def attach_run_tool_dependency(
        self, record: RunToolDependencyRecord
    ) -> RunToolDependencyRecord:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def count_unresolved_followup_dependencies(
        self,
        *,
        org_id: str,
        extractor_kind: str,
        extractor_name: str,
        tool_name: str,
    ) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def list_run_tool_dependencies(self, run_id: str) -> list[RunToolDependencyRecord]:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def resolve_pending_tool_call(
        self,
        call_id: str,
        *,
        result: dict[str, Any],
        resolved_at: datetime | None = None,
        valid_for_seconds: int,
    ) -> PendingToolCallRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def update_resolved_pending_tool_call_result(
        self,
        call_id: str,
        *,
        result: dict[str, Any],
        resolved_at: datetime | None = None,
        valid_for_seconds: int,
    ) -> PendingToolCallRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def mark_pending_tool_call_not_applicable(
        self,
        call_id: str,
        *,
        resolved_at: datetime | None = None,
        valid_for_seconds: int,
    ) -> PendingToolCallRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def claim_ready_agent_run(
        self,
        *,
        org_id: str,
        worker_id: str,
        now: datetime | None = None,
        claim_ttl_seconds: int = 600,
    ) -> AgentRunRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def claim_finalization_failed_agent_run(
        self,
        *,
        org_id: str,
        worker_id: str,
        now: datetime | None = None,
        claim_ttl_seconds: int = 600,
    ) -> AgentRunRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def list_resumable_work_org_ids(
        self,
        *,
        now: datetime | None = None,
        limit: int = 1000,
    ) -> list[str]:
        """Return distinct org_ids that have actionable resumable-extraction work.

        Cross-org maintenance query (intentionally NOT scoped to ``self.org_id``):
        the resume scheduler uses it to discover every org that has a run ready
        to resume, a run awaiting finalization retry, or a pending tool call that
        can be expired, so per-org workers can be driven for all of them.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def consume_run_tool_dependencies(self, run_id: str) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")
