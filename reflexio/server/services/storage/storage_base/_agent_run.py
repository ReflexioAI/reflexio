"""Shared storage types and helpers for resumable extraction agent runs.

Residual holder for the ``_agent_run.py`` decomposition. The 18 shared
non-class symbols live in the leaf ``agent_run/_models.py`` and are re-exported
here so ``storage_base/__init__.py`` keeps importing them from ``._agent_run``
byte-identically. ``AgentRunMixin`` survives as a thin COMPOSITE ABC:
``AgentRunStoreABC`` supplies the agent-run lifecycle stubs, and the remaining
pending-tool-call / run-tool-dependency stubs plus the public helper
staticmethods stay inline here until Tasks 3-4 peel them out.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .agent_run import AgentRunStoreABC
from .agent_run._models import (
    NOT_APPLICABLE_ANSWER,
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    PendingToolCallUpsertResult,
    PriorAnswerMatch,
    RunToolDependencyKind,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    canonical_json,
    embedding_similarity,
    human_feedback_scope,
    is_not_applicable_tool_result,
    normalize_dedup_text,
    not_applicable_tool_result,
)

__all__ = [
    "NOT_APPLICABLE_ANSWER",
    "AgentBinding",
    "AgentRunMixin",
    "AgentRunRecord",
    "AgentRunStatus",
    "PendingToolCallRecord",
    "PendingToolCallStatus",
    "PendingToolCallUpsertResult",
    "PriorAnswerMatch",
    "RunToolDependencyKind",
    "RunToolDependencyRecord",
    "build_pending_tool_call_dedup_key",
    "build_scope_hash",
    "canonical_json",
    "embedding_similarity",
    "human_feedback_scope",
    "is_not_applicable_tool_result",
    "normalize_dedup_text",
    "not_applicable_tool_result",
]


class AgentRunMixin(AgentRunStoreABC):
    """Backend-neutral helpers shared by resumable extraction storage backends."""

    build_scope_hash = staticmethod(build_scope_hash)
    human_feedback_scope = staticmethod(human_feedback_scope)
    build_pending_tool_call_dedup_key = staticmethod(build_pending_tool_call_dedup_key)

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

    def delete_expired_pending_tool_calls(
        self,
        *,
        now: int,
        grace_seconds: int,
        limit: int = 1000,
    ) -> int:
        """Physically delete terminal 'expired'-status rows past the grace window.

        Only rows with ``status = 'expired'`` (the terminal marker set by
        ``expire_pending_tool_calls``) are candidates. RESOLVED rows are never
        deleted — even if their ``expires_at`` is in the past — because their
        ``valid_until`` may still be live and their cached result is needed by
        resumable extraction resume paths.

        The deletion cutoff is ``now - grace_seconds`` expressed as an ISO-8601
        string (``expires_at`` is stored as TEXT/ISO in SQLite and as
        ``timestamp with time zone`` in Postgres).

        Args:
            now: Current Unix epoch seconds.
            grace_seconds: Extra TTL buffer; a row is only deleted if its
                ``expires_at < datetime(now - grace_seconds)``.
            limit: Maximum rows to delete in one call (default 1000).

        Returns:
            Number of rows actually deleted.
        """
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

    def consume_run_tool_dependencies(self, run_id: str) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")
