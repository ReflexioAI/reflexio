"""Shared storage types and helpers for resumable extraction agent runs.

Residual holder for the ``_agent_run.py`` decomposition. The 18 shared
non-class symbols live in the leaf ``agent_run/_models.py`` and are re-exported
here so ``storage_base/__init__.py`` keeps importing them from ``._agent_run``
byte-identically. ``AgentRunMixin`` survives as a thin COMPOSITE ABC:
``AgentRunStoreABC`` supplies the agent-run lifecycle stubs,
``PendingToolCallStoreABC`` supplies the pending-tool-call stubs, and the
remaining run-tool-dependency stubs plus the public helper staticmethods stay
inline here until Task 4 peels them out.
"""

from __future__ import annotations

from .agent_run import AgentRunStoreABC, PendingToolCallStoreABC
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


class AgentRunMixin(AgentRunStoreABC, PendingToolCallStoreABC):
    """Backend-neutral helpers shared by resumable extraction storage backends."""

    build_scope_hash = staticmethod(build_scope_hash)
    human_feedback_scope = staticmethod(human_feedback_scope)
    build_pending_tool_call_dedup_key = staticmethod(build_pending_tool_call_dedup_key)

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

    def consume_run_tool_dependencies(self, run_id: str) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")
