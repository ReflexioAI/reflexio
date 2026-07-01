"""Shared storage types and helpers for resumable extraction agent runs.

Residual holder for the ``_agent_run.py`` decomposition. The 18 shared
non-class symbols live in the leaf ``agent_run/_models.py`` and are re-exported
here so ``storage_base/__init__.py`` keeps importing them from ``._agent_run``
byte-identically. ``AgentRunMixin`` survives as a thin COMPOSITE ABC:
``AgentRunStoreABC`` supplies the agent-run lifecycle stubs,
``PendingToolCallStoreABC`` supplies the pending-tool-call stubs,
``RunToolDependencyStoreABC`` supplies the run-tool-dependency stubs, and only
the public helper staticmethods stay inline here. All twenty-three agent-run
methods now live in the three ``agent_run`` sub-mixins.
"""

from __future__ import annotations

from .agent_run import (
    AgentRunStoreABC,
    PendingToolCallStoreABC,
    RunToolDependencyStoreABC,
)
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


class AgentRunMixin(
    AgentRunStoreABC, PendingToolCallStoreABC, RunToolDependencyStoreABC
):
    """Backend-neutral helpers shared by resumable extraction storage backends."""

    build_scope_hash = staticmethod(build_scope_hash)
    human_feedback_scope = staticmethod(human_feedback_scope)
    build_pending_tool_call_dedup_key = staticmethod(build_pending_tool_call_dedup_key)
