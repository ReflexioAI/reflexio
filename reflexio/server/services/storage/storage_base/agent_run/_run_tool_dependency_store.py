"""Backend-neutral RunToolDependencyStore contract (the RunToolDependencyStore bucket).

Extracted verbatim from ``_agent_run.py`` (the RunToolDependencyStore bucket):
the four run-tool-dependency methods. This is the LAST bucket — after this
extraction all twenty-three agent-run methods live in the three ``agent_run``
sub-mixins and the residual ``_agent_run.py`` module is helpers-only. The
composite ``AgentRunMixin`` now inherits all three ABCs and retains only the
shared ``build_scope_hash`` / ``human_feedback_scope`` /
``build_pending_tool_call_dedup_key`` staticmethods.
"""

from __future__ import annotations

from ._models import RunToolDependencyRecord


class RunToolDependencyStoreABC:
    """Backend-neutral run-tool-dependency store contract."""

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
