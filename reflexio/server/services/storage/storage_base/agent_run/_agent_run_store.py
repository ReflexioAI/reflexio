"""Backend-neutral AgentRunStore contract (the AgentRunStore bucket).

Extracted from ``_agent_run.py`` (the AgentRunStore bucket): the agent-run
lifecycle and provenance lookup methods. The residual ``AgentRunMixin``
composite stays composed alongside this class and holds the remaining pending-tool-call /
run-tool-dependency stubs plus the shared ``build_scope_hash`` /
``human_feedback_scope`` / ``build_pending_tool_call_dedup_key`` staticmethods.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ._models import AgentRunRecord, AgentRunStatus


class AgentRunStoreABC:
    """Backend-neutral agent-run lifecycle store contract."""

    def create_agent_run(self, record: AgentRunRecord) -> AgentRunRecord:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def get_agent_run(self, run_id: str) -> AgentRunRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def get_agent_run_finalization_receipt(
        self,
        *,
        run_id: str,
        entity_type: str,
    ) -> list[str] | None:
        """Return persisted learning ids for a completed run finalization."""
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def save_agent_run_finalization_receipt(
        self,
        *,
        run_id: str,
        entity_type: str,
        learning_ids: list[str],
    ) -> bool:
        """Persist an immutable run-to-learning binding and report insert ownership."""
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def get_latest_finalized_agent_run_for_request(
        self,
        *,
        org_id: str,
        extractor_kind: str,
        user_id: str | None,
        request_id: str,
    ) -> AgentRunRecord | None:
        """Return the newest finalized extraction run for one logical request."""
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
        expected_statuses: tuple[AgentRunStatus, ...] | None = None,
    ) -> AgentRunRecord | None:
        raise NotImplementedError(f"{type(self).__name__} does not support agent runs")

    def fail_running_agent_runs_for_request(
        self,
        *,
        org_id: str,
        extractor_kind: str,
        user_id: str | None,
        request_id: str,
        last_error: str,
    ) -> int:
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
