from __future__ import annotations

from abc import ABC, abstractmethod

from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
    SubjectWriteBarrier,
)


class GovernanceMixin(ABC):
    """Mixin for backend-neutral governance storage primitives."""

    @abstractmethod
    def hide_governance_agent_playbooks_for_rebuild(self, purge_id: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def apply_governance_user_data_delete(
        self, purge_id: str, user_id: str
    ) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def apply_governance_agent_playbook_rebuild(
        self,
        purge_id: str,
        agent_playbook_id: int,
        remaining_source_windows: list[dict[str, object]],
        content: str | None,
        trigger: str | None,
        rationale: str | None,
        blocking_issue: dict[str, object] | None,
        expanded_terms: str | None,
        tags: list[str] | None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def complete_purge_operation_with_audit(
        self, purge_id: str, audit_event: AuditEvent
    ) -> PurgeOperation:
        raise NotImplementedError

    @abstractmethod
    def begin_subject_erasure_barrier(
        self, subject_ref: str, purge_id: str
    ) -> SubjectWriteBarrier:
        raise NotImplementedError

    @abstractmethod
    def assert_subject_writable(self, subject_ref: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def complete_subject_erasure_barrier_after_empty_check(
        self, purge_id: str, audit_event: AuditEvent
    ) -> PurgeOperation:
        raise NotImplementedError

    @abstractmethod
    def fail_subject_erasure_barrier(
        self,
        subject_ref: str,
        purge_id: str,
        error_code: str,
        error_detail: str,
    ) -> SubjectWriteBarrier:
        raise NotImplementedError

    @abstractmethod
    def get_subject_write_barrier(self, subject_ref: str) -> SubjectWriteBarrier | None:
        raise NotImplementedError
