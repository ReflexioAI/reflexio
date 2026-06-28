from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
    PurgeOperationTarget,
)
from reflexio.models.config_schema import GovernanceRetentionConfig


class GovernanceMixin:
    """Mixin for backend-neutral governance storage primitives."""

    @abstractmethod
    def append_audit_event(self, event: AuditEvent) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_audit_events(
        self, subject_ref: str | None = None, *, org_id: str | None = None
    ) -> list[AuditEvent]:
        raise NotImplementedError

    @abstractmethod
    def begin_purge_operation(
        self,
        purge_id: str,
        idempotency_key: str,
        operation_type: Literal["user_erasure", "org_purge"],
        scope_type: Literal["user", "org"],
        subject_ref: str | None,
        request_ref: str,
    ) -> PurgeOperation:
        raise NotImplementedError

    @abstractmethod
    def record_purge_target(
        self,
        purge_id: str,
        target_name: str,
        phase: str,
        status: Literal["pending", "running", "failed", "complete"],
        target_ref: str = "",
        detail: dict[str, object] | None = None,
        deleted_count: int = 0,
        error_detail: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_purge_targets(
        self, purge_id: str, phase: str | None = None
    ) -> list[PurgeOperationTarget]:
        raise NotImplementedError

    @abstractmethod
    def purge_targets_prepared(self, purge_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def prepare_governance_erase_targets(
        self,
        purge_id: str,
        user_id: str,
        owned_user_playbook_ids: set[int] | None = None,
    ) -> None:
        raise NotImplementedError

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
    def fail_purge_operation(
        self, purge_id: str, error_code: str, error_detail: str
    ) -> PurgeOperation:
        raise NotImplementedError

    @abstractmethod
    def get_purge_operation(self, purge_id: str) -> PurgeOperation:
        raise NotImplementedError

    @abstractmethod
    def gc_governance_retention(self, *, config: GovernanceRetentionConfig) -> int:
        raise NotImplementedError
