from __future__ import annotations

from abc import ABC, abstractmethod

from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
)


class GovernanceEraseExecutionMixin(ABC):
    """Backend-neutral governance-erase-execution store contract.

    Extracted verbatim from ``_governance.py`` (the GovernanceEraseExecution
    bucket). The residual ``GovernanceMixin`` ABC stays composed alongside this
    mixin and holds the remaining rebuild-hide / agent-playbook-rebuild abstract
    methods.
    """

    @abstractmethod
    def apply_governance_user_data_delete(
        self, purge_id: str, user_id: str
    ) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def complete_purge_operation_with_audit(
        self, purge_id: str, audit_event: AuditEvent
    ) -> PurgeOperation:
        raise NotImplementedError
