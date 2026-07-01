from __future__ import annotations

from abc import ABC, abstractmethod

from reflexio.models.api_schema.domain.governance import AuditEvent
from reflexio.models.config_schema import GovernanceRetentionConfig


class AuditEventStoreMixin(ABC):
    """Backend-neutral audit-event store contract.

    Extracted verbatim from ``_governance.py`` (the AuditEventStore bucket). The
    residual ``GovernanceMixin`` ABC stays composed alongside this mixin and
    holds the remaining purge / barrier abstract methods.
    """

    @abstractmethod
    def append_audit_event(self, event: AuditEvent) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_audit_events(
        self, subject_ref: str | None = None, *, org_id: str | None = None
    ) -> list[AuditEvent]:
        raise NotImplementedError

    @abstractmethod
    def gc_governance_retention(self, *, config: GovernanceRetentionConfig) -> int:
        raise NotImplementedError
