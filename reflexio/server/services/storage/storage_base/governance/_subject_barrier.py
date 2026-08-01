from __future__ import annotations

from abc import ABC, abstractmethod

from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    PurgeOperation,
    SubjectWriteBarrier,
)
from reflexio.server.services.storage.governance_claims import PurgeExecutionClaim


class SubjectBarrierMixin(ABC):
    """Backend-neutral subject-erasure-barrier store contract.

    Extracted verbatim from ``_governance.py`` (the SubjectBarrier bucket). The
    residual ``GovernanceMixin`` ABC stays composed alongside this mixin and
    holds the remaining rebuild-hide / governance-erase-execution / purge-complete
    abstract methods.
    """

    @abstractmethod
    def begin_subject_erasure_barrier(
        self,
        subject_ref: str,
        purge_id: str,
        execution_claim: PurgeExecutionClaim | None = None,
    ) -> SubjectWriteBarrier:
        raise NotImplementedError

    @abstractmethod
    def assert_subject_writable(self, subject_ref: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def complete_subject_erasure_barrier_after_empty_check(
        self,
        purge_id: str,
        audit_event: AuditEvent,
        execution_claim: PurgeExecutionClaim | None = None,
    ) -> PurgeOperation:
        raise NotImplementedError

    @abstractmethod
    def fail_subject_erasure_barrier(
        self,
        subject_ref: str,
        purge_id: str,
        error_code: str,
        error_detail: str,
        execution_claim: PurgeExecutionClaim | None = None,
    ) -> SubjectWriteBarrier:
        raise NotImplementedError

    @abstractmethod
    def get_subject_write_barrier(self, subject_ref: str) -> SubjectWriteBarrier | None:
        raise NotImplementedError
