from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from reflexio.models.api_schema.domain.governance import (
    PurgeOperation,
    PurgeOperationTarget,
)
from reflexio.server.services.storage.governance_claims import PurgeExecutionClaim


class PurgeOperationStoreMixin(ABC):
    """Backend-neutral purge-operation store contract.

    Extracted verbatim from ``_governance.py`` (the PurgeOperationStore bucket).
    The residual ``GovernanceMixin`` ABC stays composed alongside this mixin and
    holds the remaining rebuild-hide / governance-erase-execution / barrier
    abstract methods.
    """

    @abstractmethod
    def begin_purge_operation(
        self,
        purge_id: str,
        idempotency_key: str,
        operation_type: Literal["user_erasure", "org_purge"],
        scope_type: Literal["user", "org"],
        subject_ref: str | None,
        request_ref: str,
        authoritative_user_id: str | None = None,
    ) -> PurgeOperation:
        raise NotImplementedError

    @abstractmethod
    def claim_purge_operation_execution(
        self,
        purge_id: str,
        *,
        lease_owner: str,
        lease_ttl_seconds: int,
    ) -> PurgeExecutionClaim | None:
        """Atomically claim or take over a stale purge execution."""
        raise NotImplementedError

    @abstractmethod
    def assert_purge_operation_execution_claim(
        self, purge_id: str, execution_claim: PurgeExecutionClaim
    ) -> None:
        """Raise when the purge execution claim no longer owns the live fence."""
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
        execution_claim: PurgeExecutionClaim | None = None,
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
    def fail_purge_operation(
        self, purge_id: str, error_code: str, error_detail: str
    ) -> PurgeOperation:
        raise NotImplementedError

    @abstractmethod
    def get_purge_operation(self, purge_id: str) -> PurgeOperation:
        raise NotImplementedError
