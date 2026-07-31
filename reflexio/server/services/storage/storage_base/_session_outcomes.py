"""Session outcome storage contract."""

from abc import abstractmethod
from dataclasses import dataclass

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    SessionOutcomeFailureReason,
    SessionOutcomeRecord,
    SetSessionOutcomeRequest,
)


@dataclass(frozen=True)
class SessionOutcomeWriteResult:
    recorded: bool
    user_id: str | None = None
    source: str | None = None
    reason: SessionOutcomeFailureReason | None = None
    context_changed: bool = False
    outcome_id: str | None = None
    outcome_revision: int | None = None
    outcome_contract_digest: str | None = None
    finalized_trajectory_digest: str | None = None


@dataclass(frozen=True)
class SessionOutcomeContext:
    """Writer-visible state used to validate one outcome attempt."""

    user_id: str | None = None
    source: str | None = None
    first_request_at: int | None = None
    existing: bool = False
    user_contract_violation: bool = False
    source_contract_violation: bool = False


class SessionOutcomeStoreMixin:
    @abstractmethod
    def get_session_outcome_context(self, session_id: str) -> SessionOutcomeContext:
        """Resolve a durable duplicate or the canonical first request on the writer."""
        raise NotImplementedError

    @abstractmethod
    def record_session_outcome(
        self,
        request: SetSessionOutcomeRequest,
        *,
        created_at: int,
        expected_context: SessionOutcomeContext,
    ) -> SessionOutcomeWriteResult:
        raise NotImplementedError

    @abstractmethod
    def get_session_outcomes(
        self, request: GetSessionOutcomesRequest
    ) -> list[SessionOutcomeRecord]:
        raise NotImplementedError

    @abstractmethod
    def clear_session_outcomes_for_user(self, user_id: str) -> dict[str, int]:
        """Delete subject-owned outcomes."""
        raise NotImplementedError
