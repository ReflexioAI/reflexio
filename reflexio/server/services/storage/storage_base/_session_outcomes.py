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

    def __post_init__(self) -> None:
        identity = (
            self.outcome_id,
            self.outcome_revision,
            self.outcome_contract_digest,
            self.finalized_trajectory_digest,
        )
        if any(value is None for value in identity) and not all(
            value is None for value in identity
        ):
            raise ValueError(
                "outcome identity fields must be all populated or all null"
            )


@dataclass(frozen=True)
class SessionOutcomeContext:
    """Writer-visible state used to validate one outcome attempt."""

    user_id: str | None = None
    source: str | None = None
    first_request_at: int | None = None
    existing: bool = False
    user_contract_violation: bool = False
    source_contract_violation: bool = False
    #: Whether the row that already exists was INFERRED by the offline tuner
    #: rather than reported by the customer. Only such a row may be rewritten
    #: -- displaced by a customer's report, or refreshed by the tuner itself --
    #: and only a rewritable row needs the validity checks re-run: for every
    #: other existing row the sole legal write is a byte-exact retry, which has
    #: nothing left to validate. Meaningless when ``existing`` is False.
    existing_is_inferred: bool = False


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
        is_inferred: bool = False,
    ) -> SessionOutcomeWriteResult:
        """Record one outcome.

        ``is_inferred`` marks a write the offline tuner produced from a judge
        verdict rather than a customer's own report. It is NOT on
        ``SetSessionOutcomeRequest`` on purpose: that model is the public
        request body, so a field on it would let a caller declare their own
        outcome displaceable. Only an internal caller can pass this.

        Over an existing INFERRED row it also selects which rewrite happens: a
        customer's write (``False``) displaces the row, archiving it and
        advancing ``outcome_revision``; the tuner's own write (``True``)
        refreshes it in place, archiving nothing and moving no
        customer-visible field. A byte-exact repeat of either is an idempotent
        no-op rather than a rewrite. Over a CUSTOMER's row neither is allowed.
        """
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
