"""Facade handling of the internal cutover keywords on ``mark_session_outcome``."""

from reflexio.models.api_schema.domain import (
    SessionOutcomeFailureReason,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.storage_base._session_outcomes import (
    OutcomePrefixPrecondition,
    SessionOutcomeContext,
    SessionOutcomeWriteResult,
)

_REQUEST = SetSessionOutcomeRequest(
    session_id="s1", outcome=SessionOutcomeKind.SUCCESS, occurred_at=1_700_000_000
)


def _storage(reflexio_mock, result: SessionOutcomeWriteResult):
    storage = reflexio_mock.request_context.storage
    storage.get_session_outcome_context.return_value = SessionOutcomeContext(
        user_id="u1", source="published", first_request_at=1
    )
    storage.record_session_outcome.return_value = result
    return storage


def test_unset_keywords_never_reach_storage(reflexio_mock) -> None:
    storage = _storage(reflexio_mock, SessionOutcomeWriteResult(recorded=False))

    reflexio_mock.mark_session_outcome(_REQUEST, is_inferred=True)

    # A backend that predates the keywords must keep working.
    assert set(storage.record_session_outcome.call_args.kwargs) == {
        "created_at",
        "expected_context",
        "is_inferred",
    }


def test_set_keywords_are_forwarded(reflexio_mock) -> None:
    storage = _storage(reflexio_mock, SessionOutcomeWriteResult(recorded=False))
    precondition = OutcomePrefixPrecondition(verdict_settled_at=5, interaction_count=2)

    reflexio_mock.mark_session_outcome(
        _REQUEST,
        is_inferred=True,
        trajectory_through_request_id="r2",
        prefix_precondition=precondition,
    )

    kwargs = storage.record_session_outcome.call_args.kwargs
    assert kwargs["trajectory_through_request_id"] == "r2"
    assert kwargs["prefix_precondition"] is precondition


def test_prefix_moved_is_conflicting_finalization_without_retry(
    reflexio_mock,
) -> None:
    storage = _storage(
        reflexio_mock, SessionOutcomeWriteResult(recorded=False, prefix_moved=True)
    )

    response = reflexio_mock.mark_session_outcome(
        _REQUEST, is_inferred=True, trajectory_through_request_id="r2"
    )

    assert response.success is False
    assert response.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    storage.record_session_outcome.assert_called_once()
