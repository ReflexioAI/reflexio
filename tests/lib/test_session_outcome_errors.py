"""Publisher failures distinguish programming defects from unavailable storage."""

from unittest.mock import MagicMock

import pytest

from reflexio.lib._session_outcome import SessionOutcomeMixin
from reflexio.models.api_schema.domain import (
    SessionOutcomeFailureReason,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.storage_base import SessionOutcomeContext


@pytest.mark.parametrize(
    "failure",
    [
        TypeError("narrowed override"),
        AttributeError("missing capability"),
        StorageError("offline"),
    ],
)
@pytest.mark.parametrize("with_cutover", [False, True])
@pytest.mark.parametrize(
    "operation", ["get_session_outcome_context", "record_session_outcome"]
)
def test_outcome_facade_surfaces_programming_errors(failure, operation, with_cutover):
    facade = object.__new__(SessionOutcomeMixin)
    facade.request_context = MagicMock()
    facade.request_context.org_id = "test"
    storage = facade.request_context.storage
    storage.get_session_outcome_context.return_value = SessionOutcomeContext(
        user_id="user", source="source", first_request_at=1
    )
    getattr(storage, operation).side_effect = failure
    request = SetSessionOutcomeRequest(
        session_id="session", outcome=SessionOutcomeKind.SUCCESS, occurred_at=2
    )
    kwargs = {"trajectory_through_request_id": "r2"} if with_cutover else {}
    if isinstance(failure, StorageError):
        response = facade.mark_session_outcome(request, **kwargs)
        assert response.success is False
        assert response.reason == SessionOutcomeFailureReason.STORAGE_ERROR
    else:
        with pytest.raises(type(failure), match=str(failure)):
            facade.mark_session_outcome(request, **kwargs)
