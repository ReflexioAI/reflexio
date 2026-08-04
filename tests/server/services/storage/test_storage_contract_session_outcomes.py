"""Session outcome storage contract."""

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    Request,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.storage_base import BaseStorage


def test_first_write_preserves_outcome_fields(storage: BaseStorage) -> None:
    storage.add_request(
        Request(
            request_id="r1",
            user_id="u1",
            session_id="s1",
            source="published",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="s1",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
        label="booked",
        value=12.0,
        metadata={"crm": "test"},
    )
    context = storage.get_session_outcome_context("s1")
    first = storage.record_session_outcome(
        request, created_at=102, expected_context=context
    )
    duplicate = storage.record_session_outcome(
        request.model_copy(update={"outcome": "failure"}),
        created_at=103,
        expected_context=context,
    )

    assert first.recorded is True
    assert duplicate.recorded is False
    assert duplicate.source == "published"
    records = storage.get_session_outcomes(GetSessionOutcomesRequest(label="booked"))
    assert len(records) == 1
    assert records[0].outcome == SessionOutcomeKind.SUCCESS
    assert records[0].value == 12.0
    assert records[0].metadata == {"crm": "test"}


def test_unknown_session_is_rejected(storage: BaseStorage) -> None:
    context = storage.get_session_outcome_context("missing")
    result = storage.record_session_outcome(
        SetSessionOutcomeRequest(
            session_id="missing",
            outcome=SessionOutcomeKind.FAILURE,
            occurred_at=1,
        ),
        created_at=2,
        expected_context=context,
    )
    assert result.recorded is False
    assert result.reason == "unknown_session"


def test_stale_first_request_context_cannot_commit(storage: BaseStorage) -> None:
    storage.add_request(
        Request(
            request_id="later",
            user_id="u1",
            session_id="race",
            source="later-source",
            created_at=200,
        )
    )
    stale = storage.get_session_outcome_context("race")
    storage.add_request(
        Request(
            request_id="earlier",
            user_id="u1",
            session_id="race",
            source="canonical-source",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="race", outcome=SessionOutcomeKind.FAILURE, occurred_at=201
    )

    stale_result = storage.record_session_outcome(
        request, created_at=202, expected_context=stale
    )
    fresh = storage.get_session_outcome_context("race")
    fresh_result = storage.record_session_outcome(
        request, created_at=202, expected_context=fresh
    )

    assert stale_result.context_changed is True
    assert fresh_result.recorded is True
    assert fresh_result.source == "canonical-source"


def test_empty_request_source_is_preserved(storage: BaseStorage) -> None:
    storage.add_request(
        Request(
            request_id="empty-source-r1",
            user_id="u1",
            session_id="empty-source",
            source="",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="empty-source",
        outcome=SessionOutcomeKind.FAILURE,
        occurred_at=101,
    )
    result = storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context("empty-source"),
    )

    assert result.source == ""
    [record] = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["empty-source"])
    )
    assert record.source == ""


def test_clear_outcomes_survives_governance_secret_rotation(
    storage: BaseStorage, monkeypatch
) -> None:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "old-secret")
    storage.add_request(
        Request(
            request_id="rotated-r1",
            user_id="rotated-user",
            session_id="rotated-session",
            source="test",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="rotated-session",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
    )
    storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context("rotated-session"),
    )

    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "new-secret")
    counts = storage.clear_session_outcomes_for_user("rotated-user")

    assert counts == {"session_outcomes": 1}
    assert storage.get_session_outcomes(GetSessionOutcomesRequest()) == []
