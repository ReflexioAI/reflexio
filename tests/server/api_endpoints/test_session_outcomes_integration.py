"""HTTP contracts for caller-authored session outcomes."""

import inspect
import logging
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from reflexio.client.client import ReflexioClient
from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    Request,
    SessionOutcomeFailureReason,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.server.cache.reflexio_cache import get_reflexio


def test_source_is_derived_from_tiebroken_first_request(
    client_with_org: tuple[TestClient, str], caplog
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    storage.add_request(
        Request(
            request_id="z-later-tiebreak",
            user_id="u1",
            session_id="source-session",
            source="later-source",
            created_at=100,
        )
    )
    storage.add_request(
        Request(
            request_id="a-earliest-tiebreak",
            user_id="u1",
            session_id="source-session",
            source="canonical-source",
            created_at=100,
        )
    )
    with caplog.at_level(logging.WARNING, logger="reflexio.lib._session_outcome"):
        response = client.post(
            "/api/session_outcome",
            json={
                "session_id": "source-session",
                "outcome": "success",
                "occurred_at": 101,
                "source": "caller-cannot-override",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert {
        "success": True,
        "recorded": True,
        "message": "Outcome recorded",
        "user_id": "u1",
        "source": "canonical-source",
    }.items() <= body.items()
    assert body["outcome_id"]
    assert body["outcome_revision"] == 1
    assert len(body["outcome_contract_digest"]) == 64
    assert len(body["finalized_trajectory_digest"]) == 64
    assert "stripped unknown fields: source" in caplog.text
    assert "multiple sources for session source-session" in caplog.text

    read = client.post(
        "/api/get_session_outcomes", json={"session_ids": ["source-session"]}
    )
    assert read.status_code == 200
    assert read.json()["session_outcomes"][0]["source"] == "canonical-source"
    assert (
        "source"
        not in inspect.signature(ReflexioClient.mark_session_outcome).parameters
    )


def test_retry_survives_ordinary_session_deletion(
    client_with_org: tuple[TestClient, str],
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    storage.add_request(
        Request(
            request_id="delete-r1",
            user_id="u1",
            session_id="delete-session",
            source="published",
            created_at=100,
        )
    )
    payload = {
        "session_id": "delete-session",
        "outcome": "failure",
        "occurred_at": 101,
    }
    assert client.post("/api/session_outcome", json=payload).json()["recorded"] is True

    assert storage.delete_session("delete-session") == 1
    retry = client.post("/api/session_outcome", json=payload)

    assert retry.status_code == 200
    assert retry.json()["success"] is True
    assert retry.json()["recorded"] is False
    assert retry.json()["source"] == "published"


def test_conflicting_retry_is_not_accepted_only_by_session_identity(
    client_with_org: tuple[TestClient, str],
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    storage.add_request(
        Request(
            request_id="conflict-r1",
            user_id="u1",
            session_id="conflict-session",
            source="published",
            created_at=100,
        )
    )
    first = client.post(
        "/api/session_outcome",
        json={
            "session_id": "conflict-session",
            "outcome": "success",
            "occurred_at": 101,
        },
    )
    conflict = client.post(
        "/api/session_outcome",
        json={
            "session_id": "conflict-session",
            "outcome": "failure",
            "occurred_at": 101,
        },
    )

    assert first.status_code == 200
    assert first.json()["recorded"] is True
    assert conflict.status_code == 200
    assert conflict.json()["success"] is False
    assert conflict.json()["recorded"] is False
    assert conflict.json()["reason"] == "conflicting_finalization"


def test_retry_compares_metadata_as_json_values(
    client_with_org: tuple[TestClient, str],
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    storage.add_request(
        Request(
            request_id="semantic-metadata-r1",
            user_id="u1",
            session_id="semantic-metadata-session",
            source="published",
            created_at=100,
        )
    )
    payload = {
        "session_id": "semantic-metadata-session",
        "outcome": "success",
        "occurred_at": 101,
        "metadata": {"label": "same", "nested": {"one": 1, "two": 2}},
    }
    first = client.post("/api/session_outcome", json=payload)
    assert first.json()["recorded"] is True
    storage.conn.execute(  # type: ignore[attr-defined]
        "UPDATE session_outcomes SET metadata = ? WHERE session_id = ?",
        (
            '{ "nested": { "two": 2, "one": 1 }, "label": "same" }',
            "semantic-metadata-session",
        ),
    )
    storage.conn.commit()  # type: ignore[attr-defined]

    retry = client.post("/api/session_outcome", json=payload)

    assert retry.status_code == 200
    assert retry.json()["success"] is True
    assert retry.json()["recorded"] is False
    assert retry.json().get("reason") is None


def test_retry_rejects_metadata_with_different_json_value_types(
    client_with_org: tuple[TestClient, str],
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    storage.add_request(
        Request(
            request_id="typed-metadata-r1",
            user_id="u1",
            session_id="typed-metadata-session",
            source="published",
            created_at=100,
        )
    )
    first = client.post(
        "/api/session_outcome",
        json={
            "session_id": "typed-metadata-session",
            "outcome": "success",
            "occurred_at": 101,
            "metadata": {"nested": {"value": True}},
        },
    )
    assert first.json()["recorded"] is True

    retry = client.post(
        "/api/session_outcome",
        json={
            "session_id": "typed-metadata-session",
            "outcome": "success",
            "occurred_at": 101,
            "metadata": {"nested": {"value": 1}},
        },
    )

    assert retry.status_code == 200
    assert retry.json()["success"] is False
    assert retry.json()["recorded"] is False
    assert retry.json()["reason"] == "conflicting_finalization"


@pytest.mark.parametrize(
    "stored_metadata",
    [
        pytest.param("", id="empty"),
        pytest.param("{malformed", id="malformed"),
        pytest.param("[" * 10_000 + "]" * 10_000, id="pathological-nesting"),
    ],
)
def test_retry_rejects_invalid_stored_metadata(
    client_with_org: tuple[TestClient, str], stored_metadata: str
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    session_id = f"invalid-stored-metadata-{len(stored_metadata)}"
    storage.add_request(
        Request(
            request_id=f"{session_id}-r1",
            user_id="u1",
            session_id=session_id,
            source="published",
            created_at=100,
        )
    )
    payload = {
        "session_id": session_id,
        "outcome": "success",
        "occurred_at": 101,
    }
    first = client.post("/api/session_outcome", json=payload)
    assert first.json()["recorded"] is True
    storage.conn.execute(  # type: ignore[attr-defined]
        "UPDATE session_outcomes SET metadata = ? WHERE session_id = ?",
        (stored_metadata, session_id),
    )
    storage.conn.commit()  # type: ignore[attr-defined]

    retry = client.post("/api/session_outcome", json=payload)

    assert retry.status_code == 200
    assert retry.json()["success"] is False
    assert retry.json()["recorded"] is False
    assert retry.json()["reason"] == "conflicting_finalization"


def test_outcome_validation_boundaries(
    client_with_org: tuple[TestClient, str],
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    storage.add_request(
        Request(
            request_id="bounds-r1",
            user_id="u1",
            session_id="bounds-session",
            source="published",
            created_at=100,
        )
    )
    base = {"session_id": "bounds-session", "outcome": "failure"}

    before = client.post(
        "/api/session_outcome", json={**base, "occurred_at": 99}
    ).json()
    future = client.post(
        "/api/session_outcome",
        json={**base, "occurred_at": int(time.time()) + 86401},
    ).json()
    oversized = client.post(
        "/api/session_outcome",
        json={**base, "occurred_at": 101, "metadata": {"x": "a" * 16384}},
    )
    blank_label = client.post(
        "/api/session_outcome",
        json={**base, "occurred_at": 101, "label": "   "},
    )
    invalid_range = client.post(
        "/api/get_session_outcomes", json={"start_time": 2, "end_time": 1}
    )

    assert before["reason"] == "occurred_before_session"
    assert future["reason"] == "occurred_in_future"
    assert oversized.status_code == 422
    assert blank_label.status_code == 422
    assert invalid_range.status_code == 422

    for invalid_value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError, match="valid JSON values"):
            SetSessionOutcomeRequest(
                session_id="bounds-session",
                outcome=SessionOutcomeKind.FAILURE,
                occurred_at=101,
                metadata={"nested": {"invalid": invalid_value}},
            )


@pytest.mark.parametrize("invalid_token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_metadata_returns_422_without_persistence(
    client_with_org: tuple[TestClient, str], invalid_token: str
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    session_id = f"invalid-metadata-{invalid_token}"
    storage.add_request(
        Request(
            request_id=f"{session_id}-r1",
            user_id="u1",
            session_id=session_id,
            source="published",
            created_at=100,
        )
    )

    response = client.post(
        "/api/session_outcome",
        content=(
            '{"session_id":"'
            + session_id
            + '","outcome":"failure","occurred_at":101,'
            + '"metadata":{"nested":{"invalid":'
            + invalid_token
            + "}}}"
        ),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == (
        "Value error, metadata must contain only valid JSON values"
    )
    assert (
        storage.get_session_outcomes(
            GetSessionOutcomesRequest(session_ids=[session_id])
        )
        == []
    )


def test_outcome_warning_values_are_sanitized_and_bounded(
    client_with_org: tuple[TestClient, str], caplog
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    session_id = "unsafe\nsession"
    for request_id, source in (("first", "one"), ("second", "two")):
        storage.add_request(
            Request(
                request_id=request_id,
                user_id="u1",
                session_id=session_id,
                source=source,
                created_at=100,
            )
        )
    extras = {f"field-{index}\n{'x' * 100}": index for index in range(6)}

    with caplog.at_level(logging.WARNING, logger="reflexio.lib._session_outcome"):
        response = client.post(
            "/api/session_outcome",
            json={
                "session_id": session_id,
                "outcome": "success",
                "occurred_at": 101,
                **extras,
            },
        )

    assert response.status_code == 200
    messages = [record.getMessage() for record in caplog.records]
    unknown_warning = next(
        message for message in messages if "stripped unknown fields" in message
    )
    source_warning = next(
        message for message in messages if "multiple sources" in message
    )
    assert "\n" not in unknown_warning
    assert "+1 more" in unknown_warning
    assert len(unknown_warning) < 500
    assert source_warning.endswith("unsafe?session")


def test_acceptance_hook_runs_before_persistence_at_exact_deadline(
    client_with_org: tuple[TestClient, str], monkeypatch
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    for suffix in ("equal", "late"):
        storage.add_request(
            Request(
                request_id=f"deadline-{suffix}-r1",
                user_id="u1",
                session_id=f"deadline-{suffix}",
                source="published",
                created_at=100,
            )
        )

    def provider(_org_id, _request, received_at, _user_id):
        return (
            SessionOutcomeFailureReason.AFTER_OUTCOME_WINDOW
            if received_at > 200
            else None
        )

    monkeypatch.setattr(
        "reflexio.lib._session_outcome.get_service", lambda _key: provider
    )
    monkeypatch.setattr("reflexio.lib._session_outcome.time", lambda: 200)
    equal = client.post(
        "/api/session_outcome",
        json={"session_id": "deadline-equal", "outcome": "success", "occurred_at": 150},
    )
    monkeypatch.setattr("reflexio.lib._session_outcome.time", lambda: 201)
    late = client.post(
        "/api/session_outcome",
        json={"session_id": "deadline-late", "outcome": "success", "occurred_at": 150},
    )

    assert equal.json()["recorded"] is True
    assert late.json()["reason"] == "after_outcome_window"
    assert (
        storage.get_session_outcomes(
            GetSessionOutcomesRequest(session_ids=["deadline-late"])
        )
        == []
    )


def test_erased_session_without_mapping_is_unknown(
    client_with_org: tuple[TestClient, str],
) -> None:
    client, org_id = client_with_org
    storage = get_reflexio(org_id=org_id).get_storage()
    storage.add_request(
        Request(
            request_id="erase-r1",
            user_id="erased-user",
            session_id="erased-session",
            source="published",
            created_at=100,
        )
    )
    recorded = client.post(
        "/api/session_outcome",
        json={
            "session_id": "erased-session",
            "outcome": "success",
            "occurred_at": 101,
        },
    )
    assert recorded.json()["recorded"] is True
    storage.clear_user_data("erased-user")

    response = client.post(
        "/api/session_outcome",
        json={
            "session_id": "erased-session",
            "outcome": "failure",
            "occurred_at": 101,
        },
    )

    assert response.status_code == 200
    assert response.json()["reason"] == "unknown_session"
    assert (
        storage.conn.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) FROM session_outcomes WHERE session_id = ?",
            ("erased-session",),
        ).fetchone()[0]
        == 0
    )
