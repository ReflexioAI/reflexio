from hashlib import sha256

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.domain.entities import (
    GetSessionOutcomesResponse,
    SessionOutcomeRecord,
)
from reflexio.models.api_schema.domain.enums import SessionOutcomeKind
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_json_bytes,
    outcome_contract_digest,
    trajectory_digest,
)


def _outcome_contract_digest(**changes: object) -> str:
    payload: dict[str, object] = {
        "source": "customer_webhook",
        "schema_version": 1,
        "allowed_values": ("success", "failure", "unknown"),
        "finalization_rule": "first_write",
    }
    payload.update(changes)
    return outcome_contract_digest(**payload)  # type: ignore[arg-type]


def test_canonical_json_bytes_ignores_object_key_order() -> None:
    assert canonical_json_bytes({"b": [2, {"d": 4, "c": 3}], "a": 1}) == (
        b'{"a":1,"b":[2,{"c":3,"d":4}]}'
    )


def test_outcome_contract_digest_changes_when_source_changes() -> None:
    assert _outcome_contract_digest(
        source="customer_webhook"
    ) != _outcome_contract_digest(source="customer_batch")


def test_outcome_contract_digest_changes_when_schema_version_changes() -> None:
    assert _outcome_contract_digest(schema_version=1) != _outcome_contract_digest(
        schema_version=2
    )


def test_outcome_contract_digest_normalizes_allowed_value_order() -> None:
    assert _outcome_contract_digest(
        allowed_values=("success", "failure", "unknown")
    ) == _outcome_contract_digest(allowed_values=("unknown", "success", "failure"))


def test_outcome_contract_digest_changes_when_allowed_values_change() -> None:
    assert _outcome_contract_digest(
        allowed_values=("success", "failure", "unknown")
    ) != _outcome_contract_digest(allowed_values=("success", "failure"))


def test_outcome_contract_digest_changes_when_finalization_rule_changes() -> None:
    assert _outcome_contract_digest(
        finalization_rule="first_write"
    ) != _outcome_contract_digest(finalization_rule="replaceable")


def test_trajectory_digest_changes_when_trajectory_data_changes() -> None:
    assert trajectory_digest({"messages": [{"role": "user", "content": "one"}]}) != (
        trajectory_digest({"messages": [{"role": "user", "content": "two"}]})
    )


def test_session_outcome_record_accepts_unknown_and_serializes_identities() -> None:
    record = SessionOutcomeRecord(
        outcome_id="outcome-1",
        outcome_revision=1,
        user_id="user-1",
        session_id="session-1",
        outcome=SessionOutcomeKind.UNKNOWN,
        occurred_at=1,
        source="customer_webhook",
        outcome_contract_digest="a" * 64,
        finalized_trajectory_digest="b" * 64,
        created_at=2,
    )

    response = GetSessionOutcomesResponse(success=True, session_outcomes=[record])

    assert response.model_dump(mode="json")["session_outcomes"] == [
        {
            "outcome_id": "outcome-1",
            "outcome_revision": 1,
            "user_id": "user-1",
            "session_id": "session-1",
            "outcome": "unknown",
            "occurred_at": 1,
            "source": "customer_webhook",
            "label": None,
            "value": None,
            "metadata": None,
            "outcome_contract_digest": "a" * 64,
            "finalized_trajectory_digest": "b" * 64,
            "created_at": 2,
        }
    ]


@pytest.mark.parametrize(
    "field_name", ["outcome_contract_digest", "finalized_trajectory_digest"]
)
@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "not-a-digest"])
def test_session_outcome_record_rejects_non_sha256_identity_digests(
    field_name: str, digest: str
) -> None:
    payload: dict[str, object] = {
        "outcome_id": "outcome-1",
        "outcome_revision": 1,
        "user_id": "user-1",
        "session_id": "session-1",
        "outcome": "success",
        "occurred_at": 1,
        "source": "customer_webhook",
        "outcome_contract_digest": "a" * 64,
        "finalized_trajectory_digest": "b" * 64,
        "created_at": 2,
    }
    payload[field_name] = digest

    with pytest.raises(ValidationError, match="lowercase SHA-256 hex"):
        SessionOutcomeRecord(**payload)  # type: ignore[arg-type]


def test_trajectory_digest_matches_sha256_of_canonical_json() -> None:
    assert trajectory_digest({"b": 2, "a": 1}) == sha256(b'{"a":1,"b":2}').hexdigest()
