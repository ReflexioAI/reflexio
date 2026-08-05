from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from itertools import combinations
from typing import Any, Protocol

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.domain.entities import (
    GetSessionOutcomesRequest,
    GetSessionOutcomesResponse,
    InteractionData,
    ManualPlaybookGenerationRequest,
    ManualProfileGenerationRequest,
    PublishUserInteractionRequest,
    Request,
    RerunPlaybookGenerationRequest,
    RerunProfileGenerationRequest,
    SessionOutcomeRecord,
    SetSessionOutcomeResponse,
)
from reflexio.models.api_schema.domain.enums import SessionOutcomeKind
from reflexio.models.api_schema.retriever_schema import (
    GetRequestsRequest,
    GetUserProfilesRequest,
    SearchUserProfileRequest,
)
from reflexio.server.services.storage import session_outcome_identity
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_json_bytes,
    canonical_session_trajectory,
    outcome_contract_digest,
    trajectory_digest,
)
from reflexio.server.services.storage.storage_base import SessionOutcomeWriteResult


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


class _HasSource(Protocol):
    @property
    def source(self) -> str | None: ...


_OUTCOME_SOURCE_MODEL_FACTORIES: tuple[Callable[[str], _HasSource], ...] = (
    lambda source: Request(
        request_id="request-1",
        user_id="user-1",
        session_id="session-1",
        source=source,
    ),
    lambda source: PublishUserInteractionRequest(
        user_id="user-1",
        session_id="session-1",
        interaction_data_list=[InteractionData(content="hello")],
        source=source,
    ),
    lambda source: SessionOutcomeRecord(
        user_id="user-1",
        session_id="session-1",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=1,
        source=source,
        created_at=2,
    ),
    lambda source: SetSessionOutcomeResponse(success=True, source=source),
    lambda source: GetSessionOutcomesRequest(source=source),
    lambda source: GetRequestsRequest(source=source),
    lambda source: SearchUserProfileRequest(user_id="user-1", source=source),
    lambda source: GetUserProfilesRequest(user_id="user-1", source=source),
    lambda source: RerunProfileGenerationRequest(source=source),
    lambda source: ManualProfileGenerationRequest(source=source),
    lambda source: ManualPlaybookGenerationRequest(source=source),
    lambda source: RerunPlaybookGenerationRequest(source=source),
)
_OUTCOME_SOURCE_MODEL_IDS = (
    "request",
    "publish",
    "record",
    "set-response",
    "get-outcomes",
    "get-requests",
    "search-profiles",
    "get-profiles",
    "rerun-profiles",
    "manual-profiles",
    "manual-playbooks",
    "rerun-playbooks",
)


@pytest.mark.parametrize(
    "factory", _OUTCOME_SOURCE_MODEL_FACTORIES, ids=_OUTCOME_SOURCE_MODEL_IDS
)
def test_outcome_source_models_accept_machine_label(
    factory: Callable[[str], _HasSource],
) -> None:
    assert factory("support-agent:v2").source == "support-agent:v2"
    assert factory("a" * 128).source == "a" * 128


@pytest.mark.parametrize(
    "factory", _OUTCOME_SOURCE_MODEL_FACTORIES, ids=_OUTCOME_SOURCE_MODEL_IDS
)
@pytest.mark.parametrize(
    "source",
    [
        "alice@example.com",
        "https://example.com/hook",
        "support agent",
        " support-agent",
        "support/agent",
        "Support-Agent",
        "support-agént",
        "a" * 129,
    ],
)
def test_outcome_source_models_reject_sensitive_or_free_form_values(
    factory: Callable[[str], _HasSource],
    source: str,
) -> None:
    with pytest.raises(ValidationError):
        factory(source)


@pytest.mark.parametrize(
    "factory", _OUTCOME_SOURCE_MODEL_FACTORIES, ids=_OUTCOME_SOURCE_MODEL_IDS
)
def test_outcome_source_models_preserve_empty_source(
    factory: Callable[[str], _HasSource],
) -> None:
    assert factory("").source == ""


def test_optional_outcome_source_models_preserve_absence() -> None:
    assert SetSessionOutcomeResponse(success=False).source is None
    assert GetSessionOutcomesRequest().source is None
    assert GetRequestsRequest().source is None
    assert SearchUserProfileRequest(user_id="user-1").source is None
    assert GetUserProfilesRequest(user_id="user-1").source is None
    assert RerunProfileGenerationRequest().source is None
    assert ManualProfileGenerationRequest().source is None
    assert ManualPlaybookGenerationRequest().source is None
    assert RerunPlaybookGenerationRequest().source is None


def test_outcome_source_json_schema_exposes_exact_constraints() -> None:
    source_schema = PublishUserInteractionRequest.model_json_schema()["properties"][
        "source"
    ]

    assert {
        "maxLength": 128,
        "pattern": "^[a-z0-9][a-z0-9._:-]{0,127}$",
        "type": "string",
    } in source_schema["anyOf"]


def test_outcome_contract_digest_is_stable_for_valid_machine_label() -> None:
    payload = {
        "allowed_values": ["failure", "success", "unknown"],
        "finalization_rule": "first_write",
        "schema_version": 1,
        "source": "support-agent:v2",
    }

    assert (
        _outcome_contract_digest(source="support-agent:v2")
        == sha256(canonical_json_bytes(payload)).hexdigest()
    )


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


def test_canonical_session_trajectory_normalizes_sqlite_and_postgres_rows() -> None:
    sqlite_request = {
        "request_id": "parity-request",
        "user_id": "parity-user",
        "created_at": "2023-11-14T22:13:20+00:00",
        "source": "parity-source",
        "agent_version": "parity-agent",
        "session_id": "parity-session",
        "evaluation_only": 0,
        "retrieval_experiment_id": None,
        "retrieval_experiment_arm": None,
    }
    postgres_request = {
        **sqlite_request,
        "created_at": datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC),
        "evaluation_only": False,
    }
    sqlite_interaction = {
        "interaction_id": 4242,
        "user_id": "parity-user",
        "request_id": "parity-request",
        "created_at": "2023-11-14T22:13:21+00:00",
        "content": "Parity trajectory",
        "role": "User",
        "token_count": 3,
        "user_action": "none",
        "user_action_description": "",
        "interacted_image_url": "",
        "image_encoding": "",
        "shadow_content": "",
        "expert_content": "",
        "tools_used": '[{"tool_data":{"confidence":0.75},"tool_name":"rank"}]',
        "citations": "[]",
        "retrieved_learnings": "[]",
    }
    postgres_interaction = {
        **sqlite_interaction,
        "created_at": datetime(2023, 11, 14, 22, 13, 21, tzinfo=UTC),
        "tools_used": [{"tool_name": "rank", "tool_data": {"confidence": 0.75}}],
        "citations": [],
        "retrieved_learnings": [],
    }

    sqlite_projection = canonical_session_trajectory(
        "parity-session",
        [sqlite_request],
        {"parity-request": [sqlite_interaction]},
    )
    postgres_projection = canonical_session_trajectory(
        "parity-session",
        [postgres_request],
        {"parity-request": [postgres_interaction]},
    )

    assert sqlite_projection == postgres_projection
    assert sqlite_projection["requests"][0]["request"]["evaluation_only"] is False
    assert trajectory_digest(sqlite_projection) == (
        "73d0f738bb5a3c7668787c678230c4758b68923e12a16e3851b401db780e4272"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_trajectory_digest_rejects_non_finite_nested_floats(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        trajectory_digest({"nested": [{"value": value}]})


@pytest.mark.parametrize(
    "container_factory",
    [
        lambda value: {"nested": value},
        lambda value: [value],
        lambda value: (value,),
    ],
    ids=["mapping", "list", "tuple"],
)
def test_trajectory_digest_accepts_maximum_canonical_json_depth(
    container_factory,
) -> None:
    value: object = "leaf"
    for _ in range(session_outcome_identity.MAX_CANONICAL_TRAJECTORY_JSON_DEPTH):
        value = container_factory(value)

    assert trajectory_digest(value)


@pytest.mark.parametrize(
    "container_factory",
    [
        lambda value: {"nested": value},
        lambda value: [value],
        lambda value: (value,),
    ],
    ids=["mapping", "list", "tuple"],
)
def test_trajectory_digest_rejects_over_maximum_canonical_json_depth(
    container_factory,
) -> None:
    value: object = "leaf"
    for _ in range(session_outcome_identity.MAX_CANONICAL_TRAJECTORY_JSON_DEPTH + 1):
        value = container_factory(value)

    with pytest.raises(
        ValueError, match="canonical trajectory JSON exceeds maximum depth"
    ):
        trajectory_digest(value)


@pytest.mark.parametrize(
    ("empty_container", "container_factory"),
    [
        ({}, lambda value: {"nested": value}),
        ([], lambda value: [value]),
        ((), lambda value: (value,)),
    ],
    ids=["mapping", "list", "tuple"],
)
def test_trajectory_digest_accepts_maximum_empty_container_depth(
    empty_container,
    container_factory,
) -> None:
    value: object = empty_container
    for _ in range(session_outcome_identity.MAX_CANONICAL_TRAJECTORY_JSON_DEPTH - 1):
        value = container_factory(value)

    assert trajectory_digest(value)


@pytest.mark.parametrize(
    ("empty_container", "container_factory"),
    [
        ({}, lambda value: {"nested": value}),
        ([], lambda value: [value]),
        ((), lambda value: (value,)),
    ],
    ids=["mapping", "list", "tuple"],
)
def test_trajectory_digest_rejects_over_maximum_empty_container_depth(
    empty_container,
    container_factory,
) -> None:
    value: object = empty_container
    for _ in range(session_outcome_identity.MAX_CANONICAL_TRAJECTORY_JSON_DEPTH):
        value = container_factory(value)

    with pytest.raises(
        ValueError, match="canonical trajectory JSON exceeds maximum depth"
    ):
        trajectory_digest(value)


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


def test_session_outcome_record_accepts_legacy_all_null_identity() -> None:
    record = SessionOutcomeRecord(
        outcome_id=None,
        outcome_revision=None,
        user_id="legacy-user",
        session_id="legacy-session",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=1,
        source="customer_webhook",
        outcome_contract_digest=None,
        finalized_trajectory_digest=None,
        created_at=2,
    )

    assert record.outcome_id is None
    assert record.outcome_revision is None
    assert record.outcome_contract_digest is None
    assert record.finalized_trajectory_digest is None


def test_session_outcome_record_rejects_partial_legacy_identity() -> None:
    with pytest.raises(ValidationError, match="all populated or all null"):
        SessionOutcomeRecord(
            outcome_id="outcome-1",
            outcome_revision=None,
            user_id="legacy-user",
            session_id="legacy-session",
            outcome=SessionOutcomeKind.SUCCESS,
            occurred_at=1,
            source="customer_webhook",
            outcome_contract_digest=None,
            finalized_trajectory_digest=None,
            created_at=2,
        )


@pytest.mark.parametrize(
    "populated_fields",
    [
        fields
        for populated_count in range(1, 4)
        for fields in combinations(
            (
                "outcome_id",
                "outcome_revision",
                "outcome_contract_digest",
                "finalized_trajectory_digest",
            ),
            populated_count,
        )
    ],
)
def test_session_outcome_write_result_rejects_partial_identity(
    populated_fields: tuple[str, ...],
) -> None:
    identity = {
        "outcome_id": "outcome-1",
        "outcome_revision": 1,
        "outcome_contract_digest": "a" * 64,
        "finalized_trajectory_digest": "b" * 64,
    }

    with pytest.raises(ValueError, match="all populated or all null"):
        SessionOutcomeWriteResult(
            recorded=False,
            **{
                field_name: value
                for field_name, value in identity.items()
                if field_name in populated_fields
            },
        )


@pytest.mark.parametrize(
    "populated_fields",
    [
        fields
        for populated_count in range(1, 4)
        for fields in combinations(
            (
                "outcome_id",
                "outcome_revision",
                "outcome_contract_digest",
                "finalized_trajectory_digest",
            ),
            populated_count,
        )
    ],
)
def test_set_session_outcome_response_rejects_partial_identity(
    populated_fields: tuple[str, ...],
) -> None:
    identity = {
        "outcome_id": "outcome-1",
        "outcome_revision": 1,
        "outcome_contract_digest": "a" * 64,
        "finalized_trajectory_digest": "b" * 64,
    }

    with pytest.raises(ValidationError, match="all populated or all null"):
        SetSessionOutcomeResponse(
            success=True,
            **{
                field_name: value
                for field_name, value in identity.items()
                if field_name in populated_fields
            },
        )


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {
            "outcome_id": "outcome-1",
            "outcome_revision": 1,
            "outcome_contract_digest": "a" * 64,
            "finalized_trajectory_digest": "b" * 64,
        },
    ],
)
def test_write_result_and_response_accept_complete_identity_shapes(
    identity: dict[str, Any],
) -> None:
    write_result = SessionOutcomeWriteResult(recorded=False, **identity)
    response = SetSessionOutcomeResponse(success=True, **identity)

    assert write_result.outcome_id == response.outcome_id


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
