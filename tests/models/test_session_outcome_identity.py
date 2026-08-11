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
    CanonicalTrajectoryDigestAccumulator,
    canonical_json_bytes,
    canonical_session_trajectory,
    canonical_trajectory_bytes,
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


_STRICT_OUTCOME_SOURCE_INPUT_FACTORIES: tuple[Callable[[str], _HasSource], ...] = (
    lambda source: PublishUserInteractionRequest(
        user_id="user-1",
        session_id="session-1",
        interaction_data_list=[InteractionData(content="hello")],
        source=source,
    ),
    lambda source: GetSessionOutcomesRequest(source=source),
    lambda source: GetRequestsRequest(source=source),
    lambda source: SearchUserProfileRequest(user_id="user-1", source=source),
    lambda source: GetUserProfilesRequest(user_id="user-1", source=source),
    lambda source: RerunProfileGenerationRequest(source=source),
    lambda source: ManualProfileGenerationRequest(source=source),
    lambda source: ManualPlaybookGenerationRequest(source=source),
    lambda source: RerunPlaybookGenerationRequest(source=source),
)
_STRICT_OUTCOME_SOURCE_INPUT_IDS = (
    "publish",
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
    "factory",
    _STRICT_OUTCOME_SOURCE_INPUT_FACTORIES,
    ids=_STRICT_OUTCOME_SOURCE_INPUT_IDS,
)
def test_outcome_source_inputs_accept_machine_label(
    factory: Callable[[str], _HasSource],
) -> None:
    assert factory("support-agent:v2").source == "support-agent:v2"
    assert factory("a" * 128).source == "a" * 128


@pytest.mark.parametrize(
    "factory",
    _STRICT_OUTCOME_SOURCE_INPUT_FACTORIES,
    ids=_STRICT_OUTCOME_SOURCE_INPUT_IDS,
)
@pytest.mark.parametrize(
    "source",
    [
        "Legacy Source",
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
def test_outcome_source_inputs_reject_sensitive_or_free_form_values(
    factory: Callable[[str], _HasSource],
    source: str,
) -> None:
    with pytest.raises(ValidationError):
        factory(source)


@pytest.mark.parametrize(
    "factory",
    _STRICT_OUTCOME_SOURCE_INPUT_FACTORIES,
    ids=_STRICT_OUTCOME_SOURCE_INPUT_IDS,
)
def test_outcome_source_inputs_preserve_empty_source(
    factory: Callable[[str], _HasSource],
) -> None:
    assert factory("").source == ""


_PERSISTED_OUTCOME_SOURCE_MODEL_FACTORIES: tuple[Callable[[str], _HasSource], ...] = (
    lambda source: Request(
        request_id="request-1",
        user_id="user-1",
        session_id="session-1",
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
)


@pytest.mark.parametrize(
    "factory",
    _PERSISTED_OUTCOME_SOURCE_MODEL_FACTORIES,
    ids=("request", "outcome", "finalization-response"),
)
@pytest.mark.parametrize(
    "source",
    ["Legacy Source", "legacy/source", "legacy-sourcé", "x" * 256],
)
def test_persisted_outcome_source_models_preserve_legacy_values(
    factory: Callable[[str], _HasSource],
    source: str,
) -> None:
    assert factory(source).source == source


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


def test_canonical_trajectory_bytes_returns_exact_utf8_representation() -> None:
    assert canonical_trajectory_bytes(
        {"value": 1.5, "message": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"}
    ) == (b'{"message":"caf\xc3\xa9","value":1.5}')


def test_canonical_trajectory_bytes_ignores_mapping_insertion_order() -> None:
    first = {"messages": [{"content": "hello", "role": "user"}], "session": "s1"}
    second = {"session": "s1", "messages": [{"role": "user", "content": "hello"}]}

    assert canonical_trajectory_bytes(first) == canonical_trajectory_bytes(second)


@pytest.mark.parametrize(
    "value",
    [
        {"value": float("nan")},
        {"value": 2**53},
        {"value": object()},
        {1: "non-string-key"},
        {"value": "\ud800"},
    ],
    ids=[
        "non-finite-float",
        "non-ijson-integer",
        "unsupported-object",
        "key",
        "surrogate",
    ],
)
def test_canonical_trajectory_bytes_rejects_the_same_values_as_digest(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)) as digest_error:
        trajectory_digest(value)

    with pytest.raises(type(digest_error.value)) as bytes_error:
        canonical_trajectory_bytes(value)

    assert str(bytes_error.value) == str(digest_error.value)


def test_trajectory_digest_hashes_canonical_trajectory_bytes() -> None:
    trajectory = {"messages": [{"role": "user", "content": "hello"}]}

    assert (
        trajectory_digest(trajectory)
        == sha256(canonical_trajectory_bytes(trajectory)).hexdigest()
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


def test_streaming_trajectory_digest_matches_materialized_multi_request_json() -> None:
    request_rows = [
        {
            "request_id": "request-b",
            "user_id": "user-1",
            "created_at": datetime(2026, 8, 10, 9, 0, tzinfo=UTC),
            "source": "workflow:v2",
            "agent_version": "agent-2",
            "session_id": "session-1",
            "evaluation_only": False,
            "retrieval_experiment_id": "experiment-1",
            "retrieval_experiment_arm": "treatment",
        },
        {
            "request_id": "request-a",
            "user_id": "user-1",
            "created_at": "2026-08-10T09:01:00+00:00",
            "source": "workflow:v2",
            "agent_version": "agent-2",
            "session_id": "session-1",
            "evaluation_only": True,
            "retrieval_experiment_id": None,
            "retrieval_experiment_arm": None,
        },
    ]
    interactions_by_request = {
        "request-b": [
            {
                "interaction_id": 11,
                "user_id": "user-1",
                "request_id": "request-b",
                "created_at": datetime(2026, 8, 10, 9, 0, 1, tzinfo=UTC),
                "content": "nested payload",
                "role": "User",
                "token_count": 3,
                "user_action": "none",
                "user_action_description": None,
                "interacted_image_url": "",
                "image_encoding": None,
                "shadow_content": "",
                "expert_content": None,
                "tools_used": '[{"input":{"z":1,"a":[true,null,{"x":"y"}]} }]',
                "citations": [{"metadata": {"beta": 2, "alpha": 1}}],
                "retrieved_learnings": {"items": [[1, 2], {"nested": [3.5]}]},
            },
            {
                "interaction_id": "12",
                "user_id": "user-1",
                "request_id": "request-b",
                "created_at": "2026-08-10T09:00:02+00:00",
                "content": "second",
                "role": "Assistant",
                "token_count": None,
                "user_action": "none",
                "user_action_description": "",
                "interacted_image_url": None,
                "image_encoding": "",
                "shadow_content": None,
                "expert_content": "",
                "tools_used": [],
                "citations": "[]",
                "retrieved_learnings": None,
            },
        ],
        "request-a": [],
    }
    materialized = canonical_session_trajectory(
        "session-1", request_rows, interactions_by_request
    )
    accumulator = CanonicalTrajectoryDigestAccumulator("session-1")

    for request_row in request_rows:
        accumulator.start_request(request_row)
        for interaction_row in interactions_by_request[request_row["request_id"]]:
            accumulator.add_interaction(interaction_row)
        accumulator.finish_request()

    assert accumulator.hexdigest() == trajectory_digest(materialized)


def test_streaming_trajectory_byte_count_matches_each_materialized_prefix() -> None:
    request_row = _streaming_request_row()
    interaction_row = _streaming_interaction_row()
    accumulator = CanonicalTrajectoryDigestAccumulator("session-1")

    assert accumulator.byte_count_if_finalized() == len(
        canonical_trajectory_bytes(canonical_session_trajectory("session-1", [], {}))
    )

    accumulator.start_request(request_row)
    assert accumulator.byte_count_if_finalized() == len(
        canonical_trajectory_bytes(
            canonical_session_trajectory("session-1", [request_row], {})
        )
    )

    accumulator.add_interaction(interaction_row)
    materialized = canonical_session_trajectory(
        "session-1", [request_row], {"request-1": [interaction_row]}
    )
    assert accumulator.byte_count_if_finalized() == len(
        canonical_trajectory_bytes(materialized)
    )

    accumulator.finish_request()
    assert accumulator.byte_count_if_finalized() == len(
        canonical_trajectory_bytes(materialized)
    )
    assert accumulator.hexdigest() == trajectory_digest(materialized)


def test_streaming_trajectory_byte_count_handles_one_huge_interaction() -> None:
    request_row = _streaming_request_row()
    interaction_row = _streaming_interaction_row()
    interaction_row["content"] = "x" * 100_000
    materialized = canonical_session_trajectory(
        "session-1", [request_row], {"request-1": [interaction_row]}
    )
    accumulator = CanonicalTrajectoryDigestAccumulator("session-1")

    accumulator.start_request(request_row)
    accumulator.add_interaction(interaction_row)

    assert accumulator.byte_count_if_finalized() == len(
        canonical_trajectory_bytes(materialized)
    )


def _streaming_request_row() -> dict[str, object]:
    return {
        "request_id": "request-1",
        "user_id": "user-1",
        "created_at": "2026-08-10T09:00:00+00:00",
        "source": "workflow:v2",
        "agent_version": "agent-2",
        "session_id": "session-1",
        "evaluation_only": False,
        "retrieval_experiment_id": None,
        "retrieval_experiment_arm": None,
    }


def _streaming_interaction_row(*, tools_used: object = ()) -> dict[str, object]:
    return {
        "interaction_id": 1,
        "user_id": "user-1",
        "request_id": "request-1",
        "created_at": "2026-08-10T09:00:01+00:00",
        "content": "nested payload",
        "role": "User",
        "token_count": 3,
        "user_action": "none",
        "user_action_description": "",
        "interacted_image_url": "",
        "image_encoding": "",
        "shadow_content": "",
        "expert_content": "",
        "tools_used": tools_used,
        "citations": [],
        "retrieved_learnings": [],
    }


@pytest.mark.parametrize(
    ("nested_container_count", "raises"),
    [(95, False), (96, True)],
    ids=["exact-boundary", "over-boundary"],
)
def test_streaming_trajectory_digest_has_materialized_depth_parity(
    nested_container_count: int,
    raises: bool,
) -> None:
    tools_used: object = "leaf"
    for _ in range(nested_container_count):
        tools_used = [tools_used]
    request_row = _streaming_request_row()
    interaction_row = _streaming_interaction_row(tools_used=tools_used)
    materialized = canonical_session_trajectory(
        "session-1", [request_row], {"request-1": [interaction_row]}
    )
    accumulator = CanonicalTrajectoryDigestAccumulator("session-1")
    accumulator.start_request(request_row)

    if raises:
        with pytest.raises(
            ValueError, match="canonical trajectory JSON exceeds maximum depth"
        ):
            trajectory_digest(materialized)
        with pytest.raises(
            ValueError, match="canonical trajectory JSON exceeds maximum depth"
        ):
            accumulator.add_interaction(interaction_row)
    else:
        accumulator.add_interaction(interaction_row)
        accumulator.finish_request()
        assert accumulator.hexdigest() == trajectory_digest(materialized)


@pytest.mark.parametrize(
    "first_error",
    ["bad-request", "no-request", "wrong-request", "bad-json", "unfinished"],
)
def test_streaming_trajectory_digest_is_permanently_poisoned_after_error(
    first_error: str,
) -> None:
    accumulator = CanonicalTrajectoryDigestAccumulator("session-1")
    request_row = _streaming_request_row()

    with pytest.raises((KeyError, RuntimeError, ValueError)):
        if first_error == "bad-request":
            accumulator.start_request({})
        elif first_error == "no-request":
            accumulator.add_interaction(_streaming_interaction_row())
        else:
            accumulator.start_request(request_row)
            if first_error == "wrong-request":
                accumulator.add_interaction(
                    {
                        **_streaming_interaction_row(),
                        "request_id": "request-2",
                    }
                )
            elif first_error == "bad-json":
                accumulator.add_interaction(_streaming_interaction_row(tools_used="{"))
            else:
                accumulator.hexdigest()

    for operation in (
        lambda: accumulator.start_request(request_row),
        lambda: accumulator.add_interaction(_streaming_interaction_row()),
        accumulator.finish_request,
        accumulator.hexdigest,
    ):
        with pytest.raises(
            RuntimeError,
            match=r"canonical trajectory digest accumulator is invalid$",
        ):
            operation()
