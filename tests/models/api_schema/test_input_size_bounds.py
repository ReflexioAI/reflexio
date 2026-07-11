"""SEC-007: input-size bounds on user-input request models.

Guards against a single request amplifying into memory exhaustion via an
unbounded batch (list cardinality) or an unbounded free-text/base64 field.
Each capped field must:
  - reject ``N+1`` items / over-length strings with ``ValidationError``;
  - accept an at-cap payload;
  - keep existing ``min_length=1`` behavior (empty list still rejected).
"""

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.common import ToolUsed
from reflexio.models.api_schema.domain.entities import (
    AddUserPlaybookRequest,
    Citation,
    DeleteAgentPlaybooksByIdsRequest,
    DeleteProfilesByIdsRequest,
    DeleteRequestsByIdsRequest,
    DeleteUserPlaybooksByIdsRequest,
    InteractionData,
    PublishUserInteractionRequest,
    RetrievedLearning,
    RetrievedLearningSnapshot,
    UserPlaybook,
)


def _playbook() -> UserPlaybook:
    return UserPlaybook(
        agent_version="v1",
        request_id="r1",
        trigger="t",
        content="c",
        rationale="r",
    )


# ---------------------------------------------------------------------------
# Part 1: list cardinality caps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "field", "item", "cap"),
    [
        (DeleteRequestsByIdsRequest, "request_ids", "x", 10_000),
        (DeleteProfilesByIdsRequest, "profile_ids", "x", 10_000),
        (DeleteAgentPlaybooksByIdsRequest, "agent_playbook_ids", 1, 10_000),
        (DeleteUserPlaybooksByIdsRequest, "user_playbook_ids", 1, 10_000),
    ],
)
def test_delete_by_ids_list_cardinality(
    model: type, field: str, item: object, cap: int
) -> None:
    # at cap: accepted
    model(**{field: [item] * cap})
    # N+1: rejected
    with pytest.raises(ValidationError):
        model(**{field: [item] * (cap + 1)})
    # empty: still rejected (min_length=1 preserved)
    with pytest.raises(ValidationError):
        model(**{field: []})


def test_add_user_playbook_request_cardinality() -> None:
    cap = 1_000
    AddUserPlaybookRequest(user_playbooks=[_playbook()] * cap)
    with pytest.raises(ValidationError):
        AddUserPlaybookRequest(user_playbooks=[_playbook()] * (cap + 1))
    with pytest.raises(ValidationError):
        AddUserPlaybookRequest(user_playbooks=[])


def _publish(**overrides: object) -> PublishUserInteractionRequest:
    kwargs: dict[str, object] = {
        "user_id": "u1",
        "session_id": "s1",
        "interaction_data_list": [InteractionData(content="hi")],
    }
    kwargs.update(overrides)
    return PublishUserInteractionRequest(**kwargs)  # type: ignore[arg-type]


def test_interaction_data_list_cardinality() -> None:
    cap = 1_000
    _publish(interaction_data_list=[InteractionData(content="hi")] * cap)
    with pytest.raises(ValidationError):
        _publish(interaction_data_list=[InteractionData(content="hi")] * (cap + 1))
    with pytest.raises(ValidationError):
        _publish(interaction_data_list=[])


# ---------------------------------------------------------------------------
# Part 2: InteractionData string + nested-list bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "cap"),
    [
        ("content", 1_000_000),
        ("shadow_content", 1_000_000),
        ("expert_content", 1_000_000),
        ("image_encoding", 15_000_000),
        ("user_action_description", 10_000),
        ("role", 1_000),
    ],
)
def test_interaction_data_string_bounds(field: str, cap: int) -> None:
    # at limit: accepted
    InteractionData(**{field: "a" * cap})  # type: ignore[arg-type]
    # over limit: rejected
    with pytest.raises(ValidationError):
        InteractionData(**{field: "a" * (cap + 1)})  # type: ignore[arg-type]


def test_interaction_data_interacted_image_url_bound() -> None:
    cap = 2_048
    # must be a valid image URL (SSRF validator); a data: URI passes without
    # host checks, so pad one to exactly the cap.
    prefix = "data:image/png;base64,"
    base = prefix + "A" * (cap - len(prefix))
    assert len(base) == cap
    InteractionData(interacted_image_url=base)
    with pytest.raises(ValidationError):
        InteractionData(interacted_image_url=base + "A")


def test_interaction_data_nested_list_bounds() -> None:
    cap = 1_000
    tool = ToolUsed(tool_name="t")
    citation = Citation(kind="playbook", real_id="x")
    InteractionData(tools_used=[tool] * cap, citations=[citation] * cap)
    with pytest.raises(ValidationError):
        InteractionData(tools_used=[tool] * (cap + 1))
    with pytest.raises(ValidationError):
        InteractionData(citations=[citation] * (cap + 1))


def test_retrieved_learning_snapshot_bounds_and_aggregate_bytes() -> None:
    at_content_cap = RetrievedLearningSnapshot(content="a" * 100_000)
    with pytest.raises(ValidationError):
        RetrievedLearningSnapshot(content="a" * 100_001)

    learning = RetrievedLearning(
        kind="profile", learning_id="p1", snapshot=at_content_cap
    )
    _publish(
        interaction_data_list=[InteractionData(retrieved_learnings=[learning] * 100)]
    )
    with pytest.raises(ValidationError, match="at most 10 MiB"):
        _publish(
            interaction_data_list=[
                InteractionData(retrieved_learnings=[learning] * 105)
            ]
        )


def test_publish_request_string_bounds() -> None:
    cap = 1_000
    _publish(source="s" * cap)
    _publish(agent_version="v" * cap)
    with pytest.raises(ValidationError):
        _publish(source="s" * (cap + 1))
    with pytest.raises(ValidationError):
        _publish(agent_version="v" * (cap + 1))
