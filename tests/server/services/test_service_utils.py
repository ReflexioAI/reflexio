"""Tests for service_utils module."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.common import ToolUsed
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
    UserActionType,
)
from reflexio.server.services.service_utils import (
    _CONTENT_TRUNCATION_MARKER,
    DEFAULT_MAX_INTERACTION_CONTENT_TOKENS,
    _get_content_token_encoding,
    _resolve_max_interaction_content_tokens,
    format_interactions_to_history_string,
    format_sessions_to_history_string,
    slice_content_by_tokens,
)

_ENV_MAX_TOKENS = "REFLEXIO_MAX_INTERACTION_CONTENT_TOKENS"


def test_format_interactions_to_history_string_with_content():
    """Test formatting interactions with text content."""
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user",
            request_id="test_request",
            content="I love Italian food",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user",
            request_id="test_request",
            content="I also enjoy sushi",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
    ]

    result = format_interactions_to_history_string(interactions)
    expected = "user: ```I love Italian food```\nuser: ```I also enjoy sushi```"
    assert result == expected


def test_format_interactions_to_history_string_with_actions():
    """Test formatting interactions with user actions."""
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user",
            request_id="test_request",
            content="",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.CLICK,
            user_action_description="menu item",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user",
            request_id="test_request",
            content="",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.SCROLL,
            user_action_description="to bottom",
        ),
    ]

    result = format_interactions_to_history_string(interactions)
    expected = "user: ```click menu item```\nuser: ```scroll to bottom```"
    assert result == expected


def test_format_interactions_to_history_string_mixed():
    """Test formatting interactions with both content and actions."""
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user",
            request_id="test_request",
            content="I love sushi",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user",
            request_id="test_request",
            content="",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.CLICK,
            user_action_description="menu item",
        ),
    ]

    result = format_interactions_to_history_string(interactions)
    expected = "user: ```I love sushi```\nuser: ```click menu item```"
    assert result == expected


def test_format_interactions_to_history_string_with_content_and_action():
    """Test formatting interaction with both content and action in same interaction."""
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user",
            request_id="test_request",
            content="I love sushi",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.CLICK,
            user_action_description="sushi restaurant",
        ),
    ]

    result = format_interactions_to_history_string(interactions)
    expected = "user: ```I love sushi```\nuser: ```click sushi restaurant```"
    assert result == expected


def test_format_interactions_to_history_string_empty():
    """Test formatting empty interactions list."""
    interactions = []
    result = format_interactions_to_history_string(interactions)
    assert result == ""


def test_format_interactions_to_history_string_with_tools_used_placeholder():
    """Tool-only assistant turns from the Claude Code hook are stored with the
    literal placeholder content "(tool call)" so that the renderer (which
    skips empty content) still emits a line carrying the [used tool: ...]
    prefix. Pinning this contract prevents a future change to either the
    hook or the renderer from silently dropping tool-only turns from
    playbook-extraction context.
    """
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user",
            request_id="test_request",
            content="(tool call)",
            role="assistant",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
            tools_used=[
                ToolUsed(
                    tool_name="search_docs",
                    tool_data={"input": {"query": "TypeError async handler"}},
                ),
            ],
        ),
    ]

    result = format_interactions_to_history_string(interactions)
    # The renderer must include both the [used tool: ...] marker AND the
    # placeholder content; the playbook-extraction prompts key off the
    # marker prefix to detect tool usage.
    assert "[used tool: search_docs(" in result
    assert "TypeError async handler" in result
    assert "(tool call)" in result
    assert result.startswith("assistant: ```[used tool:")


def test_format_interactions_to_history_string_multiple_roles():
    """Test formatting interactions with different roles."""
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="test_user",
            request_id="test_request",
            content="Can you help me?",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="test_user",
            request_id="test_request",
            content="Of course, I can help!",
            role="assistant",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
    ]

    result = format_interactions_to_history_string(interactions)
    expected = "user: ```Can you help me?```\nassistant: ```Of course, I can help!```"
    assert result == expected


def _create_request(request_id: str, created_at: int) -> Request:
    """Helper function to create a Request object for testing."""
    return Request(
        request_id=request_id,
        user_id="test_user",
        created_at=created_at,
    )


def _create_interaction(
    interaction_id: int, content: str, role: str, created_at: int
) -> Interaction:
    """Helper function to create an Interaction object for testing."""
    return Interaction(
        interaction_id=interaction_id,
        user_id="test_user",
        request_id="test_request",
        content=content,
        role=role,
        created_at=created_at,
        user_action=UserActionType.NONE,
        user_action_description="",
    )


def test_format_sessions_to_history_string_empty():
    """Test formatting empty sessions list."""
    result = format_sessions_to_history_string([])
    assert result == ""


def test_format_sessions_to_history_string_single_group():
    """Test formatting a single session.

    Header includes the session date so downstream extraction agents have
    a temporal anchor for relative-time references in the conversation.
    """
    base_time = int(datetime.now(UTC).timestamp())
    iso = datetime.fromtimestamp(base_time, tz=UTC).strftime("%Y-%m-%d")

    session_data = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_1", base_time),
        interactions=[
            _create_interaction(1, "Hello", "user", base_time),
            _create_interaction(2, "Hi there!", "assistant", base_time + 1),
        ],
    )

    result = format_sessions_to_history_string([session_data])
    expected = (
        f"=== Session: group_1 (date: {iso}) ===\n"
        "user: ```Hello```\nassistant: ```Hi there!```"
    )
    assert result == expected


def test_format_sessions_to_history_string_consolidates_same_group():
    """Test that multiple requests with the same group name are consolidated under one header."""
    base_time = int(datetime.now(UTC).timestamp())

    # Three separate requests, all with the same session_id name
    session_id_1 = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_1", base_time),
        interactions=[
            _create_interaction(1, "First message", "user", base_time),
            _create_interaction(2, "First response", "assistant", base_time + 1),
        ],
    )

    session_id_2 = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_2", base_time + 100),
        interactions=[
            _create_interaction(3, "Second message", "user", base_time + 100),
            _create_interaction(4, "Second response", "assistant", base_time + 101),
        ],
    )

    session_id_3 = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_3", base_time + 200),
        interactions=[
            _create_interaction(5, "Third message", "user", base_time + 200),
            _create_interaction(6, "Third response", "assistant", base_time + 201),
        ],
    )

    result = format_sessions_to_history_string(
        [session_id_1, session_id_2, session_id_3]
    )

    iso = datetime.fromtimestamp(base_time, tz=UTC).strftime("%Y-%m-%d")
    # All interactions should be under a single header
    expected = (
        f"=== Session: group_1 (date: {iso}) ===\n"
        "user: ```First message```\n"
        "assistant: ```First response```\n"
        "user: ```Second message```\n"
        "assistant: ```Second response```\n"
        "user: ```Third message```\n"
        "assistant: ```Third response```"
    )
    assert result == expected


def test_format_sessions_to_history_string_multiple_groups():
    """Test formatting multiple different sessions."""
    base_time = int(datetime.now(UTC).timestamp())

    group_a = RequestInteractionDataModel(
        session_id="session_a",
        request=_create_request("req_a", base_time),
        interactions=[
            _create_interaction(1, "Message A", "user", base_time),
        ],
    )

    group_b = RequestInteractionDataModel(
        session_id="session_b",
        request=_create_request("req_b", base_time + 100),
        interactions=[
            _create_interaction(2, "Message B", "user", base_time + 100),
        ],
    )

    result = format_sessions_to_history_string([group_a, group_b])
    iso_a = datetime.fromtimestamp(base_time, tz=UTC).strftime("%Y-%m-%d")
    iso_b = datetime.fromtimestamp(base_time + 100, tz=UTC).strftime("%Y-%m-%d")
    expected = (
        f"=== Session: session_a (date: {iso_a}) ===\n"
        "user: ```Message A```\n\n"
        f"=== Session: session_b (date: {iso_b}) ===\n"
        "user: ```Message B```"
    )
    assert result == expected


def test_format_sessions_to_history_string_mixed_groups():
    """Test multiple sessions with some sharing the same name."""
    base_time = int(datetime.now(UTC).timestamp())

    # Two requests in group_1
    group_1_req_1 = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_1", base_time),
        interactions=[
            _create_interaction(1, "Group 1 - Request 1", "user", base_time),
        ],
    )

    group_1_req_2 = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_2", base_time + 100),
        interactions=[
            _create_interaction(2, "Group 1 - Request 2", "user", base_time + 100),
        ],
    )

    # One request in group_2 (comes between the two group_1 requests in terms of time)
    group_2_req = RequestInteractionDataModel(
        session_id="group_2",
        request=_create_request("req_3", base_time + 50),
        interactions=[
            _create_interaction(3, "Group 2 - Request 1", "user", base_time + 50),
        ],
    )

    result = format_sessions_to_history_string(
        [group_1_req_1, group_2_req, group_1_req_2]
    )

    iso_1 = datetime.fromtimestamp(base_time, tz=UTC).strftime("%Y-%m-%d")
    iso_2 = datetime.fromtimestamp(base_time + 50, tz=UTC).strftime("%Y-%m-%d")
    # Groups should be sorted by earliest request timestamp
    # group_1 (base_time) should come before group_2 (base_time + 50)
    expected = (
        f"=== Session: group_1 (date: {iso_1}) ===\n"
        "user: ```Group 1 - Request 1```\n"
        "user: ```Group 1 - Request 2```\n\n"
        f"=== Session: group_2 (date: {iso_2}) ===\n"
        "user: ```Group 2 - Request 1```"
    )
    assert result == expected


def test_format_sessions_to_history_string_preserves_order_within_group():
    """Test that requests within the same group are ordered by created_at."""
    base_time = int(datetime.now(UTC).timestamp())

    # Create requests out of order
    late_request = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_late", base_time + 200),
        interactions=[
            _create_interaction(3, "Late message", "user", base_time + 200),
        ],
    )

    early_request = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_early", base_time),
        interactions=[
            _create_interaction(1, "Early message", "user", base_time),
        ],
    )

    middle_request = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_middle", base_time + 100),
        interactions=[
            _create_interaction(2, "Middle message", "user", base_time + 100),
        ],
    )

    # Pass them out of order
    result = format_sessions_to_history_string(
        [late_request, early_request, middle_request]
    )

    iso = datetime.fromtimestamp(base_time, tz=UTC).strftime("%Y-%m-%d")
    # Should be sorted by created_at within the group
    expected = (
        f"=== Session: group_1 (date: {iso}) ===\n"
        "user: ```Early message```\n"
        "user: ```Middle message```\n"
        "user: ```Late message```"
    )
    assert result == expected


def test_slice_content_by_tokens_within_budget_unchanged():
    """Content at or below the budget is returned verbatim."""
    content = "short content well under budget"
    assert slice_content_by_tokens(content, 512) == content


def test_slice_content_by_tokens_none_disables_slicing():
    """A None budget disables slicing even for very long content."""
    content = " ".join(str(i) for i in range(2000))
    assert slice_content_by_tokens(content, None) == content


def test_slice_content_by_tokens_empty_content():
    """Empty content is returned unchanged regardless of budget."""
    assert slice_content_by_tokens("", 512) == ""


def test_slice_content_by_tokens_keeps_head_and_tail():
    """Over-budget content keeps the first half + last half with a marker."""
    encoding = _get_content_token_encoding()
    content = " ".join(str(i) for i in range(2000))
    tokens = encoding.encode(content)
    assert len(tokens) > 512  # precondition: actually over budget

    result = slice_content_by_tokens(content, 512)

    head = encoding.decode(tokens[:256])
    tail = encoding.decode(tokens[-256:])
    expected = f"{head}{_CONTENT_TRUNCATION_MARKER}{tail}"
    assert result == expected
    assert _CONTENT_TRUNCATION_MARKER in result
    # The sliced result is materially shorter than the original.
    assert len(encoding.encode(result)) < len(tokens)


def test_resolve_max_tokens_unset_uses_default(monkeypatch):
    """Unset env falls back to the 512 default."""
    monkeypatch.delenv(_ENV_MAX_TOKENS, raising=False)
    assert (
        _resolve_max_interaction_content_tokens()
        == DEFAULT_MAX_INTERACTION_CONTENT_TOKENS
    )


def test_resolve_max_tokens_valid_override(monkeypatch):
    """A positive integer env value is used as-is."""
    monkeypatch.setenv(_ENV_MAX_TOKENS, "1024")
    assert _resolve_max_interaction_content_tokens() == 1024


@pytest.mark.parametrize("raw", ["0", "-1", "-512"])
def test_resolve_max_tokens_non_positive_disables(monkeypatch, raw):
    """A value <= 0 disables slicing (returns None)."""
    monkeypatch.setenv(_ENV_MAX_TOKENS, raw)
    assert _resolve_max_interaction_content_tokens() is None


def test_resolve_max_tokens_invalid_falls_back_to_default(monkeypatch):
    """A malformed env value falls back to the default."""
    monkeypatch.setenv(_ENV_MAX_TOKENS, "not-an-int")
    assert (
        _resolve_max_interaction_content_tokens()
        == DEFAULT_MAX_INTERACTION_CONTENT_TOKENS
    )


def test_format_interactions_slices_long_content(monkeypatch):
    """format_interactions_to_history_string slices content over the budget."""
    monkeypatch.setenv(_ENV_MAX_TOKENS, "8")
    long_content = " ".join(str(i) for i in range(500))
    interactions = [
        _create_interaction(1, long_content, "user", 1_700_000_000),
    ]

    result = format_interactions_to_history_string(interactions)

    assert _CONTENT_TRUNCATION_MARKER in result
    assert long_content not in result  # full content was not emitted verbatim
    assert result.startswith("user: ```")
    assert result.endswith("```")


def test_format_interactions_no_slice_when_disabled(monkeypatch):
    """Disabling via <=0 leaves content verbatim."""
    monkeypatch.setenv(_ENV_MAX_TOKENS, "0")
    long_content = " ".join(str(i) for i in range(500))
    interactions = [
        _create_interaction(1, long_content, "user", 1_700_000_000),
    ]

    result = format_interactions_to_history_string(interactions)

    assert _CONTENT_TRUNCATION_MARKER not in result
    assert long_content in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
