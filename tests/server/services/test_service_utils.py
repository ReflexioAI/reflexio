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
    format_interactions_to_history_string,
    format_sessions_to_history_string,
)


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
        app_id="test_app",
        agent_id="test_agent",
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
    """Test formatting a single session."""
    base_time = int(datetime.now(UTC).timestamp())

    session_data = RequestInteractionDataModel(
        session_id="group_1",
        request=_create_request("req_1", base_time),
        interactions=[
            _create_interaction(1, "Hello", "user", base_time),
            _create_interaction(2, "Hi there!", "assistant", base_time + 1),
        ],
    )

    result = format_sessions_to_history_string([session_data])
    expected = "=== Session: group_1 ===\nuser: ```Hello```\nassistant: ```Hi there!```"
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

    # All interactions should be under a single header
    expected = (
        "=== Session: group_1 ===\n"
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
    expected = (
        "=== Session: session_a ===\n"
        "user: ```Message A```\n\n"
        "=== Session: session_b ===\n"
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

    # Groups should be sorted by earliest request timestamp
    # group_1 (base_time) should come before group_2 (base_time + 50)
    expected = (
        "=== Session: group_1 ===\n"
        "user: ```Group 1 - Request 1```\n"
        "user: ```Group 1 - Request 2```\n\n"
        "=== Session: group_2 ===\n"
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

    # Should be sorted by created_at within the group
    expected = (
        "=== Session: group_1 ===\n"
        "user: ```Early message```\n"
        "user: ```Middle message```\n"
        "user: ```Late message```"
    )
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
