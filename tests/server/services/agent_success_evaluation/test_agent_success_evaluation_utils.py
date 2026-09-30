"""Tests for agent success evaluation utility functions."""

import json
import re
from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    construct_agent_success_evaluation_messages_from_sessions,
)


def _render_agent_success_prompt() -> str:
    return PromptManager().render_prompt(
        "agent_success_evaluation",
        {
            "agent_context_prompt": "Test agent",
            "success_definition_prompt": "Complete the requested task",
            "tool_can_use": "No tools",
            "interactions": "user: ```hello```",
        },
    )


def test_agent_success_prompt_v1_3_0_is_active_and_renders() -> None:
    prompt_manager = PromptManager()

    assert prompt_manager.get_active_version("agent_success_evaluation") == "1.3.0"
    assert "Step 4: Count corrective user turns" in _render_agent_success_prompt()
    assert "[Metadata Definition]" not in _render_agent_success_prompt()


def test_agent_success_prompt_examples_are_valid_json() -> None:
    """Every rendered output example must honor the prompt's JSON contract."""
    examples = re.findall(
        r"```json\n(.*?)\n```", _render_agent_success_prompt(), flags=re.DOTALL
    )

    assert len(examples) == 2
    for example in examples:
        json.loads(example)


@pytest.mark.parametrize(
    ("case", "expected_fragment"),
    [
        ("no correction", "the user's initial request"),
        ("factual correction", "wrong limit"),
        ("incomplete-answer revision", "omits rollback steps"),
        ("approach redirection", "rejects it and asks to use the API"),
        ("several issues in one turn", "fixes several details"),
        ("repeated corrections", "later has to correct it again"),
        ("same-topic new question", "different question about the same topic"),
        ("new deliverable", "separate implementation plan"),
        ("clarification answer", "agent's clarification question"),
    ],
)
def test_agent_success_prompt_documents_correction_rubric_cases(
    case: str, expected_fragment: str
) -> None:
    del case
    assert expected_fragment in _render_agent_success_prompt()


def test_construct_agent_success_evaluation_messages_with_sessions():
    """Test that construct_agent_success_evaluation_messages_from_sessions formats interactions correctly in the rendered prompt."""
    # Create test interactions with both content and actions
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="user_123",
            request_id="req_1",
            content="The agent helped me complete my task successfully",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="user_123",
            request_id="req_1",
            content="I used the search tool",
            role="assistant",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
        Interaction(
            interaction_id=3,
            user_id="user_123",
            request_id="req_1",
            content="Great!",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.CLICK,
            user_action_description="search button",
        ),
    ]

    # Create test request
    test_request = Request(
        request_id="req_1",
        user_id="user_123",
        source="test",
        agent_version="v1.0",
        session_id="test_group",
        created_at=int(datetime.now(UTC).timestamp()),
    )

    # Create RequestInteractionDataModel
    request_interaction_data_models = [
        RequestInteractionDataModel(
            request=test_request,
            interactions=interactions,
            session_id="test_group",
        )
    ]

    # Create prompt manager
    prompt_manager = PromptManager()

    # Call the function
    messages = construct_agent_success_evaluation_messages_from_sessions(
        prompt_manager=prompt_manager,
        request_interaction_data_models=request_interaction_data_models,
        agent_context_prompt="Test agent context",
        success_definition_prompt="Evaluate if the agent successfully completed the task",
        tool_can_use="search, calculator",
    )

    # Validate that messages were created
    assert len(messages) > 0, "No messages were created"

    # Find the user message that contains the interactions
    found_interactions = False
    for message in messages:
        # Messages are dicts with 'role' and 'content' keys
        if isinstance(message, dict) and "content" in message:
            # Content can be a string or a list of content blocks
            content = message.get("content", "")
            if isinstance(content, list):
                # Extract text from content blocks
                extracted_text = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        extracted_text += item.get("text", "")
                content = extracted_text
            else:
                content = str(content)

            # Check if this message contains the interaction section
            if (
                "[Interactions]" in content
                or "User and agent interactions:" in content
                or "user: ```The agent helped me complete my task successfully```"
                in content  # Check directly
            ):
                # Validate the interactions are formatted correctly in the rendered prompt
                assert (
                    "user: ```The agent helped me complete my task successfully```"
                    in content
                ), (
                    "Expected 'user: ```The agent helped me complete my task successfully```' in prompt"
                )
                assert "assistant: ```I used the search tool```" in content, (
                    "Expected 'assistant: ```I used the search tool```' in prompt"
                )
                assert "user: ```Great!```" in content, (
                    "Expected 'user: ```Great!```' in prompt"
                )
                assert "user: ```click search button```" in content, (
                    "Expected 'user: ```click search button```' in prompt"
                )

                # Also verify success definition and tools are in the content
                assert (
                    "Evaluate if the agent successfully completed the task" in content
                ), "Expected success definition in prompt"
                assert "search, calculator" in content, "Expected tools in prompt"
                assert (
                    "Count corrective user turns across the entire session" in content
                )
                assert (
                    "Topic continuity alone is not evidence of a correction" in content
                )
                assert '"number_of_correction_per_session"' in content

                found_interactions = True
                break

    assert found_interactions, "Did not find interactions in the rendered prompt"


def test_construct_agent_success_evaluation_messages_with_empty_sessions():
    """Test that construct_agent_success_evaluation_messages_from_sessions handles empty sessions."""
    # Empty sessions list
    request_interaction_data_models = []

    # Create prompt manager
    prompt_manager = PromptManager()

    # Call the function
    messages = construct_agent_success_evaluation_messages_from_sessions(
        prompt_manager=prompt_manager,
        request_interaction_data_models=request_interaction_data_models,
        agent_context_prompt="Test agent context",
        success_definition_prompt="Evaluate if the agent successfully completed the task",
        tool_can_use="search, calculator",
    )

    # Should still create messages (user message with prompt)
    assert len(messages) > 0, "No messages were created for empty sessions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _session_of(
    n_interactions: int, filler_words: int
) -> list[RequestInteractionDataModel]:
    """Build one session whose interactions each stay inside the per-interaction cap."""
    now = int(datetime.now(UTC).timestamp())
    interactions = []
    for i in range(n_interactions):
        if i == 0:
            marker = "FIRST-TASK-STATEMENT"
        elif i == n_interactions - 1:
            marker = "FINAL-OUTCOME-STATEMENT"
        elif i == n_interactions // 2:
            marker = "MIDDLE-OF-SESSION-STATEMENT"
        else:
            marker = f"step-{i}"
        interactions.append(
            Interaction(
                interaction_id=i,
                user_id="u",
                request_id="req",
                content=f"{marker} " + " ".join(["filler"] * filler_words),
                role="user" if i % 2 == 0 else "assistant",
                created_at=now,
                user_action=UserActionType.NONE,
                user_action_description="",
            )
        )
    request = Request(
        request_id="req",
        user_id="u",
        source="test",
        agent_version="v1",
        session_id="s",
        created_at=now,
    )
    return [
        RequestInteractionDataModel(
            request=request, interactions=interactions, session_id="s"
        )
    ]


def _rendered_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            parts.extend(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            parts.append(str(content))
    return "\n".join(parts)


def _build(models: list[RequestInteractionDataModel]) -> str:
    return _rendered_text(
        construct_agent_success_evaluation_messages_from_sessions(
            prompt_manager=PromptManager(),
            request_interaction_data_models=models,
            agent_context_prompt="agent",
            success_definition_prompt="done",
            tool_can_use="none",
        )
    )


def test_long_session_transcript_is_bounded_keeping_task_and_outcome(monkeypatch):
    """41 in-budget interactions must not build an unbounded prompt.

    Regression for claude-smart#162: each interaction was capped, the SUM was
    not, and a 41-interaction session was rejected with 'Prompt is too long'.
    Every interaction here is far below the 512-token per-interaction cap, so
    only the aggregate budget can be what bounds the result.
    """
    import reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils as utils
    from reflexio.server.services.service_utils import (
        _CONTENT_TRUNCATION_MARKER,
        _get_content_token_encoding,
    )

    monkeypatch.setattr(utils, "EVALUATION_TRANSCRIPT_TOKEN_LIMIT", 1_000)
    models = _session_of(41, filler_words=60)

    rendered = _build(models)
    unbounded = utils.format_sessions_to_history_string(models)
    encoding = _get_content_token_encoding()
    assert len(encoding.encode(unbounded)) > 1_000, "fixture must exceed the budget"

    assert _CONTENT_TRUNCATION_MARKER.strip() in rendered
    assert "FIRST-TASK-STATEMENT" in rendered, "the task at the head was dropped"
    assert "FINAL-OUTCOME-STATEMENT" in rendered, "the outcome at the tail was dropped"
    assert "MIDDLE-OF-SESSION-STATEMENT" not in rendered


def test_session_within_budget_is_unchanged():
    """A normal session must reach the judge verbatim, with no truncation marker."""
    from reflexio.server.services.service_utils import _CONTENT_TRUNCATION_MARKER

    rendered = _build(_session_of(4, filler_words=5))

    assert _CONTENT_TRUNCATION_MARKER.strip() not in rendered
    for marker in (
        "FIRST-TASK-STATEMENT",
        "MIDDLE-OF-SESSION-STATEMENT",
        "FINAL-OUTCOME-STATEMENT",
    ):
        assert marker in rendered
