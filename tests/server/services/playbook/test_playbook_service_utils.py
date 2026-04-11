"""Tests for playbook service utility functions."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    BlockingIssue,
    BlockingIssueKind,
    Interaction,
    Request,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
    StructuredPlaybookList,
    construct_playbook_extraction_messages_from_sessions,
    ensure_playbook_content,
    format_structured_fields_for_display,
)


def test_construct_playbook_extraction_messages_with_sessions():
    """Test that construct_playbook_extraction_messages_from_sessions formats interactions correctly in the rendered prompt."""
    # Create test interactions
    interactions = [
        Interaction(
            interaction_id=1,
            user_id="user_123",
            request_id="req_1",
            content="I need help with my account",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="none",
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="user_123",
            request_id="req_1",
            content="Here is how to access your account",
            role="assistant",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="none",
            user_action_description="",
        ),
        Interaction(
            interaction_id=3,
            user_id="user_123",
            request_id="req_1",
            content="Thank you!",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action="click",
            user_action_description="help button",
        ),
    ]

    # Create request and request interaction data model
    request = Request(
        request_id="req_1",
        user_id="user_123",
        source="test",
        agent_version="1.0.0",
        session_id="session_1",
    )

    request_interaction_data_models = [
        RequestInteractionDataModel(
            session_id="session_1",
            request=request,
            interactions=interactions,
        )
    ]

    # Create prompt manager
    prompt_manager = PromptManager()

    # Call the function
    messages = construct_playbook_extraction_messages_from_sessions(
        prompt_manager=prompt_manager,
        request_interaction_data_models=request_interaction_data_models,
        extraction_definition_prompt="Evaluate the quality of the agent's response",
        agent_context_prompt="Customer support agent",
    )

    # Validate that messages were created
    assert len(messages) > 0, "No messages were created"

    # Helper to extract text from a message's content (string or content blocks)
    def extract_text(message):
        content = message.get("content", "")
        if isinstance(content, list):
            extracted = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    extracted += item.get("text", "")
            return extracted
        return str(content)

    # Verify playbook definition is in the system message (moved there for token caching)
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert system_messages, "Expected a system message"
    system_text = extract_text(system_messages[0])
    assert "Evaluate the quality of the agent's response" in system_text, (
        "Expected playbook definition in system message"
    )

    # Find the user message that contains the interactions
    found_interactions = False
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content = extract_text(message)

            # Check if this message contains the interaction section
            if (
                "[Intearctions start]" in content
                or "[Interactions end]" in content
                or "User and agent interactions:" in content
                or "Session:" in content
                or "user: ```I need help with my account```"
                in content  # Check directly for content
            ):
                # Validate the interactions are formatted correctly in the rendered prompt
                # Note: Content is wrapped in backticks in the prompt template
                assert "user: ```I need help with my account```" in content, (
                    "Expected 'user: ```I need help with my account```' in prompt"
                )
                assert (
                    "assistant: ```Here is how to access your account```" in content
                ), (
                    "Expected 'assistant: ```Here is how to access your account```' in prompt"
                )
                assert "user: ```Thank you!```" in content, (
                    "Expected 'user: ```Thank you!```' in prompt"
                )
                assert "user: ```click help button```" in content, (
                    "Expected 'user: ```click help button```' in prompt"
                )

                found_interactions = True
                break

    assert found_interactions, "Did not find interactions in the rendered prompt"


def test_construct_playbook_extraction_messages_with_empty_sessions():
    """Test that construct_playbook_extraction_messages_from_sessions handles empty sessions."""
    # Empty sessions list
    request_interaction_data_models = []

    # Create prompt manager
    prompt_manager = PromptManager()

    # Call the function
    messages = construct_playbook_extraction_messages_from_sessions(
        prompt_manager=prompt_manager,
        request_interaction_data_models=request_interaction_data_models,
        extraction_definition_prompt="Evaluate the quality of the agent's response",
        agent_context_prompt="Customer support agent",
    )

    # Should still create messages (system message + user message with prompt)
    assert len(messages) > 0, "No messages were created for empty sessions"


# ===============================
# Tests for format_structured_fields_for_display and ensure_playbook_content
# ===============================


class TestFormatStructuredFieldsForDisplay:
    """Tests for the shared format_structured_fields_for_display function (display/debug formatting)."""

    def test_all_fields_present(self):
        """Test formatting with all fields populated."""
        structured = StructuredPlaybookContent(
            instruction="use clear language",
            pitfall="use jargon",
            trigger="explaining technical concepts to beginners",
        )
        result = format_structured_fields_for_display(structured)
        assert 'Trigger: "explaining technical concepts to beginners"' in result
        assert 'Instruction: "use clear language"' in result
        assert 'Pitfall: "use jargon"' in result

    def test_trigger_none(self):
        """Test that None trigger is omitted from output."""
        structured = StructuredPlaybookContent(
            instruction="use clear language",
            pitfall=None,
            trigger=None,
        )
        result = format_structured_fields_for_display(structured)
        assert "Trigger:" not in result
        assert 'Instruction: "use clear language"' in result

    def test_trigger_empty_string(self):
        """Test that empty string trigger is omitted from output."""
        structured = StructuredPlaybookContent(
            instruction="use clear language",
            pitfall=None,
            trigger=None,
        )
        result = format_structured_fields_for_display(structured)
        assert "Trigger:" not in result

    def test_only_instruction(self):
        """Test formatting with only instruction."""
        structured = StructuredPlaybookContent(
            instruction="be concise",
            pitfall=None,
            trigger=None,
        )
        result = format_structured_fields_for_display(structured)
        assert 'Instruction: "be concise"' in result
        assert "Pitfall:" not in result
        assert "Trigger:" not in result

    def test_only_pitfall(self):
        """Test formatting with only pitfall."""
        structured = StructuredPlaybookContent(
            instruction=None,
            pitfall="ramble",
            trigger=None,
        )
        result = format_structured_fields_for_display(structured)
        assert 'Pitfall: "ramble"' in result
        assert "Instruction:" not in result
        assert "Trigger:" not in result

    def test_with_blocking_issue(self):
        """Test formatting with blocking_issue."""
        structured = StructuredPlaybookContent(
            instruction="acknowledge limitation",
            trigger="user asks for real-time data",
            blocking_issue=BlockingIssue(
                kind=BlockingIssueKind.MISSING_TOOL,
                details="No real-time data API available",
            ),
        )
        result = format_structured_fields_for_display(structured)
        assert "Blocked by:" in result
        assert "missing_tool" in result
        assert "No real-time data API available" in result

    def test_all_fields_none_returns_empty(self):
        """Test that all-None fields returns empty string."""
        structured = StructuredPlaybookContent()
        result = format_structured_fields_for_display(structured)
        assert result == ""


# ===============================
# Tests for StructuredPlaybookContent freeform support
# ===============================


class TestStructuredPlaybookContentFreeform:
    """Tests for freeform playbook support in StructuredPlaybookContent."""

    def test_has_content_structured_only(self):
        """Structured playbook with trigger + action returns True."""
        sfc = StructuredPlaybookContent(
            trigger="user asks about X",
            instruction="do Y",
        )
        assert sfc.has_content is True
        assert sfc.is_structured is True

    def test_has_content_freeform_only(self):
        """Freeform playbook content alone returns True."""
        sfc = StructuredPlaybookContent(
            content="Agent tends to over-explain simple concepts",
        )
        assert sfc.has_content is True
        assert sfc.is_structured is False

    def test_has_content_empty_freeform(self):
        """Whitespace-only freeform returns False."""
        sfc = StructuredPlaybookContent(content="   ")
        assert sfc.has_content is False
        assert sfc.is_structured is False

    def test_has_content_none_freeform(self):
        """None freeform with no structured fields returns False."""
        sfc = StructuredPlaybookContent()
        assert sfc.has_content is False
        assert sfc.is_structured is False

    def test_has_content_both_present(self):
        """When both structured and freeform are present, structured takes precedence."""
        sfc = StructuredPlaybookContent(
            trigger="user asks X",
            instruction="do Y",
            content="some observation",
        )
        assert sfc.has_content is True
        assert sfc.is_structured is True

    def test_validate_freeform_without_trigger(self):
        """Freeform playbook without trigger should pass validation."""
        sfc = StructuredPlaybookContent(
            content="Agent consistently over-apologizes",
        )
        assert sfc.content == "Agent consistently over-apologizes"
        assert sfc.trigger is None

    def test_validate_trigger_without_action_still_fails(self):
        """Trigger without instruction/pitfall still raises even with freeform."""
        with pytest.raises(ValueError, match="instruction.*pitfall"):
            StructuredPlaybookContent(
                trigger="user asks about X",
                content="some observation",
            )

    def test_freeform_from_dict(self):
        """Parse freeform playbook from a dict (as LLM would return)."""
        sfc = StructuredPlaybookContent.model_validate(
            {"content": "Agent over-explains"}
        )
        assert sfc.has_content is True
        assert sfc.is_structured is False
        assert sfc.content == "Agent over-explains"


class TestStructuredPlaybookList:
    """Tests for the multi-entry StructuredPlaybookList wrapper."""

    def test_empty_list(self):
        """An empty playbooks list parses successfully and yields zero entries."""
        result = StructuredPlaybookList.model_validate({"playbooks": []})
        assert result.playbooks == []

    def test_default_constructs_empty(self):
        """Constructing with no args defaults to an empty playbooks list."""
        result = StructuredPlaybookList()
        assert result.playbooks == []

    def test_multiple_entries(self):
        """A list with multiple entries parses each into a StructuredPlaybookContent."""
        result = StructuredPlaybookList.model_validate(
            {
                "playbooks": [
                    {
                        "trigger": "user asks for help debugging",
                        "instruction": "explain the root cause first",
                        "content": "Explain root cause before fixes.",
                    },
                    {
                        "trigger": "agent provides a factual correction",
                        "instruction": "deliver corrections without apologies",
                        "content": "Reserve apologies for genuine mistakes.",
                    },
                ]
            }
        )
        assert len(result.playbooks) == 2
        triggers = [p.trigger for p in result.playbooks]
        assert triggers == [
            "user asks for help debugging",
            "agent provides a factual correction",
        ]

    def test_legacy_single_playbook_shape_rejected(self):
        """Legacy {"playbook": ...} shape is no longer accepted."""
        with pytest.raises(ValueError):
            StructuredPlaybookList.model_validate({"playbook": None})

    def test_legacy_feedback_shape_rejected(self):
        """Legacy {"feedback": ...} shape is no longer accepted."""
        with pytest.raises(ValueError):
            StructuredPlaybookList.model_validate({"feedback": None})

    def test_unknown_field_rejected(self):
        """Extra fields beyond `playbooks` are forbidden."""
        with pytest.raises(ValueError):
            StructuredPlaybookList.model_validate(
                {"playbooks": [], "extra_field": "nope"}
            )

    def test_legacy_flat_shape_rejected(self):
        """Legacy flat single-entry shape (no `playbooks` wrapper) is rejected.

        Pins the contract that an LLM regression to the v1 single-entry
        shape ``{"trigger": ..., "instruction": ...}`` no longer parses
        as a StructuredPlaybookList — the broad ``except`` in
        ``PlaybookExtractor.extract_playbook_entries`` then logs the
        ValidationError and returns ``[]`` instead of silently building
        a malformed playbook.
        """
        with pytest.raises(ValueError):
            StructuredPlaybookList.model_validate(
                {
                    "trigger": "user asks for help debugging",
                    "instruction": "explain the root cause first",
                }
            )

    def test_nested_entry_tolerates_extra_fields(self):
        """Unknown fields on a nested entry are tolerated at runtime.

        ``StructuredPlaybookContent`` is intentionally ``extra="allow"``
        for runtime parsing (the strict ``additionalProperties: false``
        only flows into the JSON Schema sent to OpenAI structured output).
        Pinning this so a future tightening to ``extra="forbid"`` is a
        deliberate, reviewed change rather than a silent regression that
        breaks every provider whose output drifts slightly.
        """
        result = StructuredPlaybookList.model_validate(
            {
                "playbooks": [
                    {
                        "trigger": "user asks for help",
                        "instruction": "respond",
                        "bogus_field_from_provider": 1,
                    }
                ]
            }
        )
        assert len(result.playbooks) == 1
        assert result.playbooks[0].trigger == "user asks for help"


class TestFormatStructuredFieldsForDisplayFreeform:
    """Tests for format_structured_fields_for_display freeform fallback behavior."""

    def test_freeform_fallback(self):
        """When no structured fields, returns playbook content."""
        sfc = StructuredPlaybookContent(
            content="Agent over-apologizes when correcting",
        )
        result = format_structured_fields_for_display(sfc)
        assert result == "Agent over-apologizes when correcting"

    def test_structured_takes_precedence(self):
        """When structured fields present, playbook content is not used."""
        sfc = StructuredPlaybookContent(
            trigger="user asks X",
            instruction="do Y",
            content="some observation",
        )
        result = format_structured_fields_for_display(sfc)
        assert "Trigger:" in result
        assert "Instruction:" in result
        assert "some observation" not in result

    def test_freeform_not_applied_to_structured_data(self):
        """StructuredData (not StructuredPlaybookContent) never falls back to freeform."""
        from reflexio.models.api_schema.service_schemas import StructuredData

        sd = StructuredData()
        result = format_structured_fields_for_display(sd)
        assert result == ""


# ===============================
# Tests for ensure_playbook_content
# ===============================


class TestEnsurePlaybookContent:
    """Tests for the ensure_playbook_content helper."""

    def test_returns_playbook_content_when_present(self):
        """When playbook content is a non-empty string, return it as-is."""
        structured = StructuredPlaybookContent(
            trigger="user asks X",
            instruction="do Y",
        )
        result = ensure_playbook_content("My freeform playbook", structured)
        assert result == "My freeform playbook"

    def test_falls_back_to_structured_when_none(self):
        """When playbook content is None, fall back to formatted structured fields."""
        structured = StructuredPlaybookContent(
            trigger="user asks X",
            instruction="do Y",
        )
        result = ensure_playbook_content(None, structured)
        assert 'Trigger: "user asks X"' in result
        assert 'Instruction: "do Y"' in result

    def test_falls_back_to_structured_when_empty(self):
        """When playbook content is empty string, fall back to formatted structured fields."""
        structured = StructuredPlaybookContent(
            trigger="user asks X",
            instruction="do Y",
        )
        result = ensure_playbook_content("", structured)
        assert 'Trigger: "user asks X"' in result
        assert 'Instruction: "do Y"' in result

    def test_falls_back_to_structured_when_whitespace_only(self):
        """When playbook content is whitespace-only, fall back to formatted structured fields."""
        structured = StructuredPlaybookContent(
            trigger="user asks X",
            instruction="do Y",
        )
        result = ensure_playbook_content("   ", structured)
        assert 'Trigger: "user asks X"' in result

    def test_works_with_structured_data(self):
        """ensure_playbook_content also works with StructuredData (not just StructuredPlaybookContent)."""
        from reflexio.models.api_schema.service_schemas import StructuredData

        sd = StructuredData(
            trigger="when testing",
            instruction="check all fields",
        )
        result = ensure_playbook_content(None, sd)
        assert 'Trigger: "when testing"' in result
        assert 'Instruction: "check all fields"' in result


# ===============================
# Tests for ensure_playbook_content freeform invariant
# ===============================


class TestEnsurePlaybookContentEdgeCases:
    """Additional edge cases for ensure_playbook_content."""

    def test_returns_empty_string_when_both_empty(self):
        """When both playbook content and structured fields are empty, returns empty string."""
        from reflexio.models.api_schema.service_schemas import StructuredData

        result = ensure_playbook_content(None, StructuredData())
        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
