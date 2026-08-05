"""Tests for playbook service utility functions."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.common import ToolUsed
from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
    StructuredPlaybookList,
    StructuredReferencedExtractedPlaybookList,
    build_playbook_prompt_context,
    construct_playbook_extraction_messages_from_sessions,
    dedupe_and_drop_empty,
    ensure_playbook_content,
    format_structured_fields_for_display,
    uses_evidence_grounded_extraction,
)


@pytest.mark.parametrize("wrapper", ["candidate", "playbook", "lesson"])
def test_strict_extraction_schema_unwraps_complete_candidate(wrapper):
    candidate = {
        "Rationale": "A visible correction supports a future rule.",
        "Evidence Kind": "correction",
        "Trigger": "When the task begins",
        "Content": "Apply the corrected behavior.",
        "Evidence Refs": ["T1"],
    }

    output = StructuredReferencedExtractedPlaybookList.model_validate(
        {"playbooks": [{wrapper: candidate}]}
    )

    assert len(output.playbooks) == 1
    assert output.playbooks[0].evidence_kind == "correction"
    assert output.playbooks[0].evidence_refs == ["T1"]


def test_strict_extraction_schema_recovers_nested_alias_contract():
    output = StructuredReferencedExtractedPlaybookList.model_validate(
        {
            "playbooks": [
                {
                    "supported_signal": {
                        "candidates": [
                            {
                                "Reason": "The user corrected the behavior.",
                                "Type": "correction",
                                "Future Trigger": "When the task begins",
                                "Action": "Apply the corrected behavior.",
                                "Sources": ["T2"],
                            }
                        ]
                    }
                }
            ]
        }
    )

    assert output.playbooks[0].content == "Apply the corrected behavior."
    assert output.playbooks[0].evidence_refs == ["T2"]


def test_strict_extraction_schema_drops_evidence_object_but_keeps_valid_sibling():
    candidate = {
        "rationale": "A visible correction supports a future rule.",
        "evidence_kind": "correction",
        "trigger": "When the task begins",
        "content": "Apply the corrected behavior.",
        "evidence_refs": ["T1"],
    }

    output = StructuredReferencedExtractedPlaybookList.model_validate(
        {
            "playbooks": [
                {"evidence_ref": "T1"},
                candidate,
            ]
        }
    )

    assert len(output.playbooks) == 1
    assert output.playbooks[0].content == "Apply the corrected behavior."


def test_strict_extraction_schema_treats_evidence_only_list_as_empty():
    output = StructuredReferencedExtractedPlaybookList.model_validate(
        {
            "playbooks": [
                {"evidence_ref": "T1"},
                {"Evidence Ref": "T2"},
            ]
        }
    )

    assert output.playbooks == []


def test_strict_evidence_schema_follows_selected_candidate_prompt_version():
    """Strict extraction follows the independently selected prompt family."""
    assert uses_evidence_grounded_extraction(PromptManager(), expert=False)
    assert not uses_evidence_grounded_extraction(PromptManager(), expert=True)

    manager = PromptManager(
        version_override={
            "playbook_extraction_context": "4.5.0",
        }
    )
    assert not uses_evidence_grounded_extraction(manager, expert=False)
    assert not uses_evidence_grounded_extraction(manager, expert=True)


def test_prompt_context_uses_local_turn_refs_and_retains_provenance():
    """The model sees local labels while code retains real request/interaction IDs."""
    interaction = Interaction(
        interaction_id=84721,
        user_id="user_123",
        request_id="request-secret-42",
        content="Use the regional deployment checklist.",
        role="user",
        created_at=1_700_000_000,
        user_action=UserActionType.NONE,
        user_action_description="",
    )
    request = Request(
        request_id="request-secret-42",
        user_id="user_123",
        source="test",
        agent_version="1.0.0",
        session_id="session-secret-9",
        created_at=1_700_000_000,
    )
    context = build_playbook_prompt_context(
        [
            RequestInteractionDataModel(
                session_id="session-secret-9",
                request=request,
                interactions=[interaction],
            )
        ],
        label_turns=True,
    )

    assert "[T1]" in context.text
    assert "84721" not in context.text
    assert "request-secret-42" not in context.text
    assert "session-secret-9" not in context.text
    assert context.evidence_sources["T1"].interaction_id == 84721
    assert context.evidence_sources["T1"].request_id == "request-secret-42"
    assert context.evidence_sources["T1"].evidence_texts == (
        "Use the regional deployment checklist.",
    )
    assert context.evidence_sources["T1"].role == "user"
    assert context.evidence_sources["T1"].request_source == "test"
    assert context.evidence_units["T1"].source_span == (
        "Use the regional deployment checklist."
    )
    assert context.evidence_units["T1"].interaction_id == 84721


def test_prompt_context_does_not_label_turns_unless_requested():
    interaction = Interaction(
        interaction_id=12,
        user_id="user",
        request_id="request",
        content="Keep the active prompt contract unchanged.",
        role="user",
    )
    request = Request(
        request_id="request",
        user_id="user",
        source="test",
        agent_version="1",
        session_id="session",
    )

    context = build_playbook_prompt_context(
        [
            RequestInteractionDataModel(
                session_id="session",
                request=request,
                interactions=[interaction],
            )
        ]
    )

    assert "[T1]" not in context.text
    assert context.evidence_sources["T1"].interaction_id == 12


def test_prompt_context_maps_one_turn_reference_to_all_visible_sources():
    interaction = Interaction(
        interaction_id=13,
        user_id="user",
        request_id="request",
        content="First paragraph.\n\nSecond paragraph.",
        role="assistant",
        tools_used=[ToolUsed(tool_name="lookup", tool_data={"key": "value"})],
        user_action=UserActionType.CLICK,
        user_action_description="confirm",
    )
    request = Request(
        request_id="request",
        user_id="user",
        source="test",
        agent_version="1",
        session_id="session",
    )

    context = build_playbook_prompt_context(
        [
            RequestInteractionDataModel(
                session_id="session",
                request=request,
                interactions=[interaction],
            )
        ],
        label_turns=True,
    )

    assert list(context.evidence_units) == ["T1"]
    assert context.evidence_units["T1"].source_span == (
        '[used tool: lookup({"key": "value"})]\n\n'
        "First paragraph.\n\nSecond paragraph.\n\nclick confirm"
    )
    assert "[T1] assistant: ```[used tool:" in context.text
    assert "[T1] assistant: ```click confirm```" in context.text


def test_prompt_context_labels_turns_in_message_order_not_database_id_order():
    """Replay/imported IDs must not reorder the conversation shown to the model."""
    interactions = [
        Interaction(
            interaction_id=900,
            user_id="user_123",
            request_id="request-1",
            content="First turn",
            role="user",
            created_at=100,
        ),
        Interaction(
            interaction_id=100,
            user_id="user_123",
            request_id="request-1",
            content="Second turn",
            role="assistant",
            created_at=101,
        ),
    ]
    request = Request(
        request_id="request-1",
        user_id="user_123",
        source="test",
        agent_version="1.0.0",
        session_id="session-1",
        created_at=100,
    )

    context = build_playbook_prompt_context(
        [
            RequestInteractionDataModel(
                session_id="session-1",
                request=request,
                interactions=interactions,
            )
        ],
        label_turns=True,
    )

    assert context.text.index("[T1] user: ```First turn```") < context.text.index(
        "[T2] assistant: ```Second turn```"
    )
    assert context.evidence_sources["T1"].interaction_id == 900
    assert context.evidence_sources["T2"].interaction_id == 100


def test_prompt_context_preserves_request_boundaries_sources_and_chronology():
    requests = []
    for index, (created_at, source, content) in enumerate(
        [(100, "web", "First request"), (200, "api", "Second request")], start=1
    ):
        request_id = f"private-request-{index}"
        requests.append(
            RequestInteractionDataModel(
                session_id="private-session",
                request=Request(
                    request_id=request_id,
                    user_id="user",
                    source=source,
                    agent_version="1",
                    session_id="private-session",
                    created_at=created_at,
                ),
                interactions=[
                    Interaction(
                        interaction_id=1000 + index,
                        user_id="user",
                        request_id=request_id,
                        content=content,
                        role="user",
                        created_at=created_at,
                    )
                ],
            )
        )

    context = build_playbook_prompt_context(list(reversed(requests)), label_turns=True)

    assert context.text.index("[R1] Request (source: web)") < context.text.index(
        "[R2] Request (source: api)"
    )
    assert context.text.index("[T1] user: ```First request```") < context.text.index(
        "[T2] user: ```Second request```"
    )
    assert "private-request" not in context.text
    assert "private-session" not in context.text


def test_prompt_context_interleaves_linked_sessions_by_visible_turn_time():
    """A later-persisted quick reply must follow the question it answers."""

    def request_model(
        request_id: str,
        session_id: str,
        request_created_at: int,
        interaction_created_at: int,
        content: str,
    ) -> RequestInteractionDataModel:
        return RequestInteractionDataModel(
            session_id=session_id,
            request=Request(
                request_id=request_id,
                user_id="user",
                source="test",
                agent_version="1",
                session_id=session_id,
                created_at=request_created_at,
            ),
            interactions=[
                Interaction(
                    interaction_id=interaction_created_at,
                    user_id="user",
                    request_id=request_id,
                    content=content,
                    role="user",
                    created_at=interaction_created_at,
                )
            ],
        )

    context = build_playbook_prompt_context(
        [
            request_model("downstream", "main", 102, 120, "Generate now"),
            request_model("answer", "quick-reply", 103, 110, "History"),
            request_model("question", "main", 101, 100, "Which angle?"),
        ],
        label_turns=True,
    )

    assert context.text.index("[T1] user: ```Which angle?```") < context.text.index(
        "[T2] user: ```History```"
    )
    assert context.text.index("[T2] user: ```History```") < context.text.index(
        "[T3] user: ```Generate now```"
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
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
        Interaction(
            interaction_id=2,
            user_id="user_123",
            request_id="req_1",
            content="Here is how to access your account",
            role="assistant",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.NONE,
            user_action_description="",
        ),
        Interaction(
            interaction_id=3,
            user_id="user_123",
            request_id="req_1",
            content="Thank you!",
            role="user",
            created_at=int(datetime.now(UTC).timestamp()),
            user_action=UserActionType.CLICK,
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
                assert "[T3] user: ```click help button```" in content, (
                    "Expected the user action in the labelled turn"
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


def test_extraction_prompt_keeps_general_triggers_specific_actions():
    """Active extraction prompts should preserve retrieve-general/action-specific guidance."""
    prompt_manager = PromptManager()
    context_prompt = prompt_manager.render_prompt(
        "playbook_extraction_context",
        variables={
            "agent_context_prompt": "agent",
            "extraction_definition_prompt": "definition",
            "tool_can_use": "tools",
        },
    )
    main_prompt = prompt_manager.render_prompt(
        "playbook_extraction_main",
        variables={"interactions": "interaction text"},
    )

    context_normalized = " ".join(context_prompt.replace("*", "").split())
    main_normalized = " ".join(main_prompt.replace("*", "").split())
    assert "earliest observable future condition" in main_normalized
    assert "later mistake, correction" in main_normalized
    assert "earliest observable task class" in context_normalized
    assert "Do not wait for the mistake" in context_normalized
    assert "broad enough to retrieve the supported lesson" in context_normalized
    assert "narrow enough not to apply outside its evidence" in context_normalized


def test_consolidation_prompt_preserves_operational_surfaces():
    """Consolidation must not merge concrete fanout rules into vague sweep rules."""
    prompt_manager = PromptManager()
    rendered = prompt_manager.render_prompt(
        "playbook_consolidation",
        variables={
            "new_playbook_count": 1,
            "new_playbooks": "[NEW-0]\nContent: x",
            "existing_playbooks": "[EXISTING-0]\nContent: y",
        },
    )
    normalized = " ".join(rendered.split())

    assert "Preserve concrete operational surfaces" in normalized
    assert "target groups, SGs, health checks, task defs" in normalized
    assert "generic sweep/audit rule" in normalized


def _playbook(content: str, trigger: str | None = "When debugging") -> UserPlaybook:
    return UserPlaybook(
        agent_version="1.0",
        request_id="request_1",
        content=content,
        trigger=trigger,
    )


def test_dedupe_and_drop_empty_removes_blank_content():
    """Blank persisted playbooks are dropped before storage."""
    playbooks = [
        _playbook(""),
        _playbook("   \n\t"),
        _playbook("Run the narrow verification first."),
    ]

    assert dedupe_and_drop_empty(playbooks) == [playbooks[2]]


def test_dedupe_and_drop_empty_collapses_case_and_whitespace_duplicates():
    """Same-batch byte/fold-equivalent duplicates keep the first row."""
    first = _playbook("Run the narrow verification first.", " When Debugging ")
    duplicate = _playbook("  run the narrow verification first.  ", "when debugging")
    different_trigger = _playbook(
        "Run the narrow verification first.", "When preparing a PR"
    )

    assert dedupe_and_drop_empty([first, duplicate, different_trigger]) == [
        first,
        different_trigger,
    ]


# ===============================
# Tests for format_structured_fields_for_display and ensure_playbook_content
# ===============================


class TestFormatStructuredFieldsForDisplay:
    """Tests for the shared format_structured_fields_for_display function (display/debug formatting)."""

    def test_trigger_present(self):
        """Test formatting with trigger populated."""
        structured = StructuredPlaybookContent(
            trigger="explaining technical concepts to beginners",
        )
        result = format_structured_fields_for_display(structured)
        assert 'Trigger: "explaining technical concepts to beginners"' in result

    def test_trigger_none(self):
        """Test that None trigger is omitted from output."""
        structured = StructuredPlaybookContent(
            trigger=None,
        )
        result = format_structured_fields_for_display(structured)
        assert "Trigger:" not in result

    def test_trigger_empty_string(self):
        """Test that empty string trigger is omitted from output."""
        structured = StructuredPlaybookContent(
            trigger=None,
        )
        result = format_structured_fields_for_display(structured)
        assert "Trigger:" not in result

    def test_all_fields_none_returns_empty_string(self):
        """All-None structured fields produce no fallback content."""
        structured = StructuredPlaybookContent()
        result = format_structured_fields_for_display(structured)
        assert result == ""


# ===============================
# Tests for StructuredPlaybookContent freeform support
# ===============================


class TestStructuredPlaybookContentFreeform:
    """Tests for freeform playbook support in StructuredPlaybookContent."""

    def test_has_content_structured_only(self):
        """Structured playbook with trigger + content returns True."""
        sfc = StructuredPlaybookContent(
            trigger="user asks about X",
            content="do Y",
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
                        "content": "Explain root cause before fixes.",
                    },
                    {
                        "trigger": "agent provides a factual correction",
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
        shape ``{"trigger": ..., "content": ...}`` no longer parses
        as a StructuredPlaybookList — the broad ``except`` in
        ``PlaybookExtractor.extract_playbook_entries`` then logs the
        ValidationError and returns ``[]`` instead of silently building
        a malformed playbook.
        """
        with pytest.raises(ValueError):
            StructuredPlaybookList.model_validate(
                {
                    "trigger": "user asks for help debugging",
                    "content": "explain the root cause first",
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
                        "content": "respond helpfully",
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
            content="some observation",
        )
        result = format_structured_fields_for_display(sfc)
        assert "Trigger:" in result
        assert "some observation" not in result


# ===============================
# Tests for ensure_playbook_content
# ===============================


class TestEnsurePlaybookContent:
    """Tests for the ensure_playbook_content helper."""

    def test_returns_playbook_content_when_present(self):
        """When playbook content is a non-empty string, return it as-is."""
        structured = StructuredPlaybookContent(
            trigger="user asks X",
            content="do Y",
        )
        result = ensure_playbook_content("My freeform playbook", structured)
        assert result == "My freeform playbook"

    def test_falls_back_to_structured_when_none(self):
        """When playbook content is None, fall back to formatted structured fields."""
        structured = StructuredPlaybookContent(
            trigger="user asks X",
            content="do Y",
        )
        result = ensure_playbook_content(None, structured)
        assert 'Trigger: "user asks X"' in result

    def test_falls_back_to_structured_when_empty(self):
        """When playbook content is empty string, fall back to formatted structured fields."""
        structured = StructuredPlaybookContent(
            trigger="user asks X",
            content="do Y",
        )
        result = ensure_playbook_content("", structured)
        assert 'Trigger: "user asks X"' in result

    def test_falls_back_to_structured_when_whitespace_only(self):
        """When playbook content is whitespace-only, fall back to formatted structured fields."""
        structured = StructuredPlaybookContent(
            trigger="user asks X",
            content="do Y",
        )
        result = ensure_playbook_content("   ", structured)
        assert 'Trigger: "user asks X"' in result


# ===============================
# Tests for ensure_playbook_content freeform invariant
# ===============================


class TestEnsurePlaybookContentEdgeCases:
    """Additional edge cases for ensure_playbook_content."""

    def test_returns_empty_string_when_both_empty(self):
        """When playbook content is missing and no structured fields are set, fallback content is empty."""
        result = ensure_playbook_content(None, StructuredPlaybookContent())
        assert result == ""


# ===============================
# Tests for expert and incremental message construction
# ===============================


class TestConstructExpertPlaybookExtractionMessages:
    """Tests for construct_expert_playbook_extraction_messages."""

    def _make_expert_interactions(self):
        """Create interactions with expert_content for testing."""
        return [
            Interaction(
                interaction_id=1,
                user_id="user_1",
                request_id="req_1",
                content="How do I reset my password?",
                role="user",
                created_at=int(datetime.now(UTC).timestamp()),
            ),
            Interaction(
                interaction_id=2,
                user_id="user_1",
                request_id="req_1",
                content="Click on forgot password on the login page.",
                role="assistant",
                created_at=int(datetime.now(UTC).timestamp()),
                expert_content="Navigate to Settings > Security > Reset Password. Include the 48-hour cooling period warning.",
            ),
        ]

    def _make_request_data(self, interactions):
        request = Request(
            request_id="req_1",
            user_id="user_1",
            source="test",
            agent_version="1.0",
            session_id="session_1",
        )
        return [
            RequestInteractionDataModel(
                session_id="session_1",
                request=request,
                interactions=interactions,
            )
        ]

    def test_expert_messages_constructed_with_comparison_pairs(self):
        """Expert extraction should include comparison pairs in user message."""
        from reflexio.server.services.playbook.playbook_service_utils import (
            construct_expert_playbook_extraction_messages,
        )

        interactions = self._make_expert_interactions()
        ridms = self._make_request_data(interactions)
        prompt_manager = PromptManager()

        messages = construct_expert_playbook_extraction_messages(
            prompt_manager=prompt_manager,
            request_interaction_data_models=ridms,
            agent_context_prompt="Customer support agent",
            extraction_definition_prompt="Evaluate agent quality",
        )

        assert len(messages) > 0

        # Extract all text from messages
        all_text = ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        all_text += item.get("text", "")
            else:
                all_text += str(content)

        # System message should contain the agent context
        system_text = ""
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_text += item.get("text", "")
                else:
                    system_text += str(content)

        assert "Evaluate agent quality" in system_text
        assert "Customer support agent" in system_text

        # Should include comparison pair content
        assert "Agent Response" in all_text or "Expert Response" in all_text
        assert "[T1]" not in all_text

    def test_expert_prompt_no_instruction_pitfall(self):
        """Expert extraction prompt should not reference instruction or pitfall fields."""
        from reflexio.server.services.playbook.playbook_service_utils import (
            construct_expert_playbook_extraction_messages,
        )

        interactions = self._make_expert_interactions()
        ridms = self._make_request_data(interactions)
        prompt_manager = PromptManager()

        messages = construct_expert_playbook_extraction_messages(
            prompt_manager=prompt_manager,
            request_interaction_data_models=ridms,
            agent_context_prompt="Test agent",
            extraction_definition_prompt="Test focus",
        )

        # Check the system message doesn't have instruction/pitfall in the output schema
        system_text = ""
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_text += item.get("text", "")
                else:
                    system_text += str(content)

        # The new v3 prompt should NOT have instruction/pitfall in its output schema
        assert '"instruction"' not in system_text
        assert '"pitfall"' not in system_text


class TestHasExpertContent:
    """Tests for has_expert_content utility function."""

    def test_returns_true_when_expert_content_present(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            has_expert_content,
        )

        interactions = [
            Interaction(
                interaction_id=1,
                user_id="u1",
                request_id="r1",
                content="agent response",
                role="assistant",
                expert_content="better response",
            ),
        ]
        assert has_expert_content(interactions) is True

    def test_returns_false_when_no_expert_content(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            has_expert_content,
        )

        interactions = [
            Interaction(
                interaction_id=1,
                user_id="u1",
                request_id="r1",
                content="agent response",
                role="assistant",
            ),
        ]
        assert has_expert_content(interactions) is False

    def test_returns_false_for_empty_list(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            has_expert_content,
        )

        assert has_expert_content([]) is False


class TestEvidenceGroupLabels:
    """Group labels expose which candidates share provenance, without IDs."""

    @staticmethod
    def _row(source_ids: list[int]) -> UserPlaybook:
        return UserPlaybook(
            user_playbook_id=0,
            agent_version="v1",
            request_id="r1",
            playbook_name="pb",
            content="c",
            trigger="t",
            source="test",
            source_interaction_ids=source_ids,
        )

    def test_transitively_linked_rows_share_one_label(self):
        from reflexio.server.services.playbook.components.consolidator import (
            PlaybookConsolidator,
        )

        # 1-2 and 2-3 overlap, so all three are one connected component even
        # though rows 1 and 3 share nothing directly.
        labels = PlaybookConsolidator._evidence_group_labels(
            [self._row([1, 2]), self._row([2, 3]), self._row([3, 4])]
        )

        assert labels[0] == labels[1] == labels[2] != "none"

    def test_disjoint_rows_get_distinct_labels(self):
        from reflexio.server.services.playbook.components.consolidator import (
            PlaybookConsolidator,
        )

        labels = PlaybookConsolidator._evidence_group_labels(
            [self._row([1]), self._row([2])]
        )

        assert labels[0] != labels[1]
        assert "none" not in labels

    def test_rows_without_provenance_are_unlabelled(self):
        from reflexio.server.services.playbook.components.consolidator import (
            PlaybookConsolidator,
        )

        labels = PlaybookConsolidator._evidence_group_labels(
            [self._row([]), self._row([7])]
        )

        assert labels[0] == "none"
        assert labels[1] != "none"

    def test_labels_never_contain_interaction_ids(self):
        from reflexio.server.services.playbook.components.consolidator import (
            PlaybookConsolidator,
        )

        labels = PlaybookConsolidator._evidence_group_labels(
            [self._row([98765]), self._row([98765])]
        )

        assert all("98765" not in label for label in labels)


class TestEvidenceKindAliases:
    """Both extraction schemas must accept exactly the same spellings."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("observed_failure", "observed-failure"),
            ("failure", "observed-failure"),
            ("Rejection", "rejected-approach"),
            ("expert", "expert-gap"),
            ("SUCCESS", "verified-success"),
            ("correction", "correction"),
        ],
    )
    def test_alias_is_canonicalized_for_both_schemas(self, raw, expected):
        strict = StructuredReferencedExtractedPlaybookList.model_validate(
            {
                "playbooks": [
                    {
                        "rationale": "why",
                        "evidence_kind": raw,
                        "trigger": "when",
                        "content": "do",
                        "evidence_refs": ["T1"],
                    }
                ]
            }
        )
        legacy = StructuredPlaybookContent.model_validate(
            {"content": "do", "trigger": "when", "evidence_kind": raw}
        )

        assert strict.playbooks[0].evidence_kind == expected
        assert legacy.evidence_kind == expected


class TestConsolidationSearchKeys:
    """A cached retrieval is only reusable while the search keys hold."""

    @staticmethod
    def _row(trigger: str, content: str = "c") -> UserPlaybook:
        return UserPlaybook(
            user_playbook_id=0,
            agent_version="v1",
            request_id="r1",
            playbook_name="pb",
            content=content,
            trigger=trigger,
            source="test",
        )

    def test_unchanged_survivors_keep_the_cache_valid(self):
        from reflexio.server.services.playbook.service import (
            _consolidation_search_keys,
        )

        before = _consolidation_search_keys([self._row("when X"), self._row("when Y")])
        after = _consolidation_search_keys([self._row("when X")])

        assert after <= before

    def test_a_revised_trigger_invalidates_the_cache(self):
        from reflexio.server.services.playbook.service import (
            _consolidation_search_keys,
        )

        before = _consolidation_search_keys([self._row("when X")])
        after = _consolidation_search_keys([self._row("when X, narrowly")])

        assert not after <= before

    def test_content_is_the_key_when_trigger_is_absent(self):
        from reflexio.server.services.playbook.service import (
            _consolidation_search_keys,
        )

        assert _consolidation_search_keys([self._row("", content="fallback")]) == {
            "fallback"
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
