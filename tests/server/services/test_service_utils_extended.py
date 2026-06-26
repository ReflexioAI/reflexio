"""Extended tests for service_utils -- covers functions not tested in test_service_utils.py."""

import base64
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    ToolUsed,
)
from reflexio.server.services.service_utils import (
    MessageConstructionConfig,
    PromptConfig,
    _image_data_url_from_encoding,
    construct_messages_from_interactions,
    extract_json_from_string,
    format_interactions_to_history_string,
    format_messages_for_logging,
)

# ---------------------------------------------------------------------------
# format_interactions_to_history_string -- tools_used branch
# ---------------------------------------------------------------------------


def test_format_interactions_with_tools_used():
    tool = ToolUsed(tool_name="search", tool_data={"input": {"query": "test"}})
    interaction = Interaction(
        interaction_id=1,
        user_id="u1",
        request_id="r1",
        content="Here are the results",
        role="assistant",
        created_at=int(datetime.now(UTC).timestamp()),
        tools_used=[tool],
    )

    result = format_interactions_to_history_string([interaction])
    assert '[used tool: search({"input": {"query": "test"}})]' in result
    assert "assistant: ```[used tool: search" in result
    assert "Here are the results```" in result


# ---------------------------------------------------------------------------
# extract_json_from_string
# ---------------------------------------------------------------------------


def test_extract_json_from_code_block():
    text = '```json\n{"key": "value"}\n```'
    result = extract_json_from_string(text)
    assert result == {"key": "value"}


def test_extract_json_from_braces():
    text = 'some text {"key": "value"} more text'
    result = extract_json_from_string(text)
    assert result == {"key": "value"}


def test_extract_json_python_booleans():
    text = '{"flag": True, "other": False, "none_val": None}'
    result = extract_json_from_string(text)
    assert result == {"flag": True, "other": False, "none_val": None}


def test_extract_json_single_quotes():
    text = "{'key': 'value'}"
    result = extract_json_from_string(text)
    assert result == {"key": "value"}


def test_extract_json_invalid():
    result = extract_json_from_string("no json here")
    assert result == {}


# ---------------------------------------------------------------------------
# construct_messages_from_interactions
# ---------------------------------------------------------------------------


def _make_prompt_manager() -> MagicMock:
    pm = MagicMock()
    pm.render_prompt.return_value = "rendered prompt"
    return pm


def test_construct_messages_with_image_url():
    pm = _make_prompt_manager()
    interaction = Interaction(
        interaction_id=1,
        user_id="u1",
        request_id="r1",
        content="describe this",
        role="user",
        created_at=int(datetime.now(UTC).timestamp()),
        interacted_image_url="https://example.com/img.png",
    )
    config = MessageConstructionConfig(
        prompt_manager=pm,
        user_prompt_config=PromptConfig(prompt_id="p1", variables={}),
    )

    messages = construct_messages_from_interactions([interaction], config)
    user_msg = messages[-1]
    assert user_msg["role"] == "user"
    # Content should be a list (mixed text + image)
    assert isinstance(user_msg["content"], list)
    image_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"] == "https://example.com/img.png"


def test_construct_messages_with_image_encoding():
    pm = _make_prompt_manager()
    image_encoding = base64.b64encode(b"RIFF\x00\x00\x00\x00WEBP").decode("ascii")
    interaction = Interaction(
        interaction_id=1,
        user_id="u1",
        request_id="r1",
        content="describe this",
        role="user",
        created_at=int(datetime.now(UTC).timestamp()),
        image_encoding=image_encoding,
    )
    config = MessageConstructionConfig(
        prompt_manager=pm,
        user_prompt_config=PromptConfig(prompt_id="p1", variables={}),
    )

    messages = construct_messages_from_interactions([interaction], config)
    user_msg = messages[-1]
    assert isinstance(user_msg["content"], list)
    image_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
    assert len(image_blocks) == 1
    assert (
        image_blocks[0]["image_url"]["url"]
        == f"data:image/webp;base64,{image_encoding}"
    )


def test_image_data_url_from_encoding_rejects_unknown_signature():
    image_encoding = base64.b64encode(b"not-an-image").decode("ascii")

    with pytest.raises(ValueError, match="Unsupported image signature"):
        _image_data_url_from_encoding(image_encoding)


def test_construct_messages_text_flattening():
    pm = _make_prompt_manager()
    config = MessageConstructionConfig(
        prompt_manager=pm,
        user_prompt_config=PromptConfig(prompt_id="p1", variables={}),
    )

    messages = construct_messages_from_interactions([], config)
    user_msg = messages[-1]
    assert user_msg["role"] == "user"
    # All-text content should be flattened to a plain string
    assert isinstance(user_msg["content"], str)
    assert user_msg["content"] == "rendered prompt"


def test_construct_messages_with_system_prompt():
    pm = _make_prompt_manager()
    config = MessageConstructionConfig(
        prompt_manager=pm,
        system_prompt_config=PromptConfig(prompt_id="sys", variables={"k": "v"}),
        user_prompt_config=PromptConfig(prompt_id="usr", variables={}),
    )

    messages = construct_messages_from_interactions([], config)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "rendered prompt"
    assert messages[1]["role"] == "user"


def test_construct_messages_empty():
    pm = _make_prompt_manager()
    config = MessageConstructionConfig(prompt_manager=pm)

    messages = construct_messages_from_interactions([], config)
    assert messages == []


# ---------------------------------------------------------------------------
# format_messages_for_logging
# ---------------------------------------------------------------------------


def test_format_messages_for_logging_string_content():
    messages = [{"role": "user", "content": "Hello world"}]
    result = format_messages_for_logging(messages)
    assert "Message 1:" in result
    assert "role: user" in result
    assert "Hello world" in result


def test_format_messages_for_logging_list_content():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img.png"},
                },
            ],
        }
    ]
    result = format_messages_for_logging(messages)
    assert "Message 1:" in result
    assert "role: user" in result
    assert "Describe this image" in result
    assert "image_url" in result


def test_format_messages_for_logging_renders_assistant_tool_calls_sdk_shape():
    """Assistant messages with SDK-object tool_calls must render id/name/arguments.

    Before this fix, an assistant message with ``content=None`` and only
    ``tool_calls`` looked like ``content: null`` with zero visibility into
    the tools the model invoked.
    """
    from types import SimpleNamespace

    tc = SimpleNamespace(
        id="call_abc",
        function=SimpleNamespace(
            name="flag_cross_entity_conflict",
            arguments='{"candidate_index":0,"reason":"contradicts profile"}',
        ),
    )
    messages = [{"role": "assistant", "content": None, "tool_calls": [tc]}]

    result = format_messages_for_logging(messages)

    assert "role: assistant" in result
    assert "content: null" in result
    assert "tool_calls:" in result
    assert "- id: call_abc" in result
    assert "name: flag_cross_entity_conflict" in result
    # Arguments should be parsed + re-serialised for readability
    assert '"candidate_index": 0' in result
    assert '"reason": "contradicts profile"' in result


def test_format_messages_for_logging_renders_assistant_tool_calls_dict_shape():
    """Pass-through serialisation sometimes produces dict-shaped tool_calls."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "type": "function",
                    "function": {
                        "name": "emit_profile",
                        "arguments": '{"content":"User likes Go","time_to_live":"infinity"}',
                    },
                }
            ],
        }
    ]

    result = format_messages_for_logging(messages)

    assert "- id: call_xyz" in result
    assert "name: emit_profile" in result
    assert '"content": "User likes Go"' in result


def test_format_messages_for_logging_renders_tool_call_id_on_tool_role():
    """Tool-role messages must surface tool_call_id so readers can correlate."""
    messages = [
        {"role": "tool", "tool_call_id": "call_abc", "content": '{"flagged": 0}'},
    ]

    result = format_messages_for_logging(messages)

    assert "role: tool" in result
    assert "tool_call_id: call_abc" in result
    assert '{"flagged": 0}' in result


def test_format_messages_for_logging_handles_malformed_arguments_json():
    """Tool_call arguments that aren't valid JSON should fall back to raw string."""
    from types import SimpleNamespace

    tc = SimpleNamespace(
        id="call_bad",
        function=SimpleNamespace(name="emit", arguments="not valid json {"),
    )
    messages = [{"role": "assistant", "content": None, "tool_calls": [tc]}]

    result = format_messages_for_logging(messages)

    # Formatter must not crash, and should preserve the raw string
    assert "name: emit" in result
    assert "not valid json {" in result


def test_format_messages_for_logging_skips_tool_calls_block_when_absent():
    """Assistant messages without tool_calls don't emit a ``tool_calls:`` header."""
    messages = [{"role": "assistant", "content": "plain text response"}]

    result = format_messages_for_logging(messages)

    assert "tool_calls:" not in result
    assert "plain text response" in result


# ---------------------------------------------------------------------------
# _format_response_for_logging — ToolCallingChatResponse rendering
# ---------------------------------------------------------------------------


def test_format_response_renders_tool_calling_chat_response_with_sdk_tool_calls():
    """ToolCallingChatResponse with SDK-shaped tool_calls renders id/name/arguments."""
    from types import SimpleNamespace

    from reflexio.server.llm.litellm_client import ToolCallingChatResponse
    from reflexio.server.services.service_utils import _format_response_for_logging

    tc = SimpleNamespace(
        id="call_abc",
        function=SimpleNamespace(name="rank", arguments='{"ordered_ids":["b1","b2"]}'),
    )
    resp = ToolCallingChatResponse(
        content=None, tool_calls=[tc], finish_reason="tool_calls"
    )

    out = _format_response_for_logging(resp)

    assert isinstance(out, str)
    assert "ToolCallingChatResponse(finish_reason='tool_calls')" in out
    assert "content: None" in out
    assert "tool_calls:" in out
    assert "- id: call_abc" in out
    assert "name: rank" in out
    # Arguments are parsed from JSON + re-serialized for readability
    assert '"ordered_ids": ["b1", "b2"]' in out


def test_format_response_renders_tool_calling_chat_response_with_empty_tool_calls():
    """ToolCallingChatResponse with no tool_calls still renders content + finish_reason."""
    from reflexio.server.llm.litellm_client import ToolCallingChatResponse
    from reflexio.server.services.service_utils import _format_response_for_logging

    resp = ToolCallingChatResponse(
        content="plain text reply", tool_calls=None, finish_reason="stop"
    )

    out = _format_response_for_logging(resp)

    assert "ToolCallingChatResponse(finish_reason='stop')" in out
    assert "content: 'plain text reply'" in out
    assert "tool_calls: []" in out


def test_format_response_passes_basemodel_through_unchanged():
    """Pydantic BaseModel responses (classic extractor / deduplicator outputs)
    must NOT be transformed — preserves existing llm_io.log shape for classic."""
    from pydantic import BaseModel

    from reflexio.server.services.service_utils import _format_response_for_logging

    class FakeClassicOutput(BaseModel):
        profiles: list[str] = []

    resp = FakeClassicOutput(profiles=["User likes polars"])

    out = _format_response_for_logging(resp)

    # The helper returned the same object — caller's %s formatter will
    # render it via str(resp) exactly as today.
    assert out is resp


def test_format_response_passes_string_through_unchanged():
    """Plain strings go straight through (tool_loop handlers return strings)."""
    from reflexio.server.services.service_utils import _format_response_for_logging

    out = _format_response_for_logging("raw string response")
    assert out == "raw string response"
