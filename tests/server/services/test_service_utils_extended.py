"""Extended tests for service_utils -- covers functions not tested in test_service_utils.py."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from reflexio.models.api_schema.service_schemas import (
    Interaction,
    ToolUsed,
)
from reflexio.server.services.service_utils import (
    MessageConstructionConfig,
    PromptConfig,
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
    interaction = Interaction(
        interaction_id=1,
        user_id="u1",
        request_id="r1",
        content="describe this",
        role="user",
        created_at=int(datetime.now(UTC).timestamp()),
        image_encoding="abc123base64data",
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
        image_blocks[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123base64data"
    )


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
