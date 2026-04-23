"""LiteLLMClient extensions for tool-calling (Task 1.3)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    ToolCallingChatResponse,
)
from reflexio.server.llm.model_defaults import ModelRole

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_tool_call_response(tool_name: str, args_json: str) -> MagicMock:
    """Build a MagicMock shaped like a litellm tool-call response."""
    tool_call = MagicMock()
    tool_call.function.name = tool_name
    tool_call.function.arguments = args_json

    message = MagicMock()
    message.content = None
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls"

    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return response


def _mock_text_response(text: str) -> MagicMock:
    """Build a MagicMock shaped like a normal litellm text response."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolCallingExtensions:
    """Tests for tools/tool_choice/model_role kwargs on LiteLLMClient."""

    def test_generate_chat_response_passes_tools_kwarg(self) -> None:
        """tools + tool_choice are forwarded to litellm.completion; result is ToolCallingChatResponse."""
        config = LiteLLMConfig(model="gpt-4o")
        client = LiteLLMClient(config)

        mock_response = _mock_tool_call_response("emit_profile", '{"name": "Alice"}')

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "emit_profile",
                    "description": "Emit a profile",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "hello"}],
                tools=tools,
                tool_choice="auto",
            )

        # The tools and tool_choice kwargs must have been forwarded
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"

        # The result must be a ToolCallingChatResponse
        assert isinstance(result, ToolCallingChatResponse)
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.name == "emit_profile"
        assert result.finish_reason == "tool_calls"
        assert result.content is None

    def test_model_role_resolves_to_angle_reader_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """model_role=ANGLE_READER resolves to the anthropic angle_reader default model."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Ensure no other provider keys interfere
        for var in (
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
            "CLAUDE_SMART_USE_LOCAL_CLI",
        ):
            monkeypatch.delenv(var, raising=False)

        config = LiteLLMConfig(model="gpt-4o")
        client = LiteLLMClient(config)

        mock_response = _mock_text_response("hi")

        with patch("litellm.completion", return_value=mock_response) as mock_completion:
            client.generate_chat_response(
                messages=[{"role": "user", "content": "hello"}],
                model_role=ModelRole.ANGLE_READER,
            )

        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_non_tool_path_unchanged(self) -> None:
        """Without tools kwarg the existing str-return path is untouched."""
        config = LiteLLMConfig(model="gpt-4o")
        client = LiteLLMClient(config)

        mock_response = _mock_text_response("hi")

        with patch("litellm.completion", return_value=mock_response):
            result = client.generate_chat_response(
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result == "hi"
        assert not isinstance(result, ToolCallingChatResponse)
