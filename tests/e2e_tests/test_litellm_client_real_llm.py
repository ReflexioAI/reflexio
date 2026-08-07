"""Real-LLM e2e tests: LiteLLM client against OpenAI and Anthropic.

The live half of ``tests/server/llm/test_litellm_client.py``. It lives here
because ``llm_mock._is_e2e_test_run`` exempts only paths under ``e2e_tests/``
from the session-wide ``litellm.completion`` patch -- from its old home these
"live" tests answered from the mock, so they asserted nothing about either
provider no matter which keys were set.

The offline half stays behind: those tests build a client from a literal config
and assert how the config resolved, need no credential, and run in the default
unit tier.

Run with:
    set -a && source .env && set +a && \
    RUN_LOW_PRIORITY=1 uv run pytest tests/e2e_tests/test_litellm_client_real_llm.py \
      -v -o 'addopts=' -n 0
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator, Iterator

import pytest

from reflexio.models.config_schema import (
    AnthropicConfig,
    APIKeyConfig,
    OpenAIConfig,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    create_litellm_client,
)
from reflexio.test_support.llm_credentials import real_provider_key
from reflexio.test_support.llm_mock import assert_litellm_unpatched
from tests.server.llm.test_litellm_client import MathResult, create_minimal_png
from tests.server.test_utils import skip_in_precommit, skip_low_priority

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_credentials,
    pytest.mark.skipif(
        not real_provider_key("OPENAI_API_KEY")
        and not real_provider_key("ANTHROPIC_API_KEY"),
        reason="Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY environment variable is set",
    ),
]


@pytest.fixture(autouse=True)
def _requires_unmocked_litellm() -> Iterator[None]:
    """Fail loudly rather than assert against the mock.

    The whole reason this file is under ``e2e_tests/``. If it is ever invoked
    in a way that leaves the global patch on, every provider assertion below
    would be checking canned mock text.
    """
    assert_litellm_unpatched()
    yield


def _get_openai_test_model() -> str:
    """Get an OpenAI model for testing."""
    return os.getenv("OPENAI_TEST_MODEL", "gpt-5.4-mini")


def _get_claude_test_model() -> str:
    """Get a Claude model for testing."""
    return os.getenv("ANTHROPIC_TEST_MODEL", "claude-sonnet-4-5-20250929")


@pytest.fixture
def openai_client() -> LiteLLMClient:
    """Create an OpenAI-based LiteLLM client."""
    if not real_provider_key("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    # GPT-5 models use reasoning tokens internally, so we need more max_tokens
    return create_litellm_client(
        model=_get_openai_test_model(),
        temperature=0.1,
        max_tokens=500,
        max_retries=2,
    )


@pytest.fixture
def claude_client() -> LiteLLMClient:
    """Create a Claude-based LiteLLM client."""
    if not real_provider_key("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    return create_litellm_client(
        model=_get_claude_test_model(),
        temperature=0.1,
        max_tokens=256,
        max_retries=2,
    )


@pytest.fixture
def test_image_bytes() -> bytes:
    """Create a test PNG image as bytes (solid red)."""
    return create_minimal_png(width=50, height=50, color=(255, 0, 0))


@pytest.fixture
def test_image_file(test_image_bytes: bytes) -> Generator[str, None, None]:
    """Create a temporary PNG image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(test_image_bytes)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestLiteLLMClientOpenAI:
    """Test LiteLLM client with OpenAI models."""

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_simple(self, openai_client: LiteLLMClient):
        """Test simple text generation with OpenAI."""
        response = openai_client.generate_response(
            "What is 2+2? Answer with just the number."
        )

        assert isinstance(response, str)
        assert "4" in response

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_system_message(self, openai_client: LiteLLMClient):
        """Test response with system message."""
        response = openai_client.generate_response(
            prompt="What color is the sky?",
            system_message="You are a pirate. Respond like a pirate.",
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_generate_chat_response(self, openai_client: LiteLLMClient):
        """Test multi-turn chat response."""
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]

        response = openai_client.generate_chat_response(messages)

        assert isinstance(response, str)
        assert "alice" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_chat_response_with_system_message(
        self, openai_client: LiteLLMClient
    ):
        """Test chat response with prepended system message."""
        messages = [
            {"role": "user", "content": "Tell me a joke."},
        ]

        response = openai_client.generate_chat_response(
            messages,
            system_message="You are a comedian who only tells very short one-liner jokes.",
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_structured_output_with_pydantic(self, openai_client: LiteLLMClient):
        """Test structured output using Pydantic model."""
        response = openai_client.generate_response(
            prompt="What is 5+5? Provide the answer and a brief explanation.",
            response_format=MathResult,
        )

        # Should be parsed to Pydantic model
        assert isinstance(response, MathResult)
        assert response.answer == 10
        assert len(response.explanation) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_embeddings(self, openai_client: LiteLLMClient):
        """Test embedding generation."""
        embedding = openai_client.get_embedding("Hello, world!")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


class TestLiteLLMClientClaude:
    """Test LiteLLM client with Claude models."""

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_simple(self, claude_client: LiteLLMClient):
        """Test simple text generation with Claude."""
        response = claude_client.generate_response(
            "What is 3+3? Answer with just the number."
        )

        assert isinstance(response, str)
        assert "6" in response

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_system_message(self, claude_client: LiteLLMClient):
        """Test response with system message."""
        response = claude_client.generate_response(
            prompt="What color is grass?",
            system_message="You are a robot. Respond in a robotic manner.",
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_generate_chat_response(self, claude_client: LiteLLMClient):
        """Test multi-turn chat response."""
        messages = [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "Blue is a nice color!"},
            {"role": "user", "content": "What is my favorite color?"},
        ]

        response = claude_client.generate_chat_response(messages)

        assert isinstance(response, str)
        assert "blue" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_structured_output_with_pydantic(self, claude_client: LiteLLMClient):
        """Test structured output using Pydantic model with Claude."""
        response = claude_client.generate_response(
            prompt="What is 7+7? Provide the answer and a brief explanation.",
            response_format=MathResult,
        )

        # Should be parsed to Pydantic model
        assert isinstance(response, MathResult)
        assert response.answer == 14
        assert len(response.explanation) > 0


class TestLiteLLMClientMultiModal:
    """Test LiteLLM client with image inputs."""

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_image_file_openai(
        self, openai_client: LiteLLMClient, test_image_file: str
    ):
        """Test image analysis with file path (OpenAI)."""
        response = openai_client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_file],
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_image_bytes_openai(
        self, openai_client: LiteLLMClient, test_image_bytes: bytes
    ):
        """Test image analysis with bytes (OpenAI)."""
        response = openai_client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_bytes],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_image_file_claude(
        self, claude_client: LiteLLMClient, test_image_file: str
    ):
        """Test image analysis with file path (Claude)."""
        response = claude_client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_file],
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_image_bytes_claude(
        self, claude_client: LiteLLMClient, test_image_bytes: bytes
    ):
        """Test image analysis with bytes (Claude)."""
        response = claude_client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_bytes],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        assert "red" in response.lower()

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_multiple_images(self, openai_client: LiteLLMClient):
        """Test analysis of multiple images."""
        red_image = create_minimal_png(20, 20, (255, 0, 0))
        blue_image = create_minimal_png(20, 20, (0, 0, 255))

        response = openai_client.generate_response(
            prompt="What are the colors of these two images? List both.",
            images=[red_image, blue_image],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        response_lower = response.lower()
        assert "red" in response_lower
        assert "blue" in response_lower

    @skip_in_precommit
    @skip_low_priority
    def test_image_with_system_message(
        self, openai_client: LiteLLMClient, test_image_bytes: bytes
    ):
        """Test image analysis with system message."""
        response = openai_client.generate_response(
            prompt="Describe this image.",
            system_message="You are an art critic. Be very brief.",
            images=[test_image_bytes],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        assert len(response) > 0


class TestLiteLLMClientModelSwitching:
    """Test switching between different models."""

    @skip_in_precommit
    @skip_low_priority
    def test_switch_from_openai_to_claude(self):
        """Test that we can create clients for different providers."""
        if not real_provider_key("OPENAI_API_KEY") or not real_provider_key(
            "ANTHROPIC_API_KEY"
        ):
            pytest.skip("Both OPENAI_API_KEY and ANTHROPIC_API_KEY required")

        # Create OpenAI client (GPT-5 needs more tokens due to reasoning)
        openai_client = create_litellm_client(
            model="gpt-5.4-mini",
            temperature=0.1,
            max_tokens=300,
        )
        openai_response = openai_client.generate_response("Say 'hello' only.")

        # Create Claude client
        claude_client = create_litellm_client(
            model="claude-sonnet-4-5-20250929",
            temperature=0.1,
            max_tokens=100,
        )
        claude_response = claude_client.generate_response("Say 'hello' only.")

        # Both should work
        assert isinstance(openai_response, str)
        assert isinstance(claude_response, str)
        assert "hello" in openai_response.lower()
        assert "hello" in claude_response.lower()


class TestLiteLLMClientEdgeCases:
    """Test edge cases that require a real provider."""

    @skip_in_precommit
    @skip_low_priority
    def test_long_conversation(self, openai_client: LiteLLMClient):
        """Test handling of longer conversations."""
        messages = []
        for i in range(5):
            messages.append({"role": "user", "content": f"Message {i + 1}: Hello!"})
            messages.append({"role": "assistant", "content": f"Response {i + 1}: Hi!"})
        messages.append(
            {"role": "user", "content": "How many times did we exchange greetings?"}
        )

        response = openai_client.generate_chat_response(messages)

        assert isinstance(response, str)
        # Should mention 5 or "five" somewhere in the response
        response_lower = response.lower()
        assert "5" in response or "five" in response_lower


class TestLiteLLMClientAPIKeyOverride:
    """Test API key configuration override against real providers."""

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_api_key_override_openai(self):
        """Test generating response using OpenAI API key from config override."""
        openai_key = real_provider_key("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Use real key from env but pass through api_key_config
        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model=_get_openai_test_model(),
            temperature=0.1,
            max_tokens=500,
            max_retries=2,
            api_key_config=api_key_config,
        )

        response = client.generate_response("What is 1+1? Answer with just the number.")

        assert isinstance(response, str)
        assert "2" in response

    @skip_in_precommit
    @skip_low_priority
    def test_generate_response_with_api_key_override_anthropic(self):
        """Test generating response using Anthropic API key from config override."""
        anthropic_key = real_provider_key("ANTHROPIC_API_KEY")
        if not anthropic_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        # Use real key from env but pass through api_key_config
        api_key_config = APIKeyConfig(anthropic=AnthropicConfig(api_key=anthropic_key))
        client = create_litellm_client(
            model=_get_claude_test_model(),
            temperature=0.1,
            max_tokens=256,
            max_retries=2,
            api_key_config=api_key_config,
        )

        response = client.generate_response("What is 4+4? Answer with just the number.")

        assert isinstance(response, str)
        assert "8" in response

    @skip_in_precommit
    @skip_low_priority
    def test_embeddings_with_api_key_override(self):
        """Test embedding generation with API key override."""
        openai_key = real_provider_key("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model="gpt-5.4-mini",
            api_key_config=api_key_config,
        )

        embedding = client.get_embedding("Test embedding with API key override")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @skip_in_precommit
    @skip_low_priority
    def test_structured_output_with_api_key_override(self):
        """Test structured output with API key override."""
        openai_key = real_provider_key("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model=_get_openai_test_model(),
            temperature=0.1,
            max_tokens=500,
            api_key_config=api_key_config,
        )

        response = client.generate_response(
            prompt="What is 3+3? Provide the answer and a brief explanation.",
            response_format=MathResult,
        )

        assert isinstance(response, MathResult)
        assert response.answer == 6
        assert len(response.explanation) > 0

    @skip_in_precommit
    @skip_low_priority
    def test_chat_response_with_api_key_override(self):
        """Test multi-turn chat response with API key override."""
        openai_key = real_provider_key("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model=_get_openai_test_model(),
            temperature=0.1,
            max_tokens=500,
            api_key_config=api_key_config,
        )

        messages = [
            {"role": "user", "content": "My favorite number is 42."},
            {"role": "assistant", "content": "That's a great number!"},
            {"role": "user", "content": "What is my favorite number?"},
        ]

        response = client.generate_chat_response(messages)

        assert isinstance(response, str)
        assert "42" in response

    @skip_in_precommit
    @skip_low_priority
    def test_image_analysis_with_api_key_override(self, test_image_bytes: bytes):
        """Test image analysis with API key override."""
        openai_key = real_provider_key("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("OPENAI_API_KEY not set")

        api_key_config = APIKeyConfig(openai=OpenAIConfig(api_key=openai_key))
        client = create_litellm_client(
            model=_get_openai_test_model(),
            temperature=0.1,
            max_tokens=500,
            api_key_config=api_key_config,
        )

        response = client.generate_response(
            prompt="What color is this image? Answer in one word.",
            images=[test_image_bytes],
            image_media_type="image/png",
        )

        assert isinstance(response, str)
        assert "red" in response.lower()
