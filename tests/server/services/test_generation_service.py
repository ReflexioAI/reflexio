import datetime
import tempfile
from datetime import UTC
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService


@pytest.fixture
def mock_llm_responses():
    """Mock all LLM calls to avoid actual API calls"""

    def mock_generate_chat_response_side_effect(messages, **kwargs):
        """Mock LLM responses for different types of calls"""
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        # Check if this is a should_extract_profile call
        if "Output just a boolean value" in prompt_content:
            return "false"  # Don't extract profiles in this test
        # For structured output parsing
        if kwargs.get("parse_structured_output", False):
            return {"add": [], "update": [], "delete": []}
        return '```json\n{"add": [], "update": [], "delete": []}\n```'

    with patch(
        "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
        side_effect=mock_generate_chat_response_side_effect,
    ):
        yield


def test_publish_request_with_session_id(mock_llm_responses):
    """
    Test that requests with a session_id are stored correctly.
    """
    user_id = "test_user_id"
    org_id = "test_org"
    session_id = "test_session_id"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        generation_service = GenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        interaction = InteractionData(
            content="test interaction",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )

        request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction],
            session_id=session_id,
        )

        # Request should succeed
        generation_service.run(request)


def test_empty_session_id_allows_multiple_requests(mock_llm_responses):
    """
    Test that multiple requests with empty session_id are allowed.
    """
    user_id = "test_user_id"
    org_id = "test_org"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        generation_service = GenerationService(
            llm_client=llm_client,
            request_context=RequestContext(org_id=org_id, storage_base_dir=temp_dir),
        )

        interaction = InteractionData(
            content="interaction without session",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )

        # Request without session_id (empty string)
        request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction],
            session_id="",  # Empty session
        )

        # Should not raise any exception
        generation_service.run(request)

        # Try another request with empty session_id - should also succeed
        another_interaction = InteractionData(
            content="another interaction without session",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )

        another_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[another_interaction],
            session_id="",
        )

        # Should not raise any exception
        generation_service.run(another_request)


# NOTE: TestWindowSizeStrideOverrides class was removed because the global
# _get_extraction_window_size() and _get_stride_size() methods were removed
# from GenerationService. Each extractor now handles its own window/stride
# calculation using the get_extractor_window_params() utility function.
# See: reflexio/server/services/extractor_interaction_utils.py
