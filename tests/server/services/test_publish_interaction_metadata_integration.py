"""Integration test: customer-stamped metadata on publish_interaction
reaches the Request row in storage.

The publish path is the customer-facing surface. F2's value depends on
metadata flowing from the customer's call through to the Request entity
that the eval pipeline reads. This test locks in that contract end-to-end
through the real ``GenerationService.run`` + real SQLite storage.
"""

import datetime
import tempfile
from datetime import UTC
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_llm_responses():
    """Mock all LLM calls to avoid actual API calls.

    Mirrors the pattern used by ``test_generation_service.py``: ``should_extract``
    returns False so we don't drive the extraction pipeline; we only care that
    the Request row gets written with the metadata.
    """

    def mock_generate_chat_response_side_effect(messages, **kwargs):
        prompt_content = ""
        for message in messages:
            if isinstance(message, dict) and "content" in message:
                prompt_content += str(message["content"])

        if "Output just a boolean value" in prompt_content:
            return "false"
        if kwargs.get("parse_structured_output", False):
            return {"add": [], "update": [], "delete": []}
        return '```json\n{"add": [], "update": [], "delete": []}\n```'

    with patch(
        "reflexio.server.llm.litellm_client.LiteLLMClient.generate_chat_response",
        side_effect=mock_generate_chat_response_side_effect,
    ):
        yield


def test_publish_interaction_carries_metadata_to_request(mock_llm_responses):
    """Customer publishes with metadata; the stored Request row reflects it.

    Drives the real publish path (GenerationService.run) against a real
    SQLite-backed RequestContext in a temp dir, then reads back the Request
    rows by session and asserts the metadata round-tripped intact.
    """
    user_id = "u-publish-metadata-test"
    org_id = "org-publish-metadata-test"
    session_id = "s-publish-metadata-test"
    expected_metadata = {"reflexio_retrieval_enabled": True}

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        request_context = RequestContext(org_id=org_id, storage_base_dir=temp_dir)
        generation_service = GenerationService(
            llm_client=llm_client,
            request_context=request_context,
        )

        interaction = InteractionData(
            content="test interaction for metadata plumbing",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )

        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction],
            session_id=session_id,
            metadata=expected_metadata,
        )

        generation_service.run(publish_request)

        storage = request_context.storage
        assert storage is not None
        requests = storage.get_requests_by_session(user_id, session_id)
        assert len(requests) >= 1, (
            "publish should have created at least one Request row"
        )
        for stored_request in requests:
            assert stored_request.metadata == expected_metadata, (
                f"metadata mismatch on stored Request {stored_request.request_id}: "
                f"got {stored_request.metadata!r}, expected {expected_metadata!r}"
            )


def test_publish_interaction_defaults_metadata_to_empty_dict(mock_llm_responses):
    """When the customer omits ``metadata``, the stored Request has an empty dict.

    Locks in backward compatibility: existing callers that never pass
    ``metadata`` keep working and the field defaults to ``{}`` (never None).
    """
    user_id = "u-publish-metadata-default"
    org_id = "org-publish-metadata-default"
    session_id = "s-publish-metadata-default"

    with tempfile.TemporaryDirectory() as temp_dir:
        llm_config = LiteLLMConfig(model="gpt-4o-mini")
        llm_client = LiteLLMClient(llm_config)
        request_context = RequestContext(org_id=org_id, storage_base_dir=temp_dir)
        generation_service = GenerationService(
            llm_client=llm_client,
            request_context=request_context,
        )

        interaction = InteractionData(
            content="test interaction without metadata",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )

        publish_request = PublishUserInteractionRequest(
            user_id=user_id,
            interaction_data_list=[interaction],
            session_id=session_id,
        )

        generation_service.run(publish_request)

        storage = request_context.storage
        assert storage is not None
        requests = storage.get_requests_by_session(user_id, session_id)
        assert len(requests) >= 1
        for stored_request in requests:
            assert stored_request.metadata == {}, (
                f"default metadata should be empty dict, got {stored_request.metadata!r}"
            )
