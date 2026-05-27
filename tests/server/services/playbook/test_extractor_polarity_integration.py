"""
Integration tests for polarity emission by the classic playbook extractor.

These tests verify the end-to-end polarity flow from LLM response through
``PlaybookExtractor.run()`` to the produced ``UserPlaybook`` objects:

* Neutral / no-failure windows produce playbooks with ``polarity="positive"``
  via the schema default — confirming the default-polarity threading.
* Failure-evidence windows with explicit ``polarity="negative"`` LLM output
  produce playbooks with negative polarity AND content prefixed by one of
  ``NEGATIVE_PREFIXES`` — confirming the negative path is preserved end-to-end.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)
from reflexio.models.config_schema import PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.playbook.playbook_extractor import PlaybookExtractor
from reflexio.server.services.playbook.playbook_generation_service import (
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
    StructuredPlaybookList,
)
from reflexio.server.services.polarity_utils import NEGATIVE_PREFIXES

pytestmark = pytest.mark.integration


# ===============================
# Fixtures
# ===============================


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage isolation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def request_context(temp_storage_dir, worker_id):
    """Create a request context with a mocked storage and per-worker org isolation."""
    org_id = f"polarity_int_{worker_id}"
    context = RequestContext(org_id=org_id, storage_base_dir=temp_storage_dir)
    context.storage = MagicMock()
    # Mock prompt manager so the extractor can render the extraction prompt
    # without depending on the on-disk prompt registry.
    context.prompt_manager = MagicMock()
    context.prompt_manager.render_prompt.return_value = "mock prompt"
    context.prompt_manager.get_active_version.return_value = "1.2.0"
    return context


@pytest.fixture
def mock_llm_client():
    """Create a mock LiteLLM client."""
    return MagicMock(spec=LiteLLMClient)


@pytest.fixture
def extractor_config():
    """Classic playbook extractor config (non-expert)."""
    return PlaybookConfig(
        extractor_name="quality_playbook",
        extraction_definition_prompt="Evaluate agent quality",
    )


@pytest.fixture
def service_config():
    """Runtime service config for the extractor."""
    return PlaybookGenerationServiceConfig(
        agent_version="1.0.0",
        request_id="test_request",
        source="api",
    )


@pytest.fixture
def neutral_request_interaction_models():
    """A neutral window — user is satisfied, no pushback / failure evidence."""
    request = Request(
        request_id="req_neutral",
        user_id="user_neutral",
        created_at=1000,
        source="api",
    )
    interactions = [
        Interaction(
            interaction_id=101,
            user_id="user_neutral",
            content="Can you summarize my last invoice?",
            request_id="req_neutral",
            created_at=1000,
            role="user",
        ),
        Interaction(
            interaction_id=102,
            user_id="user_neutral",
            content="Here is the summary of your last invoice: ...",
            request_id="req_neutral",
            created_at=1001,
            role="assistant",
        ),
        Interaction(
            interaction_id=103,
            user_id="user_neutral",
            content="Thanks, that's exactly what I needed.",
            request_id="req_neutral",
            created_at=1002,
            role="user",
        ),
    ]
    return [
        RequestInteractionDataModel(
            session_id="req_neutral",
            request=request,
            interactions=interactions,
        )
    ]


@pytest.fixture
def failure_request_interaction_models():
    """A failure-evidence window — user pushes back on the agent's response."""
    request = Request(
        request_id="req_failure",
        user_id="user_failure",
        created_at=2000,
        source="api",
    )
    interactions = [
        Interaction(
            interaction_id=201,
            user_id="user_failure",
            content="Please cancel my subscription.",
            request_id="req_failure",
            created_at=2000,
            role="user",
        ),
        Interaction(
            interaction_id=202,
            user_id="user_failure",
            content="Can you confirm you want to cancel? Are you sure?",
            request_id="req_failure",
            created_at=2001,
            role="assistant",
        ),
        Interaction(
            interaction_id=203,
            user_id="user_failure",
            content="I already said yes, stop asking me to confirm.",
            request_id="req_failure",
            created_at=2002,
            role="user",
        ),
    ]
    return [
        RequestInteractionDataModel(
            session_id="req_failure",
            request=request,
            interactions=interactions,
        )
    ]


# ===============================
# Helpers
# ===============================


def _build_extractor(
    request_context: RequestContext,
    mock_llm_client: MagicMock,
    extractor_config: PlaybookConfig,
    service_config: PlaybookGenerationServiceConfig,
) -> PlaybookExtractor:
    """Construct a PlaybookExtractor wired with mocked dependencies.

    Args:
        request_context (RequestContext): Mock-backed request context.
        mock_llm_client (MagicMock): Mock LiteLLM client whose
            ``generate_chat_response`` will be set per-test.
        extractor_config (PlaybookConfig): Classic playbook config.
        service_config (PlaybookGenerationServiceConfig): Runtime service config.

    Returns:
        PlaybookExtractor: Extractor ready for ``run()`` invocation.
    """
    return PlaybookExtractor(
        request_context=request_context,
        llm_client=mock_llm_client,
        extractor_config=extractor_config,
        service_config=service_config,
        agent_context="Test agent",
    )


# ===============================
# Tests
# ===============================


def test_classic_extractor_emits_positive_when_no_failure_evidence(
    request_context,
    mock_llm_client,
    extractor_config,
    service_config,
    neutral_request_interaction_models,
):
    """Window with neutral interactions → extracted playbooks have polarity=positive.

    Validates the default-polarity threading end-to-end: the LLM emits a
    ``StructuredPlaybookContent`` without an explicit ``polarity`` field, so the
    schema default ``"positive"`` flows through ``_build_user_playbook`` into
    the resulting ``UserPlaybook``.
    """
    request_context.storage.get_last_k_interactions_grouped.return_value = (
        neutral_request_interaction_models,
        [],
    )

    # LLM emits entries without explicit polarity — relies on the default.
    mock_llm_client.generate_chat_response.return_value = StructuredPlaybookList(
        playbooks=[
            StructuredPlaybookContent(
                trigger="user requests a summary of recent activity",
                content="Provide a concise summary using the latest record",
            ),
            StructuredPlaybookContent(
                trigger="user thanks the agent after a successful response",
                content="Acknowledge briefly and offer to help with anything else",
            ),
        ]
    )

    extractor = _build_extractor(
        request_context, mock_llm_client, extractor_config, service_config
    )

    # MOCK_LLM_RESPONSE=false ensures the mock_llm_client return value is used
    # instead of the extractor's deterministic mock branch.
    with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
        result = extractor.run()

    assert len(result) == 2, "Expected two playbooks from the neutral window"
    assert all(playbook.polarity == "positive" for playbook in result), (
        "All playbooks must default to polarity=positive when LLM omits the field"
    )


def test_classic_extractor_emits_negative_on_clear_failure(
    request_context,
    mock_llm_client,
    extractor_config,
    service_config,
    failure_request_interaction_models,
):
    """Window with user pushback → at least one playbook has polarity=negative.

    Validates that negative polarity is preserved end-to-end when the LLM emits
    it: the emitted ``UserPlaybook`` must have ``polarity == "negative"`` AND
    content starting with one of ``NEGATIVE_PREFIXES``
    (``"Avoid"``/``"Do not"``/``"Don't"``/``"Never"``).
    """
    request_context.storage.get_last_k_interactions_grouped.return_value = (
        failure_request_interaction_models,
        [],
    )

    mock_llm_client.generate_chat_response.return_value = StructuredPlaybookList(
        playbooks=[
            StructuredPlaybookContent(
                trigger="user confirms a cancellation request",
                content="Avoid asking the user to confirm a cancellation more than once",
                polarity="negative",
            ),
            # Companion positive entry — verifies a mixed-polarity window
            # produces the right mix downstream.
            StructuredPlaybookContent(
                trigger="user issues a cancellation command",
                content="Acknowledge the cancellation and proceed without redundant prompts",
            ),
        ]
    )

    extractor = _build_extractor(
        request_context, mock_llm_client, extractor_config, service_config
    )

    with patch.dict(os.environ, {"MOCK_LLM_RESPONSE": "false"}):
        result = extractor.run()

    assert len(result) == 2, "Expected both playbook entries to be emitted"

    negative_playbooks = [pb for pb in result if pb.polarity == "negative"]
    assert len(negative_playbooks) >= 1, (
        "Expected at least one playbook with polarity=negative from a failure window"
    )

    negative_playbook = negative_playbooks[0]
    assert any(
        negative_playbook.content.lstrip().startswith(prefix)
        for prefix in NEGATIVE_PREFIXES
    ), (
        f"Negative-polarity content must start with one of {NEGATIVE_PREFIXES}; "
        f"got: {negative_playbook.content!r}"
    )
