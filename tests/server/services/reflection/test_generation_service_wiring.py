"""Confirms reflection is invoked from inside GenerationService.run and
that a failure there is swallowed so the publish completes normally.

The substantive reflection behavior is exercised by
``test_reflection_service.py`` against real storage; this file is a thin
wiring check.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService

pytestmark = pytest.mark.integration


@pytest.fixture
def temp_storage_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def request_context(temp_storage_dir):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield RequestContext(org_id="test_org", storage_base_dir=temp_storage_dir)


@pytest.fixture
def llm_client():
    return LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))


@pytest.fixture
def publish_request():
    return PublishUserInteractionRequest(
        user_id="u1",
        session_id="test_session",
        interaction_data_list=[
            InteractionData(role="User", content="hello"),
            InteractionData(role="Assistant", content="hi"),
        ],
        source="cli",
        agent_version="v1",
    )


def test_run_invokes_maybe_run_reflection(request_context, llm_client, publish_request):
    """GenerationService.run must call _maybe_run_reflection after saving
    interactions and before extractor pool spins up."""
    service = GenerationService(llm_client=llm_client, request_context=request_context)
    # Stub out the generation services so we don't run real LLM extractors.
    with (
        patch(
            "reflexio.server.services.generation_service.ProfileGenerationService"
        ) as profile_cls,
        patch(
            "reflexio.server.services.generation_service.PlaybookGenerationService"
        ) as playbook_cls,
        patch.object(service, "_maybe_run_reflection") as mock_reflect,
    ):
        profile_cls.return_value.run = MagicMock()
        playbook_cls.return_value.run = MagicMock()

        result = service.run(publish_request)

    assert result.request_id is not None
    mock_reflect.assert_called_once()
    kwargs = mock_reflect.call_args.kwargs
    assert kwargs.get("user_id") == "u1"
    assert kwargs.get("source") == "cli"
    # agent_version must be threaded through so reflected playbooks
    # inherit the publish's resolved version, not "".
    assert kwargs.get("agent_version") == "v1"


def test_run_writes_request_and_interactions_before_publish_limiter(
    request_context, llm_client, publish_request
):
    """The post-write limiter must not block acknowledged async publishes from storage."""
    request_id = "req-before-learning-limiter"
    request = publish_request.model_copy(update={"request_id": request_id})
    service = GenerationService(llm_client=llm_client, request_context=request_context)
    limiter_calls = []

    @contextmanager
    def saturated_publish_limit(org_id, operation, **kwargs):
        limiter_calls.append((org_id, operation, kwargs))
        raise TimeoutError("publish limiter saturated")
        yield

    with (
        patch(
            "reflexio.server.services.generation_service.operation_limit",
            saturated_publish_limit,
        ),
        patch("reflexio.server.services.generation_service.record_usage_event"),
        pytest.raises(TimeoutError, match="publish limiter saturated"),
    ):
        service.run(request)

    storage = request_context.storage
    assert storage is not None
    assert storage.get_request(request_id) is not None
    assert len(storage.get_interactions_by_request_ids([request_id])) == 2
    assert limiter_calls == [
        ("test_org", "publish", {"wait_forever": True}),
    ]


def test_reflection_failure_does_not_break_publish(
    request_context, llm_client, publish_request, caplog
):
    """A bug in reflection must not change the publish outcome."""
    service = GenerationService(llm_client=llm_client, request_context=request_context)
    with (
        patch(
            "reflexio.server.services.generation_service.ProfileGenerationService"
        ) as profile_cls,
        patch(
            "reflexio.server.services.generation_service.PlaybookGenerationService"
        ) as playbook_cls,
        patch(
            "reflexio.server.services.reflection.service.ReflectionService.run",
            side_effect=RuntimeError("boom"),
        ),
        caplog.at_level(
            "WARNING", logger="reflexio.server.services.generation_service"
        ),
    ):
        profile_cls.return_value.run = MagicMock()
        playbook_cls.return_value.run = MagicMock()

        result = service.run(publish_request)

    assert result.request_id is not None  # publish succeeded
    assert "reflection step failed" in caplog.text
