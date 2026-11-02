"""Execution admission runs after preflight and before any external compute."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm.compute_admission import (
    ComputeAdmissionError,
    external_compute_admission,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.providers import embedding_service_provider as service
from reflexio.server.services.storage.error import ReadinessUnavailableError


def test_text_admission_refusal_prevents_dispatch_and_fallback(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="gpt-4o",
        _hidden_params={"custom_llm_provider": "openai"},
    )
    dispatch = MagicMock(return_value=response)
    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", staticmethod(dispatch)
    )
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", fallback_models=["gpt-4o-mini"], max_retries=0)
    )
    assert client.generate_response("synthetic control") == "ok"
    dispatch.reset_mock()
    reason = ReadinessUnavailableError("Controlled admission refusal")
    guard = MagicMock(side_effect=reason)
    token = external_compute_admission.set(guard)
    try:
        with pytest.raises(ComputeAdmissionError) as refused:
            client.generate_response("synthetic refused request")
        assert refused.value.reason is reason
    finally:
        external_compute_admission.reset(token)
    guard.assert_called_once()
    dispatch.assert_not_called()


def test_embedding_service_refusal_prevents_post_and_retry(monkeypatch):
    client = MagicMock()
    client.post.return_value.json.return_value = {
        "data": [{"index": 0, "embedding": [0.0, 1.0]}]
    }
    monkeypatch.setattr(service, "_http_client", lambda: client)
    arguments: dict[str, Any] = {
        "url": "http://synthetic.invalid/embeddings",
        "model": "local/test",
        "texts": ["synthetic"],
        "dimensions": 2,
        "timeout": 1.0,
        "mode": "local_service",
    }
    assert service._post_embedding_batch(**arguments) == [[0.0, 1.0]]
    client.reset_mock()
    reason = ReadinessUnavailableError("Controlled admission refusal")
    guard = MagicMock(side_effect=reason)
    token = external_compute_admission.set(guard)
    try:
        with pytest.raises(ComputeAdmissionError) as refused:
            service._post_embedding_batch(**arguments)
        assert refused.value.reason is reason
    finally:
        external_compute_admission.reset(token)
    guard.assert_called_once()
    client.post.assert_not_called()
