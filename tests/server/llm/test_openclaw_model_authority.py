"""Selected native routes and retired-default admission through the full client."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
)
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.providers import openclaw_provider

TARGET = "openclaw/anthropic/claude-sonnet-4-6"
LEGACY = "OPENCLAW_DEFAULT_MODEL"


class Answer(BaseModel):
    answer: str


@pytest.fixture
def client_transport(monkeypatch):
    monkeypatch.delenv(LEGACY, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = LiteLLMClient(
        LiteLLMConfig(model=TARGET, max_tokens=32, fallback_models=[])
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer":"ok"}', tool_calls=None, refusal=None
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="synthetic-response",
        _hidden_params={},
    )
    transport = Mock(return_value=response)
    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    return client, transport


@pytest.mark.parametrize("role", list(ModelRole))
def test_selected_openclaw_route_survives_completion_role_detection(
    client_transport, monkeypatch, role
):
    """Even an embedding *label* on completion does not change its native route.

    Actual embedding APIs use the separate embedding concern; this asserts no
    embedding support in the OpenClaw provider.
    """
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-unused-model")
    client.generate_chat_response(
        [{"role": "user", "content": "synthetic evidence"}], model_role=role
    )
    assert transport.call_count == 1
    assert transport.call_args.args[0]["model"] == TARGET


@pytest.mark.parametrize("mode", ["plain", "structured", "validated"])
def test_legacy_default_refuses_before_any_dispatch_or_fallback(
    client_transport, monkeypatch, mode
):
    client, transport = client_transport
    client.config.model = "openclaw/default"
    client.config.fallback_models = ["gpt-4o-mini"]
    monkeypatch.setenv(LEGACY, "private-old-selection")
    options: dict[str, Any] = {} if mode == "plain" else {"response_format": Answer}
    if mode == "validated":
        options["structured_output_validator"] = lambda _value: []
    with pytest.raises(ValueError, match=LEGACY) as failure:
        client.generate_chat_response(
            [{"role": "user", "content": "synthetic evidence"}], **options
        )
    assert "private-old-selection" not in str(failure.value)
    transport.assert_not_called()


@pytest.mark.parametrize("reaches_fallback", [False, True])
def test_old_default_is_checked_only_when_its_route_is_reached(
    client_transport, monkeypatch, reaches_fallback
):
    client, transport = client_transport
    client.config.model = "gpt-4o-mini"
    client.config.fallback_models = ["openclaw/default"]
    monkeypatch.setenv(LEGACY, "private-old-selection")
    if reaches_fallback:
        transport.side_effect = LiteLLMClientError("synthetic primary outage")
        with pytest.raises(ValueError, match=LEGACY):
            client.generate_response("synthetic evidence")
    else:
        client.generate_response("synthetic evidence")
    assert transport.call_count == 1
    assert transport.call_args.args[0]["model"] == "gpt-4o-mini"


@pytest.mark.parametrize("legacy", [None, "", "  "])
def test_default_delegates_without_inventing_a_model(monkeypatch, legacy):
    if legacy is None:
        monkeypatch.delenv(LEGACY, raising=False)
    else:
        monkeypatch.setenv(LEGACY, legacy)
    dispatch = Mock(return_value="ok")
    monkeypatch.setattr(openclaw_provider, "_call_cli", dispatch)
    openclaw_provider.OpenClawLLM().completion(
        model="openclaw/default", messages=[{"role": "user", "content": "hello"}]
    )
    assert dispatch.call_args.kwargs["model"] is None


def test_direct_provider_refuses_legacy_default_before_cli(monkeypatch):
    monkeypatch.setenv(LEGACY, "private-old-selection")
    dispatch = Mock(return_value="ok")
    monkeypatch.setattr(openclaw_provider, "_call_cli", dispatch)
    with pytest.raises(ValueError, match=LEGACY):
        openclaw_provider.OpenClawLLM().completion(
            model="openclaw/default", messages=[{"role": "user", "content": "hello"}]
        )
    dispatch.assert_not_called()


def test_explicit_non_cli_transport_does_not_read_native_default(
    client_transport, monkeypatch
):
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-unused-model")
    client.generate_response(
        "synthetic evidence",
        model="openclaw/default",
        custom_llm_provider="openai",
        api_base="http://127.0.0.1:1/v1",
    )
    assert transport.call_count == 1
    assert transport.call_args.args[0]["custom_llm_provider"] == "openai"


def test_explicit_cli_transport_refuses_a_prefixless_legacy_default(
    client_transport, monkeypatch
):
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-old-selection")
    with pytest.raises(ValueError, match=LEGACY):
        client.generate_response(
            "synthetic evidence", model="default", custom_llm_provider="openclaw"
        )
    transport.assert_not_called()


def test_explicit_request_target_replaces_native_client_target(
    client_transport, monkeypatch
):
    client, transport = client_transport
    monkeypatch.setenv(LEGACY, "private-unused-model")
    requested = "openclaw/openai/gpt-4o-mini"
    client.generate_response(
        "synthetic evidence", model=requested, model_role=ModelRole.EXTRACTION_AGENT
    )
    assert transport.call_count == 1
    assert transport.call_args.args[0]["model"] == requested
