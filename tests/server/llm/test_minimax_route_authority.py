"""The admitted MiniMax endpoint is the one its isolated worker dispatches."""

import multiprocessing
import socket
import time
from unittest.mock import patch

import httpx
import litellm
import pytest

from reflexio.models.config_schema import APIKeyConfig
from reflexio.server.llm._litellm_subprocess import _litellm_completion_worker
from reflexio.server.llm._minimax_quota import backup_credentials
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    check_model_input_budget,
    model_input_budget,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.test_support.llm_mock import unpatched_litellm

MODEL = "minimax/MiniMax-M2.1"
GATEWAY = "https://customer.invalid/v1"
MESSAGES = [{"role": "user", "content": "synthetic endpoint evidence"}]


@pytest.fixture
def route_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "scripted-primary")
    monkeypatch.setenv("MINIMAX_BACKUP_API_KEY", "scripted-backup")
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.delenv("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "token_counter", lambda **_kwargs: 10)


def client(**kwargs):
    return LiteLLMClient(LiteLLMConfig(model=MODEL, fallback_models=[], **kwargs))


@pytest.mark.parametrize("source", ["environment", "global"])
@pytest.mark.parametrize("output_cap", [None, 20])
def test_implicit_custom_route_requires_its_own_capacity(
    route_env, monkeypatch, source, output_cap
):
    if source == "environment":
        monkeypatch.setenv("MINIMAX_API_BASE", GATEWAY)
    else:
        monkeypatch.setattr(litellm, "api_base", GATEWAY)
    with model_input_budget(), pytest.raises(InputBudgetConfigurationError):
        params, *_ = client(max_tokens=output_cap)._build_completion_params(MESSAGES)
        check_model_input_budget(params)


def test_endpoint_precedence_and_native_backup_policy(route_env, monkeypatch):
    native, *_ = client(max_tokens=20)._build_completion_params(MESSAGES)
    assert "api_base" not in native
    assert backup_credentials(native) == ("scripted-primary", "scripted-backup")
    monkeypatch.setenv("MINIMAX_API_BASE", GATEWAY)
    environment, *_ = client(max_tokens=20)._build_completion_params(MESSAGES)
    assert environment["api_base"] == GATEWAY
    assert backup_credentials(environment) is None
    monkeypatch.setattr(litellm, "api_base", "https://global.invalid/v1")
    global_route, *_ = client(max_tokens=20)._build_completion_params(MESSAGES)
    assert global_route["api_base"] == "https://global.invalid/v1"
    configured = client(
        max_tokens=20,
        api_key_config=APIKeyConfig.model_validate(
            {
                "custom_endpoint": {
                    "api_key": "scripted-customer",
                    "api_base": "https://configured.invalid/v1",
                }
            }
        ),
    )
    configured_route, *_ = configured._build_completion_params(MESSAGES)
    assert configured_route["api_base"] == "https://configured.invalid/v1"
    for key in ("api_base", "base_url"):
        explicit, *_ = client(max_tokens=20)._build_completion_params(
            MESSAGES, **{key: "https://request.invalid/v1"}
        )
        assert explicit[key] == "https://request.invalid/v1"
        assert backup_credentials(explicit) is None


@pytest.mark.parametrize("empty", ["", "   "])
def test_blank_override_keeps_native_route_and_backup(route_env, monkeypatch, empty):
    monkeypatch.setenv("MINIMAX_API_BASE", empty)
    params, *_ = client(max_tokens=20)._build_completion_params(MESSAGES)
    assert "api_base" not in params
    assert backup_credentials(params) == ("scripted-primary", "scripted-backup")


def test_minimax_override_does_not_retarget_other_providers(route_env, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_BASE", GATEWAY)
    params, *_ = client(max_tokens=20)._build_completion_params(
        MESSAGES, model="openai/gpt-4o-mini"
    )
    assert "api_base" not in params


def test_declared_custom_capacity_and_explicit_output_admit(route_env, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_BASE", GATEWAY)
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON", f'{{"{MODEL}@{GATEWAY}":1000}}'
    )
    with model_input_budget():
        params, *_ = client(max_tokens=20)._build_completion_params(MESSAGES)
        check_model_input_budget(params)
    assert params["api_base"] == GATEWAY
    assert params["max_tokens"] == 20


def _capture_worker(params, queue, deadline):
    def send(self, request, **kwargs):
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "route-probe",
                "object": "chat.completion",
                "created": 1,
                "model": "MiniMax-M2.1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": str(request.url)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    # Exercise the pinned LiteLLM transformation, stopping only at HTTP send.
    with (
        patch.object(httpx.Client, "send", send),
        patch.object(
            socket.socket, "connect", side_effect=AssertionError("Network forbidden")
        ),
    ):
        _litellm_completion_worker(params, queue, deadline)


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_worker_uses_admitted_route_after_ambient_change(
    route_env, monkeypatch, method
):
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} unavailable on this OS")
    monkeypatch.setenv("MINIMAX_API_BASE", GATEWAY)
    params, *_ = client(max_tokens=20)._build_completion_params(MESSAGES)
    assert params["api_base"] == GATEWAY
    # Mutation after admission cannot redirect a physical attempt.
    monkeypatch.setenv("MINIMAX_API_BASE", "https://changed.invalid/v1")
    context = multiprocessing.get_context(method)
    queue = context.Queue()
    process = context.Process(
        target=_capture_worker, args=(params, queue, time.monotonic() + 30)
    )
    with unpatched_litellm():
        process.start()
    try:
        status, result = queue.get(timeout=30)
        assert status == "ok", result
        assert result.choices[0].message.content == (GATEWAY + "/chat/completions")
    finally:
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        queue.close()
        queue.join_thread()
