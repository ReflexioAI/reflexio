"""Credential failover crosses the real client guards, parser and accounting."""

import logging
import time
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from reflexio.models.config_schema import APIKeyConfig, MiniMaxConfig
from reflexio.server.llm._litellm_types import (
    LiteLLMClientError,
    ProviderRequestGuardError,
)
from reflexio.server.llm._minimax_quota import backup_credentials
from reflexio.server.llm.call_deadline import model_deadline
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.provider_circuit import (
    FIRST_OPEN_SECONDS,
    ProviderCircuitBreaker,
    circuit_route,
    provider_wait_seconds,
)
from reflexio.server.llm.token_accounting import RunTokenCapture, run_token_capture

MODEL = "minimax/MiniMax-M3"
PRIMARY = "scripted-primary"
BACKUP = "scripted-backup"
MESSAGES = [{"role": "user", "content": "diagnose this session"}]


class Diagnosis(BaseModel):
    supported_gap: bool


def exhausted() -> LiteLLMClientError:
    return LiteLLMClientError(
        "RateLimitError: MinimaxException - Token Plan usage limit reached (2056)",
        upstream_error_type="RateLimitError",
        upstream_status_code=429,
    )


def response() -> MagicMock:
    r = MagicMock()
    r.model = "MiniMax-M3"
    r.choices[0].message.content = '{"supported_gap":true}'
    r.choices[0].message.tool_calls = None
    r.choices[0].finish_reason = "stop"
    r.usage = MagicMock(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    r.usage.prompt_tokens_details = None
    r.usage.cache_creation_input_tokens = None
    r.usage.cache_read_input_tokens = None
    return r


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", PRIMARY)
    monkeypatch.setenv("MINIMAX_BACKUP_API_KEY", BACKUP)
    return LiteLLMClient(LiteLLMConfig(model=MODEL, fallback_models=[]))


def test_quota_failover_preserves_request_validation_usage_and_recovery(
    configured,
    monkeypatch,
    caplog,
):
    from reflexio.server.llm import _litellm_text_generation as generation

    now = [1000.0]
    breaker = ProviderCircuitBreaker(clock=lambda: now[0], jitter=lambda: 0.0)
    monkeypatch.setattr(generation, "provider_circuit", breaker)
    sent, guarded = [], []
    recovered = [False]

    def dispatch(params, _timeout):
        sent.append(dict(params))
        if params["api_key"] == PRIMARY and not recovered[0]:
            raise exhausted()
        return response()

    monkeypatch.setattr(configured, "_completion_with_hard_timeout", dispatch)
    totals = RunTokenCapture()
    token = run_token_capture.set(totals)
    try:
        with caplog.at_level(logging.INFO):
            for _ in range(2):
                value = configured.generate_chat_response(
                    MESSAGES,
                    response_format=Diagnosis,
                    provider_request_guard=lambda params, *_: guarded.append(
                        dict(params)
                    ),
                )
                assert value == Diagnosis(supported_gap=True)
        assert [p["api_key"] for p in sent] == [PRIMARY, BACKUP, BACKUP]
        assert [p["api_key"] for p in guarded] == [PRIMARY, BACKUP, BACKUP]
        assert all(p["model"] == MODEL for p in sent)
        assert sent[0]["messages"] == sent[1]["messages"]
        assert sent[0]["response_format"] == sent[1]["response_format"]
        assert totals.totals.prompt_tokens == 20
        assert totals.totals.completion_tokens == 4
        assert totals.completions == 2
        assert "credential_slot=backup" in caplog.text
        assert PRIMARY not in caplog.text and BACKUP not in caplog.text
        recovered[0] = True
        now[0] += FIRST_OPEN_SECONDS
        configured.generate_chat_response(MESSAGES, response_format=Diagnosis)
        assert sent[-1]["api_key"] == PRIMARY
        assert breaker.outage_retry_delay(circuit_route(sent[-1])) is None
    finally:
        run_token_capture.reset(token)


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("slow"),
        LiteLLMClientError("rate limit", upstream_status_code=429),
        LiteLLMClientError("unauthorized", upstream_status_code=401),
        LiteLLMClientError("Token Plan usage limit reached (2056)"),
        LiteLLMClientError(
            "Token Plan usage limit reached (2056)", upstream_status_code=401
        ),
        LiteLLMClientError(
            "Token Plan usage limit reached (2056)", upstream_status_code=400
        ),
        LiteLLMClientError(
            "usage limit reached in job 20560", upstream_status_code=429
        ),
    ],
)
def test_other_failures_never_spend_backup(configured, monkeypatch, error):
    sent = []

    def dispatch(params, _timeout):
        sent.append(params["api_key"])
        raise error

    monkeypatch.setattr(configured, "_completion_with_hard_timeout", dispatch)
    with pytest.raises(LiteLLMClientError):
        configured.generate_chat_response(MESSAGES)
    assert sent == [PRIMARY]


def test_both_exhausted_are_bounded_and_leave_durable_work_waiting(
    configured, monkeypatch
):
    sent = []

    def dispatch(params, _timeout):
        sent.append(params["api_key"])
        raise exhausted()

    monkeypatch.setattr(configured, "_completion_with_hard_timeout", dispatch)
    for _ in range(2):
        with pytest.raises(LiteLLMClientError) as raised:
            configured.generate_chat_response(MESSAGES)
        assert provider_wait_seconds(raised.value) is not None
    assert sent == [PRIMARY, BACKUP]


def test_backup_cannot_bypass_request_guard(configured, monkeypatch):
    sent = []

    def dispatch(params, _timeout):
        sent.append(params["api_key"])
        raise exhausted()

    def guard(params, *_args):
        if params["api_key"] == BACKUP:
            raise ProviderRequestGuardError("refused")

    monkeypatch.setattr(configured, "_completion_with_hard_timeout", dispatch)
    with pytest.raises(ProviderRequestGuardError):
        configured.generate_chat_response(MESSAGES, provider_request_guard=guard)
    assert sent == [PRIMARY]


def test_backup_cannot_extend_the_callers_deadline(configured, monkeypatch):
    from reflexio.server.llm import _litellm_text_generation as generation

    now = [time.monotonic()]
    monkeypatch.setattr(generation.time, "monotonic", lambda: now[0])
    sent = []

    def dispatch(params, _timeout):
        sent.append(params["api_key"])
        now[0] += 20
        raise exhausted()

    monkeypatch.setattr(configured, "_completion_with_hard_timeout", dispatch)
    with model_deadline(now[0] + 10), pytest.raises(LiteLLMClientError):
        configured.generate_chat_response(MESSAGES)
    assert sent == [PRIMARY]


def test_customer_key_is_not_redirected(configured, monkeypatch):
    configured = LiteLLMClient(
        LiteLLMConfig(
            model=MODEL,
            fallback_models=[],
            api_key_config=APIKeyConfig(
                minimax=MiniMaxConfig(api_key="customer-owned")
            ),
        )
    )
    sent = []

    def dispatch(params, _timeout):
        sent.append(params["api_key"])
        raise exhausted()

    monkeypatch.setattr(configured, "_completion_with_hard_timeout", dispatch)
    with pytest.raises(LiteLLMClientError):
        configured.generate_chat_response(MESSAGES)
    assert sent == ["customer-owned"]


@pytest.mark.parametrize(
    "override",
    [
        {"model": "openai/test"},
        {"api_base": "https://customer.invalid/v1"},
        {"base_url": "https://customer.invalid/v1"},
        {"api_key": "customer"},
    ],
)
def test_ineligible_routes(configured, override):
    assert backup_credentials({"model": MODEL, **override}) is None


def test_environment_custom_endpoint_cannot_receive_backup(configured, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_BASE", "https://customer.invalid/v1")
    assert backup_credentials({"model": MODEL}) is None


def test_global_custom_endpoint_cannot_receive_backup(configured, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "api_base", "https://customer.invalid/v1")
    assert backup_credentials({"model": MODEL}) is None


@pytest.mark.parametrize("backup", ["", "  ", PRIMARY])
def test_missing_or_duplicate_backup_disables_failover(configured, monkeypatch, backup):
    monkeypatch.setenv("MINIMAX_BACKUP_API_KEY", backup)
    assert backup_credentials({"model": MODEL}) is None
