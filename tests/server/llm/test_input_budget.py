"""Every built provider request is admitted before consuming durable quota."""

from unittest.mock import Mock

import pytest

from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    InputBudgetExceededError,
    declared_context_limits,
    extraction_request_guard,
)


@pytest.fixture
def counter(monkeypatch):
    counter = Mock(return_value=100)
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", counter
    )
    return counter


def params(**extra):
    return {
        "model": "test-model",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "evidence"}],
        **extra,
    }


def test_continuations_repairs_and_fallback_are_charged_per_guard_call(counter):
    reserve = Mock()
    guard = extraction_request_guard(
        context_limits={"test-model": 32768, "fallback": 32768}, reserve_attempt=reserve
    )
    for request in (
        params(),
        params(messages=[{"role": "user", "content": "repair"}]),
        params(model="fallback"),
    ):
        guard(request, 120, (), False)
    assert reserve.call_count == 3
    assert counter.call_count == 3


def test_schema_tools_margin_and_output_reserve_reject_before_quota(counter):
    counter.return_value = 1000
    reserve = Mock()
    guard = extraction_request_guard(
        context_limits={"test-model": 10000}, reserve_attempt=reserve
    )
    tools = [{"type": "function", "function": {"name": "ask_human"}}]
    with pytest.raises(InputBudgetExceededError):
        guard(
            params(
                tools=tools,
                response_format={"type": "json_schema", "schema": {"type": "object"}},
            ),
            120,
            (),
            True,
        )
    assert counter.call_args_list[0].kwargs["tools"] == tools
    assert counter.call_count == 2
    reserve.assert_not_called()


def test_custom_endpoint_cannot_inherit_model_capacity(counter):
    reserve = Mock()
    guard = extraction_request_guard(
        context_limits={"test-model": 32768}, reserve_attempt=reserve
    )
    with pytest.raises(InputBudgetConfigurationError, match="Custom endpoint"):
        guard(params(api_base="https://custom.invalid"), 120, (), False)
    reserve.assert_not_called()
    counter.assert_not_called()


def test_unknown_capacity_and_missing_output_reservation_fail_closed(
    counter, monkeypatch
):
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.get_model_info",
        Mock(return_value={"max_tokens": 4096}),
    )
    guard = extraction_request_guard(context_limits={}, reserve_attempt=Mock())
    with pytest.raises(InputBudgetConfigurationError, match="context capacity"):
        guard(params(), 120, (), False)
    guard = extraction_request_guard(
        context_limits={"test-model": 32768}, reserve_attempt=Mock()
    )
    with pytest.raises(InputBudgetConfigurationError, match="output reservation"):
        guard(params(max_tokens=None), 120, (), False)


@pytest.mark.parametrize(
    "raw", ['{"model": true}', '{"model": "32768"}', '{"model": 0}', "[]", "{"]
)
def test_invalid_limit_declarations_fail_explicitly(raw):
    with pytest.raises(InputBudgetConfigurationError):
        declared_context_limits(raw)


def test_empty_declarations_mean_unset():
    assert declared_context_limits(" ") == {}


def test_approved_64k_target_preserves_margin_and_actual_model_output_capacity(counter):
    counter.return_value = 40000
    reserve = Mock()
    guard = extraction_request_guard(
        context_limits={"test-model": 512000}, reserve_attempt=reserve
    )
    guard(params(), 120, (), False)
    reserve.assert_called_once()
    counter.return_value = 55000
    with pytest.raises(InputBudgetExceededError):
        guard(params(), 120, (), False)
    assert reserve.call_count == 1
    counter.return_value = 20000
    smaller_model = extraction_request_guard(
        context_limits={"test-model": 25000}, reserve_attempt=reserve
    )
    with pytest.raises(InputBudgetExceededError):
        smaller_model(params(), 120, (), False)
    assert reserve.call_count == 1


def test_scoped_guard_runs_at_each_actual_fallback_dispatch_and_resets(monkeypatch):
    from types import SimpleNamespace

    from reflexio.server.llm.input_budget import extraction_provider_guard
    from reflexio.server.llm.litellm_client import (
        LiteLLMClient,
        LiteLLMClientError,
        LiteLLMConfig,
    )

    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", fallback_models=["gpt-4o-mini"])
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, refusal=None),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="gpt-4o-mini",
        _hidden_params={"custom_llm_provider": "openai"},
    )
    dispatch = Mock(
        side_effect=[LiteLLMClientError("fixture transport failure"), response]
    )
    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch)
    guard = Mock()
    token = extraction_provider_guard.set(guard)
    try:
        assert (
            client.generate_chat_response([{"role": "user", "content": "evidence"}])
            == "ok"
        )
    finally:
        extraction_provider_guard.reset(token)
    assert [call.args[0]["model"] for call in guard.call_args_list] == [
        "gpt-4o",
        "gpt-4o-mini",
    ]
    assert guard.call_count == dispatch.call_count == 2
    assert extraction_provider_guard.get() is None


def test_scoped_guard_rejection_never_dispatches_or_falls_back(monkeypatch):
    from reflexio.server.llm.input_budget import extraction_provider_guard
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", fallback_models=["gpt-4o-mini"])
    )
    dispatch = Mock()
    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch)
    guard = Mock(side_effect=InputBudgetExceededError("fixture input budget"))
    token = extraction_provider_guard.set(guard)
    try:
        with pytest.raises(InputBudgetExceededError):
            client.generate_chat_response([{"role": "user", "content": "evidence"}])
    finally:
        extraction_provider_guard.reset(token)
    assert guard.call_count == 1
    dispatch.assert_not_called()


@pytest.mark.parametrize("classified", [False, True])
def test_isolated_capacity_classifier_does_not_guess_from_code_or_message(
    monkeypatch, classified
):
    import multiprocessing

    from litellm.exceptions import BadRequestError, ContextWindowExceededError

    from reflexio.server.llm import _litellm_text_generation as module
    from reflexio.server.llm._litellm_subprocess import _snapshot_completion_error
    from reflexio.server.llm.input_budget import (
        ProviderInputOverflowError,
        extraction_provider_guard,
        extraction_request_identity,
    )
    from reflexio.server.llm.litellm_client import (
        LiteLLMClient,
        LiteLLMClientError,
        LiteLLMConfig,
    )

    error_type = ContextWindowExceededError if classified else BadRequestError
    error = error_type(
        message="context window exceeds limit (2013)",
        model="gpt-4o",
        llm_provider="openai",
    )
    snapshot = _snapshot_completion_error(error, {"model": "gpt-4o"})
    assert snapshot.input_capacity_exceeded is classified
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("This isolated error fixture uses a real forked worker")
    process_context = multiprocessing.get_context("fork")
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda: process_context)

    def reject_worker(params, result_queue):
        result_queue.put(("error", snapshot))

    monkeypatch.setattr(module, "_litellm_completion_worker", reject_worker)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    guard_token = extraction_provider_guard.set(Mock())
    identity = {
        "attempt_id": "reserved",
        "request_fingerprint": "a" * 64,
        "chunk_index": 0,
    }
    identity_token = extraction_request_identity.set(identity)
    try:
        with pytest.raises(
            ProviderInputOverflowError if classified else LiteLLMClientError
        ) as caught:
            client._completion_with_hard_timeout({"model": "gpt-4o"}, 5)
        if classified:
            assert isinstance(caught.value, ProviderInputOverflowError)
            assert caught.value.identity == identity
    finally:
        extraction_request_identity.reset(identity_token)
        extraction_provider_guard.reset(guard_token)
    # Ordinary callers keep the existing error/fallback contract.
    with pytest.raises(LiteLLMClientError):
        client._completion_with_hard_timeout({"model": "gpt-4o"}, 5)


@pytest.mark.parametrize("prompt_tokens", [0, 10])
def test_durable_usage_observer_preserves_zero_usage_and_cache_subfields(
    monkeypatch, prompt_tokens
):
    from types import SimpleNamespace

    from reflexio.server.llm.input_budget import extraction_usage_observer
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=5,
        total_tokens=prompt_tokens + 5,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2 if prompt_tokens else 0),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=None,
    )
    response = SimpleNamespace(
        usage=usage, model="gpt-4o", _hidden_params={"custom_llm_provider": "openai"}
    )
    observer = Mock()
    token = extraction_usage_observer.set(observer)
    try:
        client._log_token_usage({"model": "gpt-4o"}, response)
    finally:
        extraction_usage_observer.reset(token)
    assert observer.call_count == 1
    provenance, total = observer.call_args.args
    assert provenance.model_name == "gpt-4o"
    assert total.prompt_tokens == prompt_tokens
    assert total.completion_tokens == 5
    assert total.cache_read_input_tokens == (2 if prompt_tokens else 0)
    assert total.cache_write_input_tokens == 0


def test_usage_checkpoint_failure_never_dispatches_an_extra_fallback(monkeypatch):
    from types import SimpleNamespace

    from reflexio.server.llm.input_budget import extraction_usage_observer
    from reflexio.server.llm.litellm_client import (
        LiteLLMClient,
        LiteLLMConfig,
        ProviderRequestGuardError,
    )

    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", fallback_models=["gpt-4o-mini"])
    )
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=None,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    response = SimpleNamespace(
        usage=usage, model="gpt-4o", _hidden_params={"custom_llm_provider": "openai"}
    )
    dispatch = Mock(return_value=response)
    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch)
    observer = Mock(side_effect=RuntimeError("fixture checkpoint unavailable"))
    token = extraction_usage_observer.set(observer)
    try:
        with pytest.raises(ProviderRequestGuardError, match="usage checkpoint"):
            client.generate_chat_response([{"role": "user", "content": "evidence"}])
    finally:
        extraction_usage_observer.reset(token)
    assert dispatch.call_count == 1
    assert observer.call_count == 1


@pytest.mark.parametrize(
    "model,maximum",
    [
        ("gpt-4o", 16384),
        ("anthropic/claude-sonnet-4-20250514", 64000),
        ("gemini/gemini-2.5-flash", 65536),
    ],
)
def test_native_worker_none_cap_is_explicit_before_actual_scoped_dispatch(
    counter, monkeypatch, model, maximum
):
    from types import SimpleNamespace

    from reflexio.server.llm.input_budget import extraction_provider_guard
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.get_model_info",
        Mock(return_value={"max_input_tokens": 1048576, "max_output_tokens": maximum}),
    )
    client = LiteLLMClient(
        LiteLLMConfig(model=model, max_tokens=None, fallback_models=[])
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, refusal=None),
                finish_reason="stop",
            )
        ],
        usage=None,
        model=model,
        _hidden_params={},
    )
    dispatch = Mock(return_value=response)
    monkeypatch.setattr(client, "_completion_with_hard_timeout", dispatch)
    # Ordinary calls retain the provider's existing uncapped request behavior.
    ordinary = client._build_completion_params(
        [{"role": "user", "content": "evidence"}]
    )[0]
    assert "max_tokens" not in ordinary and "max_completion_tokens" not in ordinary
    reserve = Mock()
    token = extraction_provider_guard.set(
        extraction_request_guard(
            context_limits={model: 1048576}, reserve_attempt=reserve
        )
    )
    try:
        assert (
            client.generate_chat_response([{"role": "user", "content": "evidence"}])
            == "ok"
        )
    finally:
        extraction_provider_guard.reset(token)
    assert dispatch.call_args.args[0]["max_tokens"] == maximum
    reserve.assert_called_once()
    assert client.config.max_tokens is None


@pytest.mark.parametrize(
    "override", [{"max_tokens": 4096}, {"max_completion_tokens": 12000}]
)
def test_explicit_output_caps_are_not_replaced_by_native_metadata(override):
    from reflexio.server.llm.input_budget import supply_extraction_output_reservation

    request = {
        "model": "unknown-model",
        "api_base": "https://custom.invalid",
        **override,
    }
    supply_extraction_output_reservation(request)
    assert request == {
        "model": "unknown-model",
        "api_base": "https://custom.invalid",
        **override,
    }


def test_custom_none_output_cap_cannot_inherit_native_maximum(counter, monkeypatch):
    from reflexio.server.llm.input_budget import supply_extraction_output_reservation

    lookup = Mock(return_value={"max_output_tokens": 16384})
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.get_model_info", lookup
    )
    with pytest.raises(
        InputBudgetConfigurationError, match="explicit output reservation"
    ):
        supply_extraction_output_reservation(
            params(max_tokens=None, api_base="https://custom.invalid")
        )
    lookup.assert_not_called()


@pytest.mark.unit
def test_nonrejecting_measurement_uses_the_admission_counter(counter):
    from pydantic import BaseModel

    from reflexio.server.llm.input_budget import (
        check_request_input_budget,
        measure_request_input_tokens,
    )

    class Output(BaseModel):
        decision: bool

    counter.return_value = 1000
    request = params(
        tools=[{"type": "function", "function": {"name": "ask_human"}}],
        tool_choice="auto",
        response_format=Output,
    )
    assert measure_request_input_tokens(request) == 2400
    assert counter.call_args_list[0].kwargs["tools"] == request["tools"]
    assert counter.call_args_list[0].kwargs["tool_choice"] == "auto"
    assert '"decision"' in counter.call_args_list[1].kwargs["text"]
    with pytest.raises(InputBudgetExceededError):
        check_request_input_budget(request, context_limits={"test-model": 10000})
    assert (
        check_request_input_budget(request, context_limits={"test-model": 20000})
        == 2400
    )


@pytest.mark.unit
def test_nonrejecting_measurement_keeps_counting_failure_typed(counter):
    from reflexio.server.llm.input_budget import measure_request_input_tokens

    counter.side_effect = RuntimeError("counter unavailable")
    with pytest.raises(InputBudgetConfigurationError, match="counting failed"):
        measure_request_input_tokens(params())
