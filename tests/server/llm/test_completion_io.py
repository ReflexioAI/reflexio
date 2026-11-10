"""Qualification hooks preserve original completion IO and context boundaries."""

from contextvars import Context
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.server.llm.completion_io import completion_io_scope
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.test_support.llm_mock import inline_test_completion


@pytest.fixture
def physical_completion(monkeypatch):
    from reflexio.server.llm import _model_compat as generation

    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o", timeout=10))
    monkeypatch.setattr(client, "_run_completion", inline_test_completion)
    dispatch = Mock(return_value=object())
    monkeypatch.setattr(generation, "model_completion", dispatch)
    return client, dispatch


def test_default_delegates_original_completion(physical_completion):
    client, dispatch = physical_completion
    params = {"model": "gpt-4o", "timeout": 10}
    assert client._completion_with_hard_timeout(params, 10) is dispatch.return_value
    dispatch.assert_called_once_with(params)


@pytest.mark.parametrize("fails", [False, True])
def test_recording_observes_original_success_or_error(physical_completion, fails):
    client, dispatch = physical_completion
    error = RuntimeError("synthetic provider error")
    if fails:
        dispatch.side_effect = error
    observed = []
    params = {"model": "gpt-4o", "timeout": 10}

    def record(request, hard_timeout, invoke):
        assert request is params
        assert hard_timeout == 10
        try:
            result = invoke()
        except RuntimeError as exc:
            observed.append(exc)
            raise
        observed.append(result)
        return result

    with completion_io_scope(record):
        if fails:
            with pytest.raises(RuntimeError) as caught:
                client._completion_with_hard_timeout(params, 10)
            assert caught.value is error
        else:
            assert (
                client._completion_with_hard_timeout(params, 10)
                is dispatch.return_value
            )
    assert observed == [error if fails else dispatch.return_value]
    dispatch.assert_called_once_with(params)


def test_nested_replay_resets_after_exception_and_does_not_leak_context(
    physical_completion,
):
    client, dispatch = physical_completion
    outer_response, inner_response = object(), object()
    with completion_io_scope(lambda *_: outer_response):
        assert client._completion_with_hard_timeout({}, 10) is outer_response
        with pytest.raises(ValueError), completion_io_scope(lambda *_: inner_response):
            assert client._completion_with_hard_timeout({}, 10) is inner_response
            raise ValueError("leave inner scope")
        assert client._completion_with_hard_timeout({}, 10) is outer_response
        assert (
            Context().run(client._completion_with_hard_timeout, {}, 10)
            is dispatch.return_value
        )
    assert client._completion_with_hard_timeout({}, 10) is dispatch.return_value
    assert dispatch.call_count == 2


def test_budget_abort_does_not_dispatch(physical_completion):
    client, dispatch = physical_completion

    def exhausted(*_args: Any) -> Any:
        raise RuntimeError("qualification physical-call allowance exhausted")

    with completion_io_scope(exhausted), pytest.raises(RuntimeError, match="allowance"):
        client._completion_with_hard_timeout({}, 10)
    dispatch.assert_not_called()


def test_recording_dispatch_retains_original_timeout(physical_completion):
    from reflexio.server.llm.litellm_client import LLMHardTimeoutError

    client, dispatch = physical_completion
    observed = []

    def record(_request, _hard_timeout, invoke):
        try:
            return invoke()
        except LLMHardTimeoutError as exc:
            observed.append(exc)
            raise

    with completion_io_scope(record), pytest.raises(LLMHardTimeoutError):
        client._completion_with_hard_timeout({}, 0)
    assert len(observed) == 1
    dispatch.assert_not_called()


def test_public_request_guard_runs_before_replay(physical_completion):
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError

    client, dispatch = physical_completion
    replay = Mock()

    def refuse(*_args):
        raise ProviderRequestGuardError("synthetic admission refusal")

    with completion_io_scope(replay), pytest.raises(ProviderRequestGuardError):
        client.generate_response("synthetic", provider_request_guard=refuse)
    replay.assert_not_called()
    dispatch.assert_not_called()


def test_public_replay_still_parses_and_accounts_response(physical_completion):
    import litellm

    from reflexio.server.llm.token_accounting import RunTokenCapture, run_token_capture

    client, dispatch = physical_completion
    response = litellm.ModelResponse(
        model="gpt-4o",
        choices=[{"message": {"role": "assistant", "content": "saved response"}}],
        usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    )
    capture = RunTokenCapture()
    token = run_token_capture.set(capture)
    try:
        with completion_io_scope(lambda *_: response):
            assert client.generate_response("synthetic") == "saved response"
    finally:
        run_token_capture.reset(token)
    assert capture.totals.prompt_tokens == 11
    assert capture.totals.completion_tokens == 3
    dispatch.assert_not_called()
