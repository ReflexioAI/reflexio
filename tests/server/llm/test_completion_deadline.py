"""A received completion is accounted, but cannot refresh its acceptance budget."""

import queue
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import litellm
import pytest
from pydantic import BaseModel

from reflexio.server import search_runtime
from reflexio.server.llm import _litellm_text_generation as generation
from reflexio.server.llm import call_deadline
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.token_accounting import RunTokenCapture, run_token_capture
from reflexio.test_support.llm_mock import use_isolated_completion


@pytest.fixture
def completion(monkeypatch):
    clock = SimpleNamespace(now=1000.0, startup=0.0, receive=0.0, construct=0.0)
    local_time = SimpleNamespace(
        monotonic=lambda: clock.now,
        perf_counter=time.perf_counter,
        time=time.time,
        sleep=time.sleep,
    )
    monkeypatch.setattr(generation, "time", local_time)
    monkeypatch.setattr(search_runtime, "time", local_time)
    monkeypatch.setattr(call_deadline, "monotonic", local_time.monotonic)
    response = litellm.ModelResponse(
        model="gpt-4o",
        choices=[{"message": {"role": "assistant", "content": "received"}}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    worker = Mock(pid=None, exitcode=0)
    worker.is_alive.return_value = False

    def start():
        clock.now += clock.startup
        worker.pid = 123

    worker.start.side_effect = start
    result_queue = Mock()

    def receive(**_kwargs):
        clock.now += clock.receive
        return "ok", response

    result_queue.get.side_effect = receive

    def construct(**_kwargs):
        clock.now += clock.construct
        return worker

    process = Mock(side_effect=construct)
    monkeypatch.setattr(
        generation.multiprocessing,
        "get_context",
        lambda: SimpleNamespace(Queue=lambda **_: result_queue, Process=process),
    )
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o", max_retries=0, timeout=1))
    use_isolated_completion(client)
    usage = Mock(wraps=client._log_token_usage)
    monkeypatch.setattr(client, "_log_token_usage", usage)
    capture = RunTokenCapture()
    token = run_token_capture.set(capture)
    try:
        yield SimpleNamespace(
            clock=clock,
            worker=worker,
            queue=result_queue,
            process=process,
            client=client,
            usage=usage,
            response=response,
            capture=capture,
        )
    finally:
        run_token_capture.reset(token)


@pytest.mark.parametrize("slow_stage", ["startup", "receive"])
@pytest.mark.parametrize("scope", ["model", "search", "rung"])
@pytest.mark.parametrize("elapsed", [0.03, 0.075])
def test_public_late_response_is_accounted_but_not_accepted(
    completion, slow_stage, scope, elapsed
):
    setattr(completion.clock, slow_stage, elapsed)
    if scope == "search":
        context = search_runtime._scope.set(
            search_runtime.SearchScope(started=1000.0, deadline=1000.03)
        )
    else:
        context = None
    completion.client.config.timeout = 0.03 if scope == "rung" else 1
    try:
        with (
            call_deadline.model_deadline(1000.03)
            if scope == "model"
            else nullcontext(),
            pytest.raises(
                (generation.LiteLLMClientError, search_runtime.SearchDeadlineError)
            ),
        ):
            completion.client.generate_response("synthetic")
    finally:
        if context is not None:
            search_runtime._scope.reset(context)
    assert completion.worker.start.call_count == 1
    assert completion.usage.call_count == 1
    assert completion.usage.call_args.args[1] is completion.response
    assert completion.capture.completions == 1
    assert completion.capture.totals.prompt_tokens == 1
    assert completion.capture.totals.completion_tokens == 1
    assert completion.worker.close.called and completion.queue.close.called


def test_fast_public_response_is_still_accepted(completion):
    completion.clock.startup = completion.clock.receive = 0.001
    with call_deadline.model_deadline(1000.03):
        assert completion.client.generate_response("synthetic") == "received"
    assert completion.usage.call_count == 1


def test_expired_model_deadline_does_not_construct_worker(completion):
    with (
        call_deadline.model_deadline(999.0),
        pytest.raises(TimeoutError),
    ):
        completion.client._completion_with_deadline(
            {"model": "gpt-4o"}, generation.time.monotonic() + (1)
        )
    completion.process.assert_not_called()
    completion.worker.start.assert_not_called()


def test_construction_exhausting_budget_does_not_start_worker(completion):
    completion.clock.construct = 0.075
    with pytest.raises(generation.LLMHardTimeoutError):
        completion.client._completion_with_deadline(
            {"model": "gpt-4o"}, generation.time.monotonic() + (0.03)
        )
    completion.worker.start.assert_not_called()
    assert completion.worker.close.called and completion.queue.close.called


def test_exited_child_final_read_spends_only_remaining_budget(completion):
    waits = []

    def empty(*, timeout):
        waits.append(timeout)
        completion.clock.now += timeout
        raise queue.Empty

    completion.clock.startup = 0.02
    completion.queue.get.side_effect = empty
    with pytest.raises(generation.LiteLLMClientError):
        completion.client._completion_with_deadline(
            {"model": "gpt-4o"}, generation.time.monotonic() + (0.03)
        )
    assert sum(waits) <= 0.010001
    assert completion.clock.now <= 1000.030001


def test_enclosing_expiry_prevents_another_fallback(completion):
    completion.client.config.fallback_models = ["gpt-4o-mini"]
    completion.clock.receive = 0.075
    with (
        call_deadline.model_deadline(1000.03),
        pytest.raises(generation.LiteLLMClientError),
    ):
        completion.client.generate_response("synthetic")
    assert completion.worker.start.call_count == 1
    assert completion.usage.call_count == 1


def test_public_parsing_cannot_return_output_after_deadline(completion, monkeypatch):
    parse = completion.client._maybe_parse_structured_output

    def slow_parse(*args):
        completion.clock.now += 0.075
        return parse(*args)

    monkeypatch.setattr(completion.client, "_maybe_parse_structured_output", slow_parse)
    with (
        call_deadline.model_deadline(1000.03),
        pytest.raises(generation.LiteLLMClientError),
    ):
        completion.client.generate_response("synthetic")
    assert completion.usage.call_count == 1


@pytest.mark.parametrize("budget", [0, -0.01])
def test_spent_attempt_budget_cannot_be_restarted(completion, budget):
    with pytest.raises(generation.LLMHardTimeoutError):
        completion.client._completion_with_deadline(
            {"model": "gpt-4o"}, generation.time.monotonic() + (budget)
        )
    completion.process.assert_not_called()


def test_provider_guard_spending_budget_cannot_dispatch(completion):
    def guard(*_args):
        completion.clock.now += 0.075

    with (
        call_deadline.model_deadline(1000.03),
        pytest.raises(generation.LiteLLMClientError),
    ):
        completion.client.generate_response("synthetic", provider_request_guard=guard)
    completion.process.assert_not_called()
    completion.usage.assert_not_called()


@pytest.mark.parametrize("model_budget,search_budget", [(0.02, 0.03), (0.03, 0.02)])
def test_nested_scopes_use_the_tighter_deadline(
    completion, model_budget, search_budget
):
    completion.clock.receive = 0.025
    token = search_runtime._scope.set(
        search_runtime.SearchScope(started=1000.0, deadline=1000.0 + search_budget)
    )
    try:
        with (
            call_deadline.model_deadline(1000.0 + model_budget),
            call_deadline.model_deadline(1001.0),
            pytest.raises(
                (generation.LiteLLMClientError, search_runtime.SearchDeadlineError)
            ),
        ):
            completion.client.generate_response("synthetic")
    finally:
        search_runtime._scope.reset(token)
    assert completion.usage.call_count == 1


def test_validation_spending_enclosing_budget_cannot_return_success(completion):
    class Answer(BaseModel):
        value: str

    completion.response.choices[0].message.content = '{"value":"ready"}'

    def validate(_output):
        completion.clock.now += 0.075
        return []

    with (
        call_deadline.model_deadline(1000.03),
        pytest.raises(generation.LiteLLMClientError),
    ):
        completion.client.generate_response(
            "synthetic", response_format=Answer, structured_output_validator=validate
        )
    assert completion.usage.call_count == 1


def test_child_startup_expiry_prevents_provider_dispatch(monkeypatch):
    from reflexio.server.llm import _litellm_subprocess as subprocess

    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        subprocess, "time", SimpleNamespace(monotonic=lambda: clock.now), raising=False
    )
    monkeypatch.setattr(subprocess, "_reset_llm_client_state_after_fork", lambda: None)

    def register():
        clock.now += 1

    monkeypatch.setattr(subprocess, "_register_local_cli_providers", register)
    dispatch = Mock()
    monkeypatch.setattr(subprocess, "model_completion", dispatch)
    result_queue = Mock()
    subprocess._litellm_completion_worker({"model": "gpt-4o"}, result_queue, 1000.5)
    dispatch.assert_not_called()
    status, error = result_queue.put.call_args.args[0]
    assert status == "error"
    assert error.type_name == "LLMHardTimeoutError"


def test_semantic_validator_cannot_accept_after_attempt_expiry(completion):
    from pydantic import BaseModel

    class Output(BaseModel):
        value: str

    completion.response.choices[0].message.content = '{"value":"ok"}'
    completion.client.config.timeout = 0.03

    def validate(_output):
        completion.clock.now += 0.075
        return []

    with pytest.raises(generation.LiteLLMClientError, match="attempt deadline"):
        completion.client.generate_chat_response(
            [{"role": "user", "content": "synthetic"}],
            response_format=Output,
            structured_output_validator=validate,
        )
    assert completion.usage.call_count == 1
