"""A loaded completion worker spends its caller's existing attempt budget."""

from queue import Queue
from types import SimpleNamespace
from typing import Any

import pytest

from reflexio.server.llm import _litellm_subprocess as worker


def test_child_startup_spends_provider_transport_allowance(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        worker,
        "_reset_llm_client_state_after_fork",
        lambda: clock.__setitem__(0, 104.0),
    )
    monkeypatch.setattr(worker, "_register_local_cli_providers", lambda: None)
    monkeypatch.setattr(worker, "_picklable_completion_result", lambda result: result)
    dispatched = []
    monkeypatch.setattr(
        worker,
        "model_completion",
        lambda params: dispatched.append(params) or SimpleNamespace(choices=[]),
    )
    params = {"model": "offline-model", "timeout": 10.0}
    result: Any = Queue()

    worker._litellm_completion_worker(params, result, 110.0)

    assert result.get_nowait()[0] == "ok"
    assert dispatched == [{"model": "offline-model", "timeout": 6.0}]
    assert params == {"model": "offline-model", "timeout": 10.0}


@pytest.mark.parametrize("startup_finished", [110.0, 111.0])
def test_expired_child_never_dispatches(monkeypatch, startup_finished):
    monkeypatch.setattr(worker.time, "monotonic", lambda: startup_finished)
    monkeypatch.setattr(worker, "_reset_llm_client_state_after_fork", lambda: None)
    monkeypatch.setattr(worker, "_register_local_cli_providers", lambda: None)
    dispatched = []
    monkeypatch.setattr(
        worker, "model_completion", lambda params: dispatched.append(params)
    )
    result: Any = Queue()

    worker._litellm_completion_worker(
        {"model": "offline-model", "timeout": 10.0}, result, 110.0
    )

    status, error = result.get_nowait()
    assert status == "error"
    assert error.type_name == "LLMHardTimeoutError"
    assert dispatched == []


def test_mock_execution_cannot_qualify_a_live_provider(monkeypatch):
    from reflexio.server.llm._litellm_text_generation import TextGenerationMixin
    from reflexio.test_support.llm_mock import (
        assert_litellm_unpatched,
        unpatched_litellm,
    )

    with unpatched_litellm(), monkeypatch.context() as local_patch:
        local_patch.setattr(TextGenerationMixin, "_execute_completion", lambda *_: None)
        with pytest.raises(AssertionError, match="execution adapter"):
            assert_litellm_unpatched()


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_deadline_cannot_dispatch(monkeypatch, deadline):
    monkeypatch.setattr(worker, "_reset_llm_client_state_after_fork", lambda: None)
    monkeypatch.setattr(worker, "_register_local_cli_providers", lambda: None)
    dispatched = []
    monkeypatch.setattr(
        worker, "model_completion", lambda params: dispatched.append(params)
    )
    result: Any = Queue()
    worker._litellm_completion_worker(
        {"model": "offline-model", "timeout": 10}, result, deadline
    )
    assert result.get_nowait()[0] == "error"
    assert dispatched == []


def test_child_shares_the_same_attempt_with_nested_calls(monkeypatch):
    from reflexio.server.llm.call_deadline import model_call_deadline

    monkeypatch.setattr(worker.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(worker, "_reset_llm_client_state_after_fork", lambda: None)
    monkeypatch.setattr(worker, "_register_local_cli_providers", lambda: None)
    observed = []
    monkeypatch.setattr(
        worker,
        "model_completion",
        lambda _params: observed.append(model_call_deadline.get()) or "done",
    )
    result: Any = Queue()
    worker._litellm_completion_worker(
        {"model": "offline-model", "timeout": 10}, result, 110.0
    )
    assert result.get_nowait() == ("ok", "done")
    assert observed == [110.0]


@pytest.mark.parametrize(
    "helper_name", ["use_mock_completion", "use_isolated_completion"]
)
def test_execution_selection_cannot_mutate_client_class(helper_name):
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.test_support import llm_mock

    with pytest.raises(TypeError, match="instance, never a class"):
        getattr(llm_mock, helper_name)(LiteLLMClient)


def test_live_guard_rejects_subclass_executor_shadow(monkeypatch):
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.test_support.llm_mock import (
        assert_litellm_unpatched,
        unpatched_litellm,
    )

    with unpatched_litellm(), monkeypatch.context() as local_patch:
        local_patch.setattr(LiteLLMClient, "_execute_completion", lambda *_: None)
        with pytest.raises(AssertionError, match="execution adapter"):
            assert_litellm_unpatched()
