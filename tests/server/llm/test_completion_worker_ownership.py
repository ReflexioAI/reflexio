"""Completion callers and live isolated workers share one provider permit."""

import multiprocessing
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.server.llm import _litellm_text_generation as generation
from reflexio.server.llm import _provider_concurrency as admission
from reflexio.server.llm import call_deadline
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig


@pytest.fixture
def provider_permit(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    released = threading.Event()
    release = semaphore.release

    def release_and_note():
        release()
        released.set()

    monkeypatch.setattr(semaphore, "release", release_and_note)
    monkeypatch.setattr(admission, "_provider_key", lambda _model: "owned-worker")
    monkeypatch.setattr(admission, "_get_semaphore", lambda _provider: semaphore)
    return released


class LingeringWorker:
    """Even after kill, capacity stays held until the OS reports process exit."""

    def __init__(self, *, fail_start=False):
        self.pid = None
        self.exitcode = None
        self.fail_start = fail_start
        self.killed = threading.Event()
        self.allow_exit = threading.Event()
        self.reaping = threading.Event()
        self.closed = threading.Event()
        self.join_timeouts = []

    def start(self):
        if self.fail_start:
            raise RuntimeError("synthetic start failure")
        self.pid = 123

    def is_alive(self):
        return self.pid is not None and self.exitcode is None

    def terminate(self):
        pass

    def kill(self):
        self.killed.set()

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if timeout is None:
            self.reaping.set()
            assert self.allow_exit.wait(5), "test did not release its owned worker"
            self.exitcode = -9

    def close(self):
        assert not self.is_alive()
        self.closed.set()


@pytest.mark.parametrize("outcome", ["success", "receive-error"])
def test_live_worker_retains_permit_after_caller_finishes(
    monkeypatch, provider_permit, outcome
):
    worker = LingeringWorker()
    result_queue = Mock()
    if outcome == "success":
        result_queue.get.return_value = ("ok", "ready")
    else:
        result_queue.get.side_effect = RuntimeError("synthetic receiver failure")
    context = SimpleNamespace(
        Queue=lambda **_kwargs: result_queue,
        Process=lambda **_kwargs: worker,
    )
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    try:
        with admission.provider_slot("gpt-4o"):
            if outcome == "success":
                assert (
                    client._completion_with_hard_timeout({"model": "gpt-4o"}, 30)
                    == "ready"
                )
            else:
                with pytest.raises(RuntimeError, match="synthetic receiver failure"):
                    client._completion_with_hard_timeout({"model": "gpt-4o"}, 30)
        assert worker.killed.is_set()
        assert worker.reaping.wait(2)
        assert not provider_permit.is_set()
        assert not worker.closed.is_set()
        assert not result_queue.close.called
        with (
            pytest.raises(admission.ProviderCapSaturatedError),
            admission.provider_slot("gpt-4o", max_wait=0),
        ):
            pytest.fail("live worker's provider capacity was reused")
    finally:
        worker.allow_exit.set()
    assert provider_permit.wait(2)
    assert worker.closed.is_set()
    assert worker.join_timeouts == [None]
    result_queue.close.assert_called_once()
    result_queue.join_thread.assert_called_once()
    with admission.provider_slot("gpt-4o", max_wait=0):
        pass


def test_start_failure_closes_unstarted_worker_and_queue(monkeypatch, provider_permit):
    worker = LingeringWorker(fail_start=True)
    result_queue = Mock()
    context = SimpleNamespace(
        Queue=lambda **_kwargs: result_queue,
        Process=lambda **_kwargs: worker,
    )
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    with (
        pytest.raises(RuntimeError, match="synthetic start failure"),
        admission.provider_slot("gpt-4o"),
    ):
        client._completion_with_hard_timeout({"model": "gpt-4o"}, 30)
    assert provider_permit.is_set()
    assert worker.closed.is_set()
    assert not worker.killed.is_set()
    assert worker.join_timeouts == []
    result_queue.close.assert_called_once()
    result_queue.join_thread.assert_called_once()


def _publish_then_linger(_params, result_queue, _deadline):
    result_queue.put(("ok", {"pid": os.getpid(), "value": "ready"}))
    time.sleep(10)


def test_rejected_late_public_result_keeps_live_worker_permit(
    monkeypatch, provider_permit
):
    clock = [1000.0]
    monkeypatch.setattr(
        generation,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter),
    )
    monkeypatch.setattr(call_deadline, "monotonic", lambda: clock[0])
    worker = LingeringWorker()
    result_queue = Mock()

    def receive(**_kwargs):
        clock[0] += 0.075
        return "ok", SimpleNamespace(usage=None)

    result_queue.get.side_effect = receive
    context = SimpleNamespace(
        Queue=lambda **_: result_queue, Process=lambda **_: worker
    )
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    try:
        with (
            call_deadline.model_deadline(1000.03),
            pytest.raises(generation.LiteLLMClientError, match="attempt deadline"),
        ):
            client.generate_response("synthetic")
        assert worker.reaping.wait(2)
        assert not provider_permit.is_set()
        with (
            pytest.raises(admission.ProviderCapSaturatedError),
            admission.provider_slot("gpt-4o", max_wait=0),
        ):
            pytest.fail("late result freed a still-live worker's capacity")
    finally:
        worker.allow_exit.set()
    assert provider_permit.wait(2)


def test_multiple_retained_owners_release_capacity_only_after_last_exit(
    provider_permit,
):
    with admission.provider_slot("gpt-4o"):
        first = admission.retain_current_provider_slot()
        second = admission.retain_current_provider_slot()
    first()
    assert not provider_permit.is_set()
    with (
        pytest.raises(admission.ProviderCapSaturatedError),
        admission.provider_slot("gpt-4o", max_wait=0),
    ):
        pytest.fail("another owner is still using this permit")
    second()
    assert provider_permit.is_set()


def test_unknown_nested_provider_does_not_retain_enclosing_permit(
    monkeypatch, provider_permit
):
    monkeypatch.setattr(
        admission,
        "_provider_key",
        lambda model: None if model == "unknown" else "owned-worker",
    )
    with admission.provider_slot("gpt-4o"), admission.provider_slot("unknown"):
        release_unknown = admission.retain_current_provider_slot()
    assert provider_permit.is_set()
    # The unknown context's callback must not release or retain the outer slot.
    with admission.provider_slot("gpt-4o", max_wait=0):
        release_unknown()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="synthetic worker fixture requires fork",
)
def test_actual_lingering_worker_is_reaped_without_blocking_caller(
    monkeypatch, provider_permit
):
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(generation.multiprocessing, "get_context", lambda: context)
    monkeypatch.setattr(generation, "_litellm_completion_worker", _publish_then_linger)
    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    monkeypatch.setattr(client, "_should_process_isolate_completion", lambda *_: True)
    started = time.monotonic()
    with admission.provider_slot("gpt-4o"):
        result = client._completion_with_hard_timeout({"model": "gpt-4o"}, 30)
    elapsed = time.monotonic() - started
    assert result["value"] == "ready"
    assert provider_permit.wait(3)
    with pytest.raises(ProcessLookupError):
        os.kill(result["pid"], 0)
    assert elapsed < 1.5
