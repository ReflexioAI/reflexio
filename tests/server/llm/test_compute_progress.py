"""Unretained progress and durable retention define whether input retry is free."""

import pytest

from reflexio.server.llm.compute_admission import (
    ComputeAdmissionError,
    ComputeInterruptedError,
    admit_external_compute,
    compute_attempt_scope,
    external_compute_admission,
    record_compute_result,
    retain_compute_result,
)
from reflexio.server.services.storage.error import (
    ExecutionInputBusyError,
    ReadinessUnavailableError,
    execution_input_is_busy,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("progress", ["accepted", "completed"])
def test_unretained_compute_cannot_authorize_free_input_replay(progress):
    with pytest.raises(ComputeInterruptedError) as raised, compute_attempt_scope():
        if progress == "accepted":
            admit_external_compute()
        else:
            record_compute_result()
        raise ExecutionInputBusyError("Synthetic private input value")
    assert not execution_input_is_busy(raised.value)
    assert raised.value.__cause__ is None
    assert "private" not in str(raised.value)


def test_guard_refusal_before_dispatch_remains_input_only():
    def blocked():
        raise ExecutionInputBusyError("Synthetic pre-dispatch input")

    token = external_compute_admission.set(blocked)
    try:
        with pytest.raises(ComputeAdmissionError) as raised, compute_attempt_scope():
            admit_external_compute()
    finally:
        external_compute_admission.reset(token)
    assert execution_input_is_busy(raised.value)


def test_retained_child_does_not_create_a_parent_replay_obligation():
    with pytest.raises(ExecutionInputBusyError), compute_attempt_scope():
        with compute_attempt_scope():
            admit_external_compute()
            record_compute_result()
            retain_compute_result()
        raise ExecutionInputBusyError("Cached parent writes are busy")


def test_retained_child_cannot_refund_earlier_unretained_parent_work():
    with pytest.raises(ComputeInterruptedError), compute_attempt_scope():
        admit_external_compute()
        with compute_attempt_scope():
            record_compute_result()
            retain_compute_result()
        raise ExecutionInputBusyError("Parent progress is still unretained")


def test_unretained_child_progress_propagates_until_parent_retention():
    with pytest.raises(ComputeInterruptedError), compute_attempt_scope():
        with compute_attempt_scope():
            record_compute_result()
        raise ExecutionInputBusyError("Child result still volatile")
    with pytest.raises(ExecutionInputBusyError), compute_attempt_scope():
        with compute_attempt_scope():
            record_compute_result()
        retain_compute_result()
        raise ExecutionInputBusyError("Parent result has committed")


@pytest.mark.parametrize("completed", [False, True])
def test_readiness_can_refund_only_without_a_completed_result(completed):
    expected = ComputeInterruptedError if completed else ReadinessUnavailableError
    with pytest.raises(expected), compute_attempt_scope():
        admit_external_compute()
        if completed:
            record_compute_result()
        raise ReadinessUnavailableError("Synthetic readiness deferral")


def test_nested_dispatch_reserves_once_even_after_child_retention():
    from reflexio.server.llm.compute_admission import reserve_compute_attempt

    admitted = []
    with compute_attempt_scope(reserve=lambda: admitted.append(1)):
        with compute_attempt_scope():
            reserve_compute_attempt()
            admit_external_compute()
            retain_compute_result()
        with compute_attempt_scope():
            admit_external_compute()
    assert admitted == [1]


def test_failed_reservation_cannot_enter_provider_or_create_progress():
    def blocked():
        raise ExecutionInputBusyError("Actual owner writer refused")

    with compute_attempt_scope(reserve=blocked) as progress:
        with pytest.raises(ComputeAdmissionError) as raised:
            admit_external_compute()
        assert not progress.unretained
    assert execution_input_is_busy(raised.value)


def test_parallel_children_share_one_completed_owner_reservation():
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from threading import Event

    admitted = []
    entered, release = Event(), Event()

    def reserve():
        admitted.append(1)
        entered.set()
        assert release.wait(5)

    def dispatch():
        with compute_attempt_scope():
            admit_external_compute()

    with (
        compute_attempt_scope(reserve=reserve),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(copy_context().run, dispatch)
        assert entered.wait(5)
        second = pool.submit(copy_context().run, dispatch)
        release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert admitted == [1]
