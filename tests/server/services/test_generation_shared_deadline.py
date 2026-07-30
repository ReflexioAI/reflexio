"""The generation timeout is ONE budget shared by both awaited futures.

Profile and playbook generation run concurrently but are awaited in sequence.
Passing each ``future.result()`` the same constant handed the second service a
fresh full budget on top of whatever the first had already consumed, so a 600s
setting really allowed up to 1200s and the "timed out after 600s" log line
misreported the budget it had enforced.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

from reflexio.server.services.generation_service import _SharedDeadline


def test_remaining_shrinks_as_the_budget_is_spent(monkeypatch: pytest.MonkeyPatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    budget = _SharedDeadline(600.0)
    assert budget.remaining() == pytest.approx(600.0)

    clock["now"] += 250.0
    assert budget.remaining() == pytest.approx(350.0)

    clock["now"] += 100.0
    assert budget.remaining() == pytest.approx(250.0)


def test_remaining_clamps_at_zero_once_exhausted(monkeypatch: pytest.MonkeyPatch):
    """An overspent budget must fail the next wait, not wrap to a negative."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    budget = _SharedDeadline(600.0)
    clock["now"] += 900.0

    assert budget.remaining() == 0.0


def test_second_future_inherits_only_the_leftover_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    """The regression: two sequential waits must not each get a full budget."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    budget = _SharedDeadline(600.0)
    granted: list[float] = []

    # First service resolves immediately but burns 590s of wall clock.
    first: Future = Future()
    first.set_result("profile")
    granted.append(budget.remaining())
    first.result(timeout=granted[-1])
    clock["now"] += 590.0

    # Second service is still running; it may only have what is left.
    second: Future = Future()
    granted.append(budget.remaining())

    assert granted[0] == pytest.approx(600.0)
    assert granted[1] == pytest.approx(10.0)
    with pytest.raises(FuturesTimeoutError):
        second.result(timeout=0.01)


def test_exhausted_budget_does_not_wait_for_a_pending_future():
    """A spent budget fails fast instead of blocking for another full timeout."""
    budget = _SharedDeadline(0.0)
    pending: Future = Future()

    started = time.monotonic()
    with pytest.raises(FuturesTimeoutError):
        pending.result(timeout=budget.remaining())

    assert time.monotonic() - started < 1.0
