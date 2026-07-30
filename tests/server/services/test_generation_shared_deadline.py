"""The generation timeout is ONE budget shared by both awaited futures.

Profile and playbook generation run concurrently but are awaited in sequence.
Passing each ``future.result()`` the same constant handed the second service a
fresh full budget on top of whatever the first had already consumed, so a 600s
setting really allowed up to 1200s and the "timed out after 600s" log line
misreported the budget it had enforced.

These drive ``_completed_within_shared_budget`` — the loop BOTH production call
sites use — and assert on the timeout actually handed to each ``result()``
call, so they fail if either site reverts to a per-future constant.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

import pytest

from reflexio.server.services.generation_service import (
    _completed_within_shared_budget,
    _SharedDeadline,
)


class _RecordingFuture:
    """Future stand-in that records the timeout it was awaited with."""

    def __init__(self, granted: list[float], outcome: Any = "value") -> None:
        self._granted = granted
        self._outcome = outcome

    def result(self, timeout: float | None = None) -> Any:
        self._granted.append(float(timeout if timeout is not None else -1.0))
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


def _drain(
    awaited: list[tuple[Any, str]], warnings: list[str]
) -> list[tuple[str, Any]]:
    return list(
        _completed_within_shared_budget(
            awaited,  # type: ignore[arg-type]
            request_id="req-1",
            warnings=warnings,
            timeout_seconds=600.0,
        )
    )


def test_second_future_gets_only_the_leftover_budget(monkeypatch: pytest.MonkeyPatch):
    """The regression: two sequential waits must not each get a full budget."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    granted: list[float] = []

    class _SlowFirst(_RecordingFuture):
        def result(self, timeout: float | None = None) -> Any:
            value = super().result(timeout)
            clock["now"] += 590.0  # first service burns most of the budget
            return value

    warnings: list[str] = []
    completed = _drain(
        [
            (_SlowFirst(granted), "profile_generation"),
            (_RecordingFuture(granted), "playbook_generation"),
        ],
        warnings,
    )

    assert [name for name, _ in completed] == [
        "profile_generation",
        "playbook_generation",
    ]
    assert granted[0] == pytest.approx(600.0)
    assert granted[1] == pytest.approx(10.0)
    assert warnings == []


def test_exhausted_budget_grants_the_second_future_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    """An overspent budget must clamp to 0, not wrap into a negative."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    granted: list[float] = []

    class _Overrunning(_RecordingFuture):
        def result(self, timeout: float | None = None) -> Any:
            value = super().result(timeout)
            clock["now"] += 900.0
            return value

    warnings: list[str] = []
    _drain(
        [
            (_Overrunning(granted), "profile_generation"),
            (_RecordingFuture(granted), "playbook_generation"),
        ],
        warnings,
    )

    assert granted[1] == 0.0


def test_timeout_warning_reports_the_budget_actually_enforced(
    monkeypatch: pytest.MonkeyPatch,
):
    """Naming the full 600s total would misreport the wait that just expired."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    granted: list[float] = []

    class _SlowFirst(_RecordingFuture):
        def result(self, timeout: float | None = None) -> Any:
            super().result(timeout)
            clock["now"] += 590.0
            return "value"

    warnings: list[str] = []
    completed = _drain(
        [
            (_SlowFirst(granted), "profile_generation"),
            (
                _RecordingFuture(granted, FuturesTimeoutError()),
                "playbook_generation",
            ),
        ],
        warnings,
    )

    assert [name for name, _ in completed] == ["profile_generation"]
    assert len(warnings) == 1
    assert "playbook_generation timed out after 10.0s" in warnings[0]
    assert "shared 600s budget" in warnings[0]


def test_a_failing_service_is_reported_and_does_not_stop_the_other():
    warnings: list[str] = []
    granted: list[float] = []

    completed = _drain(
        [
            (_RecordingFuture(granted, RuntimeError("boom")), "profile_generation"),
            (_RecordingFuture(granted, "plan"), "playbook_generation"),
        ],
        warnings,
    )

    assert completed == [("playbook_generation", "plan")]
    assert warnings == ["profile_generation failed: boom"]


def test_remaining_shrinks_as_the_budget_is_spent(monkeypatch: pytest.MonkeyPatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    budget = _SharedDeadline(600.0)
    assert budget.remaining() == pytest.approx(600.0)

    clock["now"] += 250.0
    assert budget.remaining() == pytest.approx(350.0)


def test_exhausted_budget_does_not_wait_for_a_pending_future():
    """A spent budget fails fast instead of blocking for another full timeout."""
    budget = _SharedDeadline(0.0)
    pending: Future = Future()

    started = time.monotonic()
    with pytest.raises(FuturesTimeoutError):
        pending.result(timeout=budget.remaining())

    assert time.monotonic() - started < 1.0
