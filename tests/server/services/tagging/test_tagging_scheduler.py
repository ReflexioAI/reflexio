from __future__ import annotations

import runpy
import threading
from typing import Any

import pytest

from reflexio.server.services.tagging import tagging_scheduler
from reflexio.server.services.tagging.tagging_scheduler import (
    TaggingScheduler,
    schedule_tagging,
)


@pytest.fixture(autouse=True)
def isolated_scheduler(monkeypatch):
    # A publish test may leave unrelated future work in the shared singleton.
    monkeypatch.setattr(TaggingScheduler, "_instance", None)


class _FakeScheduler:
    def __init__(self, sink: list[tuple[Any, Any]]) -> None:
        self._sink = sink

    def schedule(self, key: Any, callback: Any) -> None:
        self._sink.append((key, callback))


def test_scheduler_fires_scheduled_callback(monkeypatch: Any) -> None:
    # Keep the debounce tiny so the test does not wait on the real delay.
    monkeypatch.setattr(tagging_scheduler, "_EFFECTIVE_DELAY_SECONDS", 0.01)
    fired = threading.Event()
    TaggingScheduler.get_instance().schedule(("org", None, "user", "v1"), fired.set)
    assert fired.wait(timeout=5)


def test_scheduler_drain_waits_for_scheduled_callback(monkeypatch: Any) -> None:
    # Keep the debounce tiny so the test does not wait on the real delay.
    monkeypatch.setattr(tagging_scheduler, "_EFFECTIVE_DELAY_SECONDS", 0.01)
    fired = threading.Event()
    scheduler = TaggingScheduler.get_instance()
    scheduler.schedule(("org", None, "drain-user", "v1"), fired.set)

    assert scheduler.drain(timeout_seconds=2.0)
    assert fired.is_set()


def test_scheduler_coalesces_same_scope_to_latest_callback(monkeypatch: Any) -> None:
    now = [100.0]
    monkeypatch.setattr(tagging_scheduler, "_EFFECTIVE_DELAY_SECONDS", 10.0)
    monkeypatch.setattr(tagging_scheduler.time, "monotonic", lambda: now[0])
    first_fired = threading.Event()
    latest_fired = threading.Event()
    scheduler = TaggingScheduler()

    scheduler.schedule(("org", None, "coalesced-user", "v1"), first_fired.set)
    scheduler.schedule(("org", None, "coalesced-user", "v1"), latest_fired.set)

    with scheduler._mutex:
        first_entry, latest_entry = scheduler._heap
        assert first_entry[0] == latest_entry[0]
        assert first_entry[1] != latest_entry[1]

    now[0] = 111.0
    scheduler._wake_event.set()
    assert latest_fired.wait(timeout=5)
    assert not first_fired.is_set()


def test_schedule_tagging_skips_when_no_user(monkeypatch: Any) -> None:
    scheduled: list[tuple[Any, Any]] = []
    monkeypatch.setattr(
        TaggingScheduler,
        "get_instance",
        classmethod(lambda _cls: _FakeScheduler(scheduled)),
    )

    # Empty user_id must not enqueue anything (and must not touch the deps).
    schedule_tagging(
        org_id="o",
        user_id="",
        agent_version="v",
        request_context=None,  # type: ignore[arg-type]
        llm_client=None,  # type: ignore[arg-type]
    )
    assert scheduled == []

    schedule_tagging(
        org_id="o",
        user_id="u",
        agent_version="v",
        request_context=None,  # type: ignore[arg-type]
        llm_client=None,  # type: ignore[arg-type]
    )
    assert len(scheduled) == 1
    assert scheduled[0][0] == ("o", None, "u", "v")


def test_test_mode_cannot_change_tagging_policy(monkeypatch):
    monkeypatch.setenv("IS_TEST_ENV", "true")
    policy = runpy.run_path(tagging_scheduler.__file__)
    assert policy["_EFFECTIVE_DELAY_SECONDS"] == 15


# --- provider outage -----------------------------------------------------------------

_OUTAGE_ROUTE = "test/tagging-model||-"


def _open_breaker() -> None:
    from reflexio.server.llm.provider_circuit import (
        OPEN_AFTER_FAILURES,
        provider_circuit,
    )

    for _ in range(OPEN_AFTER_FAILURES):
        provider_circuit.admit(_OUTAGE_ROUTE, model="test/tagging-model").failed()


def _circuit_refusal() -> BaseException:
    from reflexio.server.llm.provider_circuit import ProviderCircuitOpenError

    refused = ProviderCircuitOpenError("Provider circuit open; request not sent")
    refused.circuit_route = _OUTAGE_ROUTE
    return refused


class _RecordingScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any, float]] = []

    def schedule(self, key: Any, callback: Any, *, delay_seconds: float = 0.0) -> None:
        self.calls.append((key, callback, delay_seconds))


def test_a_pass_refused_by_an_open_breaker_waits_for_the_provider(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """PYTHON-FASTAPI-18W's sibling: an outage dropped the pass at ERROR."""
    _open_breaker()
    recorder = _RecordingScheduler()
    monkeypatch.setattr(
        TaggingScheduler, "get_instance", classmethod(lambda _cls: recorder)
    )
    key = ("org", None, "outage-user", "v1")

    def callback() -> None:
        raise _circuit_refusal()

    with caplog.at_level("INFO"):
        TaggingScheduler._run_callback(key, callback)

    assert [(k, c) for k, c, _ in recorder.calls] == [(key, callback)]
    assert recorder.calls[0][2] > 0
    assert [r for r in caplog.records if r.levelname == "ERROR"] == []
    assert any("event=tagging_provider_wait" in r.getMessage() for r in caplog.records)


def test_an_ordinary_tagging_failure_is_still_reported_and_dropped(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = _RecordingScheduler()
    monkeypatch.setattr(
        TaggingScheduler, "get_instance", classmethod(lambda _cls: recorder)
    )

    def callback() -> None:
        raise ValueError("tagging bug")

    with caplog.at_level("INFO"):
        TaggingScheduler._run_callback(("org", None, "bug-user", "v1"), callback)

    assert recorder.calls == []
    assert any(
        r.levelname == "ERROR" and "Tagging callback failed" in r.getMessage()
        for r in caplog.records
    )


def test_an_outage_delay_lengthens_but_never_shortens_the_debounce(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(tagging_scheduler, "_EFFECTIVE_DELAY_SECONDS", 15.0)
    monkeypatch.setattr(tagging_scheduler.time, "monotonic", lambda: 100.0)
    scheduler = TaggingScheduler()
    scheduler.schedule(("org", None, "a", "v1"), lambda: None, delay_seconds=600.0)
    scheduler.schedule(("org", None, "b", "v1"), lambda: None, delay_seconds=0.0)
    with scheduler._mutex:
        fire = {key[2]: when for when, _seq, key in scheduler._heap}
    assert fire == {"a": 700.0, "b": 115.0}
