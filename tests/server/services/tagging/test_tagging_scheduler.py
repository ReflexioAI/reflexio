from __future__ import annotations

import threading
from typing import Any

from reflexio.server.services.tagging import tagging_scheduler
from reflexio.server.services.tagging.tagging_scheduler import (
    TaggingScheduler,
    schedule_tagging,
)


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


def test_tagging_callback_streak_is_per_job_type(transient_failure_classifier) -> None:
    """One-shot keyed work: any later tagging success ends the streak."""
    from reflexio.server import background_work

    def failing() -> None:
        raise transient_failure_classifier("server closed the connection")

    TaggingScheduler._run_callback(("org_1", None, "u1", "v1"), failing)
    assert set(background_work._streaks) == {"tagging-callback"}

    TaggingScheduler._run_callback(("org_2", None, "u2", "v1"), lambda: None)
    assert background_work._streaks == {}
