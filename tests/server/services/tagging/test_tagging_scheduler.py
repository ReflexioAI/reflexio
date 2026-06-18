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
    TaggingScheduler.get_instance().schedule(("org", "user", "v1"), fired.set)
    assert fired.wait(timeout=5)


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
    assert scheduled[0][0] == ("o", "u", "v")
