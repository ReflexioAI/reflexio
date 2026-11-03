"""A timeout cannot mint new maintenance capacity on the next tick."""

import threading
from typing import Never

import pytest

from reflexio.server.services.lineage import gc_scheduler

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("next_orgs", [["a", "b"], ["c", "d"]])
def test_scheduler_retains_capacity_until_timed_out_work_retires(
    next_orgs, monkeypatch
):
    def unexpected_context(_org_id: str) -> Never:
        raise AssertionError("The tick test supplies its sweep directly")

    scheduler = gc_scheduler.LineageGCScheduler(
        request_context_factory=unexpected_context, bootstrap_org_id="a"
    )
    monkeypatch.setattr(gc_scheduler, "_ORG_SWEEP_TIMEOUT_SECONDS", 0.02)
    release = threading.Event()
    third_started = threading.Event()
    condition = threading.Condition()
    active = 0
    peak = 0

    def blocked_sweep(_org_id):
        nonlocal active, peak
        with condition:
            active += 1
            peak = max(peak, active)
            if active > 2:
                third_started.set()
            condition.notify_all()
        try:
            assert release.wait(5), "Test cleanup did not release maintenance work"
        finally:
            with condition:
                active -= 1
                condition.notify_all()

    monkeypatch.setattr(scheduler, "_sweep_org", blocked_sweep)
    try:
        scheduler._gc_tick(["a", "b"], max_workers=2)
        with condition:
            assert condition.wait_for(lambda: active == 2, timeout=2)
        scheduler._gc_tick(next_orgs, max_workers=2)
        assert not third_started.wait(0.2), (
            "A later tick replaced capacity still occupied by timed-out work"
        )
        assert peak == 2
    finally:
        release.set()
        scheduler._on_stopped()
        with condition:
            assert condition.wait_for(lambda: active == 0, timeout=2)
