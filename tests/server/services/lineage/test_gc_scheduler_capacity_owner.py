"""A timeout cannot mint new maintenance capacity on the next tick."""

import threading
from typing import Never

import pytest

from reflexio.server.services.lineage import gc_scheduler

pytestmark = pytest.mark.unit


def test_width_one_stays_on_scheduler_thread(monkeypatch):
    def unused(_org: str) -> Never:
        raise AssertionError("Sweep is supplied directly")

    scheduler = gc_scheduler.LineageGCScheduler(
        request_context_factory=unused, bootstrap_org_id="a"
    )
    threads = []
    monkeypatch.setattr(
        scheduler, "_sweep_org", lambda _: threads.append(threading.get_ident())
    )
    try:
        scheduler._gc_tick(["a", "b"], max_workers=1)
        assert threads == [threading.get_ident()] * 2
        assert scheduler._fanout_owner is None
    finally:
        scheduler.stop()


def test_real_stop_start_reuses_only_retired_capacity(monkeypatch):
    def unused(_org: str) -> Never:
        raise AssertionError("Sweep is supplied directly")

    scheduler = gc_scheduler.LineageGCScheduler(
        request_context_factory=unused, bootstrap_org_id="a"
    )
    calls = []
    errors = []
    completed = [threading.Event(), threading.Event()]

    def loop():
        try:
            scheduler._gc_tick(["a"], max_workers=2)
        except Exception as exc:
            errors.append(type(exc).__name__)
        finally:
            completed[0 if not completed[0].is_set() else 1].set()

    monkeypatch.setattr(scheduler, "_run_loop", loop)
    monkeypatch.setattr(scheduler, "_sweep_org", calls.append)
    try:
        scheduler.start()
        assert completed[0].wait(2)
        scheduler.stop()
        scheduler.start()
        assert completed[1].wait(2)
        assert not errors, errors
        assert calls == ["a", "a"]
    finally:
        scheduler.stop()


def test_restart_waits_for_old_work_instead_of_replacing_its_capacity(monkeypatch):
    def unused(_org: str) -> Never:
        raise AssertionError("Sweep is supplied directly")

    scheduler = gc_scheduler.LineageGCScheduler(
        request_context_factory=unused, bootstrap_org_id="a"
    )
    release = threading.Event()
    finished_tick = threading.Event()
    owner_work = []
    monkeypatch.setattr(gc_scheduler, "_ORG_SWEEP_TIMEOUT_SECONDS", 0.01)

    def sweep(_org):
        assert release.wait(5)

    def loop():
        try:
            scheduler._gc_tick(["a"], max_workers=2)
            owner = scheduler._fanout_owner
            assert owner is not None
            if not owner_work:
                work = owner.submit("a", sweep)
                assert work is not None
                owner_work.append(work)
        finally:
            finished_tick.set()

    monkeypatch.setattr(scheduler, "_sweep_org", sweep)
    monkeypatch.setattr(scheduler, "_run_loop", loop)
    try:
        scheduler.start()
        assert finished_tick.wait(2)
        scheduler.stop()
        old_owner = scheduler._fanout_owner
        scheduler.start()
        assert not scheduler.is_running()
        assert scheduler._fanout_owner is old_owner
        release.set()
        assert owner_work
        owner_work[0].future.result(timeout=2)
        finished_tick.clear()
        scheduler.start()
        assert finished_tick.wait(2)
        assert scheduler._fanout_owner is not old_owner
    finally:
        release.set()
        scheduler.stop()


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
