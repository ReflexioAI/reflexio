"""Worker classification survives thread boundaries without copying tenant state."""

from threading import Event

import pytest

from reflexio.server.background_work import background_work, is_background_work
from reflexio.server.org_fanout import iterate_orgs_bounded
from reflexio.server.scheduling import ThreadedScheduler


def test_nested_scope_restores_after_failure():
    assert not is_background_work()
    with background_work():
        with pytest.raises(ValueError), background_work():
            raise ValueError("worker failed")
        assert is_background_work()
    assert not is_background_work()


def test_overridden_scheduler_loop_is_background():
    ran = Event()
    observed = []

    class Scheduler(ThreadedScheduler):
        def _run_loop(self):
            observed.append(is_background_work())
            ran.set()

    scheduler = Scheduler(thread_name="classification-test")
    scheduler.start()
    assert ran.wait(1)
    scheduler.stop()
    assert observed == [True]
    assert not is_background_work()


def test_org_fanout_workers_are_background():
    observed = []
    assert (
        iterate_orgs_bounded(
            ["one", "two"],
            lambda _: observed.append(is_background_work()),
            max_workers=2,
            per_org_timeout_seconds=1,
        )
        == []
    )
    assert observed == [True, True]
    assert not is_background_work()
