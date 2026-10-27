"""Search waiters must not outlive their budget behind another constructor."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.server import search_runtime
from reflexio.server.cache import reflexio_cache as cache


class _ObservedLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.waiting = threading.Event()
        self.after_acquire = lambda: None

    def acquire(self, blocking=True, timeout=-1):
        if self.lock.acquire(blocking=False):
            acquired = True
        else:
            self.waiting.set()
            acquired = self.lock.acquire(blocking=blocking, timeout=timeout)
        if acquired:
            self.after_acquire()
        return acquired

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


@pytest.fixture
def construction(monkeypatch):
    cache.clear_reflexio_cache()
    started, release = threading.Event(), threading.Event()
    ready = Mock()
    ready.loaded_config_version.return_value = ("db", 1)
    ready.current_config_version.return_value = ("db", 1)
    lock = _ObservedLock()
    org_id = "construction-deadline"
    index = hash((org_id, None)) % cache._REFLEXIO_CONSTRUCTION_LOCK_STRIPES
    stripes: list[Any] = list(cache._reflexio_construction_locks)
    stripes[index] = lock
    monkeypatch.setattr(cache, "_reflexio_construction_locks", tuple(stripes))

    def build(**_kwargs):
        started.set()
        assert release.wait(timeout=5)
        return ready

    constructor = Mock(side_effect=build)
    monkeypatch.setattr(cache, "Reflexio", constructor)
    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(cache.get_reflexio, org_id)
        assert started.wait(timeout=2)

        def search(scope):
            token = search_runtime._scope.set(scope)
            try:
                return cache.get_reflexio(org_id)
            finally:
                search_runtime._scope.reset(token)

        try:
            yield SimpleNamespace(
                lock=lock,
                release=release,
                ready=ready,
                constructor=constructor,
                start=lambda scope: executor.submit(search, scope),
                org_id=org_id,
            )
        finally:
            release.set()
            assert owner.result(timeout=2) is ready
    cache.clear_reflexio_cache()


@pytest.mark.parametrize("cancel", [False, True], ids=["deadline", "disconnect"])
def test_search_stops_waiting_before_constructor_finishes(construction, cancel):
    scope = search_runtime.SearchScope()
    scope.deadline = time.monotonic() + (30 if cancel else 0.1)
    waiter = construction.start(scope)
    assert construction.lock.waiting.wait(timeout=1)
    if cancel:
        scope.cancel("disconnected")

    with pytest.raises(search_runtime.SearchDeadlineError):
        waiter.result(timeout=1)
    assert not construction.release.is_set()
    construction.constructor.assert_called_once()
    timing = scope.snapshot(time.monotonic())
    assert timing["phases_ms"]["reflexio.cache.construction_wait"] > 0
    assert "reflexio.cache.construct" not in timing["phases_ms"]


def test_waiter_reuses_published_instance_and_reports_wait(construction):
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 5)
    waiter = construction.start(scope)
    assert construction.lock.waiting.wait(timeout=1)
    construction.release.set()
    assert waiter.result(timeout=2) is construction.ready
    construction.constructor.assert_called_once()
    timing = scope.snapshot(time.monotonic())
    assert timing["phases_ms"]["reflexio.cache.construction_wait"] > 0
    assert "reflexio.cache.construct" not in timing["phases_ms"]


def test_cancellation_at_lock_handoff_preserves_ready_entry(construction):
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 5)
    waiter = construction.start(scope)
    assert construction.lock.waiting.wait(timeout=1)
    construction.lock.after_acquire = lambda: scope.cancel("disconnected")
    construction.release.set()
    with pytest.raises(search_runtime.SearchDeadlineError):
        waiter.result(timeout=2)
    assert cache.get_reflexio(construction.org_id) is construction.ready
    construction.constructor.assert_called_once()
