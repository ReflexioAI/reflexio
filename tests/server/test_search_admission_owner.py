"""HTTP search owns admission until its actual work has drained."""

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from reflexio.server import operation_limiter as limiter
from reflexio.server import search_runtime as runtime
from reflexio.server.routes._common import _run_limited_api

from .test_search_runtime import call


def test_ingress_owner_covers_every_registered_search_route():
    from reflexio.server.routes.search import router

    assert {
        route.path for route in router.routes if isinstance(route, APIRoute)
    } == runtime.SEARCH_HTTP_PATHS


@pytest.mark.parametrize("value", ["", "0", "0.25", "invalid"])
def test_removed_queue_clock_is_rejected_at_http_startup(monkeypatch, value):
    monkeypatch.setenv("REFLEXIO_SEARCH_CONCURRENCY_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="REFLEXIO_SEARCH_CONCURRENCY_TIMEOUT_SECONDS"):
        runtime.SearchRuntimeMiddleware(FastAPI())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/api/search",
        "/api/search_profiles",
        "/api/rerank_user_profiles",
        "/api/search_interactions",
        "/api/search_user_playbooks",
        "/api/search_agent_playbooks",
    ],
)
async def test_every_search_route_joins_original_ingress_deadline(path):
    app = FastAPI()
    observed = []

    @app.post(path)
    def search():
        scope = runtime.current()
        assert scope is not None
        observed.append(scope)
        assert scope.deadline == pytest.approx(scope.started + 3)
        return {"success": True}

    messages = await call(runtime.SearchRuntimeMiddleware(app, timeout=3), path=path)
    assert messages[0]["status"] == 200
    assert len(observed) == 1


def test_cancelled_admission_wakes_before_capacity_is_released(monkeypatch):
    monkeypatch.setenv("REFLEXIO_SEARCH_CONCURRENCY_LIMIT", "1")
    org = "cancel-admission-owner"
    state = limiter._state_for(org, "search")
    assert state.semaphore.acquire(blocking=False)
    waiting = threading.Event()
    errors = []
    called = []
    scope = runtime.SearchScope(deadline=time.monotonic() + 20)
    # Observe a real blocked acquisition without timing the worker's dispatch.
    acquire = state.semaphore.acquire

    def observed_acquire(*args, **kwargs):
        waiting.set()
        return acquire(*args, **kwargs)

    monkeypatch.setattr(state.semaphore, "acquire", observed_acquire)

    def run():
        token = runtime._scope.set(scope)
        try:
            _run_limited_api(org, "search", lambda: called.append(True))
        except BaseException as exc:
            errors.append(exc)
        finally:
            runtime._scope.reset(token)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert waiting.wait(5)
        with scope.lock:
            scope.cancel("disconnected")
        worker.join(0.5)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], runtime.SearchDeadlineError)
        assert called == []
        assert state.waiting == 0 and state.active == 0
    finally:
        state.semaphore.release()
        worker.join(5)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_http_permit_outlives_returned_route_with_owned_descendant(
    monkeypatch, cancel_cleanup
):
    monkeypatch.setenv("REFLEXIO_SEARCH_CONCURRENCY_LIMIT", "1")
    app = FastAPI()
    started, release = threading.Event(), threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    org = "descendant-admission-owner"

    def descendant():
        started.set()
        release.wait()

    @app.post("/api/search")
    def search():
        def run():
            runtime.track(executor.submit(descendant))
            assert started.wait(5)
            return {"success": True}

        return _run_limited_api(org, "search", run)

    middleware = runtime.SearchRuntimeMiddleware(app, timeout=5)
    try:
        messages = await call(middleware)
        assert messages[0]["status"] == 200
        if cancel_cleanup:
            for task in middleware.tasks:
                task.cancel()
            await asyncio.sleep(0)
        state = limiter._state_for(org, "search")
        assert state.active == 1
        assert not state.semaphore.acquire(blocking=False)
    finally:
        release.set()
        await asyncio.gather(*middleware.tasks, return_exceptions=True)
        executor.shutdown(wait=True)
    assert state.active == 0
    assert state.semaphore.acquire(blocking=False)
    state.semaphore.release()


@pytest.mark.asyncio
async def test_cleanup_cancellation_between_snapshots_retains_new_descendant(
    monkeypatch,
):
    scope = runtime.SearchScope()
    completed = Future()
    completed.set_result(None)
    scope.futures[completed] = False
    descendant = Future()
    assert descendant.set_running_or_notify_cancel()
    injected = asyncio.Event()
    sleep = asyncio.sleep
    task = None

    async def cancel_at_quiescence(delay):
        if asyncio.current_task() is task and not injected.is_set():
            token = runtime._scope.set(scope)
            try:
                # A completed worker may have registered more owned work.
                scope.futures.pop(completed, None)
                runtime.track(descendant, cancel_queued=False)
            finally:
                runtime._scope.reset(token)
            injected.set()
            assert task is not None
            task.cancel()
        await sleep(delay)

    monkeypatch.setattr(runtime.asyncio, "sleep", cancel_at_quiescence)
    task = asyncio.create_task(runtime._drain_workers(scope, lambda: None))
    try:
        await asyncio.wait_for(injected.wait(), 5)
        await sleep(0)
        assert not task.done()
        assert scope.cancelled
    finally:
        descendant.set_result(None)
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
