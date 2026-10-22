"""Handoff timing preserves the shared AnyIO limiter and cancellation contract."""

import asyncio
import threading
from contextlib import contextmanager
from contextvars import ContextVar

import anyio
import pytest
import pytest_asyncio
from anyio.to_thread import current_default_thread_limiter
from starlette.concurrency import run_in_threadpool

from reflexio.server import search_runtime as runtime

pytestmark = pytest.mark.unit


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


@contextmanager
def scoped(state):
    token = runtime._scope.set(state)
    try:
        yield
    finally:
        runtime._scope.reset(token)


@pytest_asyncio.fixture
async def limiter():
    limiter = current_default_thread_limiter()
    previous = limiter.total_tokens
    limiter.total_tokens = 1
    try:
        yield limiter
    finally:
        limiter.total_tokens = previous


@pytest.mark.asyncio
async def test_unscoped_call_delegates_once_with_unchanged_arguments(monkeypatch):
    import starlette.concurrency

    calls = []

    def work(value, *, suffix):
        return value + suffix

    async def offload(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(starlette.concurrency, "run_in_threadpool", offload)
    assert await runtime.run_in_threadpool(work, "a", suffix="b") == "ab"
    assert calls == [(work, ("a",), {"suffix": "b"})]


@pytest.mark.asyncio
async def test_occupied_shared_limiter_is_measured_without_bypass(limiter):
    entered, release = threading.Event(), threading.Event()
    invoked = []
    context = ContextVar("request_identity", default="absent")
    state = runtime.SearchScope()

    def occupy():
        entered.set()
        assert release.wait(3)

    def work(value, *, suffix):
        invoked.append(context.get())
        return value + suffix

    blocker = asyncio.create_task(run_in_threadpool(occupy))
    request = None
    try:
        await until(entered.is_set)
        token = context.set("request-one")
        try:
            with scoped(state):
                request = asyncio.create_task(
                    runtime.run_in_threadpool(work, "a", suffix="b")
                )
        finally:
            context.reset(token)
        await until(lambda: limiter.statistics().tasks_waiting == 1)
        assert not invoked and limiter.borrowed_tokens == 1
        release.set()
        assert await request == "ab"
        assert invoked == ["request-one"]
        snapshot = state.snapshot(runtime.time.monotonic())
        assert snapshot["phase_counts"] == {
            "search.threadpool.dispatch_wait": 1,
            "search.threadpool.execution": 1,
            "search.threadpool.return_wait": 1,
        }
        assert snapshot["phases_ms"]["search.threadpool.dispatch_wait"] > 0
        assert all(value >= 0 for value in snapshot["phases_ms"].values())
        assert state.active == {}
    finally:
        release.set()
        await blocker
        if request:
            await request


@pytest.mark.asyncio
async def test_queued_cancellation_records_wait_without_executing(limiter):
    entered, release = threading.Event(), threading.Event()
    holder, invoked = {}, []
    state = runtime.SearchScope()

    def occupy():
        entered.set()
        assert release.wait(3)

    async def request():
        with scoped(state), anyio.CancelScope() as cancellation:
            holder["cancel"] = cancellation
            await runtime.run_in_threadpool(lambda: invoked.append(True))

    blocker = asyncio.create_task(run_in_threadpool(occupy))
    task = None
    try:
        await until(entered.is_set)
        task = asyncio.create_task(request())
        await until(lambda: limiter.statistics().tasks_waiting == 1)
        holder["cancel"].cancel()
        await task
        assert not invoked and limiter.borrowed_tokens == 1
        snapshot = state.snapshot(runtime.time.monotonic())
        assert snapshot["phase_counts"] == {"search.threadpool.dispatch_wait": 1}
        assert snapshot["phases_ms"]["search.threadpool.dispatch_wait"] > 0
        assert state.active == {}
    finally:
        release.set()
        await blocker
        if task:
            await task


@pytest.mark.asyncio
async def test_running_cancellation_retains_permit_and_active_execution(limiter):
    entered, release = threading.Event(), threading.Event()
    holder = {}
    state = runtime.SearchScope()

    def work():
        entered.set()
        assert release.wait(3)
        return "finished"

    async def request():
        with scoped(state), anyio.CancelScope() as cancellation:
            holder["cancel"] = cancellation
            holder["result"] = await runtime.run_in_threadpool(work)

    task = asyncio.create_task(request())
    try:
        await until(entered.is_set)
        holder["cancel"].cancel()
        await asyncio.sleep(0)
        assert not task.done() and limiter.borrowed_tokens == 1
        snapshot = state.snapshot(runtime.time.monotonic())
        assert snapshot["active_phases"] == ["search.threadpool.execution"]
        assert snapshot["phases_ms"]["search.threadpool.execution"] >= 0
        release.set()
        await task
        assert holder["result"] == "finished"
        assert limiter.borrowed_tokens == 0 and state.active == {}
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_worker_exception_identity_and_context_are_preserved(limiter):
    state = runtime.SearchScope()
    error = ValueError("synthetic failure")

    def fail():
        assert runtime.current() is state
        raise error

    with scoped(state), pytest.raises(ValueError) as caught:
        await runtime.run_in_threadpool(fail)
    assert caught.value is error
    assert limiter.borrowed_tokens == 0 and state.active == {}
    assert state.phase_counts == {
        "search.threadpool.dispatch_wait": 1,
        "search.threadpool.execution": 1,
        "search.threadpool.return_wait": 1,
    }


@pytest.mark.asyncio
async def test_return_handoff_is_distinct_from_worker_execution(monkeypatch, limiter):
    import starlette.concurrency

    returned = asyncio.Event()
    resume = asyncio.Event()
    state = runtime.SearchScope()

    async def delayed_return(func, *args, **kwargs):
        result = await run_in_threadpool(func, *args, **kwargs)
        returned.set()
        await resume.wait()
        return result

    monkeypatch.setattr(starlette.concurrency, "run_in_threadpool", delayed_return)
    with scoped(state):
        task = asyncio.create_task(runtime.run_in_threadpool(lambda: "done"))
    try:
        await returned.wait()
        assert limiter.borrowed_tokens == 0
        assert state.phase_counts["search.threadpool.execution"] == 1
        assert "search.threadpool.return_wait" not in state.phase_counts
        resume.set()
        assert await task == "done"
        snapshot = state.snapshot(runtime.time.monotonic())
        assert snapshot["phases_ms"]["search.threadpool.return_wait"] > 0
        assert state.phase_counts["search.threadpool.execution"] == 1
    finally:
        resume.set()
        await task
