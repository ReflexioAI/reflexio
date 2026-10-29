"""HTTP search owns admission until its actual work has drained."""

import ast
import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI

from reflexio.server import operation_limiter as limiter
from reflexio.server import search_runtime as runtime
from reflexio.server.routes._common import _run_limited_api

from .test_search_runtime import call


def test_ingress_owner_covers_all_route_module_search_admission_callers():
    required = {}
    entries = {"_run_limited_api", "run_with_operation_limit", "operation_limit"}
    for source in (Path(runtime.__file__).parent / "routes").rglob("*.py"):
        tree = ast.parse(source.read_text())
        aliases = entries | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in entries
        }
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            searches = []
            for call_node in ast.walk(function):
                if not isinstance(call_node, ast.Call):
                    continue
                name = getattr(
                    call_node.func, "id", getattr(call_node.func, "attr", None)
                )
                if name not in aliases:
                    continue
                operation = (
                    call_node.args[1]
                    if len(call_node.args) > 1
                    else next(
                        (
                            kw.value
                            for kw in call_node.keywords
                            if kw.arg == "operation"
                        ),
                        None,
                    )
                )
                if isinstance(operation, ast.Constant) and operation.value == "search":
                    searches.append(call_node)
            if not searches:
                continue
            routes = [
                dec
                for dec in function.decorator_list
                if isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr in {"get", "post", "put", "patch", "delete"}
            ]
            assert routes, f"Unmapped search admission: {source.name}:{function.lineno}"
            for route in routes:
                assert isinstance(route.func, ast.Attribute)
                assert route.args and isinstance(route.args[0], ast.Constant)
                required[(route.func.attr.upper(), route.args[0].value)] = (
                    f"{source.name}:{function.lineno}"
                )
    assert required
    covered = runtime.SEARCH_HTTP_ROUTES
    missing = required.keys() - covered
    assert not missing, {route: required[route] for route in missing}


@pytest.mark.parametrize("success", [True, False])
def test_storage_stats_keeps_owned_admission_and_business_outcome(monkeypatch, success):
    from reflexio.models.api_schema.retriever_schema import StorageStatsResponse

    from .routes.test_search_exposure_boundary import _client, _search_results

    events = []
    monkeypatch.setattr(
        runtime, "record_health", lambda name, *_args, **kw: events.append((name, kw))
    )
    with _search_results([]) as reflexio:
        reflexio.storage_stats.return_value = StorageStatsResponse(
            success=success, profile_count=2, playbook_count=3
        )
        response = _client("dashboard").get(
            "/api/storage_stats", params={"user_id": "user-1"}
        )
    assert response.status_code == 200, response.text
    assert response.json()["profile_count"] == 2
    assert response.json()["success"] is success
    assert (
        "search.requests",
        {"outcome": "success" if success else "application_failure", "status": "200"},
    ) in events


@pytest.mark.parametrize("recording_scope", [False, True])
def test_search_admission_requires_deadline_or_explicit_wait(recording_scope):
    token = runtime._scope.set(runtime.SearchScope() if recording_scope else None)
    try:
        with (
            pytest.raises(ValueError, match="explicit.*wait"),
            limiter.operation_limit("missing-search-owner", "search"),
        ):
            pass
    finally:
        runtime._scope.reset(token)


@pytest.mark.parametrize("wait_forever", [False, True])
def test_internal_search_admission_keeps_explicit_wait_choice(wait_forever):
    token = runtime._scope.set(None)
    try:
        with limiter.operation_limit(
            "explicit-search-wait",
            "search",
            timeout_seconds=None if wait_forever else 0,
            wait_forever=wait_forever,
        ):
            assert limiter._state_for("explicit-search-wait", "search").active == 1
    finally:
        runtime._scope.reset(token)


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
