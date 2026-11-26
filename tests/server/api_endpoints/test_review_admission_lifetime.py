"""A successful review response owns capacity until physical work ends."""

import asyncio
import gc
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from fastapi.middleware.asyncexitstack import AsyncExitStackMiddleware
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from reflexio.models.api_schema.service_schemas import ReviewUserPlaybooksRequest
from reflexio.server import operation_limiter, usage_metrics
from reflexio.server.route_inventory import api_route_contexts
from reflexio.server.routes import playbooks

_PAYLOAD = {
    "start_time": "2026-07-01T00:00:00Z",
    "end_time": "2026-07-02T00:00:00Z",
    "top_k": 25,
    "report_only": False,
}


@pytest.fixture(autouse=True)
def isolated_admission(monkeypatch):
    monkeypatch.setenv("REFLEXIO_AGGREGATION_CONCURRENCY_LIMIT", "1")
    monkeypatch.setenv("REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS", "0")
    operation_limiter.reset_operation_limiters_for_tests()
    yield
    operation_limiter.reset_operation_limiters_for_tests()


def test_concurrent_http_review_cannot_acknowledge_an_unavailable_slot(
    test_app, mock_reflexio, patched_reflexio
):
    started = threading.Event()
    finish = threading.Event()

    def review(*_args, **_kwargs):
        started.set()
        assert finish.wait(5), "test did not release the running review"

    mock_reflexio.review_user_playbooks.side_effect = review
    first_client = TestClient(test_app, raise_server_exceptions=False)
    second_client = TestClient(test_app, raise_server_exceptions=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(
            first_client.post, "/api/review_user_playbooks", json=_PAYLOAD
        )
        try:
            assert started.wait(5), "first review did not start"
            second = second_client.post("/api/review_user_playbooks", json=_PAYLOAD)
            assert second.status_code == 503
            mock_reflexio.review_user_playbooks.assert_called_once()
        finally:
            finish.set()
        assert first.result(timeout=5).json()["success"] is True


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
def test_failed_response_delivery_releases_an_unstarted_review(error_type):
    reflexio = MagicMock()
    response = playbooks._AdmittedReviewResponse(
        reflexio, "delivery", ReviewUserPlaybooksRequest.model_validate(_PAYLOAD)
    )

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        raise error_type()

    async def run():
        with pytest.raises(error_type):
            await response({"type": "http"}, receive, send)

    asyncio.run(run())
    reflexio.review_user_playbooks.assert_not_called()
    with operation_limiter.try_aggregation_capacity("delivery") as admitted:
        assert admitted


def test_response_construction_failure_releases_capacity(monkeypatch):
    def fail_run_id():
        raise RuntimeError("response preparation failed")

    monkeypatch.setattr(playbooks, "new_review_run_id", fail_run_id)
    with pytest.raises(RuntimeError, match="response preparation failed"):
        playbooks._AdmittedReviewResponse(
            MagicMock(),
            "construction",
            ReviewUserPlaybooksRequest.model_validate(_PAYLOAD),
        )
    with operation_limiter.try_aggregation_capacity("construction") as admitted:
        assert admitted


def test_cancelled_response_retains_capacity_until_physical_review_finishes(
    monkeypatch,
):
    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        released = asyncio.Event()
        finish = threading.Event()
        release_count = 0
        original_release = operation_limiter._release_operation

        def release(state):
            nonlocal release_count
            original_release(state)
            release_count += 1
            loop.call_soon_threadsafe(released.set)

        monkeypatch.setattr(operation_limiter, "_release_operation", release)

        def review(*_args, **_kwargs):
            loop.call_soon_threadsafe(started.set)
            assert finish.wait(5), "test did not release the physical review"

        reflexio = MagicMock()
        reflexio.review_user_playbooks.side_effect = review
        response = playbooks._AdmittedReviewResponse(
            reflexio, "cancelled", ReviewUserPlaybooksRequest.model_validate(_PAYLOAD)
        )

        async def receive():
            return {"type": "http.disconnect"}

        async def send(_message):
            pass

        task = asyncio.create_task(response({"type": "http"}, receive, send))
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert release_count == 0
            with operation_limiter.try_aggregation_capacity("cancelled") as admitted:
                assert not admitted
        finally:
            finish.set()
            await asyncio.wait_for(released.wait(), 5)
            if not task.done():
                await task
        assert release_count == 1
        reflexio.review_user_playbooks.assert_called_once()
        with operation_limiter.try_aggregation_capacity("cancelled") as admitted:
            assert admitted

    asyncio.run(run())


def test_cancelled_endpoint_cannot_orphan_a_constructed_reservation(
    test_app, mock_reflexio, patched_reflexio, monkeypatch
):
    async def run():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        cancelled = threading.Event()
        constructed = threading.Event()
        released = threading.Event()
        original_init = playbooks._AdmittedReviewResponse.__init__
        original_release = operation_limiter._release_operation
        task = None

        def release(state):
            original_release(state)
            released.set()

        monkeypatch.setattr(operation_limiter, "_release_operation", release)

        def cancel_request():
            assert task is not None
            task.cancel()
            cancelled.set()

        def construct(response, *args, **kwargs):
            original_init(response, *args, **kwargs)
            loop.call_soon_threadsafe(cancel_request)
            if threading.get_ident() != loop_thread:
                assert cancelled.wait(5), "request cancellation did not reach the loop"
            constructed.set()

        monkeypatch.setattr(playbooks._AdmittedReviewResponse, "__init__", construct)
        # Orphan cleanup must not depend on when cyclic GC happens to run.
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            route = next(
                route
                for route in api_route_contexts(test_app)
                if route.path == "/api/review_user_playbooks"
            )

            async def receive():
                return {
                    "type": "http.request",
                    "body": json.dumps(_PAYLOAD).encode(),
                    "more_body": False,
                }

            async def send(_message):
                await asyncio.sleep(0)

            # Exercise the actual FastAPI route task whose endpoint handoff
            # can be cancelled, without an outer middleware masking it.
            scope = {
                "type": "http",
                "app": test_app,
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/review_user_playbooks",
                "raw_path": b"/api/review_user_playbooks",
                "root_path": "",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 12345),
                "server": ("testserver", 80),
            }
            route_app = AsyncExitStackMiddleware(route.app)
            task = asyncio.create_task(route_app(scope, receive, send))
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert await asyncio.to_thread(constructed.wait, 5)
            assert await asyncio.to_thread(released.wait, 5), (
                "cancelled endpoint did not release its constructed reservation"
            )
            mock_reflexio.review_user_playbooks.assert_not_called()
            with operation_limiter.try_aggregation_capacity("test-org") as admitted:
                assert admitted, "cancelled endpoint orphaned its reservation"
        finally:
            gc.collect()
            if gc_enabled:
                gc.enable()

    asyncio.run(run())


def test_blocking_admission_recorder_cannot_stall_the_request_loop(
    test_app, mock_reflexio, patched_reflexio, monkeypatch
):
    async def run():
        loop = asyncio.get_running_loop()
        entered = threading.Event()
        finish = threading.Event()
        loop_tick = threading.Event()
        responsive_before_release = []

        def record(event):
            if event.event_category == "limiter":
                entered.set()
                assert finish.wait(5), "test did not release the blocking recorder"
            return usage_metrics.UsageEventDeliveryStatus.EXEMPT

        monkeypatch.setattr(usage_metrics, "_recorder", record)

        def observe_loop():
            try:
                if entered.wait(5):
                    loop.call_soon_threadsafe(loop_tick.set)
                    responsive_before_release.append(loop_tick.wait(1))
            finally:
                finish.set()

        observer = threading.Thread(target=observe_loop, daemon=True)
        observer.start()
        try:
            async with AsyncClient(
                transport=ASGITransport(app=test_app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/review_user_playbooks", json=_PAYLOAD
                )
            assert response.status_code == 200
            mock_reflexio.review_user_playbooks.assert_called_once()
            assert responsive_before_release == [True]
        finally:
            finish.set()
            await asyncio.to_thread(observer.join, 5)
            assert not observer.is_alive()

    asyncio.run(run())


def test_cancelled_awaiter_reclaims_an_already_constructed_response(monkeypatch):
    async def run():
        published = asyncio.Event()
        original_run = playbooks.run_in_threadpool

        async def hold_result(func, *args, **kwargs):
            response = await original_run(func, *args, **kwargs)
            published.set()
            await asyncio.Event().wait()
            return response

        monkeypatch.setattr(playbooks, "run_in_threadpool", hold_result)
        reflexio = MagicMock()
        gc_enabled = gc.isenabled()
        gc.disable()
        task = asyncio.create_task(
            playbooks._AdmittedReviewResponse.create(
                reflexio,
                "published",
                ReviewUserPlaybooksRequest.model_validate(_PAYLOAD),
            )
        )
        try:
            await asyncio.wait_for(published.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            reflexio.review_user_playbooks.assert_not_called()
            with operation_limiter.try_aggregation_capacity("published") as admitted:
                assert admitted
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            gc.collect()
            if gc_enabled:
                gc.enable()

    asyncio.run(run())
