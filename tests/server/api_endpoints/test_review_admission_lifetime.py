"""A successful review response owns capacity until physical work ends."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from reflexio.models.api_schema.service_schemas import ReviewUserPlaybooksRequest
from reflexio.server import operation_limiter
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
