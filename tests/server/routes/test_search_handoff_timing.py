"""Completed request landmarks expose implicit dispatch without bypassing it."""

import asyncio
import json
import logging
import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest
from anyio.to_thread import current_default_thread_limiter
from fastapi import Request
from starlette.concurrency import run_in_threadpool

from reflexio.models.api_schema.retriever_schema import UnifiedSearchResponse
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server import search_runtime as runtime
from reflexio.server import tracing
from reflexio.server.api import create_app
from reflexio.server.routes import search
from reflexio.server.routes._metering import _stamp_search_dependencies_done

pytestmark = pytest.mark.unit


def test_completed_landmarks_preserve_union_counts_and_skip_deadline_checks():
    state = runtime.SearchScope(started=10, deadline=9)
    token = runtime._scope.set(state)
    try:
        runtime.record_phase_interval("handoff", 11, 15)
        runtime.record_phase_interval("handoff", 12, 16)
    finally:
        runtime._scope.reset(token)
    snapshot = state.snapshot(20)
    assert snapshot["phases_ms"] == {"handoff": 5000}
    assert snapshot["phase_work_ms"] == {"handoff": 8000}
    assert snapshot["phase_counts"] == {"handoff": 2}
    assert snapshot["unattributed_ms"] == 5000
    assert state.active == {}


def test_completed_landmark_without_request_scope_is_a_noop():
    token = runtime._scope.set(None)
    try:
        runtime.record_phase_interval("handoff", 11, 15)
        assert runtime.current() is None
    finally:
        runtime._scope.reset(token)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["uncontended", "contended", "missing_timestamp"])
async def test_real_search_route_records_handoff_without_tracer(
    monkeypatch, caplog, mode
):
    monkeypatch.setattr(tracing, "_tracer", None)
    caplog.set_level(logging.INFO, logger=runtime.timing_logger.name)
    effects = []
    reflexio = MagicMock()
    reflexio.request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )

    def retrieve(*_args, **_kwargs):
        effects.append("retrieved")
        return UnifiedSearchResponse(success=True, msg="OK")

    reflexio.unified_search.side_effect = retrieve
    monkeypatch.setattr(search.reflexio_cache, "get_reflexio", lambda **_kw: reflexio)
    monkeypatch.setattr(search, "enqueue_search_metering", lambda **_kw: None)
    monkeypatch.setattr(search, "observe_completed_search", lambda _value: None)
    app = create_app(
        get_org_id=lambda: "synthetic-org",
        get_caller_type=lambda: "dashboard",
    )
    limiter = current_default_thread_limiter()
    previous = limiter.total_tokens
    limiter.total_tokens = 1
    stamped, entered, release = (
        asyncio.Event(),
        threading.Event(),
        threading.Event(),
    )
    stamps = []

    async def last_dependency(request: Request):
        if mode != "missing_timestamp":
            await runtime.run_in_threadpool(_stamp_search_dependencies_done, request)
            stamps.append(request.state.search_deps_done_monotonic)
        stamped.set()
        if mode == "contended":
            await until(entered.is_set)

    app.dependency_overrides[_stamp_search_dependencies_done] = last_dependency

    def occupy():
        entered.set()
        assert release.wait(3)

    request = blocker = None
    minimum_gap_ms = 0
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(
                client.post("/api/search", json={"query": "private-query"})
            )
            await asyncio.wait_for(stamped.wait(), timeout=3)
            if mode == "contended":
                # Independent work uses the same owner, from outside the request
                # context. The real sync route must wait for this permit.
                blocker = asyncio.create_task(run_in_threadpool(occupy))
                await until(lambda: limiter.statistics().tasks_waiting == 1)
                assert limiter.borrowed_tokens == 1
                assert effects == []
                await asyncio.sleep(0.02)
                minimum_gap_ms = (time.monotonic() - stamps[0]) * 1000
                release.set()
            response = await asyncio.wait_for(request, timeout=3)
            assert response.status_code == 200, response.text
            assert response.json()["success"] is True
            assert effects == ["retrieved"]
        records = [
            json.loads(record.getMessage().split("event=search_timing ", 1)[1])
            for record in caplog.records
            if record.getMessage().startswith("event=search_timing ")
        ]
        assert len(records) == 1
        phases = records[0]["phases_ms"]
        if mode == "missing_timestamp":
            assert "search.deps_to_body" not in phases
        else:
            assert 0 <= phases["search.deps_to_body"] <= records[0]["total_ms"]
            assert phases["search.deps_to_body"] >= minimum_gap_ms - 1
        assert "private-query" not in json.dumps(records)
        assert "synthetic-org" not in json.dumps(records)
    finally:
        release.set()
        if blocker is not None:
            await blocker
        if request is not None:
            await request
        limiter.total_tokens = previous
