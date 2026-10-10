"""Slow-request diagnostics preserve responses and omit sensitive context."""

import asyncio
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reflexio.server import middleware
from reflexio.server import slow_request_diagnostics as diagnostics
from reflexio.server.middleware import TimeoutMiddleware


def test_stack_snapshot_omits_locals_and_source():
    secret = "sensitive-payload-must-never-appear"
    stacks = diagnostics._thread_stacks()
    assert "test_stack_snapshot_omits_locals_and_source" in stacks
    assert secret not in stacks
    assert len(stacks) <= 65536


@pytest.mark.parametrize("delay,threshold,expected_logs", [(0, 60, 0), (0.01, 0, 1)])
def test_real_http_response_and_watchdog_cleanup(
    monkeypatch, caplog, delay, threshold, expected_logs
):
    monkeypatch.setattr(diagnostics, "SLOW_SECONDS", threshold)
    monkeypatch.setattr(diagnostics, "_last_logged", None)
    tasks = []

    async def tracked_watchdog(path, correlation_id):
        tasks.append(asyncio.current_task())
        await diagnostics.watch_slow_request(path, correlation_id)

    monkeypatch.setattr(middleware, "watch_slow_request", tracked_watchdog)
    app = FastAPI()
    app.add_middleware(TimeoutMiddleware)

    @app.post("/api/search_profiles")
    async def search():
        await asyncio.sleep(delay)
        return {"success": True}

    with caplog.at_level(logging.WARNING), TestClient(app) as client:
        response = client.post("/api/search_profiles")
        assert response.status_code == 200
        assert response.json() == {"success": True}
        # The request owns the watchdog; no pending task survives its response.
        assert len(tasks) == 1
        assert tasks[0].done()
    assert (
        sum("event=slow_request_stacks" in r.message for r in caplog.records)
        == expected_logs
    )


def test_diagnostic_failure_does_not_change_http_response(monkeypatch):
    monkeypatch.setattr(diagnostics, "SLOW_SECONDS", 0)
    monkeypatch.setattr(diagnostics, "_last_logged", None)

    def denied_snapshot():
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(diagnostics, "_thread_stacks", denied_snapshot)
    app = FastAPI()
    app.add_middleware(TimeoutMiddleware)

    @app.delete("/api/account")
    async def delete():
        await asyncio.sleep(0.01)
        return {"deleted": True}

    with TestClient(app) as client:
        response = client.delete("/api/account")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}


def test_concurrent_diagnostics_are_rate_limited(monkeypatch, caplog):
    monkeypatch.setattr(diagnostics, "SLOW_SECONDS", 0)
    monkeypatch.setattr(diagnostics, "_last_logged", None)

    async def run():
        await asyncio.gather(
            diagnostics.watch_slow_request("/api/search_profiles", "first"),
            diagnostics.watch_slow_request("/api/account", "second"),
        )

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    assert sum("event=slow_request_stacks" in r.message for r in caplog.records) == 1
