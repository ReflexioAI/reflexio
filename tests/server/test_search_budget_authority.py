"""One explicit search budget, followed by remaining transport commit time."""

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reflexio.server import middleware as http_middleware
from reflexio.server import search_runtime as runtime


@pytest.mark.parametrize(
    ("configured", "expected"), [(None, 5), ("", 5), ("8", 8), ("0", 19), ("40", 19)]
)
@pytest.mark.parametrize("path", sorted(runtime.SEARCH_HTTP_PATHS))
def test_search_work_uses_the_explicit_budget_within_original_http_backstop(
    monkeypatch, configured, expected, path
):
    monkeypatch.delenv("REFLEXIO_SEARCH_DEADLINE_ENABLED", raising=False)
    if configured is None:
        monkeypatch.delenv("REFLEXIO_SEARCH_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("REFLEXIO_SEARCH_TIMEOUT_SECONDS", configured)
    monkeypatch.setattr(http_middleware, "backstop_for", lambda *_args: 19.0)
    app = FastAPI()

    @app.post(path)
    async def search():
        # Exercise the real worker context handoff, not only constructor parsing.
        return {
            "remaining": await runtime.run_in_threadpool(lambda: runtime.remaining(30))
        }

    with TestClient(runtime.SearchRuntimeMiddleware(app)) as client:
        response = client.post(path, json={})
    assert response.status_code == 200, response.text
    assert expected - 1 < response.json()["remaining"] <= expected


@pytest.mark.parametrize("configured", ["-1", "nan", "inf", "-inf", "not-a-number"])
def test_invalid_budget_refuses_startup_without_echoing_values(monkeypatch, configured):
    monkeypatch.setenv("REFLEXIO_SEARCH_TIMEOUT_SECONDS", configured)
    with pytest.raises(ValueError, match="REFLEXIO_SEARCH_TIMEOUT_SECONDS"):
        runtime.SearchRuntimeMiddleware(FastAPI())


@pytest.mark.parametrize("old_value", ["", "true", "false", "operator-secret"])
def test_obsolete_boolean_requires_explicit_budget_adoption(monkeypatch, old_value):
    monkeypatch.setenv("REFLEXIO_SEARCH_DEADLINE_ENABLED", old_value)
    monkeypatch.setenv("REFLEXIO_SEARCH_TIMEOUT_SECONDS", "8")
    with pytest.raises(ValueError, match="REFLEXIO_SEARCH_TIMEOUT_SECONDS") as error:
        runtime.SearchRuntimeMiddleware(FastAPI())
    if old_value == "operator-secret":
        assert old_value not in str(error.value)


@pytest.mark.parametrize("timeout", [-1.0, math.inf, math.nan])
def test_internal_timeout_override_cannot_create_an_unbounded_or_expired_scope(timeout):
    with pytest.raises(ValueError):
        runtime.SearchRuntimeMiddleware(FastAPI(), timeout=timeout)


def test_accepted_finalization_uses_remaining_http_budget_without_restarting_clock(
    monkeypatch,
):
    monkeypatch.setattr(http_middleware, "backstop_for", lambda *_args: 17.0)
    app = FastAPI()
    deadlines = []

    @app.post("/api/search")
    async def search():
        state = runtime.current()
        assert state is not None
        ingress = state.started + 17

        def record():
            finalization = runtime.current()
            assert finalization is not None
            deadlines.append((ingress, finalization.deadline, runtime.remaining(30)))

        runtime.on_response_accepted("durable-exposure", record)
        return {"ok": True}

    with TestClient(runtime.SearchRuntimeMiddleware(app, timeout=2)) as client:
        response = client.post("/api/search", json={})
    assert response.status_code == 200, response.text
    assert len(deadlines) == 1
    ingress, actual, remaining = deadlines[0]
    assert actual == ingress
    assert 16 < remaining <= 17
