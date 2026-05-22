"""Verify /healthz/eval returns the EvalHealth snapshot + liveness color."""

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reflexio.server.api_endpoints import health_api
from reflexio.server.services.agent_success_evaluation import _eval_health
from reflexio.server.services.agent_success_evaluation._eval_health import SkipReason


def _client() -> TestClient:
    app = FastAPI()
    health_api.install(app)
    return TestClient(app)


def test_eval_health_returns_skip_counts_and_failures() -> None:
    _eval_health._HEALTH.__init__()
    _eval_health.record_skip(SkipReason.ALREADY_EVALUATED)
    _eval_health.record_producer_failure()
    _eval_health.record_tick(monotonic_ts=time.monotonic())

    response = _client().get("/healthz/eval")

    assert response.status_code == 200
    body = response.json()
    assert body["skip_counts"]["already_evaluated"] == 1
    assert body["producer_failures_24h"] == 1
    assert body["last_tick_monotonic"] is not None
    assert body["liveness"] == "green"


def test_eval_health_returns_red_when_scheduler_silent_for_30_min() -> None:
    _eval_health._HEALTH.__init__()
    _eval_health.record_tick(monotonic_ts=time.monotonic() - 31 * 60)

    response = _client().get("/healthz/eval")

    assert response.json()["liveness"] == "red"


def test_eval_health_returns_amber_when_scheduler_silent_for_5_min() -> None:
    _eval_health._HEALTH.__init__()
    _eval_health.record_tick(monotonic_ts=time.monotonic() - 6 * 60)

    response = _client().get("/healthz/eval")

    assert response.json()["liveness"] == "amber"
