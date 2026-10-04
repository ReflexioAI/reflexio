"""Configuration reaches real decorated routes without relaxing their limits."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from reflexio.server import rate_limit


@pytest.fixture
def isolated_limiter(monkeypatch):
    monkeypatch.setattr(rate_limit, "_configured_key_func", None, raising=False)
    limiter = Limiter(key_func=rate_limit.get_rate_limit_key, storage_uri="memory://")
    monkeypatch.setattr(rate_limit, "limiter", limiter)
    return limiter


def app_with_early_route(limiter, budget):
    app = FastAPI()
    app.state.limiter = limiter

    def rate_exceeded(request: Request, exc: Exception):
        assert isinstance(exc, RateLimitExceeded)
        return _rate_limit_exceeded_handler(request, exc)

    app.add_exception_handler(RateLimitExceeded, rate_exceeded)

    @app.get("/early")
    @limiter.limit(budget)
    async def early(request: Request):
        return {"ok": True}

    return app


def validated_fixture_key(request: Request) -> str:
    token = request.headers.get("authorization", "")
    if token in {"Bearer fixture-a", "Bearer fixture-b"}:
        return token
    return get_remote_address(request)


def test_early_routes_keep_valid_token_buckets_and_invalid_ip_budget(isolated_limiter):
    app = app_with_early_route(isolated_limiter, "120/minute")
    rate_limit.configure_rate_limiter(validated_fixture_key)
    with TestClient(app) as client:
        for token in ("fixture-a", "fixture-b"):
            headers = {"Authorization": "Bearer " + token}
            for _ in range(120):
                assert client.get("/early", headers=headers).status_code == 200
            assert client.get("/early", headers=headers).status_code == 429
        for index in range(120):
            assert (
                client.get(
                    "/early", headers={"Authorization": f"Bearer junk-{index}"}
                ).status_code
                == 200
            )
        assert (
            client.get(
                "/early", headers={"Authorization": "Bearer junk-next"}
            ).status_code
            == 429
        )


def test_reconfigure_and_reset_apply_to_early_and_late_routes(isolated_limiter):
    app = app_with_early_route(isolated_limiter, "1/minute")
    rate_limit.configure_rate_limiter(validated_fixture_key)

    @app.get("/late")
    @isolated_limiter.limit("1/minute")
    async def late(request: Request):
        return {"ok": True}

    with TestClient(app) as client:
        headers = {"Authorization": "Bearer fixture-a"}
        for path in ("/early", "/late"):
            assert client.get(path, headers=headers).status_code == 200
            assert client.get(path, headers=headers).status_code == 429
        rate_limit.configure_rate_limiter(
            lambda request: "reconfigured:" + validated_fixture_key(request)
        )
        for path in ("/early", "/late"):
            assert client.get(path, headers=headers).status_code == 200
            assert client.get(path, headers=headers).status_code == 429
        rate_limit.configure_rate_limiter(rate_limit.get_rate_limit_key)
        for path in ("/early", "/late"):
            assert client.get(path, headers=headers).status_code == 200
            assert (
                client.get(
                    path, headers={"Authorization": "Bearer fixture-b"}
                ).status_code
                == 429
            )
