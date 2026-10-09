"""Exercise the real standalone routes and middleware, without lifespan I/O."""

import asyncio
from typing import cast

import httpx
import pytest
from starlette.types import ASGIApp
from uvicorn._types import ASGI3Application

from reflexio.server import api


def request(
    monkeypatch: pytest.MonkeyPatch,
    *,
    peer: str = "127.0.0.1",
    host: str = "localhost",
    key: str = "",
    headers: dict[str, str] | None = None,
    path: str = "/api/whoami",
    method: str = "GET",
) -> httpx.Response:
    monkeypatch.setenv("REFLEXIO_API_KEY", key)
    monkeypatch.setattr(api.app, "middleware_stack", None)

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app, client=(peer, 5000)),
            base_url=f"http://{host}",
        ) as client:
            return await client.request(method, path, headers=headers)

    return asyncio.run(run())


@pytest.mark.parametrize("peer,host", [("127.0.0.1", "localhost"), ("::1", "[::1]")])
def test_loopback_can_use_standalone_routes_without_a_key(monkeypatch, peer, host):
    response = request(monkeypatch, peer=peer, host=host)
    assert response.status_code == 200
    assert response.json()["org_id"] == "self-host-org"


@pytest.mark.parametrize("headers", [{}, {"X-Forwarded-For": "127.0.0.1"}])
def test_remote_caller_cannot_claim_local_access(monkeypatch, headers):
    response = request(monkeypatch, peer="192.0.2.1", headers=headers)
    assert response.status_code == 401


def test_outer_proxy_cannot_grant_loopback_access(monkeypatch):
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    monkeypatch.setenv("REFLEXIO_API_KEY", "")
    monkeypatch.setattr(api.app, "middleware_stack", None)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(
                # Uvicorn uses typed scope dictionaries; Starlette/httpx use
                # mappings. Both implement the same ASGI wire protocol.
                app=cast(
                    ASGIApp,
                    ProxyHeadersMiddleware(
                        cast(ASGI3Application, api.app), trusted_hosts="*"
                    ),
                ),
                client=("192.0.2.1", 5000),
            ),
            base_url="http://localhost",
        ) as client:
            return await client.get(
                "/api/whoami", headers={"X-Forwarded-For": "127.0.0.1"}
            )

    assert asyncio.run(run()).status_code == 401


@pytest.mark.parametrize("reload", [False, True])
@pytest.mark.parametrize(
    "app_module,expected",
    [("reflexio.server.api:app", False), ("enterprise:app", True)],
)
def test_entrypoint_proxy_policy_does_not_trust_headers_for_standalone(
    monkeypatch, reload, app_module, expected
):
    from unittest.mock import patch

    from reflexio.server.__main__ import main

    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    args = ["--port", "8061", "--app", app_module]
    if reload:
        args.append("--reload")
    with patch("reflexio.server.__main__.uvicorn.run") as run:
        main(args)
    assert run.call_args.kwargs["proxy_headers"] is expected


@pytest.mark.parametrize(
    "path", ["/api/add_user_playbook", "/api/publish_interaction", "/"]
)
def test_remote_mutations_are_denied_before_request_validation(monkeypatch, path):
    assert (
        request(monkeypatch, peer="192.0.2.1", method="POST", path=path).status_code
        == 401
    )


@pytest.mark.parametrize(
    "origin", ["https://evil.example", "null", "http://localhost.evil.example"]
)
def test_browser_origins_cannot_mutate_local_server(monkeypatch, origin):
    assert (
        request(
            monkeypatch,
            method="POST",
            path="/api/add_user_playbook",
            headers={"Origin": origin},
        ).status_code
        == 401
    )


def test_loopback_browser_origin_is_allowed(monkeypatch):
    assert (
        request(monkeypatch, headers={"Origin": "http://localhost:8062"}).status_code
        == 200
    )


def test_dns_rebinding_hostname_is_denied(monkeypatch):
    assert request(monkeypatch, host="evil.example").status_code == 401


@pytest.mark.parametrize("peer", ["127.0.0.1", "192.0.2.1"])
@pytest.mark.parametrize(
    "authorization", [None, "Bearer wrong", "Basic configured-secret"]
)
def test_configured_key_is_required_for_every_data_client(
    monkeypatch, peer, authorization
):
    headers = {"Authorization": authorization} if authorization else {}
    assert (
        request(
            monkeypatch, peer=peer, key="configured-secret", headers=headers
        ).status_code
        == 401
    )


def test_matching_bearer_key_allows_remote_data_access(monkeypatch):
    response = request(
        monkeypatch,
        peer="192.0.2.1",
        host="server.example",
        key="configured-secret",
        headers={"Authorization": "Bearer configured-secret"},
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/health",
        "/healthz",
        "/healthz/eval",
        "/meta/version",
        "/docs",
        "/redoc",
        "/openapi.json",
    ],
)
@pytest.mark.parametrize("key", ["", "configured-secret"])
def test_health_and_documentation_remain_public(monkeypatch, path, key):
    assert request(monkeypatch, peer="192.0.2.1", key=key, path=path).status_code == 200


def test_preflight_does_not_require_a_bearer_key(monkeypatch):
    response = request(
        monkeypatch,
        peer="192.0.2.1",
        key="configured-secret",
        path="/api/whoami",
        method="OPTIONS",
        headers={
            "Origin": "https://client.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200


def test_custom_auth_app_does_not_install_standalone_policy(monkeypatch):
    from reflexio.server.local_access import LocalAccessMiddleware

    app = api.create_app(get_org_id=lambda: "authenticated-org", require_auth=True)
    assert not any(item.cls is LocalAccessMiddleware for item in app.user_middleware)


def test_entrypoint_default_listener_is_loopback():
    from reflexio.server.__main__ import _build_parser

    assert _build_parser().parse_args(["--port", "8061"]).host == "127.0.0.1"


@pytest.mark.parametrize("key", ["", "configured-secret"])
def test_standalone_swagger_advertises_the_bearer_authentication_option(
    monkeypatch, key
):
    monkeypatch.setenv("REFLEXIO_API_KEY", key)
    monkeypatch.setattr(api.app, "openapi_schema", None)
    schema = api.app.openapi()
    assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
    expected = [{"BearerAuth": []}] if key else [{"BearerAuth": []}, {}]
    assert schema["paths"]["/api/whoami"]["get"]["security"] == expected
    for path in ["/health", "/healthz", "/healthz/eval", "/meta/version"]:
        assert schema["paths"][path]["get"]["security"] == []
