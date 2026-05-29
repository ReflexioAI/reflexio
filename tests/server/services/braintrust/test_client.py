"""Tests for the BraintrustClient HTTP wrapper.

Replaces the underlying `httpx.Client` so we can simulate Braintrust
responses without hitting the network.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from reflexio.server.services.braintrust.client import (
    BraintrustAuthError,
    BraintrustClient,
    BraintrustHTTPError,
)


def _stub_client(*, status: int, payload: Any) -> httpx.Client:
    """Build a fake httpx.Client that returns the same response for every GET."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_validate_key_returns_true_on_2xx(monkeypatch) -> None:
    c = BraintrustClient("sk-test")
    c._client = _stub_client(status=200, payload={"ok": True})
    assert c.validate_key() is True


def test_validate_key_returns_false_on_401() -> None:
    c = BraintrustClient("sk-bad")
    c._client = _stub_client(status=401, payload={"error": "Unauthorized"})
    assert c.validate_key() is False


def test_list_organizations_unwraps_objects_envelope() -> None:
    c = BraintrustClient("sk")
    c._client = _stub_client(
        status=200,
        payload={"objects": [{"id": "ws_1", "name": "My WS"}]},
    )
    orgs = c.list_organizations()
    assert orgs == [{"id": "ws_1", "name": "My WS"}]


def test_list_projects_unwraps_bare_list() -> None:
    """Some endpoints return a bare list; we accept either shape."""
    c = BraintrustClient("sk")
    c._client = _stub_client(
        status=200,
        payload=[{"id": "p_1", "name": "Prod"}],
    )
    assert c.list_projects("ws_1") == [{"id": "p_1", "name": "Prod"}]


def test_list_experiments_passes_since_param() -> None:
    """The `since` query param is included only when provided."""
    seen_params: list[dict[str, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params.multi_items()))
        return httpx.Response(200, json={"objects": []}, request=request)

    c = BraintrustClient("sk")
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    c.list_experiments("p_1", since_ts=1700000000)
    c.list_experiments("p_2", since_ts=None)

    assert ("since", "1700000000") in seen_params[0].items()
    assert "since" not in seen_params[1]


def test_http_error_raises_with_status_and_body() -> None:
    c = BraintrustClient("sk")
    c._client = _stub_client(status=500, payload={"error": "boom"})
    with pytest.raises(BraintrustHTTPError) as exc:
        c.list_organizations()
    assert exc.value.status_code == 500
    assert "boom" in exc.value.body


def test_transport_error_is_normalized_to_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("network down")

    c = BraintrustClient("sk")
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(BraintrustHTTPError) as exc:
        c.list_organizations()
    assert exc.value.status_code == 503
    assert "network down" in exc.value.body


def test_auth_error_is_raised_on_403_for_list_methods() -> None:
    c = BraintrustClient("sk-bad")
    c._client = _stub_client(status=403, payload={"error": "Forbidden"})
    with pytest.raises(BraintrustAuthError):
        c.list_organizations()


def test_authorization_header_is_set() -> None:
    """Sanity: the client sends `Authorization: Bearer ...`.

    Because we replace `_client` with a fresh one that doesn't inherit
    the constructor headers, we re-attach them on the replacement.
    """
    seen_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={"objects": []}, request=request)

    c = BraintrustClient("sk-customer")
    c._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {c.api_key}"},
    )
    c.list_organizations()
    assert seen_headers == ["Bearer sk-customer"]
