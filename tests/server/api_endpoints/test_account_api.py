"""Tests for the whoami + my_config endpoints.

Covers both the publisher-API shim (``account_api``) and the FastAPI
routes. The my_config gate is security-critical — it defaults closed
in OS/self-host and only opens when auth is enforced or the opt-in
env var is set.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from reflexio.server.api import app


@pytest.fixture(autouse=True)
def reset_gate():
    """Reset the per-app auth gate and the env var between tests.

    ``app.state.my_config_enabled`` is scoped to the FastAPI instance
    but the tests share a module-level ``app``, so we still snap it
    back. The env var is snapped back for the same reason.
    """
    prev_enabled = bool(getattr(app.state, "my_config_enabled", False))
    prev_env = os.environ.get("REFLEXIO_ALLOW_MY_CONFIG")
    app.state.my_config_enabled = False
    os.environ.pop("REFLEXIO_ALLOW_MY_CONFIG", None)
    yield
    app.state.my_config_enabled = prev_enabled
    if prev_env is None:
        os.environ.pop("REFLEXIO_ALLOW_MY_CONFIG", None)
    else:
        os.environ["REFLEXIO_ALLOW_MY_CONFIG"] = prev_env


class TestWhoamiEndpoint:
    def test_returns_masked_storage_for_self_host(self):
        client = TestClient(app)
        response = client.get("/api/whoami")

        assert response.status_code == 200
        body = response.json()
        # Self-host always resolves to DEFAULT_ORG_ID
        assert body["success"] is True
        assert body["org_id"]
        # Default SQLite should be reported as "sqlite"
        assert body["storage_type"] == "sqlite"
        # Storage label is always populated and never contains raw creds
        assert body["storage_configured"] is True
        # The label never leaks unexpected sensitive substrings
        assert "password" not in (body.get("storage_label") or "")

    def test_exception_returns_generic_message_no_leakage(self):
        """A crash in ``get_reflexio`` must not leak exception text.

        Connection strings, file paths and SQL errors are the worst
        offenders — we assert the specific secret substring does NOT
        appear in the response body while ``success=False``.
        """
        with patch(
            "reflexio.server.api_endpoints.account_api.get_reflexio",
            side_effect=RuntimeError("boom-secret-conn-string"),
        ):
            client = TestClient(app)
            response = client.get("/api/whoami")

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert body["message"] == "Failed to load storage configuration"
        assert "boom-secret-conn-string" not in response.text


class TestMyConfigEndpoint:
    def test_disabled_by_default_in_self_host(self):
        """Without auth enforcement or env opt-in, my_config is closed."""
        client = TestClient(app)
        response = client.get("/api/my_config")

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert "disabled" in body["message"].lower()

    def test_opt_in_env_var_enables_endpoint(self):
        os.environ["REFLEXIO_ALLOW_MY_CONFIG"] = "true"
        client = TestClient(app)
        response = client.get("/api/my_config")

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["storage_type"] == "sqlite"

    def test_auth_enforced_mode_enables_endpoint(self):
        """When Bearer auth is enforced app-wide, my_config is reachable.

        Simulates what ``create_app`` does for the enterprise flow by
        setting ``app.state.my_config_enabled`` directly — this is the
        same flag the factory sets when ``get_org_id`` + ``require_auth``
        are provided together.
        """
        app.state.my_config_enabled = True
        client = TestClient(app)
        response = client.get("/api/my_config")

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True

    def test_exception_returns_generic_message_no_leakage(self):
        """A crash in ``get_reflexio`` must not leak exception text.

        We have to open the endpoint gate to actually reach the body
        of ``my_config`` — the closed-by-default path short-circuits
        before hitting ``get_reflexio`` at all.
        """
        app.state.my_config_enabled = True
        with patch(
            "reflexio.server.api_endpoints.account_api.get_reflexio",
            side_effect=RuntimeError("boom-secret-db-url"),
        ):
            client = TestClient(app)
            response = client.get("/api/my_config")

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert body["message"] == "Failed to load storage configuration"
        assert "boom-secret-db-url" not in response.text
