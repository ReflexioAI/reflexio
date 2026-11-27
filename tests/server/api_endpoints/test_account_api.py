"""Tests for the whoami + my_config endpoints.

Covers both the publisher-API shim (``account_api``) and the FastAPI
routes. The my_config gate is security-critical — it defaults closed
unless the host provides an authenticated organization resolver.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import Header, HTTPException
from fastapi.testclient import TestClient

from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.api import create_app
from reflexio.server.services.configurator.configurator import DefaultConfigurator

# These test account behavior independently of the standalone listener policy.
# Actual standalone access is exercised in test_local_access.py.
app = create_app()


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

    @pytest.mark.parametrize("legacy_value", ["true", "false", "unexpected"])
    def test_retired_env_cannot_enable_export(self, monkeypatch, legacy_value):
        monkeypatch.setenv("REFLEXIO_ALLOW_MY_CONFIG", legacy_value)
        with patch("reflexio.server.api_endpoints.account_api.get_reflexio") as load:
            response = TestClient(app).get("/api/my_config")
        assert response.status_code == 200
        assert response.json()["success"] is False
        assert "storage_config" not in response.json()
        load.assert_not_called()

    def test_authenticated_host_exports_for_the_resolved_org(self):
        tokens = {"Bearer synthetic-one": "org-one", "Bearer synthetic-two": "org-two"}

        def authenticated_org(authorization: str = Header(default="")):
            if authorization not in tokens:
                raise HTTPException(status_code=401)
            return tokens[authorization]

        def load_org(*, org_id):
            configurator = DefaultConfigurator(
                org_id,
                config=Config(
                    storage_config=StorageConfigSQLite(
                        db_path=f"/synthetic/{org_id}.sqlite3"
                    )
                ),
            )
            return SimpleNamespace(
                request_context=SimpleNamespace(configurator=configurator)
            )

        client = TestClient(create_app(get_org_id=authenticated_org, require_auth=True))
        with patch(
            "reflexio.server.api_endpoints.account_api.get_reflexio",
            side_effect=load_org,
        ) as load:
            assert client.get("/api/my_config").status_code == 401
            load.assert_not_called()
            for token, org_id in tokens.items():
                response = client.get(
                    "/api/my_config", headers={"Authorization": token}
                )
                assert response.status_code == 200
                assert (
                    response.json()["storage_config"]["db_path"]
                    == f"/synthetic/{org_id}.sqlite3"
                )

    def test_exception_returns_generic_message_no_leakage(self, monkeypatch):
        """A crash in ``get_reflexio`` must not leak exception text.

        We have to open the endpoint gate to actually reach the body
        of ``my_config`` — the closed-by-default path short-circuits
        before hitting ``get_reflexio`` at all.
        """
        monkeypatch.setattr(app.state, "my_config_enabled", True)
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
