"""An organization without storage gets an explicit 503, not an unhandled 500.

Sentry PYTHON-FASTAPI-16E/16F: ``get_storage()`` raised a bare ``RuntimeError``
on ``/api/search`` (and enterprise data-source routes), which surfaced as an
unhandled 500 reported to Sentry at error level. The control,
``test_an_unexpected_runtime_error_is_still_a_500``, keeps a real server fault
on the 500 path that reaches Sentry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.lib._base import STORAGE_NOT_CONFIGURED_MSG, ReflexioBase
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.api import create_app


def _client() -> TestClient:
    return TestClient(
        create_app(
            get_org_id=lambda: "org-1",
            get_caller_type=lambda: "production_agent",
        ),
        raise_server_exceptions=False,
    )


def _reflexio_without_storage() -> MagicMock:
    reflexio = MagicMock()
    reflexio.request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )
    reflexio.request_context.storage = None
    # The real accessor, bound to a context whose storage is None.
    reflexio.get_storage.side_effect = lambda: ReflexioBase.get_storage(reflexio)
    reflexio._get_storage.side_effect = lambda: ReflexioBase._get_storage(reflexio)
    return reflexio


def test_search_without_storage_is_an_explicit_503() -> None:
    reflexio = _reflexio_without_storage()
    with patch(
        "reflexio.server.routes.search.reflexio_cache.get_reflexio",
        return_value=reflexio,
    ):
        response = _client().post(
            "/api/search", json={"query": "q", "user_id": "user-1"}
        )

    assert response.status_code == 503
    assert response.json() == {"detail": STORAGE_NOT_CONFIGURED_MSG}
    reflexio.unified_search.assert_not_called()


def test_storage_not_configured_is_not_reported_as_a_server_error() -> None:
    """Sentry's Starlette integration reports a handled exception only when it
    carries a 5xx ``status_code``; the typed error deliberately has none."""
    from reflexio.lib._base import StorageNotConfiguredError

    with pytest.raises(StorageNotConfiguredError) as exc_info:
        ReflexioBase._get_storage(_reflexio_without_storage())
    assert isinstance(exc_info.value, RuntimeError)
    assert not hasattr(exc_info.value, "status_code")


def test_an_unexpected_runtime_error_is_still_a_500() -> None:
    reflexio = MagicMock()
    reflexio.request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )
    reflexio.get_storage.side_effect = RuntimeError("database exploded")
    with patch(
        "reflexio.server.routes.search.reflexio_cache.get_reflexio",
        return_value=reflexio,
    ):
        response = _client().post(
            "/api/search", json={"query": "q", "user_id": "user-1"}
        )

    assert response.status_code == 500
