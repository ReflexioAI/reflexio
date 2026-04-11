"""Shared fixtures for API endpoint tests."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.server.api import create_app


@pytest.fixture
def test_app():
    """Create a FastAPI test app with a fixed org_id (no auth)."""
    return create_app(get_org_id=lambda: "test-org")


@pytest.fixture
def client(test_app):
    """TestClient wrapping the test app."""
    return TestClient(test_app, raise_server_exceptions=False)


@pytest.fixture
def mock_reflexio():
    """A MagicMock Reflexio instance for patching get_reflexio."""
    return MagicMock()


@pytest.fixture
def patched_reflexio(mock_reflexio):
    """Patch get_reflexio to return mock_reflexio for all tests using this fixture."""
    with (
        patch(
            "reflexio.server.cache.reflexio_cache.get_reflexio",
            return_value=mock_reflexio,
        ) as mock_get,
        patch(
            "reflexio.server.api.get_reflexio",
            return_value=mock_reflexio,
        ),
    ):
        yield mock_get
