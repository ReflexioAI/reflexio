"""Shared fixtures for API endpoint tests."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.models.config_schema import AgentSuccessConfig
from reflexio.server.api import create_app
from reflexio.server.cache.reflexio_cache import (
    get_reflexio,
    invalidate_reflexio_cache,
)


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
    """A MagicMock Reflexio instance for patching get_reflexio.

    ``get_org_config`` mirrors whatever a test wired onto ``get_config``.
    Config *write* paths read through ``get_org_config`` so they never persist
    a narrower scope's overlay back onto the org document, while read paths use
    ``get_config``; in OSS the two return the same object. Without this mirror
    a route calling ``get_org_config`` gets a bare ``MagicMock`` instead of a
    ``Config`` and 500s, so every test wiring only ``get_config`` would have to
    remember to set both.

    ``return_value`` is read directly rather than calling ``get_config()`` so
    the mirror does not register a spurious call on it.
    """
    mock = MagicMock()
    configurator = mock.request_context.configurator
    configurator.get_org_config.side_effect = lambda: (
        configurator.get_config.return_value
    )
    return mock


@pytest.fixture
def patched_reflexio(mock_reflexio):
    """Patch get_reflexio to return mock_reflexio for all tests using this fixture.

    After the A2 route-module split the domain routers resolve ``get_reflexio``
    through the ``reflexio_cache`` module (``reflexio_cache.get_reflexio(...)``),
    so patching the source binding intercepts every route module + the metering
    helpers with a single target.
    """
    with patch(
        "reflexio.server.cache.reflexio_cache.get_reflexio",
        return_value=mock_reflexio,
    ) as mock_get:
        yield mock_get


@pytest.fixture
def client_with_org():
    """A TestClient bound to a fresh unique org_id with SQLite storage.

    The org is generated per-test so the in-process per-org Reflexio cache
    doesn't bleed configuration between tests. We also stub
    ``storage.get_session_ids_in_window`` to return an empty list — the
    regenerate endpoint queries it synchronously to size the job, and
    we don't want the test outcome to depend on whatever happens to live
    in the developer's local SQLite DB. The cache entry is evicted on
    teardown.

    Returns:
        tuple[TestClient, str]: ``(client, org_id)`` pair.
    """
    org_id = f"test-regen-{uuid.uuid4().hex[:12]}"
    app = create_app(get_org_id=lambda: org_id)
    client = TestClient(app, raise_server_exceptions=False)
    # Warm the cache so subsequent get_reflexio() calls inside the
    # endpoint hand back the same instance the test can inspect.
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    assert storage is not None  # SQLite default — should never be None here.
    with patch.object(storage, "get_session_ids_in_window", return_value=[]):
        try:
            yield client, org_id
        finally:
            invalidate_reflexio_cache(org_id=org_id)


@pytest.fixture
def client_with_org_and_evaluator(client_with_org):
    """Same as ``client_with_org`` but with one configured AgentSuccessConfig.

    Adds the ``overall_success`` evaluator so the /regenerate
    POST passes the "known evaluation_name" gate.

    Returns:
        tuple[TestClient, str]: ``(client, org_id)`` pair.
    """
    client, org_id = client_with_org
    reflexio = get_reflexio(org_id=org_id)
    reflexio.request_context.configurator.set_config_by_name(
        "agent_success_config",
        AgentSuccessConfig(
            evaluation_name="overall_success",
            success_definition_prompt=(
                "Evaluate whether the agent successfully completed the task."
            ),
        ),
    )
    return client, org_id
