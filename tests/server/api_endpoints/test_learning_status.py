"""Tests for the learning_status API surface (Task 7).

Covers:
- Deferred publish response carries learning_status="deferred".
- Sync publish (wait_for_response=True) leaves learning_status absent.
- GET /api/learning_status returns a valid status for a known request.
- GET /api/learning_status returns 404 for an unknown request (never
  reports "done" for a request that never existed).
- ReflexioClient.get_learning_status deserialises the status string.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from reflexio.client import ReflexioClient
from reflexio.models.api_schema.domain.entities import Request
from reflexio.models.api_schema.service_schemas import PublishUserInteractionResponse
from reflexio.server.api import create_app
from reflexio.server.cache.reflexio_cache import get_reflexio, invalidate_reflexio_cache

_VALID_STATUS_VALUES = {"pending", "processing", "done", "failed"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _publish_payload():
    return {
        "user_id": "user-test",
        "session_id": "sess-test",
        "interaction_data_list": [
            {
                "role": "User",
                "content": "Hello",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_app():
    """FastAPI test app with a fixed org_id (no auth)."""
    return create_app(get_org_id=lambda: "test-org")


@pytest.fixture
def client(test_app):
    """TestClient wrapping test_app."""
    from fastapi.testclient import TestClient

    return TestClient(test_app, raise_server_exceptions=False)


@pytest.fixture
def patched_reflexio():
    """Patch reflexio_cache.get_reflexio with a MagicMock instance."""
    mock = MagicMock()
    with patch(
        "reflexio.server.cache.reflexio_cache.get_reflexio",
        return_value=mock,
    ):
        yield mock


@pytest.fixture
def client_with_storage():
    """TestClient + real (cache-warmed) storage for endpoint integration tests.

    Uses the same reflexio_cache warming pattern as ``client_with_org`` in
    conftest.py: warm the cache first so every ``reflexio_cache.get_reflexio``
    call inside the endpoint returns the same instance the test can seed data
    into.
    """
    from fastapi.testclient import TestClient

    org_id = f"test-ls-{uuid.uuid4().hex[:10]}"
    app = create_app(get_org_id=lambda: org_id)
    http_client = TestClient(app, raise_server_exceptions=False)
    # Warm the per-org cache entry.
    reflexio = get_reflexio(org_id=org_id)
    storage = reflexio.request_context.storage
    try:
        yield http_client, storage
    finally:
        invalidate_reflexio_cache(org_id=org_id)


@pytest.fixture
def two_org_clients():
    """Two orgs, each routed to its own org-scoped storage.

    Yields ``(client_a, client_b, storage_b)``. Each client authenticates as
    a distinct org, and ``reflexio_cache.get_reflexio`` is patched to return a
    per-org mock so Org A resolves ONLY Org A's storage and Org B resolves
    ONLY Org B's storage. This mirrors the production isolation seam
    (schema-per-org): the endpoint reads whichever storage
    ``get_reflexio(org_id)`` returns, so a request that lives under Org B is
    invisible to Org A.

    The OSS SQLite backend shares one db file across orgs (isolation is by
    separate files / enterprise schemas, not an org column), so a real-storage
    fixture cannot express cross-org isolation here — the per-org mock does.
    """
    from fastapi.testclient import TestClient

    org_a = f"test-ls-a-{uuid.uuid4().hex[:10]}"
    org_b = f"test-ls-b-{uuid.uuid4().hex[:10]}"

    # Org A's storage knows about no requests; Org B's storage owns the seeded
    # request. Tests seed Org B via ``storage_b.get_request``.
    storage_a = MagicMock()
    storage_a.get_request.return_value = None
    storage_b = MagicMock()

    def _reflexio_for(org_id: str, storage_base_dir=None):
        mock = MagicMock()
        mock.request_context.storage = storage_a if org_id == org_a else storage_b
        return mock

    with patch(
        "reflexio.server.cache.reflexio_cache.get_reflexio",
        side_effect=_reflexio_for,
    ):
        client_a = TestClient(
            create_app(get_org_id=lambda: org_a), raise_server_exceptions=False
        )
        client_b = TestClient(
            create_app(get_org_id=lambda: org_b), raise_server_exceptions=False
        )
        yield client_a, client_b, storage_b


# ---------------------------------------------------------------------------
# Deferred publish sets learning_status="deferred"
# ---------------------------------------------------------------------------


class TestDeferredPublishField:
    def test_deferred_publish_returns_learning_status_deferred(
        self, client, patched_reflexio
    ):
        response = client.post("/api/publish_interaction", json=_publish_payload())
        assert response.status_code == 200
        data = response.json()
        assert data["learning_status"] == "deferred"
        # Deferred response must carry a request_id so the caller can poll
        # GET /api/learning_status — otherwise the poll contract is broken.
        assert data.get("request_id")

    def test_sync_publish_leaves_learning_status_none(self, client, patched_reflexio):
        """Sync path returns real extraction counts — learning_status excluded."""
        mock_response = PublishUserInteractionResponse(
            success=True,
            message="Interaction processed",
            profiles_added=0,
            playbooks_added=0,
        )

        def run_immediately(**kwargs):
            return kwargs["fn"]()

        with (
            patch(
                "reflexio.server.routes._common.run_with_operation_limit",
                side_effect=run_immediately,
            ),
            patch(
                "reflexio.server.api_endpoints.publisher_api.add_user_interaction",
                return_value=mock_response,
            ),
        ):
            response = client.post(
                "/api/publish_interaction",
                params={"wait_for_response": "true"},
                json=_publish_payload(),
            )

        assert response.status_code == 200
        data = response.json()
        # response_model_exclude_none=True — field must be absent, not null
        assert "learning_status" not in data


# ---------------------------------------------------------------------------
# GET /api/learning_status
# ---------------------------------------------------------------------------


class TestLearningStatusEndpoint:
    def test_unknown_request_returns_404(self, client_with_storage):
        http_client, _ = client_with_storage
        response = http_client.get(
            "/api/learning_status", params={"request_id": "no-such-request"}
        )
        assert response.status_code == 404
        # Must NOT silently claim "done" for a request that never existed —
        # a 404 body carries an error detail, never a status field.
        body = response.json()
        assert "status" not in body

    def test_known_request_returns_valid_status(self, client_with_storage):
        http_client, storage = client_with_storage
        req = Request(
            request_id="req-ls-test-001",
            user_id="user-ls",
            session_id="sess-ls",
        )
        storage.add_request(req)

        response = http_client.get(
            "/api/learning_status", params={"request_id": req.request_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in _VALID_STATUS_VALUES

    def test_missing_request_id_param_returns_422(self, client_with_storage):
        http_client, _ = client_with_storage
        response = http_client.get("/api/learning_status")
        assert response.status_code == 422

    def test_cross_org_request_id_returns_404(self, two_org_clients):
        """Org A must NOT see Org B's request status by request_id.

        Org B owns a request; querying that request_id from Org A returns 404
        (the request is invisible cross-tenant), never Org B's real status.
        """
        client_a, client_b, storage_b = two_org_clients
        req = Request(
            request_id="req-cross-org-001",
            user_id="user-b",
            session_id="sess-b",
        )
        # Seed the request into Org B's storage only.
        storage_b.get_request.return_value = req
        storage_b.get_learning_status_for_request.return_value = "pending"

        # Sanity: Org B (the owner) can read its own status.
        owner_response = client_b.get(
            "/api/learning_status", params={"request_id": req.request_id}
        )
        assert owner_response.status_code == 200
        assert owner_response.json()["status"] in _VALID_STATUS_VALUES

        # Org A must get a 404 for Org B's request_id — no cross-tenant leak.
        cross_response = client_a.get(
            "/api/learning_status", params={"request_id": req.request_id}
        )
        assert cross_response.status_code == 404
        assert "status" not in cross_response.json()


# ---------------------------------------------------------------------------
# ReflexioClient.get_learning_status
# ---------------------------------------------------------------------------


class TestClientGetLearningStatus:
    def test_returns_status_string(self):
        """Client deserialises {"status": "pending"} and returns the string."""
        c = ReflexioClient.__new__(ReflexioClient)
        with patch.object(c, "_make_request", return_value={"status": "pending"}):
            result = c.get_learning_status("req-abc")
        assert result == "pending"

    def test_calls_correct_endpoint(self):
        """Client calls GET /api/learning_status with request_id param."""
        c = ReflexioClient.__new__(ReflexioClient)
        with patch.object(c, "_make_request", return_value={"status": "done"}) as m:
            c.get_learning_status("req-xyz")
        m.assert_called_once_with(
            "GET",
            "/api/learning_status",
            params={"request_id": "req-xyz"},
        )
