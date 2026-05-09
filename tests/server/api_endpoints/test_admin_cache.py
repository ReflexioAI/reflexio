"""Tests for ``POST /api/admin/cache/invalidate``.

Verifies that the explicit cache-eviction endpoint:
- Routes to the resolved org_id when no body is provided.
- Accepts an org_id body field that matches the caller's auth.
- Rejects mismatched org_ids with 403 (cross-org invalidate is OOS).
- Returns ``invalidated=False`` as a successful no-op when nothing
  was cached.
"""

from unittest.mock import MagicMock, patch

from reflexio.server.cache.reflexio_cache import (
    clear_reflexio_cache,
    get_reflexio,
)


class TestAdminInvalidateCacheEndpoint:
    """Behavioural tests for the admin cache invalidation endpoint."""

    def test_evicts_existing_entry_returns_invalidated_true(self, client):
        """Populating then invalidating drops the entry and returns invalidated=True."""
        clear_reflexio_cache()
        # Seed the cache with a real entry under the test-org id.
        with patch("reflexio.server.cache.reflexio_cache.Reflexio") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_reflexio("test-org")

            response = client.post("/api/admin/cache/invalidate", json={})

        assert response.status_code == 200
        data = response.json()
        assert data == {"invalidated": True, "org_id": "test-org"}

    def test_no_op_when_entry_missing_returns_false(self, client):
        """Invalidating a fresh cache returns invalidated=False, NOT an error."""
        clear_reflexio_cache()
        response = client.post("/api/admin/cache/invalidate", json={})
        assert response.status_code == 200
        data = response.json()
        assert data == {"invalidated": False, "org_id": "test-org"}

    def test_org_id_matching_caller_is_accepted(self, client):
        """A request body org_id that matches the caller's auth org passes through."""
        clear_reflexio_cache()
        response = client.post(
            "/api/admin/cache/invalidate",
            json={"org_id": "test-org"},
        )
        assert response.status_code == 200
        assert response.json()["org_id"] == "test-org"

    def test_cross_org_org_id_returns_403(self, client):
        """Body org_id mismatching the caller's auth is rejected as 403, not silently coerced."""
        response = client.post(
            "/api/admin/cache/invalidate",
            json={"org_id": "someone-else"},
        )
        assert response.status_code == 403
        assert "cross-org" in response.json()["detail"].lower()

    def test_omitted_body_org_id_uses_caller_org(self, client):
        """An empty body still works — the server resolves org_id from auth."""
        clear_reflexio_cache()
        response = client.post("/api/admin/cache/invalidate", json={})
        assert response.status_code == 200
        assert response.json()["org_id"] == "test-org"
