"""Tests for core API routes.

Verifies that FastAPI endpoints return correct status codes, response
schemas, and handle errors properly.  Uses the ``patched_reflexio``
fixture from conftest to isolate tests from real storage/LLM calls.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.retriever_schema import (
    SearchInteractionResponse,
    SearchUserProfileResponse,
    SetConfigResponse,
    UpdateUserProfileResponse,
)
from reflexio.models.api_schema.service_schemas import (
    PublishUserInteractionResponse,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite


class TestHealthEndpoints:
    """Tests for health and root endpoints — no mocking needed."""

    def test_root_returns_service_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "docs" in data

    def test_health_check_returns_healthy(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestPublishInteraction:
    """Tests for POST /api/publish_interaction."""

    @staticmethod
    def _publish_payload():
        return {
            "user_id": "user-1",
            "interaction_data_list": [
                {
                    "user_id": "user-1",
                    "session_id": "sess-1",
                    "interaction_type": "conversation",
                    "user_message": "Hello",
                    "agent_message": "Hi there!",
                }
            ],
        }

    def test_sync_publish_returns_200(self, client, patched_reflexio):
        mock_response = PublishUserInteractionResponse(
            success=True, message="Interaction processed"
        )

        with patch(
            "reflexio.server.api_endpoints.publisher_api.add_user_interaction",
            return_value=mock_response,
        ):
            response = client.post(
                "/api/publish_interaction",
                params={"wait_for_response": "true"},
                json=self._publish_payload(),
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_async_publish_returns_queued(self, client, patched_reflexio):
        """Async mode returns immediate acknowledgement without calling publisher."""
        response = client.post(
            "/api/publish_interaction",
            json=self._publish_payload(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "queued" in data["message"].lower()

    def test_publish_missing_body_returns_422(self, client):
        response = client.post("/api/publish_interaction")
        assert response.status_code == 422


class TestSearchEndpoints:
    """Tests for search endpoints."""

    def test_search_profiles_returns_200(self, client):
        mock_response = SearchUserProfileResponse(
            success=True,
            user_profiles=[],
            msg="OK",
        )

        with patch(
            "reflexio.server.api_endpoints.retriever_api.search_user_profiles",
            return_value=mock_response,
        ):
            response = client.post(
                "/api/search_profiles",
                json={"user_id": "user-1", "query": "test user"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_profiles"] == []

    def test_search_interactions_returns_200(self, client):
        mock_response = SearchInteractionResponse(
            success=True,
            interactions=[],
            msg="OK",
        )

        with patch(
            "reflexio.server.api_endpoints.retriever_api.search_interactions",
            return_value=mock_response,
        ):
            response = client.post(
                "/api/search_interactions",
                json={"user_id": "user-1", "query": "hello"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["interactions"] == []

    def test_search_profiles_missing_body_returns_422(self, client):
        response = client.post("/api/search_profiles")
        assert response.status_code == 422


class TestUpdateUserProfileRoute:
    """Tests for PUT /api/update_user_profile."""

    def test_dispatches_to_publisher_api(self, client):
        mock_response = UpdateUserProfileResponse(
            success=True, msg="User profile updated successfully"
        )
        with patch(
            "reflexio.server.api_endpoints.publisher_api.update_user_profile",
            return_value=mock_response,
        ) as mock_dispatch:
            response = client.put(
                "/api/update_user_profile",
                json={
                    "user_id": "user-1",
                    "profile_id": "p1",
                    "content": "updated content",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert mock_dispatch.call_count == 1
        kwargs = mock_dispatch.call_args.kwargs
        assert kwargs["org_id"] == "test-org"
        assert kwargs["request"].profile_id == "p1"
        assert kwargs["request"].content == "updated content"

    def test_missing_required_fields_returns_422(self, client):
        response = client.put(
            "/api/update_user_profile",
            json={"user_id": "user-1"},  # profile_id missing
        )
        assert response.status_code == 422


class TestUpdateConfigRoute:
    """Tests for POST /api/update_config (PATCH-style partial update).

    The endpoint fetches the existing config, shallow-merges the partial
    payload over it, and round-trips through ``Config(**merged)`` so
    Pydantic rejects unknown fields. Storage validation lives in
    ``reflexio.set_config``; we mock it out and assert the merged dict
    that arrives there.
    """

    @staticmethod
    def _existing_config() -> Config:
        # Platform-aware temp path — Ruff S108 flags hardcoded ``/tmp``.
        # The path isn't read or written; we just need a valid string
        # for the SQLite config so ``set_config`` round-trips through
        # the merged Config without failing validation.
        db_path = str(Path(tempfile.gettempdir()) / "existing.db")
        return Config(storage_config=StorageConfigSQLite(db_path=db_path))

    def _wire_mock(self, mock_reflexio: MagicMock, existing: Config) -> None:
        configurator = MagicMock()
        configurator.get_config.return_value = existing
        mock_reflexio.request_context.configurator = configurator
        mock_reflexio.set_config.return_value = SetConfigResponse(
            success=True, msg="Configuration set successfully"
        )

    def test_partial_dict_succeeds(self, client, patched_reflexio, mock_reflexio):
        existing = self._existing_config()
        self._wire_mock(mock_reflexio, existing)

        with patch("reflexio.server.api.invalidate_reflexio_cache") as mock_invalidate:
            response = client.post(
                "/api/update_config",
                json={"extraction_backend": "agentic"},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True

        # The reflexio.set_config call receives a merged Config with the
        # new field flipped AND the existing storage_config preserved.
        assert mock_reflexio.set_config.call_count == 1
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        assert merged.extraction_backend == "agentic"
        assert merged.storage_config == existing.storage_config

        # Cache invalidated on success.
        mock_invalidate.assert_called_once_with(org_id="test-org")

    def test_unknown_field_does_not_leak_to_set_config(
        self, client, patched_reflexio, mock_reflexio
    ):
        """Unknown top-level keys never reach reflexio.set_config.

        ``Config`` doesn't enable strict ``extra='forbid'`` validation,
        so Pydantic silently drops unknown fields rather than raising —
        but the merged ``Config`` instance handed to ``set_config`` must
        not carry the bogus attribute either way.
        """
        existing = self._existing_config()
        self._wire_mock(mock_reflexio, existing)

        response = client.post(
            "/api/update_config",
            json={"definitely_not_a_field": 42},
        )

        # If the model later switches to extra='forbid', this becomes
        # a 4xx (FastAPI rejects in request validation) and we'd still
        # want to assert set_config was not called. Pin to client-error
        # codes specifically so a 5xx regression here trips the test
        # instead of silently passing the >= 400 check.
        if response.status_code == 200:
            merged = mock_reflexio.set_config.call_args.args[0]
            assert isinstance(merged, Config)
            assert not hasattr(merged, "definitely_not_a_field")
        else:
            assert response.status_code in {400, 422}
            mock_reflexio.set_config.assert_not_called()

    def test_replaces_nested_object_wholesale(
        self, client, patched_reflexio, mock_reflexio
    ):
        existing = self._existing_config()
        self._wire_mock(mock_reflexio, existing)

        response = client.post(
            "/api/update_config",
            json={"storage_config": {"db_path": "/new/path.db"}},
        )

        assert response.status_code == 200, response.text
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        assert isinstance(merged.storage_config, StorageConfigSQLite)
        assert merged.storage_config.db_path == "/new/path.db"

    def test_does_not_invalidate_on_failure(
        self, client, patched_reflexio, mock_reflexio
    ):
        """When reflexio.set_config returns success=False, cache stays warm."""
        existing = self._existing_config()
        configurator = MagicMock()
        configurator.get_config.return_value = existing
        mock_reflexio.request_context.configurator = configurator
        mock_reflexio.set_config.return_value = SetConfigResponse(
            success=False, msg="storage validation failed"
        )

        with patch("reflexio.server.api.invalidate_reflexio_cache") as mock_invalidate:
            response = client.post(
                "/api/update_config",
                json={"extraction_backend": "agentic"},
            )

        assert response.status_code == 200
        assert response.json()["success"] is False
        mock_invalidate.assert_not_called()
