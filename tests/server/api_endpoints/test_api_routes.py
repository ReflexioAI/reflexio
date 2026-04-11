"""Tests for core API routes.

Verifies that FastAPI endpoints return correct status codes, response
schemas, and handle errors properly.  Uses the ``patched_reflexio``
fixture from conftest to isolate tests from real storage/LLM calls.
"""

from unittest.mock import patch

from reflexio.models.api_schema.retriever_schema import (
    SearchInteractionResponse,
    SearchUserProfileResponse,
)
from reflexio.models.api_schema.service_schemas import (
    PublishUserInteractionResponse,
)


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
