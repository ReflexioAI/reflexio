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

    def test_search_profiles_returns_200(self, client, patched_reflexio, mock_reflexio):
        mock_reflexio.search_user_profiles.return_value = SearchUserProfileResponse(
            success=True,
            user_profiles=[],
            msg="OK",
        )

        response = client.post(
            "/api/search_profiles",
            json={"user_id": "user-1", "query": "test user"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_profiles"] == []

    def test_search_interactions_returns_200(
        self, client, patched_reflexio, mock_reflexio
    ):
        mock_reflexio.search_interactions.return_value = SearchInteractionResponse(
            success=True,
            interactions=[],
            msg="OK",
        )

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

    # -----------------------------------------------------------------
    # R4: nested-list shallow-merge semantics
    # -----------------------------------------------------------------
    @staticmethod
    def _existing_config_with_playbooks() -> Config:
        """Existing config with a populated playbook_configs list.

        Used to verify that PATCHing a *sibling* top-level field
        preserves the playbook_configs list, AND that PATCHing
        playbook_configs itself replaces the list wholesale (no deep
        merge into list-of-dicts).
        """
        from reflexio.models.config_schema import (
            PlaybookAggregatorConfig,
            UserPlaybookExtractorConfig,
        )

        db_path = str(Path(tempfile.gettempdir()) / "existing.db")
        return Config(
            storage_config=StorageConfigSQLite(db_path=db_path),
            user_playbook_extractor_config=UserPlaybookExtractorConfig(
                extractor_name="default_playbook_extractor",
                extraction_definition_prompt="extract feedback",
                aggregation_config=PlaybookAggregatorConfig(
                    min_cluster_size=2,
                    clustering_similarity=0.45,
                ),
            ),
        )

    def test_nested_list_replaced_wholesale_when_patched(
        self, client, patched_reflexio, mock_reflexio
    ):
        """PATCH'ing user_playbook_extractor_configs replaces the entire list.

        This is the documented behavior: a partial that re-sends the
        list with only one inner field set will FAIL Pydantic
        validation because required fields like ``extractor_name`` and
        ``extraction_definition_prompt`` are missing. Callers cannot
        target a nested-list field with PATCH; they must round-trip
        the full config via /api/get_config + /api/set_config.
        """
        existing = self._existing_config_with_playbooks()
        self._wire_mock(mock_reflexio, existing)

        # Try to flip just the inner aggregation_config field
        response = client.post(
            "/api/update_config",
            json={
                "user_playbook_extractor_configs": [
                    {"aggregation_config": {"min_cluster_size": 99}}
                ]
            },
        )

        # Pydantic rejects: extractor_name and extraction_definition_prompt
        # are required on UserPlaybookExtractorConfig.
        assert response.status_code in {400, 422}, response.text
        mock_reflexio.set_config.assert_not_called()

    def test_legacy_extractor_lists_override_existing_canonical_config(
        self, client, patched_reflexio, mock_reflexio
    ):
        """Legacy list fields in PATCH payloads update the singular config."""
        existing = self._existing_config()
        self._wire_mock(mock_reflexio, existing)

        with patch("reflexio.server.api.invalidate_reflexio_cache"):
            response = client.post(
                "/api/update_config",
                json={
                    "profile_extractor_configs": [
                        {
                            "extractor_name": "legacy_profile",
                            "extraction_definition_prompt": "profile facts",
                        }
                    ],
                    "user_playbook_extractor_configs": [
                        {
                            "extractor_name": "legacy_playbook",
                            "extraction_definition_prompt": "playbook rules",
                        }
                    ],
                },
            )

        assert response.status_code == 200, response.text
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        assert merged.profile_extractor_config is not None
        assert merged.profile_extractor_config.extractor_name == "legacy_profile"
        assert merged.user_playbook_extractor_config is not None
        assert merged.user_playbook_extractor_config.extractor_name == "legacy_playbook"

    def test_legacy_empty_extractor_lists_disable_existing_extractors(
        self, client, patched_reflexio, mock_reflexio
    ):
        """Legacy empty list fields in PATCH payloads disable extraction."""
        existing = self._existing_config_with_playbooks()
        self._wire_mock(mock_reflexio, existing)

        with patch("reflexio.server.api.invalidate_reflexio_cache"):
            response = client.post(
                "/api/update_config",
                json={
                    "profile_extractor_configs": [],
                    "user_playbook_extractor_configs": [],
                },
            )

        assert response.status_code == 200, response.text
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        assert merged.profile_extractor_config is None
        assert merged.user_playbook_extractor_config is None
        assert merged.profile_extractor_configs == []
        assert merged.user_playbook_extractor_configs == []

    def test_nested_list_preserved_when_patching_unrelated_field(
        self, client, patched_reflexio, mock_reflexio
    ):
        """PATCH'ing a sibling field preserves the existing playbook_configs.

        This is the safe pattern: top-level-shallow-merge means any
        key not in the partial keeps its current value, including
        nested-list fields like ``user_playbook_extractor_configs``.
        """
        existing = self._existing_config_with_playbooks()
        self._wire_mock(mock_reflexio, existing)

        with patch("reflexio.server.api.invalidate_reflexio_cache"):
            response = client.post(
                "/api/update_config",
                json={"extraction_backend": "agentic"},
            )

        assert response.status_code == 200, response.text
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        # The partial-touched field changed
        assert merged.extraction_backend == "agentic"
        # Untouched nested list is preserved
        assert merged.user_playbook_extractor_configs is not None
        assert len(merged.user_playbook_extractor_configs) == 1
        agg = merged.user_playbook_extractor_configs[0].aggregation_config
        assert agg is not None
        assert agg.min_cluster_size == 2
        assert agg.clustering_similarity == 0.45
