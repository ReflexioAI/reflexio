"""Tests for core API routes.

Verifies that FastAPI endpoints return correct status codes, response
schemas, and handle errors properly.  Uses the ``patched_reflexio``
fixture from conftest to isolate tests from real storage/LLM calls.
"""

import tempfile
from inspect import iscoroutinefunction
from pathlib import Path
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.retriever_schema import (
    GetProfilesViewResponse,
    SearchInteractionResponse,
    SearchUserProfileResponse,
    SetConfigResponse,
    UpdateUserProfileResponse,
)
from reflexio.models.api_schema.service_schemas import (
    PublishUserInteractionResponse,
    Status,
    UserProfile,
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

    def test_health_check_is_async(self, client):
        """Keep container healthchecks off the shared sync worker threadpool."""
        route = next(route for route in client.app.routes if route.path == "/health")
        assert iscoroutinefunction(route.endpoint)


class TestPublishInteraction:
    """Tests for POST /api/publish_interaction."""

    @staticmethod
    def _publish_payload():
        # Every key inside interaction_data_list must be a real InteractionData
        # field. This fixture previously sent user_message/agent_message/
        # interaction_type -- none of which exist on the model -- so it posted
        # an interaction that bound to nothing and stored content=''. It still
        # asserted 200, which is exactly the silent failure this suite now
        # guards against.
        return {
            "user_id": "user-1",
            "session_id": "sess-1",
            "interaction_data_list": [
                {"role": "User", "content": "Hello"},
                {"role": "Agent", "content": "Hi there!"},
            ],
        }

    def test_sync_publish_returns_200(self, client, patched_reflexio):
        mock_response = PublishUserInteractionResponse(
            success=True, message="Interaction processed"
        )

        def run_immediately(**kwargs):
            return kwargs["fn"]()

        with (
            patch(
                "reflexio.server.routes._common.run_with_operation_limit",
                side_effect=run_immediately,
            ) as run_with_operation_limit,
            patch(
                "reflexio.server.api_endpoints.publisher_api.add_user_interaction",
                return_value=mock_response,
            ) as add_user_interaction,
        ):
            response = client.post(
                "/api/publish_interaction",
                params={"wait_for_response": "true"},
                json=self._publish_payload(),
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert run_with_operation_limit.call_args.kwargs["operation"] == "publish"
        add_user_interaction.assert_called_once()
        assert add_user_interaction.call_args.kwargs["use_publish_limiter"] is False

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

    def test_async_publish_rejects_contentless_interactions(
        self, client, patched_reflexio
    ):
        """A contentless publish must 422 on the async path too.

        This is the regression that matters most: the async path returns 200
        "queued" before the publisher runs, and discards the publisher's result,
        so a rejection there is invisible to the caller. Validating on the
        request model is what makes this path fail loudly. Previously this
        returned 200 and silently stored rows with content=''.
        """
        payload = self._publish_payload()
        payload["interaction_data_list"] = [{"role": "User"}]
        response = client.post("/api/publish_interaction", json=payload)
        assert response.status_code == 422
        assert "is empty" in response.text

    def test_async_publish_accepts_plugin_wire_shape(self, client, patched_reflexio):
        """Regression: a request-level key on an interaction must not break.

        Both first-party plugins build their wire payload with a denylist, so
        every turn carries ``user_id``. Rejecting unknown keys wedged them into
        a silent retry loop that never published again, so they stay ignored.
        """
        payload = self._publish_payload()
        payload["interaction_data_list"] = [
            {"role": "User", "content": "how do I do X?", "user_id": "proj-a"}
        ]
        response = client.post("/api/publish_interaction", json=payload)
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_background_publish_rejection_is_logged(
        self, client, patched_reflexio, caplog
    ):
        """A discarded background ``success=False`` must leave a trace.

        The async path hands the caller 200 "queued" and throws the publisher's
        return value away, so this log line is the only evidence a rejected
        publish ever produces. The reason string is deliberately withheld -- it
        can be an unbounded ``str(e)`` and Sentry ingests ERROR bodies unscrubbed.
        """
        import logging

        with (
            patch(
                "reflexio.server.api_endpoints.publisher_api.add_user_interaction",
                return_value=PublishUserInteractionResponse(
                    success=False, message="CUSTOMER-SECRET-XYZ"
                ),
            ),
            caplog.at_level(
                logging.ERROR, logger="reflexio.server.routes.interactions"
            ),
        ):
            response = client.post(
                "/api/publish_interaction", json=self._publish_payload()
            )
        assert response.status_code == 200
        assert "Background publish rejected" in caplog.text
        assert "CUSTOMER-SECRET-XYZ" not in caplog.text

    def test_async_publish_does_not_gate_durable_write_on_limiter(
        self, client, patched_reflexio
    ):
        with (
            patch(
                "reflexio.server.routes._common.run_with_operation_limit",
                side_effect=AssertionError("publish limiter should not wrap storage"),
            ),
            patch(
                "reflexio.server.api_endpoints.publisher_api.add_user_interaction",
                return_value=PublishUserInteractionResponse(
                    success=True, message="Interaction processed"
                ),
            ) as add_user_interaction,
        ):
            response = client.post(
                "/api/publish_interaction",
                json=self._publish_payload(),
            )

        assert response.status_code == 200
        add_user_interaction.assert_called_once()
        assert add_user_interaction.call_args.kwargs["defer_learning"] is True

    def test_publish_missing_body_returns_422(self, client):
        response = client.post("/api/publish_interaction")
        assert response.status_code == 422

    def test_async_publish_reports_unknown_field_in_warnings(
        self, client, patched_reflexio
    ):
        """A mis-keyed field is accepted but reported, never silently dropped.

        Asserts on the parsed ``warnings`` list rather than a substring of
        ``response.text``: a 422 echoes the whole request body in ``input``, so a
        substring check passes even with the feature removed entirely. That
        false-pass was demonstrated by mutation on an earlier revision.
        """
        payload = self._publish_payload()
        payload["interaction_data_list"] = [
            {"role": "User", "content": "hi", "Content": "typo"}
        ]
        response = client.post("/api/publish_interaction", json=payload)
        assert response.status_code == 200
        warnings = response.json()["warnings"]
        assert any("Content" in warning for warning in warnings), warnings
        # Names only — never the value.
        assert not any("typo" in warning for warning in warnings), warnings

    def test_async_publish_reports_skipped_empty_rows(self, client, patched_reflexio):
        payload = self._publish_payload()
        payload["interaction_data_list"] = [
            {"role": "User", "content": "REAL"},
            {"role": "Agent", "content": ""},
        ]
        response = client.post("/api/publish_interaction", json=payload)
        assert response.status_code == 200
        assert any(
            "skipped 1 empty" in warning for warning in response.json()["warnings"]
        ), response.json()["warnings"]

    def test_clean_publish_reports_no_warnings(self, client, patched_reflexio):
        response = client.post("/api/publish_interaction", json=self._publish_payload())
        assert response.status_code == 200
        assert response.json().get("warnings", []) == []


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


class TestGetAllProfilesRoute:
    """Tests for GET /api/get_all_profiles."""

    def test_exact_profile_lookup_includes_tombstones_when_requested(
        self, client, patched_reflexio, mock_reflexio
    ):
        profile = UserProfile(
            profile_id="p-superseded",
            user_id="project-1",
            content="old preference",
            last_modified_timestamp=123,
            generated_from_request_id="req-1",
            status=Status.SUPERSEDED,
        )
        mock_storage = MagicMock()
        mock_storage.get_profile_by_id.return_value = profile
        mock_reflexio.request_context.storage = mock_storage

        response = client.get(
            "/api/get_all_profiles",
            params={
                "profile_id": "p-superseded",
                "include_tombstones": "true",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_profiles"][0]["profile_id"] == "p-superseded"
        assert data["user_profiles"][0]["status"] == "superseded"
        mock_storage.get_profile_by_id.assert_called_once_with(
            "p-superseded", include_tombstones=True
        )

    def test_list_profiles_still_uses_existing_reflexio_method(
        self, client, patched_reflexio, mock_reflexio
    ):
        mock_reflexio.get_all_profiles.return_value = GetProfilesViewResponse(
            success=True,
            user_profiles=[],
            msg="Found 0 profile(s)",
        )

        response = client.get("/api/get_all_profiles", params={"limit": 10})

        assert response.status_code == 200
        mock_reflexio.get_all_profiles.assert_called_once()


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


class TestSetConfigRoute:
    """Tests for POST /api/set_config (full replacement semantics)."""

    def test_unknown_field_returns_422_before_set_config(
        self, client, patched_reflexio, mock_reflexio
    ):
        response = client.post(
            "/api/set_config",
            json={
                "storage_config": {
                    "db_path": str(Path(tempfile.gettempdir()) / "set-config.db")
                },
                "agent_sucess_config": None,
            },
        )

        assert response.status_code == 422, response.text
        mock_reflexio.set_config.assert_not_called()


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

        with patch(
            "reflexio.server.cache.reflexio_cache.invalidate_reflexio_cache"
        ) as mock_invalidate:
            response = client.post(
                "/api/update_config",
                json={"window_size": 25},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True

        # The reflexio.set_config call receives a merged Config with the
        # new field flipped AND the existing storage_config preserved.
        assert mock_reflexio.set_config.call_count == 1
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        assert merged.window_size == 25
        assert merged.storage_config == existing.storage_config

        # Cache invalidated on success.
        mock_invalidate.assert_called_once_with(org_id="test-org")

    def test_no_op_patch_skips_set_config_and_cache_invalidation(
        self, client, patched_reflexio, mock_reflexio
    ):
        existing = self._existing_config()
        self._wire_mock(mock_reflexio, existing)

        with patch(
            "reflexio.server.cache.reflexio_cache.invalidate_reflexio_cache"
        ) as mock_invalidate:
            response = client.post(
                "/api/update_config",
                json={"window_size": existing.window_size},
            )

        assert response.status_code == 200, response.text
        assert response.json() == {
            "success": True,
            "msg": "Configuration unchanged",
        }
        mock_reflexio.set_config.assert_not_called()
        mock_invalidate.assert_not_called()

    def test_unknown_field_returns_422_before_set_config(
        self, client, patched_reflexio, mock_reflexio
    ):
        """Unknown top-level keys never reach reflexio.set_config."""
        existing = self._existing_config()
        self._wire_mock(mock_reflexio, existing)

        response = client.post(
            "/api/update_config",
            json={"definitely_not_a_field": 42},
        )

        assert response.status_code == 422, response.text
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

        with patch(
            "reflexio.server.cache.reflexio_cache.invalidate_reflexio_cache"
        ) as mock_invalidate:
            response = client.post(
                "/api/update_config",
                json={"window_size": 25},
            )

        assert response.status_code == 200
        assert response.json()["success"] is False
        mock_invalidate.assert_not_called()

    # -----------------------------------------------------------------
    # R4: singular nested config patch semantics
    # -----------------------------------------------------------------
    @staticmethod
    def _existing_config_with_playbooks() -> Config:
        """Existing config with a populated playbook extractor config."""
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

    def test_nested_config_requires_full_payload_when_patched(
        self, client, patched_reflexio, mock_reflexio
    ):
        """PATCH'ing a nested config requires the full nested object."""
        existing = self._existing_config_with_playbooks()
        self._wire_mock(mock_reflexio, existing)

        response = client.post(
            "/api/update_config",
            json={
                "user_playbook_extractor_config": {
                    "aggregation_config": {"min_cluster_size": 99}
                }
            },
        )

        assert response.status_code in {400, 422}, response.text
        mock_reflexio.set_config.assert_not_called()

    def test_singular_extractor_configs_override_existing_config(
        self, client, patched_reflexio, mock_reflexio
    ):
        """Singular extractor config fields update existing config."""
        existing = self._existing_config()
        self._wire_mock(mock_reflexio, existing)

        with patch("reflexio.server.cache.reflexio_cache.invalidate_reflexio_cache"):
            response = client.post(
                "/api/update_config",
                json={
                    "profile_extractor_config": {
                        "extractor_name": "profile",
                        "extraction_definition_prompt": "profile facts",
                    },
                    "user_playbook_extractor_config": {
                        "extractor_name": "playbook",
                        "extraction_definition_prompt": "playbook rules",
                    },
                },
            )

        assert response.status_code == 200, response.text
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        assert merged.profile_extractor_config is not None
        assert merged.profile_extractor_config.extractor_name == "profile"
        assert merged.user_playbook_extractor_config is not None
        assert merged.user_playbook_extractor_config.extractor_name == "playbook"

    def test_null_extractor_configs_disable_existing_extractors(
        self, client, patched_reflexio, mock_reflexio
    ):
        """Null singular extractor config fields disable extraction."""
        existing = self._existing_config_with_playbooks()
        self._wire_mock(mock_reflexio, existing)

        with patch("reflexio.server.cache.reflexio_cache.invalidate_reflexio_cache"):
            response = client.post(
                "/api/update_config",
                json={
                    "profile_extractor_config": None,
                    "user_playbook_extractor_config": None,
                },
            )

        assert response.status_code == 200, response.text
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        assert merged.profile_extractor_config is None
        assert merged.user_playbook_extractor_config is None

    def test_nested_config_preserved_when_patching_unrelated_field(
        self, client, patched_reflexio, mock_reflexio
    ):
        """PATCH'ing a sibling field preserves the existing playbook config."""
        existing = self._existing_config_with_playbooks()
        self._wire_mock(mock_reflexio, existing)

        with patch("reflexio.server.cache.reflexio_cache.invalidate_reflexio_cache"):
            response = client.post(
                "/api/update_config",
                json={"window_size": 25},
            )

        assert response.status_code == 200, response.text
        merged = mock_reflexio.set_config.call_args.args[0]
        assert isinstance(merged, Config)
        # The partial-touched field changed
        assert merged.window_size == 25
        assert merged.user_playbook_extractor_config is not None
        agg = merged.user_playbook_extractor_config.aggregation_config
        assert agg is not None
        assert agg.min_cluster_size == 2
        assert agg.clustering_similarity == 0.45
