"""
Unit tests for ProfileGenerationService.

Covers missed lines: _build_should_run_prompt(), _process_results() deduplication
paths, _update_config_for_incremental(), run_manual_regular() edge cases,
_count_manual_generated(), error/exception handling paths, and status change helpers.
"""

import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    ManualProfileGenerationRequest,
    Request,
    Status,
    UserProfile,
)
from reflexio.models.config_schema import ProfileExtractorConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.base_generation_service import StatusChangeOperation
from reflexio.server.services.profile.profile_generation_service import (
    ProfileGenerationService,
    ProfileGenerationServiceConfig,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def llm_client():
    """Create a real LiteLLMClient (mocked at method level where needed)."""
    config = LiteLLMConfig(model="gpt-4o-mini")
    return LiteLLMClient(config)


@pytest.fixture
def request_context(temp_storage_dir):
    """Create a RequestContext with mock storage."""
    ctx = RequestContext(org_id="test_org", storage_base_dir=temp_storage_dir)
    ctx.storage = MagicMock()
    return ctx


@pytest.fixture
def service(llm_client, request_context):
    """Create a ProfileGenerationService with default flags."""
    return ProfileGenerationService(
        llm_client=llm_client,
        request_context=request_context,
    )


@pytest.fixture
def service_pending(llm_client, request_context):
    """Create a ProfileGenerationService with output_pending_status=True."""
    return ProfileGenerationService(
        llm_client=llm_client,
        request_context=request_context,
        output_pending_status=True,
    )


@pytest.fixture
def service_config():
    """Create a ProfileGenerationServiceConfig."""
    return ProfileGenerationServiceConfig(
        user_id="user_1",
        request_id="req_1",
        source="api",
    )


@pytest.fixture
def sample_profile():
    """Create a sample UserProfile."""
    return UserProfile(
        profile_id="p1",
        user_id="user_1",
        content="likes dark mode",
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="req_1",
    )


@pytest.fixture
def sample_interactions():
    """Create sample interaction objects."""
    ts = int(datetime.now(UTC).timestamp())
    return [
        Interaction(
            interaction_id=1,
            user_id="user_1",
            request_id="req_1",
            content="I prefer dark mode",
            role="user",
            created_at=ts,
        ),
        Interaction(
            interaction_id=2,
            user_id="user_1",
            request_id="req_1",
            content="Noted, dark mode enabled",
            role="assistant",
            created_at=ts + 1,
        ),
    ]


@pytest.fixture
def sample_request_interaction_models(sample_interactions):
    """Create sample RequestInteractionDataModel list."""
    request = Request(
        request_id="req_1",
        user_id="user_1",
        created_at=sample_interactions[0].created_at,
        source="api",
    )
    return [
        RequestInteractionDataModel(
            session_id="req_1",
            request=request,
            interactions=sample_interactions,
        )
    ]


# ===============================
# Test: _build_should_run_prompt
# ===============================


class TestBuildShouldRunPrompt:
    """Tests for _build_should_run_prompt with profile-specific config."""

    def test_returns_none_when_no_configs(
        self, service, sample_request_interaction_models
    ):
        """Return None when scoped_configs list is empty."""
        configs: list[ProfileExtractorConfig] = []

        with patch.object(
            service.configurator, "get_agent_context", return_value="Agent context"
        ):
            result = service._build_should_run_prompt(
                configs, sample_request_interaction_models
            )

        assert result is None

    def test_returns_prompt_with_definition_only(
        self, service, request_context, sample_request_interaction_models
    ):
        """Prompt is rendered when config has extraction_definition_prompt."""
        configs = [
            ProfileExtractorConfig(
                extractor_name="prefs",
                extraction_definition_prompt="Extract user preferences",
            ),
        ]

        with (
            patch.object(
                service.configurator, "get_agent_context", return_value="Agent context"
            ),
            patch.object(
                request_context.prompt_manager,
                "render_prompt",
                return_value="rendered prompt",
            ) as mock_render,
        ):
            result = service._build_should_run_prompt(
                configs, sample_request_interaction_models
            )

            assert result == "rendered prompt"
            variables = mock_render.call_args[0][1]
            assert (
                "definition: Extract user preferences"
                in variables["should_extract_profile_prompt"]
            )

    def test_returns_prompt_with_override(
        self, service, request_context, sample_request_interaction_models
    ):
        """Prompt includes override condition when config has should_extract_profile_prompt_override."""
        configs = [
            ProfileExtractorConfig(
                extractor_name="custom",
                extraction_definition_prompt="Basic profile",
                should_extract_profile_prompt_override="Check if user shared info",
            ),
        ]

        with (
            patch.object(
                service.configurator, "get_agent_context", return_value="Agent context"
            ),
            patch.object(
                request_context.prompt_manager,
                "render_prompt",
                return_value="rendered override",
            ) as mock_render,
        ):
            result = service._build_should_run_prompt(
                configs, sample_request_interaction_models
            )

            assert result == "rendered override"
            variables = mock_render.call_args[0][1]
            assert (
                "condition: Check if user shared info"
                in variables["should_extract_profile_prompt"]
            )

    def test_combines_multiple_configs(
        self, service, request_context, sample_request_interaction_models
    ):
        """Multiple configs are combined into numbered criteria list."""
        configs = [
            ProfileExtractorConfig(
                extractor_name="prefs",
                extraction_definition_prompt="Extract preferences",
                should_extract_profile_prompt_override="User shared preference",
            ),
            ProfileExtractorConfig(
                extractor_name="bio",
                extraction_definition_prompt="Extract biography",
            ),
        ]

        with (
            patch.object(
                service.configurator, "get_agent_context", return_value="Agent context"
            ),
            patch.object(
                request_context.prompt_manager,
                "render_prompt",
                return_value="combined prompt",
            ) as mock_render,
        ):
            result = service._build_should_run_prompt(
                configs, sample_request_interaction_models
            )

            assert result == "combined prompt"
            variables = mock_render.call_args[0][1]
            criteria = variables["should_extract_profile_prompt"]
            assert "1." in criteria
            assert "2." in criteria
            assert "Extract preferences" in criteria
            assert "Extract biography" in criteria

    def test_renders_with_multiple_definitions(
        self, service, request_context, sample_request_interaction_models
    ):
        """Multiple configs with definitions are all included in criteria."""
        configs = [
            ProfileExtractorConfig(
                extractor_name="hobbies",
                extraction_definition_prompt="Extract hobbies",
            ),
            ProfileExtractorConfig(
                extractor_name="food",
                extraction_definition_prompt="Extract food preferences",
            ),
        ]

        with (
            patch.object(
                service.configurator, "get_agent_context", return_value="Agent context"
            ),
            patch.object(
                request_context.prompt_manager,
                "render_prompt",
                return_value="partial prompt",
            ) as mock_render,
        ):
            result = service._build_should_run_prompt(
                configs, sample_request_interaction_models
            )

            assert result == "partial prompt"
            variables = mock_render.call_args[0][1]
            criteria = variables["should_extract_profile_prompt"]
            assert "Extract hobbies" in criteria
            assert "Extract food preferences" in criteria


# ===============================
# Test: _process_results
# ===============================


class TestProcessResults:
    """Tests for _process_results with deduplication paths."""

    def _setup_service_config(self, service, source="api"):
        """Helper to set service_config on the service."""
        service.service_config = ProfileGenerationServiceConfig(
            user_id="user_1",
            request_id="req_1",
            source=source,
        )

    def test_empty_results_no_action(self, service, request_context):
        """No profiles saved or deleted when results are empty."""
        self._setup_service_config(service)

        service._process_results([])

        request_context.storage.add_user_profile.assert_not_called()
        request_context.storage.delete_user_profile.assert_not_called()

    def test_empty_nested_results_no_action(self, service, request_context):
        """No profiles saved when nested result lists are empty."""
        self._setup_service_config(service)

        service._process_results([[], []])

        request_context.storage.add_user_profile.assert_not_called()

    def test_save_profiles_dedup_disabled(
        self, service, request_context, sample_profile
    ):
        """Profiles are saved directly when deduplicator is disabled."""
        self._setup_service_config(service)

        with patch(
            "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
            return_value=False,
        ):
            service._process_results([[sample_profile]])

        request_context.storage.add_user_profile.assert_called_once_with(
            "user_1", [sample_profile]
        )
        assert sample_profile.source == "api"
        assert sample_profile.status is None  # CURRENT (not pending)

    def test_save_profiles_pending_status(
        self, service_pending, request_context, sample_profile
    ):
        """Profiles get PENDING status when output_pending_status is True."""
        service_pending.service_config = ProfileGenerationServiceConfig(
            user_id="user_1",
            request_id="req_1",
            source="rerun",
        )

        with patch(
            "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
            return_value=False,
        ):
            service_pending._process_results([[sample_profile]])

        assert sample_profile.status == Status.PENDING

    def test_save_profiles_dedup_enabled(
        self, service, request_context, sample_profile
    ):
        """Deduplicator is called when enabled and profiles exist."""
        self._setup_service_config(service)

        dedup_mock = MagicMock()
        dedup_mock.deduplicate.return_value = ([sample_profile], ["old_p1"], [])

        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator",
                return_value=dedup_mock,
            ),
        ):
            service._process_results([[sample_profile]])

        dedup_mock.deduplicate.assert_called_once()
        request_context.storage.add_user_profile.assert_called_once()
        request_context.storage.delete_user_profile.assert_called_once()

    def test_dedup_with_pending_status_filter(
        self, service_pending, request_context, sample_profile
    ):
        """Deduplicator uses PENDING status filter in rerun mode."""
        service_pending.service_config = ProfileGenerationServiceConfig(
            user_id="user_1",
            request_id="req_1",
            source="rerun",
        )

        dedup_mock = MagicMock()
        dedup_mock.deduplicate.return_value = ([sample_profile], [], [])

        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator",
                return_value=dedup_mock,
            ),
        ):
            service_pending._process_results([[sample_profile]])

        dedup_mock.deduplicate.assert_called_once()

    def test_save_failure_returns_early(self, service, request_context, sample_profile):
        """When add_user_profile raises, the method returns without deleting."""
        self._setup_service_config(service)
        request_context.storage.add_user_profile.side_effect = RuntimeError("DB error")

        with patch(
            "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
            return_value=False,
        ):
            service._process_results([[sample_profile]])

        request_context.storage.delete_user_profile.assert_not_called()
        request_context.storage.add_profile_change_log.assert_not_called()

    def test_delete_superseded_failure_continues(
        self, service, request_context, sample_profile
    ):
        """When deleting superseded profile fails, processing continues."""
        self._setup_service_config(service)

        dedup_mock = MagicMock()
        dedup_mock.deduplicate.return_value = (
            [sample_profile],
            ["old_p1", "old_p2"],
            [],
        )

        request_context.storage.delete_user_profile.side_effect = RuntimeError(
            "Delete error"
        )

        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator",
                return_value=dedup_mock,
            ),
        ):
            service._process_results([[sample_profile]])

        assert request_context.storage.delete_user_profile.call_count == 2

    def test_changelog_created_after_profiles_saved(
        self, service, request_context, sample_profile
    ):
        """Profile changelog is created when new profiles are saved."""
        self._setup_service_config(service)

        with patch(
            "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
            return_value=False,
        ):
            service._process_results([[sample_profile]])

        request_context.storage.add_profile_change_log.assert_called_once()
        changelog = request_context.storage.add_profile_change_log.call_args[0][0]
        assert changelog.user_id == "user_1"
        assert changelog.request_id == "req_1"
        assert changelog.added_profiles == [sample_profile]

    def test_changelog_failure_is_handled(
        self, service, request_context, sample_profile
    ):
        """When add_profile_change_log fails, exception is caught and logged."""
        self._setup_service_config(service)
        request_context.storage.add_profile_change_log.side_effect = RuntimeError(
            "Changelog error"
        )

        with patch(
            "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
            return_value=False,
        ):
            service._process_results([[sample_profile]])

        request_context.storage.add_user_profile.assert_called_once()

    def test_changelog_with_superseded_profiles(
        self, service, request_context, sample_profile
    ):
        """Changelog includes superseded (removed) profiles from deduplication."""
        self._setup_service_config(service)

        superseded = UserProfile(
            profile_id="old_p1",
            user_id="user_1",
            content="old preference",
            last_modified_timestamp=int(datetime.now(UTC).timestamp()),
            generated_from_request_id="req_0",
        )

        dedup_mock = MagicMock()
        dedup_mock.deduplicate.return_value = (
            [sample_profile],
            [],
            [superseded],
        )

        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator",
                return_value=dedup_mock,
            ),
        ):
            service._process_results([[sample_profile]])

        changelog = request_context.storage.add_profile_change_log.call_args[0][0]
        assert changelog.removed_profiles == [superseded]

    def test_no_changelog_when_no_profiles(self, service, request_context):
        """No changelog is created when there are no new or superseded profiles."""
        self._setup_service_config(service)

        service._process_results([])

        request_context.storage.add_profile_change_log.assert_not_called()


# ===============================
# Test: _update_config_for_incremental
# ===============================


class TestUpdateConfigForIncremental:
    """Tests for _update_config_for_incremental."""

    def test_sets_incremental_flag(self, service, sample_profile):
        """Sets is_incremental=True on service_config."""
        service.service_config = ProfileGenerationServiceConfig(
            user_id="user_1",
            request_id="req_1",
        )

        service._update_config_for_incremental([[sample_profile]])

        assert service.service_config.is_incremental is True

    def test_copies_previously_extracted(self, service, sample_profile):
        """Previously extracted profiles are copied to service_config."""
        service.service_config = ProfileGenerationServiceConfig(
            user_id="user_1",
            request_id="req_1",
        )

        previously = [[sample_profile]]
        service._update_config_for_incremental(previously)

        assert len(service.service_config.previously_extracted) == 1
        assert service.service_config.previously_extracted[0][0] == sample_profile

    def test_empty_previously_extracted(self, service):
        """Empty previously_extracted list is handled correctly."""
        service.service_config = ProfileGenerationServiceConfig(
            user_id="user_1",
            request_id="req_1",
        )

        service._update_config_for_incremental([])

        assert service.service_config.is_incremental is True
        assert service.service_config.previously_extracted == []


# ===============================
# Test: run_manual_regular
# ===============================


class TestRunManualRegular:
    """Tests for run_manual_regular edge cases."""

    def test_in_progress_returns_failure(self, service):
        """Returns failure when another operation is already in progress."""
        mock_state_mgr = MagicMock()
        mock_state_mgr.check_in_progress.return_value = "Operation already in progress"

        with patch.object(
            service, "_create_state_manager", return_value=mock_state_mgr
        ):
            request = ManualProfileGenerationRequest(user_id="user_1")
            response = service.run_manual_regular(request)

        assert response.success is False
        assert response.profiles_generated == 0
        assert "already in progress" in response.msg

    def test_no_users_returns_success(self, service, request_context):
        """Returns success with 0 profiles when no users found."""
        mock_state_mgr = MagicMock()
        mock_state_mgr.check_in_progress.return_value = None

        request_context.storage.get_all_user_ids.return_value = []

        with patch.object(
            service, "_create_state_manager", return_value=mock_state_mgr
        ):
            request = ManualProfileGenerationRequest()
            response = service.run_manual_regular(request)

        assert response.success is True
        assert response.profiles_generated == 0
        assert "No users" in response.msg

    def test_specific_user_id(self, service, request_context):
        """When user_id is provided, only that user is processed."""
        mock_state_mgr = MagicMock()
        mock_state_mgr.check_in_progress.return_value = None

        request_context.storage.get_user_profile.return_value = [
            MagicMock()
        ]  # 1 profile

        with (
            patch.object(service, "_create_state_manager", return_value=mock_state_mgr),
            patch.object(service, "_run_batch_with_progress", return_value=(1, 1)),
        ):
            request = ManualProfileGenerationRequest(user_id="user_1")
            response = service.run_manual_regular(request)

        assert response.success is True
        assert response.profiles_generated == 1

    def test_exception_marks_failed(self, service):
        """Exception during processing marks operation as failed."""
        mock_state_mgr = MagicMock()
        mock_state_mgr.check_in_progress.return_value = None

        with (
            patch.object(service, "_create_state_manager", return_value=mock_state_mgr),
            patch.object(
                service,
                "_run_batch_with_progress",
                side_effect=RuntimeError("LLM timeout"),
            ),
        ):
            request = ManualProfileGenerationRequest(user_id="user_1")
            response = service.run_manual_regular(request)

        assert response.success is False
        assert "Failed" in response.msg
        mock_state_mgr.mark_progress_failed.assert_called_once()

    def test_all_users_when_no_user_id(self, service, request_context):
        """When user_id is None, all user IDs are fetched from storage."""
        mock_state_mgr = MagicMock()
        mock_state_mgr.check_in_progress.return_value = None

        request_context.storage.get_all_user_ids.return_value = ["u1", "u2", "u3"]
        request_context.storage.get_user_profile.return_value = []

        with (
            patch.object(service, "_create_state_manager", return_value=mock_state_mgr),
            patch.object(service, "_run_batch_with_progress", return_value=(3, 0)),
        ):
            request = ManualProfileGenerationRequest()
            response = service.run_manual_regular(request)

        assert response.success is True
        request_context.storage.get_all_user_ids.assert_called_once()


# ===============================
# Test: _count_manual_generated
# ===============================


class TestCountManualGenerated:
    """Tests for _count_manual_generated."""

    def test_counts_current_profiles(self, service, request_context, sample_profile):
        """Counts profiles with CURRENT status (None filter)."""
        request_context.storage.get_user_profile.return_value = [
            sample_profile,
            sample_profile,
        ]

        request = ManualProfileGenerationRequest(user_id="user_1")
        count = service._count_manual_generated(request)

        assert count == 2
        request_context.storage.get_user_profile.assert_called_once_with(
            user_id="user_1",
            status_filter=[None],
        )

    def test_returns_zero_when_no_profiles(self, service, request_context):
        """Returns 0 when no profiles exist."""
        request_context.storage.get_user_profile.return_value = []

        request = ManualProfileGenerationRequest(user_id="user_1")
        count = service._count_manual_generated(request)

        assert count == 0

    def test_no_user_id_filter(self, service, request_context):
        """When user_id is None, passes None to storage."""
        request_context.storage.get_user_profile.return_value = []

        request = ManualProfileGenerationRequest()
        service._count_manual_generated(request)

        request_context.storage.get_user_profile.assert_called_once_with(
            user_id=None,
            status_filter=[None],
        )


# ===============================
# Test: check_and_update_profiles
# ===============================


class TestCheckAndUpdateProfiles:
    """Tests for check_and_update_profiles."""

    def test_raises_not_implemented(self, service):
        """check_and_update_profiles raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            service.check_and_update_profiles([])


# ===============================
# Test: Status change helpers
# ===============================


class TestStatusChangeHelpers:
    """Tests for _has_items_with_status, _delete_items_by_status,
    _update_items_status, and _create_status_change_response."""

    def test_has_items_with_status(self, service, request_context):
        """_has_items_with_status delegates to storage.get_user_ids_with_status."""
        request_context.storage.get_user_ids_with_status.return_value = ["u1"]
        request = ProfileGenerationRequest(user_id="user_1", request_id="req_1")

        assert service._has_items_with_status(Status.PENDING, request) is True

        request_context.storage.get_user_ids_with_status.return_value = []
        assert service._has_items_with_status(None, request) is False

    def test_delete_items_by_status(self, service, request_context):
        """_delete_items_by_status delegates to storage."""
        request_context.storage.delete_all_profiles_by_status.return_value = 5
        request = ProfileGenerationRequest(user_id="user_1", request_id="req_1")

        count = service._delete_items_by_status(Status.ARCHIVED, request)

        assert count == 5
        request_context.storage.delete_all_profiles_by_status.assert_called_once_with(
            status=Status.ARCHIVED
        )

    def test_update_items_status(self, service, request_context):
        """_update_items_status delegates to storage."""
        request_context.storage.update_all_profiles_status.return_value = 3
        request = ProfileGenerationRequest(user_id="user_1", request_id="req_1")

        count = service._update_items_status(
            Status.PENDING, None, request, user_ids=["u1"]
        )

        assert count == 3
        request_context.storage.update_all_profiles_status.assert_called_once_with(
            Status.PENDING, None, user_ids=["u1"]
        )

    def test_create_upgrade_response(self, service):
        """_create_status_change_response creates UpgradeProfilesResponse."""
        response = service._create_status_change_response(
            StatusChangeOperation.UPGRADE,
            success=True,
            counts={"deleted": 2, "archived": 5, "promoted": 10},
            msg="Upgrade complete",
        )

        assert response.success is True
        assert response.profiles_deleted == 2
        assert response.profiles_archived == 5
        assert response.profiles_promoted == 10
        assert response.message == "Upgrade complete"

    def test_create_downgrade_response(self, service):
        """_create_status_change_response creates DowngradeProfilesResponse."""
        response = service._create_status_change_response(
            StatusChangeOperation.DOWNGRADE,
            success=False,
            counts={"demoted": 3, "restored": 7},
            msg="Downgrade failed",
        )

        assert response.success is False
        assert response.profiles_demoted == 3
        assert response.profiles_restored == 7
        assert response.message == "Downgrade failed"

    def test_affected_user_ids_for_upgrade(self, service, request_context):
        """_get_affected_user_ids_for_upgrade returns user IDs with PENDING when flag set."""
        request_context.storage.get_user_ids_with_status.return_value = ["u1", "u2"]

        # Use a MagicMock to simulate request with only_affected_users attribute
        request = MagicMock()
        request.only_affected_users = True

        result = service._get_affected_user_ids_for_upgrade(request)

        assert result == ["u1", "u2"]

    def test_affected_user_ids_for_upgrade_no_flag(self, service):
        """_get_affected_user_ids_for_upgrade returns None without flag."""
        request = ProfileGenerationRequest(user_id="user_1", request_id="req_1")

        result = service._get_affected_user_ids_for_upgrade(request)

        assert result is None

    def test_affected_user_ids_for_downgrade(self, service, request_context):
        """_get_affected_user_ids_for_downgrade returns user IDs with ARCHIVED when flag set."""
        request_context.storage.get_user_ids_with_status.return_value = ["u3"]

        request = MagicMock()
        request.only_affected_users = True

        result = service._get_affected_user_ids_for_downgrade(request)

        assert result == ["u3"]

    def test_affected_user_ids_for_downgrade_no_flag(self, service):
        """_get_affected_user_ids_for_downgrade returns None without flag."""
        request = ProfileGenerationRequest(user_id="user_1", request_id="req_1")

        result = service._get_affected_user_ids_for_downgrade(request)

        assert result is None


# ===============================
# Test: Service name methods
# ===============================


class TestServiceNameMethods:
    """Tests for _get_service_name, _get_base_service_name, etc."""

    def test_service_name_regular(self, service):
        """Regular service returns 'profile_generation'."""
        assert service._get_service_name() == "profile_generation"

    def test_service_name_rerun(self, service_pending):
        """Rerun service returns 'rerun_profile_generation'."""
        assert service_pending._get_service_name() == "rerun_profile_generation"

    def test_base_service_name(self, service):
        """Base service name is always 'profile_generation'."""
        assert service._get_base_service_name() == "profile_generation"

    def test_should_track_in_progress(self, service):
        """Profile generation tracks in-progress state."""
        assert service._should_track_in_progress() is True

    def test_get_lock_scope_id(self, service):
        """Lock scope ID is the user_id from the request."""
        request = ProfileGenerationRequest(user_id="user_42", request_id="req_1")
        assert service._get_lock_scope_id(request) == "user_42"

    def test_get_extractor_state_service_name(self, service):
        """Extractor state service name is 'profile_extractor'."""
        assert service._get_extractor_state_service_name() == "profile_extractor"


# ===============================
# Test: _load_generation_service_config
# ===============================


class TestLoadGenerationServiceConfig:
    """Tests for _load_generation_service_config."""

    def test_regular_mode_loads_all_profiles(self, service, request_context):
        """In regular mode, loads all profiles (no status filter)."""
        request_context.storage.get_user_profile.return_value = []

        request = ProfileGenerationRequest(
            user_id="user_1",
            request_id="req_1",
            source="api",
        )

        config = service._load_generation_service_config(request)

        assert config.user_id == "user_1"
        assert config.request_id == "req_1"
        assert config.source == "api"
        assert config.allow_manual_trigger is False
        assert config.output_pending_status is False
        request_context.storage.get_user_profile.assert_called_once_with("user_1")

    def test_rerun_mode_filters_pending_profiles(
        self, service_pending, request_context
    ):
        """In rerun mode, loads only PENDING profiles as existing data."""
        request_context.storage.get_user_profile.return_value = []

        request = ProfileGenerationRequest(
            user_id="user_1",
            request_id="req_1",
            source="rerun",
        )

        config = service_pending._load_generation_service_config(request)

        assert config.output_pending_status is True
        request_context.storage.get_user_profile.assert_called_once_with(
            "user_1", status_filter=[Status.PENDING]
        )

    def test_passes_rerun_time_filters(self, service, request_context):
        """Rerun time filters are passed through to service config."""
        request_context.storage.get_user_profile.return_value = []

        request = ProfileGenerationRequest(
            user_id="user_1",
            request_id="req_1",
            rerun_start_time=1000,
            rerun_end_time=2000,
            auto_run=False,
        )

        config = service._load_generation_service_config(request)

        assert config.rerun_start_time == 1000
        assert config.rerun_end_time == 2000
        assert config.auto_run is False

    def test_passes_extractor_names(self, service, request_context):
        """Extractor names filter is passed through to service config."""
        request_context.storage.get_user_profile.return_value = []

        request = ProfileGenerationRequest(
            user_id="user_1",
            request_id="req_1",
            extractor_names=["prefs", "bio"],
        )

        config = service._load_generation_service_config(request)

        assert config.extractor_names == ["prefs", "bio"]


# ===============================
# Test: Rerun hook implementations
# ===============================


class TestRerunHooks:
    """Tests for rerun-specific hook implementations."""

    def test_get_generated_count(self, service, request_context):
        """_get_generated_count counts PENDING profiles."""
        from reflexio.models.api_schema.service_schemas import (
            RerunProfileGenerationRequest,
        )

        request_context.storage.get_user_profile.return_value = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]

        request = RerunProfileGenerationRequest(user_id="user_1")
        count = service._get_generated_count(request)

        assert count == 3
        request_context.storage.get_user_profile.assert_called_once_with(
            user_id="user_1",
            status_filter=[Status.PENDING],
        )

    def test_create_rerun_response(self, service):
        """_create_rerun_response creates RerunProfileGenerationResponse."""
        response = service._create_rerun_response(success=True, msg="Done", count=5)

        assert response.success is True
        assert response.msg == "Done"
        assert response.profiles_generated == 5

    def test_create_run_request_for_rerun(self, service):
        """_create_run_request_for_item handles RerunProfileGenerationRequest."""
        from reflexio.models.api_schema.service_schemas import (
            RerunProfileGenerationRequest,
        )

        request = RerunProfileGenerationRequest(
            user_id=None,
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 6, 1, tzinfo=UTC),
            source="api",
            extractor_names=["prefs"],
        )

        result = service._create_run_request_for_item("user_42", request)

        assert result.user_id == "user_42"
        assert result.source == "api"
        assert result.extractor_names == ["prefs"]
        assert result.auto_run is False
        assert result.request_id.startswith("rerun_")
        assert result.rerun_start_time == int(
            datetime(2025, 1, 1, tzinfo=UTC).timestamp()
        )
        assert result.rerun_end_time == int(
            datetime(2025, 6, 1, tzinfo=UTC).timestamp()
        )

    def test_create_run_request_for_manual(self, service):
        """_create_run_request_for_item handles ManualProfileGenerationRequest."""
        request = ManualProfileGenerationRequest(
            user_id="user_1",
            source="manual",
            extractor_names=["bio"],
        )

        result = service._create_run_request_for_item("user_1", request)

        assert result.user_id == "user_1"
        assert result.source == "manual"
        assert result.extractor_names == ["bio"]
        assert result.auto_run is False
        assert result.request_id.startswith("manual_")
        assert result.rerun_start_time is None
        assert result.rerun_end_time is None

    def test_build_rerun_request_params(self, service):
        """_build_rerun_request_params creates correct parameter dict."""
        from reflexio.models.api_schema.service_schemas import (
            RerunProfileGenerationRequest,
        )

        request = RerunProfileGenerationRequest(
            user_id="user_1",
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 6, 1, tzinfo=UTC),
            source="api",
            extractor_names=["prefs"],
        )

        params = service._build_rerun_request_params(request)

        assert params["user_id"] == "user_1"
        assert params["source"] == "api"
        assert params["extractor_names"] == ["prefs"]
        assert params["start_time"] is not None
        assert params["end_time"] is not None

    def test_get_rerun_user_ids(self, service, request_context):
        """_get_rerun_user_ids delegates to storage."""
        from reflexio.models.api_schema.service_schemas import (
            RerunProfileGenerationRequest,
        )

        request_context.storage.get_rerun_user_ids.return_value = ["u1", "u2"]

        request = RerunProfileGenerationRequest(
            user_id="u1",
            start_time=datetime(2025, 1, 1, tzinfo=UTC),
            end_time=datetime(2025, 6, 1, tzinfo=UTC),
            source="api",
        )

        user_ids = service._get_rerun_user_ids(request)

        assert user_ids == ["u1", "u2"]
        request_context.storage.get_rerun_user_ids.assert_called_once()
