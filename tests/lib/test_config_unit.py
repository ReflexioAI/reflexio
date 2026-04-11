"""Unit tests for ConfigMixin and DashboardMixin.

Tests get_config, set_config for ConfigMixin and
get_dashboard_stats for DashboardMixin with mocked storage.
"""

from unittest.mock import MagicMock

from reflexio.lib._config import ConfigMixin
from reflexio.lib._dashboard import DashboardMixin
from reflexio.models.api_schema.retriever_schema import GetDashboardStatsRequest
from reflexio.models.config_schema import Config

# ---------------------------------------------------------------------------
# ConfigMixin helpers
# ---------------------------------------------------------------------------


def _make_config_mixin(*, storage_configured: bool = True) -> ConfigMixin:
    """Create a ConfigMixin instance with mocked internals."""
    mixin = object.__new__(ConfigMixin)
    mock_storage = MagicMock()

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured

    mixin.request_context = mock_request_context
    mixin.llm_client = MagicMock()
    return mixin


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_returns_config(self):
        """Returns config from configurator."""
        mixin = _make_config_mixin()
        mock_config = MagicMock(spec=Config)
        mixin.request_context.configurator.get_config.return_value = mock_config

        result = mixin.get_config()

        assert result is mock_config
        mixin.request_context.configurator.get_config.assert_called_once()

    def test_returns_none_when_no_config(self):
        """Returns None when no config is set."""
        mixin = _make_config_mixin()
        mixin.request_context.configurator.get_config.return_value = None

        result = mixin.get_config()

        assert result is None


# ---------------------------------------------------------------------------
# set_config
# ---------------------------------------------------------------------------


class TestSetConfig:
    def test_set_config_success(self):
        """Successfully sets config after validation."""
        mixin = _make_config_mixin()
        mock_storage_config = MagicMock()

        mock_config = MagicMock(spec=Config)
        mock_config.storage_config = mock_storage_config

        mixin.request_context.configurator.is_storage_config_ready_to_test.return_value = True
        mixin.request_context.configurator.test_and_init_storage_config.return_value = (
            True,
            None,
        )

        response = mixin.set_config(mock_config)

        assert response.success is True
        assert "successfully" in (response.msg or "").lower()
        mixin.request_context.configurator.set_config.assert_called_once()

    def test_set_config_storage_validation_fails(self):
        """Returns failure when storage validation fails."""
        mixin = _make_config_mixin()
        mock_config = MagicMock(spec=Config)
        mock_config.storage_config = MagicMock()

        mixin.request_context.configurator.is_storage_config_ready_to_test.return_value = True
        mixin.request_context.configurator.test_and_init_storage_config.return_value = (
            False,
            "Connection refused",
        )

        response = mixin.set_config(mock_config)

        assert response.success is False
        assert "Connection refused" in (response.msg or "")

    def test_set_config_storage_not_ready(self):
        """Returns failure when storage config is incomplete."""
        mixin = _make_config_mixin()
        mock_config = MagicMock(spec=Config)
        mock_config.storage_config = MagicMock()

        mixin.request_context.configurator.is_storage_config_ready_to_test.return_value = False

        response = mixin.set_config(mock_config)

        assert response.success is False
        assert "incomplete" in (response.msg or "").lower()

    def test_set_config_preserves_existing_storage_config(self):
        """Preserves existing storage config when none provided."""
        mixin = _make_config_mixin()
        mock_config = MagicMock(spec=Config)
        mock_config.storage_config = None

        existing_storage_config = MagicMock()
        mixin.request_context.configurator.get_current_storage_configuration.return_value = existing_storage_config
        mixin.request_context.configurator.is_storage_config_ready_to_test.return_value = True
        mixin.request_context.configurator.test_and_init_storage_config.return_value = (
            True,
            None,
        )

        response = mixin.set_config(mock_config)

        assert response.success is True
        # Verify storage_config was set to the existing one
        assert mock_config.storage_config == existing_storage_config

    def test_set_config_dict_input(self):
        """Accepts dict input and auto-converts to Config."""
        mixin = _make_config_mixin()
        mixin.request_context.configurator.get_current_storage_configuration.return_value = MagicMock()
        mixin.request_context.configurator.is_storage_config_ready_to_test.return_value = True
        mixin.request_context.configurator.test_and_init_storage_config.return_value = (
            True,
            None,
        )

        response = mixin.set_config(
            {"storage_config": {"dir_path": "/var/data/test_storage"}}
        )

        assert response.success is True

    def test_set_config_exception(self):
        """Returns failure on unexpected exception."""
        mixin = _make_config_mixin()
        mock_config = MagicMock(spec=Config)
        mock_config.storage_config = MagicMock()

        mixin.request_context.configurator.is_storage_config_ready_to_test.side_effect = RuntimeError(
            "unexpected"
        )

        response = mixin.set_config(mock_config)

        assert response.success is False
        assert "unexpected" in (response.msg or "")


# ---------------------------------------------------------------------------
# DashboardMixin helpers
# ---------------------------------------------------------------------------


def _make_dashboard_mixin(*, storage_configured: bool = True) -> DashboardMixin:
    """Create a DashboardMixin instance with mocked internals."""
    mixin = object.__new__(DashboardMixin)
    mock_storage = MagicMock()

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured

    mixin.request_context = mock_request_context
    mixin.llm_client = MagicMock()
    return mixin


def _get_dashboard_storage(mixin: DashboardMixin) -> MagicMock:
    return mixin.request_context.storage


# ---------------------------------------------------------------------------
# get_dashboard_stats
# ---------------------------------------------------------------------------


class TestGetDashboardStats:
    def test_returns_stats(self):
        """Returns dashboard stats from storage."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(mixin).get_dashboard_stats.return_value = {
            "current_period": {
                "total_profiles": 10,
                "total_interactions": 50,
                "total_playbooks": 5,
                "success_rate": 80.0,
            },
            "previous_period": {
                "total_profiles": 8,
                "total_interactions": 40,
                "total_playbooks": 4,
                "success_rate": 75.0,
            },
            "interactions_time_series": [{"timestamp": 1000, "value": 5}],
            "profiles_time_series": [{"timestamp": 1000, "value": 2}],
            "playbooks_time_series": [{"timestamp": 1000, "value": 1}],
            "evaluations_time_series": [{"timestamp": 1000, "value": 3}],
        }

        request = GetDashboardStatsRequest(days_back=30)
        response = mixin.get_dashboard_stats(request)

        assert response.success is True
        assert response.stats is not None
        assert response.stats.current_period.total_profiles == 10
        assert response.stats.previous_period.total_interactions == 40
        assert len(response.stats.interactions_time_series) == 1

    def test_storage_not_configured(self):
        """Returns empty stats when storage is not configured."""
        mixin = _make_dashboard_mixin(storage_configured=False)

        request = GetDashboardStatsRequest(days_back=30)
        response = mixin.get_dashboard_stats(request)

        assert response.success is True
        assert response.stats is not None
        assert response.stats.current_period.total_profiles == 0
        assert response.stats.current_period.total_interactions == 0
        assert response.msg is not None

    def test_dict_input(self):
        """Accepts dict input and auto-converts."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(mixin).get_dashboard_stats.return_value = {
            "current_period": {
                "total_profiles": 0,
                "total_interactions": 0,
                "total_playbooks": 0,
                "success_rate": 0.0,
            },
            "previous_period": {
                "total_profiles": 0,
                "total_interactions": 0,
                "total_playbooks": 0,
                "success_rate": 0.0,
            },
            "interactions_time_series": [],
            "profiles_time_series": [],
            "playbooks_time_series": [],
            "evaluations_time_series": [],
        }

        response = mixin.get_dashboard_stats({"days_back": 7})

        assert response.success is True
        _get_dashboard_storage(mixin).get_dashboard_stats.assert_called_once_with(
            days_back=7
        )

    def test_exception_returns_failure(self):
        """Returns failure on storage exception."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(mixin).get_dashboard_stats.side_effect = RuntimeError(
            "db error"
        )

        request = GetDashboardStatsRequest(days_back=30)
        response = mixin.get_dashboard_stats(request)

        assert response.success is False
        assert "db error" in (response.msg or "")

    def test_default_days_back(self):
        """Uses default 30 days when days_back is None."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(mixin).get_dashboard_stats.return_value = {
            "current_period": {
                "total_profiles": 0,
                "total_interactions": 0,
                "total_playbooks": 0,
                "success_rate": 0.0,
            },
            "previous_period": {
                "total_profiles": 0,
                "total_interactions": 0,
                "total_playbooks": 0,
                "success_rate": 0.0,
            },
            "interactions_time_series": [],
            "profiles_time_series": [],
            "playbooks_time_series": [],
            "evaluations_time_series": [],
        }

        request = GetDashboardStatsRequest()
        mixin.get_dashboard_stats(request)

        _get_dashboard_storage(mixin).get_dashboard_stats.assert_called_once_with(
            days_back=30
        )
