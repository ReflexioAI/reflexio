"""Unit tests for ConfigMixin and DashboardMixin.

Tests get_config, set_config for ConfigMixin and
get_dashboard_stats, get_injection_stats, get_memory_review
for DashboardMixin with mocked storage.
"""

from typing import Any, cast
from unittest.mock import MagicMock

from reflexio.lib._config import ConfigMixin
from reflexio.lib._dashboard import DashboardMixin
from reflexio.models.api_schema.retriever_schema import (
    GetDashboardStatsRequest,
    GetInjectionStatsRequest,
    GetMemoryReviewRequest,
    InjectionStat,
    MemoryReviewCandidate,
)
from reflexio.models.config_schema import Config

# ---------------------------------------------------------------------------
# ConfigMixin helpers
# ---------------------------------------------------------------------------


def _make_config_mixin(*, storage_configured: bool = True) -> Any:
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
        # normalize_config_payload is identity in the base configurator; the
        # MagicMock default would otherwise return another MagicMock and break
        # the **kwargs expansion below.
        payload = {"storage_config": {"db_path": "/var/data/test.db"}}
        mixin.request_context.configurator.normalize_config_payload.return_value = (
            payload
        )
        mixin.request_context.configurator.get_current_storage_configuration.return_value = MagicMock()
        mixin.request_context.configurator.is_storage_config_ready_to_test.return_value = True
        mixin.request_context.configurator.test_and_init_storage_config.return_value = (
            True,
            None,
        )

        response = mixin.set_config(payload)

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


def _make_dashboard_mixin(*, storage_configured: bool = True) -> Any:
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
    return cast(MagicMock, mixin.request_context.storage)


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


# ---------------------------------------------------------------------------
# get_injection_stats
# ---------------------------------------------------------------------------


class TestGetInjectionStats:
    def test_returns_stats(self):
        """Returns injection stats from storage."""
        mixin = _make_dashboard_mixin()
        stat = InjectionStat(
            entity_type="user_playbook",
            entity_id="1",
            surfaced_count=5,
            distinct_session_count=3,
            total_prompt_tokens=120,
        )
        _get_dashboard_storage(mixin).get_injection_stats.return_value = [stat]

        request = GetInjectionStatsRequest(days_back=30)
        response = mixin.get_injection_stats(request)

        assert response.success is True
        assert len(response.stats) == 1
        assert response.stats[0].entity_id == "1"
        assert response.stats[0].surfaced_count == 5

    def test_storage_not_configured(self):
        """Returns empty stats when storage is not configured."""
        mixin = _make_dashboard_mixin(storage_configured=False)

        request = GetInjectionStatsRequest(days_back=30)
        response = mixin.get_injection_stats(request)

        assert response.success is True
        assert response.stats == []
        assert response.msg is not None

    def test_dict_input(self):
        """Accepts dict input and auto-converts."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(mixin).get_injection_stats.return_value = []

        response = mixin.get_injection_stats({"days_back": 7})

        assert response.success is True
        _get_dashboard_storage(mixin).get_injection_stats.assert_called_once_with(
            days_back=7
        )

    def test_exception_returns_failure(self):
        """Returns failure on storage exception."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(mixin).get_injection_stats.side_effect = RuntimeError(
            "db error"
        )

        request = GetInjectionStatsRequest(days_back=30)
        response = mixin.get_injection_stats(request)

        assert response.success is False
        assert "db error" in (response.msg or "")

    def test_default_days_back(self):
        """Uses default 30 days when days_back is not provided."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(mixin).get_injection_stats.return_value = []

        request = GetInjectionStatsRequest()
        mixin.get_injection_stats(request)

        _get_dashboard_storage(mixin).get_injection_stats.assert_called_once_with(
            days_back=30
        )


# ---------------------------------------------------------------------------
# get_memory_review
# ---------------------------------------------------------------------------


class TestGetMemoryReview:
    def test_returns_candidates(self):
        """Returns hygiene candidates from storage."""
        mixin = _make_dashboard_mixin()
        candidate = MemoryReviewCandidate(
            entity_type="user_playbook",
            entity_id="42",
            title="stale rule",
            signals=["stale"],
            score=10,
            injection_count=0,
            citation_count=0,
        )
        _get_dashboard_storage(mixin).get_memory_review_candidates.return_value = [
            candidate
        ]

        request = GetMemoryReviewRequest(days_back=60, user_id="userA")
        response = mixin.get_memory_review(request)

        assert response.success is True
        assert len(response.candidates) == 1
        assert response.candidates[0].entity_id == "42"
        assert "stale" in response.candidates[0].signals

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_dashboard_mixin(storage_configured=False)

        request = GetMemoryReviewRequest(days_back=60, user_id="userA")
        response = mixin.get_memory_review(request)

        assert response.success is True
        assert response.candidates == []
        assert response.msg is not None

    def test_dict_input(self):
        """Accepts dict input and auto-converts."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.return_value = []

        response = mixin.get_memory_review({"days_back": 30, "user_id": "userA"})

        assert response.success is True
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.assert_called_once_with(
            days_back=30, user_id="userA", include_all_users=False
        )

    def test_signal_filter_narrows_candidates(self):
        """signal_filter keeps only candidates with at least one matching signal."""
        mixin = _make_dashboard_mixin()
        candidates = [
            MemoryReviewCandidate(
                entity_type="user_playbook",
                entity_id="1",
                title="stale",
                signals=["stale"],
                score=5,
                injection_count=0,
                citation_count=0,
            ),
            MemoryReviewCandidate(
                entity_type="user_playbook",
                entity_id="2",
                title="low cite",
                signals=["high_cost_low_cite"],
                score=7,
                injection_count=10,
                citation_count=1,
            ),
            MemoryReviewCandidate(
                entity_type="user_playbook",
                entity_id="3",
                title="stale and low cite",
                signals=["stale", "high_cost_low_cite"],
                score=9,
                injection_count=10,
                citation_count=0,
            ),
        ]
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.return_value = candidates

        request = GetMemoryReviewRequest(
            days_back=60, user_id="userA", signal_filter=["stale"]
        )
        response = mixin.get_memory_review(request)

        ids = {c.entity_id for c in response.candidates}
        assert ids == {"1", "3"}

    def test_signal_filter_none_returns_all(self):
        """signal_filter=None returns the unfiltered candidate set."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.return_value = [
            MemoryReviewCandidate(
                entity_type="user_playbook",
                entity_id="1",
                title="x",
                signals=["stale"],
                score=1,
                injection_count=0,
                citation_count=0,
            )
        ]

        request = GetMemoryReviewRequest(days_back=60, user_id="userA")
        response = mixin.get_memory_review(request)

        assert len(response.candidates) == 1

    def test_signal_filter_empty_list_filters_out_everything(self):
        """signal_filter=[] is a real (empty) filter — returns nothing.

        Distinct from signal_filter=None which returns the
        unfiltered candidate set. The Pydantic field is typed
        ``list[...] | None`` and the lib gates on
        ``is not None`` so an explicit empty list is honored.
        """
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.return_value = [
            MemoryReviewCandidate(
                entity_type="user_playbook",
                entity_id="1",
                title="stale",
                signals=["stale"],
                score=5,
                injection_count=0,
                citation_count=0,
            ),
            MemoryReviewCandidate(
                entity_type="user_playbook",
                entity_id="2",
                title="noisy",
                signals=["high_cost_low_cite"],
                score=10,
                injection_count=10,
                citation_count=1,
            ),
        ]

        request = GetMemoryReviewRequest(
            days_back=60, user_id="userA", signal_filter=[]
        )
        response = mixin.get_memory_review(request)

        assert response.candidates == []

    def test_exception_returns_failure(self):
        """Returns failure on storage exception."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.side_effect = RuntimeError("db error")

        request = GetMemoryReviewRequest(days_back=60, user_id="userA")
        response = mixin.get_memory_review(request)

        assert response.success is False
        assert "db error" in (response.msg or "")

    def test_default_days_back(self):
        """Uses default 60 days when days_back is not provided."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.return_value = []

        request = GetMemoryReviewRequest(user_id="userA")
        mixin.get_memory_review(request)

        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.assert_called_once_with(
            days_back=60, user_id="userA", include_all_users=False
        )

    def test_org_wide_review_must_be_explicit(self):
        """include_all_users=True is the explicit org-wide review path."""
        mixin = _make_dashboard_mixin()
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.return_value = []

        request = GetMemoryReviewRequest(include_all_users=True)
        response = mixin.get_memory_review(request)

        assert response.success is True
        _get_dashboard_storage(
            mixin
        ).get_memory_review_candidates.assert_called_once_with(
            days_back=60, user_id=None, include_all_users=True
        )
