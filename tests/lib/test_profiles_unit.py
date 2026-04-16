"""Unit tests for ProfilesMixin.

Tests get_profiles, get_all_profiles, search_profiles, delete_profile,
delete_all_profiles_bulk, delete_profiles_by_ids, get_profile_change_logs,
get_profile_statistics, upgrade_all_profiles, and downgrade_all_profiles
with mocked storage and services.
"""

import time
from unittest.mock import MagicMock, patch

from reflexio.lib._base import STORAGE_NOT_CONFIGURED_MSG
from reflexio.lib._profiles import ProfilesMixin
from reflexio.models.api_schema.retriever_schema import (
    GetUserProfilesRequest,
    SearchUserProfileRequest,
    UpdateUserProfileRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AddUserProfileRequest,
    AddUserProfileResponse,
    DeleteProfilesByIdsRequest,
    DeleteUserProfileRequest,
    DowngradeProfilesRequest,
    DowngradeProfilesResponse,
    ProfileChangeLog,
    Status,
    UpgradeProfilesRequest,
    UpgradeProfilesResponse,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin(*, storage_configured: bool = True) -> ProfilesMixin:
    """Create a ProfilesMixin instance with mocked internals."""
    mixin = object.__new__(ProfilesMixin)
    mock_storage = MagicMock()

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured

    mixin.request_context = mock_request_context
    mixin.llm_client = MagicMock()
    return mixin


def _get_storage(mixin: ProfilesMixin) -> MagicMock:
    return mixin.request_context.storage


def _sample_profile(**overrides) -> UserProfile:
    defaults = {
        "profile_id": "p1",
        "user_id": "user1",
        "content": "likes sushi",
        "last_modified_timestamp": int(time.time()),
        "generated_from_request_id": "req1",
    }
    defaults.update(overrides)
    return UserProfile(**defaults)


# ---------------------------------------------------------------------------
# get_profiles
# ---------------------------------------------------------------------------


class TestGetProfiles:
    def test_returns_profiles(self):
        """Successful retrieval returns profiles from storage."""
        mixin = _make_mixin()
        sample = _sample_profile()
        _get_storage(mixin).get_user_profile.return_value = [sample]

        request = GetUserProfilesRequest(user_id="user1")
        response = mixin.get_profiles(request)

        assert response.success is True
        assert len(response.user_profiles) == 1

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = GetUserProfilesRequest(user_id="user1")
        response = mixin.get_profiles(request)

        assert response.success is True
        assert response.user_profiles == []
        assert response.msg is not None

    def test_dict_input(self):
        """Accepts dict input and auto-converts."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_profile.return_value = []

        response = mixin.get_profiles({"user_id": "user1"})

        assert response.success is True
        _get_storage(mixin).get_user_profile.assert_called_once()

    def test_top_k_limit(self):
        """Applies top_k limit to results."""
        mixin = _make_mixin()
        now = int(time.time())
        profiles = [
            _sample_profile(profile_id=f"p{i}", last_modified_timestamp=now - i)
            for i in range(5)
        ]
        _get_storage(mixin).get_user_profile.return_value = profiles

        request = GetUserProfilesRequest(user_id="user1", top_k=2)
        response = mixin.get_profiles(request)

        assert response.success is True
        assert len(response.user_profiles) == 2

    def test_sorted_by_last_modified_descending(self):
        """Results are sorted by last_modified_timestamp in descending order."""
        mixin = _make_mixin()
        now = int(time.time())
        profiles = [
            _sample_profile(profile_id="p1", last_modified_timestamp=now - 100),
            _sample_profile(profile_id="p2", last_modified_timestamp=now),
            _sample_profile(profile_id="p3", last_modified_timestamp=now - 50),
        ]
        _get_storage(mixin).get_user_profile.return_value = profiles

        request = GetUserProfilesRequest(user_id="user1")
        response = mixin.get_profiles(request)

        assert response.success is True
        timestamps = [p.last_modified_timestamp for p in response.user_profiles]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_status_filter_from_parameter(self):
        """Status filter passed as parameter takes precedence."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_profile.return_value = []

        request = GetUserProfilesRequest(user_id="user1")
        mixin.get_profiles(request, status_filter=[Status.ARCHIVED])

        call_kwargs = _get_storage(mixin).get_user_profile.call_args
        assert call_kwargs[1]["status_filter"] == [Status.ARCHIVED]

    def test_status_filter_from_request(self):
        """Uses request.status_filter when parameter not given."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_profile.return_value = []

        request = GetUserProfilesRequest(
            user_id="user1", status_filter=[Status.PENDING]
        )
        mixin.get_profiles(request)

        call_kwargs = _get_storage(mixin).get_user_profile.call_args
        assert call_kwargs[1]["status_filter"] == [Status.PENDING]

    def test_default_status_filter(self):
        """Defaults to [None] status filter for current profiles."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_profile.return_value = []

        request = GetUserProfilesRequest(user_id="user1")
        mixin.get_profiles(request)

        call_kwargs = _get_storage(mixin).get_user_profile.call_args
        assert call_kwargs[1]["status_filter"] == [None]


# ---------------------------------------------------------------------------
# get_all_profiles
# ---------------------------------------------------------------------------


class TestGetAllProfiles:
    def test_returns_all(self):
        """Returns all profiles across users."""
        mixin = _make_mixin()
        sample = _sample_profile()
        _get_storage(mixin).get_all_profiles.return_value = [sample]

        response = mixin.get_all_profiles(limit=50)

        assert response.success is True
        assert len(response.user_profiles) == 1
        _get_storage(mixin).get_all_profiles.assert_called_once_with(
            limit=50, status_filter=[None]
        )

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.get_all_profiles()

        assert response.success is True
        assert response.user_profiles == []
        assert response.msg is not None

    def test_custom_status_filter(self):
        """Passes custom status filter to storage."""
        mixin = _make_mixin()
        _get_storage(mixin).get_all_profiles.return_value = []

        mixin.get_all_profiles(status_filter=[Status.ARCHIVED])

        call_kwargs = _get_storage(mixin).get_all_profiles.call_args
        assert call_kwargs[1]["status_filter"] == [Status.ARCHIVED]


# ---------------------------------------------------------------------------
# search_profiles
# ---------------------------------------------------------------------------


class TestSearchProfiles:
    def test_query_delegation(self):
        """Delegates search to storage."""
        mixin = _make_mixin()
        sample = _sample_profile()
        _get_storage(mixin).search_user_profile.return_value = [sample]

        request = SearchUserProfileRequest(user_id="user1", query="sushi")
        response = mixin.search_profiles(request)

        assert response.success is True
        assert len(response.user_profiles) == 1
        _get_storage(mixin).search_user_profile.assert_called_once()

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = SearchUserProfileRequest(user_id="user1", query="sushi")
        response = mixin.search_profiles(request)

        assert response.success is True
        assert response.user_profiles == []
        assert response.msg is not None

    def test_dict_input(self):
        """Accepts dict input and auto-converts."""
        mixin = _make_mixin()
        _get_storage(mixin).search_user_profile.return_value = []

        response = mixin.search_profiles({"user_id": "user1", "query": "test"})

        assert response.success is True

    def test_default_status_filter(self):
        """Defaults to [None] status filter for current profiles."""
        mixin = _make_mixin()
        _get_storage(mixin).search_user_profile.return_value = []

        request = SearchUserProfileRequest(user_id="user1", query="test")
        mixin.search_profiles(request)

        call_kwargs = _get_storage(mixin).search_user_profile.call_args
        assert call_kwargs[1]["status_filter"] == [None]

    def test_custom_status_filter(self):
        """Uses provided status filter."""
        mixin = _make_mixin()
        _get_storage(mixin).search_user_profile.return_value = []

        request = SearchUserProfileRequest(user_id="user1", query="test")
        mixin.search_profiles(request, status_filter=[Status.PENDING])

        call_kwargs = _get_storage(mixin).search_user_profile.call_args
        assert call_kwargs[1]["status_filter"] == [Status.PENDING]


# ---------------------------------------------------------------------------
# delete_profile
# ---------------------------------------------------------------------------


class TestDeleteProfile:
    def test_single_delete(self):
        """Deletes a profile by user_id and profile_id."""
        mixin = _make_mixin()

        request = DeleteUserProfileRequest(user_id="user1", profile_id="p1")
        response = mixin.delete_profile(request)

        assert response.success is True
        _get_storage(mixin).delete_user_profile.assert_called_once()

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()

        response = mixin.delete_profile({"user_id": "user1", "profile_id": "p1"})

        assert response.success is True

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteUserProfileRequest(user_id="user1", profile_id="p1")
        response = mixin.delete_profile(request)

        assert response.success is False

    def test_storage_exception(self):
        """Returns failure on storage exception."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_user_profile.side_effect = RuntimeError("db error")

        request = DeleteUserProfileRequest(user_id="user1", profile_id="p1")
        response = mixin.delete_profile(request)

        assert response.success is False
        assert "db error" in (response.message or "")


# ---------------------------------------------------------------------------
# delete_all_profiles_bulk
# ---------------------------------------------------------------------------


class TestDeleteAllProfilesBulk:
    def test_bulk_delete(self):
        """Deletes all profiles."""
        mixin = _make_mixin()

        response = mixin.delete_all_profiles_bulk()

        assert response.success is True
        _get_storage(mixin).delete_all_profiles.assert_called_once()

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.delete_all_profiles_bulk()

        assert response.success is False


# ---------------------------------------------------------------------------
# delete_profiles_by_ids
# ---------------------------------------------------------------------------


class TestDeleteProfilesByIds:
    def test_delete_by_ids(self):
        """Deletes profiles by their IDs."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_profiles_by_ids.return_value = 3

        request = DeleteProfilesByIdsRequest(profile_ids=["p1", "p2", "p3"])
        response = mixin.delete_profiles_by_ids(request)

        assert response.success is True
        assert response.deleted_count == 3

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_profiles_by_ids.return_value = 1

        response = mixin.delete_profiles_by_ids({"profile_ids": ["p1"]})

        assert response.success is True

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteProfilesByIdsRequest(profile_ids=["p1"])
        response = mixin.delete_profiles_by_ids(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# add_user_profile
# ---------------------------------------------------------------------------


class TestAddUserProfile:
    def test_add_user_profile_groups_by_user_id(self):
        """Profiles for two users trigger two storage calls, one per user."""
        mixin = _make_mixin()
        storage = _get_storage(mixin)

        profiles = [
            _sample_profile(profile_id="p1", user_id="user1", content="likes sushi"),
            _sample_profile(profile_id="p2", user_id="user1", content="dog owner"),
            _sample_profile(profile_id="p3", user_id="user2", content="vegetarian"),
        ]
        request = AddUserProfileRequest(user_profiles=profiles)

        response = mixin.add_user_profile(request)

        assert response.success is True
        assert response.added_count == 3
        assert storage.add_user_profile.call_count == 2

        calls_by_user = {
            call.args[0]: call.args[1]
            for call in storage.add_user_profile.call_args_list
        }
        assert set(calls_by_user.keys()) == {"user1", "user2"}
        assert len(calls_by_user["user1"]) == 2
        assert {p.profile_id for p in calls_by_user["user1"]} == {"p1", "p2"}
        assert len(calls_by_user["user2"]) == 1
        assert calls_by_user["user2"][0].profile_id == "p3"

    def test_add_user_profile_unconfigured_storage(self):
        """Returns failure response with STORAGE_NOT_CONFIGURED_MSG when storage is None."""
        mixin = _make_mixin(storage_configured=False)

        request = AddUserProfileRequest(
            user_profiles=[_sample_profile(content="likes sushi")]
        )
        response = mixin.add_user_profile(request)

        assert isinstance(response, AddUserProfileResponse)
        assert response.success is False
        assert response.message == STORAGE_NOT_CONFIGURED_MSG

    def test_add_user_profile_dict_input(self):
        """Accepts dict input and auto-converts to AddUserProfileRequest."""
        mixin = _make_mixin()

        now = int(time.time())
        response = mixin.add_user_profile(
            {
                "user_profiles": [
                    {
                        "profile_id": "p1",
                        "user_id": "user1",
                        "content": "likes sushi",
                        "last_modified_timestamp": now,
                        "generated_from_request_id": "req1",
                    },
                ]
            }
        )

        assert response.success is True
        assert response.added_count == 1
        _get_storage(mixin).add_user_profile.assert_called_once()

    def test_add_user_profile_storage_exception(self):
        """Returns failure response (with generic message) when storage raises."""
        mixin = _make_mixin()
        _get_storage(mixin).add_user_profile.side_effect = RuntimeError("db error")

        request = AddUserProfileRequest(
            user_profiles=[_sample_profile(content="likes sushi")]
        )
        response = mixin.add_user_profile(request)

        assert response.success is False
        # The implementation returns a generic message (not the raw
        # exception text) to avoid leaking storage details over HTTP.
        assert response.message is not None
        assert "db error" not in response.message


# ---------------------------------------------------------------------------
# get_profile_change_logs
# ---------------------------------------------------------------------------


class TestGetProfileChangeLogs:
    def test_returns_change_logs(self):
        """Returns change logs from storage."""
        mixin = _make_mixin()
        sample_log = ProfileChangeLog(
            id=1,
            user_id="user1",
            request_id="req1",
            added_profiles=[_sample_profile()],
            removed_profiles=[],
            mentioned_profiles=[],
        )
        _get_storage(mixin).get_profile_change_logs.return_value = [sample_log]

        response = mixin.get_profile_change_logs()

        assert response.success is True
        assert len(response.profile_change_logs) == 1

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.get_profile_change_logs()

        assert response.success is True
        assert response.profile_change_logs == []


# ---------------------------------------------------------------------------
# get_profile_statistics
# ---------------------------------------------------------------------------


class TestGetProfileStatistics:
    def test_returns_statistics(self):
        """Returns profile statistics from storage."""
        mixin = _make_mixin()
        _get_storage(mixin).get_profile_statistics.return_value = {
            "current_count": 10,
            "pending_count": 5,
            "archived_count": 3,
            "expiring_soon_count": 1,
        }

        response = mixin.get_profile_statistics()

        assert response.success is True
        assert response.current_count == 10
        assert response.pending_count == 5
        assert response.archived_count == 3
        assert response.expiring_soon_count == 1

    def test_storage_not_configured(self):
        """Returns zero counts when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.get_profile_statistics()

        assert response.success is True
        assert response.current_count == 0
        assert response.pending_count == 0
        assert response.msg is not None

    def test_exception_returns_failure(self):
        """Returns failure on storage exception."""
        mixin = _make_mixin()
        _get_storage(mixin).get_profile_statistics.side_effect = RuntimeError(
            "db error"
        )

        response = mixin.get_profile_statistics()

        assert response.success is False
        assert "db error" in (response.msg or "")


# ---------------------------------------------------------------------------
# upgrade_all_profiles
# ---------------------------------------------------------------------------


class TestUpgradeAllProfiles:
    @patch("reflexio.lib._profiles.ProfileGenerationService")
    def test_success(self, mock_service_cls):
        """Successful upgrade delegates to service."""
        mixin = _make_mixin()
        mock_service = MagicMock()
        mock_service.run_upgrade.return_value = UpgradeProfilesResponse(
            success=True, profiles_archived=2, profiles_promoted=3
        )
        mock_service_cls.return_value = mock_service

        response = mixin.upgrade_all_profiles()

        assert response.success is True
        assert response.profiles_archived == 2
        assert response.profiles_promoted == 3
        mock_service.run_upgrade.assert_called_once()

    def test_storage_not_configured(self):
        """Returns failure when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.upgrade_all_profiles()

        assert response.success is False
        assert response.message is not None

    @patch("reflexio.lib._profiles.ProfileGenerationService")
    def test_dict_input(self, mock_service_cls):
        """Accepts dict input and auto-converts."""
        mixin = _make_mixin()
        mock_service = MagicMock()
        mock_service.run_upgrade.return_value = UpgradeProfilesResponse(success=True)
        mock_service_cls.return_value = mock_service

        response = mixin.upgrade_all_profiles({"only_affected_users": True})

        assert response.success is True

    @patch("reflexio.lib._profiles.ProfileGenerationService")
    def test_none_input(self, mock_service_cls):
        """None request creates default UpgradeProfilesRequest."""
        mixin = _make_mixin()
        mock_service = MagicMock()
        mock_service.run_upgrade.return_value = UpgradeProfilesResponse(success=True)
        mock_service_cls.return_value = mock_service

        response = mixin.upgrade_all_profiles(None)

        assert response.success is True
        call_arg = mock_service.run_upgrade.call_args[0][0]
        assert isinstance(call_arg, UpgradeProfilesRequest)
        assert call_arg.only_affected_users is False


# ---------------------------------------------------------------------------
# downgrade_all_profiles
# ---------------------------------------------------------------------------


class TestDowngradeAllProfiles:
    @patch("reflexio.lib._profiles.ProfileGenerationService")
    def test_success(self, mock_service_cls):
        """Successful downgrade delegates to service."""
        mixin = _make_mixin()
        mock_service = MagicMock()
        mock_service.run_downgrade.return_value = DowngradeProfilesResponse(
            success=True, profiles_demoted=2, profiles_restored=3
        )
        mock_service_cls.return_value = mock_service

        response = mixin.downgrade_all_profiles()

        assert response.success is True
        assert response.profiles_demoted == 2
        assert response.profiles_restored == 3
        mock_service.run_downgrade.assert_called_once()

    def test_storage_not_configured(self):
        """Returns failure when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.downgrade_all_profiles()

        assert response.success is False
        assert response.message is not None

    @patch("reflexio.lib._profiles.ProfileGenerationService")
    def test_dict_input(self, mock_service_cls):
        """Accepts dict input and auto-converts."""
        mixin = _make_mixin()
        mock_service = MagicMock()
        mock_service.run_downgrade.return_value = DowngradeProfilesResponse(
            success=True
        )
        mock_service_cls.return_value = mock_service

        response = mixin.downgrade_all_profiles({"only_affected_users": True})

        assert response.success is True

    @patch("reflexio.lib._profiles.ProfileGenerationService")
    def test_none_input(self, mock_service_cls):
        """None request creates default DowngradeProfilesRequest."""
        mixin = _make_mixin()
        mock_service = MagicMock()
        mock_service.run_downgrade.return_value = DowngradeProfilesResponse(
            success=True
        )
        mock_service_cls.return_value = mock_service

        response = mixin.downgrade_all_profiles(None)

        assert response.success is True
        call_arg = mock_service.run_downgrade.call_args[0][0]
        assert isinstance(call_arg, DowngradeProfilesRequest)
        assert call_arg.only_affected_users is False


# ---------------------------------------------------------------------------
# update_user_profile
# ---------------------------------------------------------------------------


class TestUpdateUserProfile:
    def test_updates_content(self):
        """Applies content update and calls storage.update_user_profile_by_id."""
        mixin = _make_mixin()
        existing = _sample_profile(profile_id="p1", user_id="user1", content="old")
        _get_storage(mixin).get_user_profile.return_value = [existing]

        response = mixin.update_user_profile(
            UpdateUserProfileRequest(
                user_id="user1", profile_id="p1", content="new content"
            )
        )

        assert response.success is True
        storage = _get_storage(mixin)
        storage.update_user_profile_by_id.assert_called_once()
        _user_id, _profile_id, new_profile = (
            storage.update_user_profile_by_id.call_args[0]
        )
        assert _user_id == "user1"
        assert _profile_id == "p1"
        assert new_profile.content == "new content"

    def test_updates_custom_features(self):
        """Applies custom_features update while preserving content."""
        mixin = _make_mixin()
        existing = _sample_profile(profile_id="p1", user_id="user1", content="original")
        _get_storage(mixin).get_user_profile.return_value = [existing]

        response = mixin.update_user_profile(
            UpdateUserProfileRequest(
                user_id="user1",
                profile_id="p1",
                custom_features={"tier": "pro"},
            )
        )

        assert response.success is True
        _, _, new_profile = _get_storage(mixin).update_user_profile_by_id.call_args[0]
        assert new_profile.content == "original"
        assert new_profile.custom_features == {"tier": "pro"}

    def test_profile_not_found_returns_failure(self):
        """Returns success=False with descriptive msg when no matching profile."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_profile.return_value = []

        response = mixin.update_user_profile(
            UpdateUserProfileRequest(user_id="missing", profile_id="p1")
        )

        assert response.success is False
        assert "not found" in (response.msg or "").lower()
        _get_storage(mixin).update_user_profile_by_id.assert_not_called()

    def test_profile_mismatch_returns_failure(self):
        """Does not update when user has profiles but profile_id doesn't match."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_profile.return_value = [
            _sample_profile(profile_id="other", user_id="user1")
        ]

        response = mixin.update_user_profile(
            UpdateUserProfileRequest(user_id="user1", profile_id="p1")
        )

        assert response.success is False
        _get_storage(mixin).update_user_profile_by_id.assert_not_called()

    def test_storage_not_configured(self):
        """Returns success=False when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.update_user_profile(
            UpdateUserProfileRequest(user_id="u", profile_id="p")
        )

        assert response.success is False
        assert response.msg == STORAGE_NOT_CONFIGURED_MSG

    def test_dict_input(self):
        """Accepts dict input and auto-converts to UpdateUserProfileRequest."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_profile.return_value = [
            _sample_profile(profile_id="p1", user_id="user1")
        ]

        response = mixin.update_user_profile(
            {"user_id": "user1", "profile_id": "p1", "content": "updated"}
        )

        assert response.success is True
