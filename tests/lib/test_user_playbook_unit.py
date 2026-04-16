"""Unit tests for UserPlaybookMixin.

Tests get_user_playbooks, add_user_playbook, search_user_playbooks,
delete_user_playbook, upgrade_all_user_playbooks, and downgrade_all_user_playbooks
with mocked storage and services.
"""

from unittest.mock import MagicMock, patch

from reflexio.lib._user_playbook import UserPlaybookMixin
from reflexio.models.api_schema.retriever_schema import (
    GetUserPlaybooksRequest,
    SearchUserPlaybookRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AddUserPlaybookRequest,
    DeleteUserPlaybookRequest,
    DeleteUserPlaybooksByIdsRequest,
    DowngradeUserPlaybooksResponse,
    UpgradeUserPlaybooksResponse,
    UserPlaybook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin(*, storage_configured: bool = True) -> UserPlaybookMixin:
    """Create a UserPlaybookMixin instance with mocked internals."""
    mixin = object.__new__(UserPlaybookMixin)
    mock_storage = MagicMock()

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured

    mixin.request_context = mock_request_context
    mixin.llm_client = MagicMock()
    return mixin


def _get_storage(mixin: UserPlaybookMixin) -> MagicMock:
    return mixin.request_context.storage


def _sample_user_playbook(**overrides) -> UserPlaybook:
    defaults = {
        "agent_version": "v1",
        "request_id": "req-1",
        "playbook_name": "test_fb",
        "content": "test content",
    }
    defaults.update(overrides)
    return UserPlaybook(**defaults)


# ---------------------------------------------------------------------------
# get_user_playbooks
# ---------------------------------------------------------------------------


class TestGetUserPlaybooks:
    def test_returns_list(self):
        """Successful retrieval returns user playbooks from storage."""
        mixin = _make_mixin()
        sample = _sample_user_playbook()
        _get_storage(mixin).get_user_playbooks.return_value = [sample]

        request = GetUserPlaybooksRequest(limit=10)
        response = mixin.get_user_playbooks(request)

        assert response.success is True
        assert len(response.user_playbooks) == 1

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = GetUserPlaybooksRequest()
        response = mixin.get_user_playbooks(request)

        assert response.success is True
        assert response.user_playbooks == []
        assert response.msg is not None


# ---------------------------------------------------------------------------
# add_user_playbook
# ---------------------------------------------------------------------------


class TestAddUserPlaybook:
    def test_saves_playbook_directly(self):
        """Playbooks are saved directly to storage without normalization."""
        mixin = _make_mixin()
        rf = _sample_user_playbook()
        request = AddUserPlaybookRequest(user_playbooks=[rf])

        response = mixin.add_user_playbook(request)

        assert response.success is True
        assert response.added_count == 1
        _get_storage(mixin).save_user_playbooks.assert_called_once()

    def test_preserves_top_level_fields(self):
        """Top-level structured fields are preserved as-is."""
        mixin = _make_mixin()
        rf = _sample_user_playbook(trigger="when user asks")
        request = AddUserPlaybookRequest(user_playbooks=[rf])

        response = mixin.add_user_playbook(request)

        assert response.success is True
        saved = _get_storage(mixin).save_user_playbooks.call_args[0][0]
        assert saved[0].trigger == "when user asks"

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)
        rf = _sample_user_playbook()
        request = AddUserPlaybookRequest(user_playbooks=[rf])

        response = mixin.add_user_playbook(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# search_user_playbooks
# ---------------------------------------------------------------------------


class TestSearchUserPlaybooks:
    def test_basic_query(self):
        """Delegates search to storage and returns results."""
        mixin = _make_mixin()
        sample = _sample_user_playbook()
        _get_storage(mixin).search_user_playbooks.return_value = [sample]

        request = SearchUserPlaybookRequest(query="test")
        response = mixin.search_user_playbooks(request)

        assert response.success is True
        assert len(response.user_playbooks) == 1
        _get_storage(mixin).search_user_playbooks.assert_called_once()

    def test_with_filters(self):
        """Passes filter parameters through to storage."""
        mixin = _make_mixin()
        _get_storage(mixin).search_user_playbooks.return_value = []

        request = SearchUserPlaybookRequest(
            query="test",
            playbook_name="my_playbook",
            agent_version="v2",
        )
        response = mixin.search_user_playbooks(request)

        assert response.success is True
        # OS passes the full request object to storage.search_user_playbooks
        call_args = _get_storage(mixin).search_user_playbooks.call_args[0]
        passed_request = call_args[0]
        assert passed_request.playbook_name == "my_playbook"
        assert passed_request.agent_version == "v2"

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = SearchUserPlaybookRequest(query="test")
        response = mixin.search_user_playbooks(request)

        assert response.success is True
        assert response.user_playbooks == []


# ---------------------------------------------------------------------------
# delete_user_playbook
# ---------------------------------------------------------------------------


class TestDeleteUserPlaybook:
    def test_by_id(self):
        """Deletes a user playbook by ID."""
        mixin = _make_mixin()

        request = DeleteUserPlaybookRequest(user_playbook_id=42)
        response = mixin.delete_user_playbook(request)

        assert response.success is True
        _get_storage(mixin).delete_user_playbook.assert_called_once_with(42)

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteUserPlaybookRequest(user_playbook_id=42)
        response = mixin.delete_user_playbook(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# upgrade_all_user_playbooks / downgrade_all_user_playbooks
# ---------------------------------------------------------------------------


class TestUpgradeDowngradeUserPlaybooks:
    def test_upgrade_delegates_to_service(self):
        """Upgrade creates PlaybookGenerationService and delegates."""
        mixin = _make_mixin()

        mock_response = UpgradeUserPlaybooksResponse(
            success=True,
            user_playbooks_deleted=1,
            user_playbooks_archived=2,
            user_playbooks_promoted=3,
        )

        with patch(
            "reflexio.lib._user_playbook.PlaybookGenerationService"
        ) as mock_svc_cls:
            mock_svc_cls.return_value.run_upgrade.return_value = mock_response

            response = mixin.upgrade_all_user_playbooks()

        assert response.success is True
        assert response.user_playbooks_promoted == 3
        mock_svc_cls.return_value.run_upgrade.assert_called_once()

    def test_downgrade_delegates_to_service(self):
        """Downgrade creates PlaybookGenerationService and delegates."""
        mixin = _make_mixin()

        mock_response = DowngradeUserPlaybooksResponse(
            success=True,
            user_playbooks_demoted=2,
            user_playbooks_restored=3,
        )

        with patch(
            "reflexio.lib._user_playbook.PlaybookGenerationService"
        ) as mock_svc_cls:
            mock_svc_cls.return_value.run_downgrade.return_value = mock_response

            response = mixin.downgrade_all_user_playbooks()

        assert response.success is True
        assert response.user_playbooks_restored == 3
        mock_svc_cls.return_value.run_downgrade.assert_called_once()

    def test_upgrade_storage_not_configured(self):
        """Upgrade fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.upgrade_all_user_playbooks()

        assert response.success is False

    def test_downgrade_storage_not_configured(self):
        """Downgrade fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.downgrade_all_user_playbooks()

        assert response.success is False

    def test_upgrade_with_dict_input(self):
        """Upgrade accepts dict input and converts to request."""
        mixin = _make_mixin()

        mock_response = UpgradeUserPlaybooksResponse(success=True)

        with patch(
            "reflexio.lib._user_playbook.PlaybookGenerationService"
        ) as mock_svc_cls:
            mock_svc_cls.return_value.run_upgrade.return_value = mock_response

            response = mixin.upgrade_all_user_playbooks(
                {"playbook_name": "my_playbook"}
            )

        assert response.success is True

    def test_downgrade_with_dict_input(self):
        """Downgrade accepts dict input and converts to request."""
        mixin = _make_mixin()

        mock_response = DowngradeUserPlaybooksResponse(success=True)

        with patch(
            "reflexio.lib._user_playbook.PlaybookGenerationService"
        ) as mock_svc_cls:
            mock_svc_cls.return_value.run_downgrade.return_value = mock_response

            response = mixin.downgrade_all_user_playbooks(
                {"playbook_name": "my_playbook"}
            )

        assert response.success is True

    def test_upgrade_with_none_input(self):
        """Upgrade with None converts to default request (line 217->221)."""
        mixin = _make_mixin()

        mock_response = UpgradeUserPlaybooksResponse(success=True)

        with patch(
            "reflexio.lib._user_playbook.PlaybookGenerationService"
        ) as mock_svc_cls:
            mock_svc_cls.return_value.run_upgrade.return_value = mock_response

            response = mixin.upgrade_all_user_playbooks(None)

        assert response.success is True
        mock_svc_cls.return_value.run_upgrade.assert_called_once()

    def test_downgrade_with_none_input(self):
        """Downgrade with None converts to default request (line 255->259)."""
        mixin = _make_mixin()

        mock_response = DowngradeUserPlaybooksResponse(success=True)

        with patch(
            "reflexio.lib._user_playbook.PlaybookGenerationService"
        ) as mock_svc_cls:
            mock_svc_cls.return_value.run_downgrade.return_value = mock_response

            response = mixin.downgrade_all_user_playbooks(None)

        assert response.success is True
        mock_svc_cls.return_value.run_downgrade.assert_called_once()


# ---------------------------------------------------------------------------
# get_user_playbooks - dict input and error paths (lines 51, 60-61)
# ---------------------------------------------------------------------------


class TestGetUserPlaybooksDictAndError:
    def test_dict_input(self):
        """Accepts dict input and auto-converts (line 51)."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_playbooks.return_value = []

        response = mixin.get_user_playbooks({"limit": 5, "playbook_name": "my_fb"})

        assert response.success is True
        _get_storage(mixin).get_user_playbooks.assert_called_once()

    def test_storage_exception(self):
        """Returns failure on storage exception (lines 60-61)."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_playbooks.side_effect = RuntimeError("db error")

        request = GetUserPlaybooksRequest(limit=10)
        response = mixin.get_user_playbooks(request)

        assert response.success is False
        assert "db error" in (response.msg or "")


# ---------------------------------------------------------------------------
# add_user_playbook - dict input and error path (lines 80, 118-119)
# ---------------------------------------------------------------------------


class TestAddUserPlaybookDictAndError:
    def test_dict_input(self):
        """Accepts dict input and auto-converts (line 80)."""
        mixin = _make_mixin()
        rf = _sample_user_playbook()

        response = mixin.add_user_playbook({"user_playbooks": [rf.model_dump()]})

        assert response.success is True

    def test_storage_exception(self):
        """Returns failure on storage exception (lines 118-119)."""
        mixin = _make_mixin()
        _get_storage(mixin).save_user_playbooks.side_effect = RuntimeError("save error")

        rf = _sample_user_playbook()
        request = AddUserPlaybookRequest(user_playbooks=[rf])

        response = mixin.add_user_playbook(request)

        assert response.success is False
        assert "save error" in (response.message or "")


# ---------------------------------------------------------------------------
# search_user_playbooks - dict input and error path (lines 138, 148-149)
# ---------------------------------------------------------------------------


class TestSearchUserPlaybooksDictAndError:
    def test_dict_input(self):
        """Accepts dict input and auto-converts (line 138)."""
        mixin = _make_mixin()
        _get_storage(mixin).search_user_playbooks.return_value = []

        response = mixin.search_user_playbooks({"query": "test"})

        assert response.success is True

    def test_storage_exception(self):
        """Returns failure on storage exception (lines 148-149)."""
        mixin = _make_mixin()
        _get_storage(mixin).search_user_playbooks.side_effect = RuntimeError(
            "search error"
        )

        request = SearchUserPlaybookRequest(query="test")
        response = mixin.search_user_playbooks(request)

        assert response.success is False
        assert "search error" in (response.msg or "")

    def test_query_reformulation_applied(self):
        """Query reformulation modifies the request when enabled (line 145)."""
        mixin = _make_mixin()
        _get_storage(mixin).search_user_playbooks.return_value = []

        # Mock the _reformulate_query to return a reformulated query
        mixin._reformulate_query = MagicMock(return_value="rewritten query")

        request = SearchUserPlaybookRequest(query="original", enable_reformulation=True)
        response = mixin.search_user_playbooks(request)

        assert response.success is True
        # OS passes the full request object to storage.search_user_playbooks
        call_args = _get_storage(mixin).search_user_playbooks.call_args[0]
        passed_request = call_args[0]
        assert passed_request.query == "rewritten query"


# ---------------------------------------------------------------------------
# delete_user_playbook - dict input (line 167)
# ---------------------------------------------------------------------------


class TestDeleteUserPlaybookDict:
    def test_dict_input(self):
        """Accepts dict input and auto-converts (line 167)."""
        mixin = _make_mixin()

        response = mixin.delete_user_playbook({"user_playbook_id": 99})

        assert response.success is True
        _get_storage(mixin).delete_user_playbook.assert_called_once_with(99)


# ---------------------------------------------------------------------------
# delete_user_playbooks_by_ids_bulk - dict input and storage_not_configured (lines 184-189)
# ---------------------------------------------------------------------------


class TestDeleteUserPlaybooksByIdsBulk:
    def test_deletes_by_ids(self):
        """Deletes user playbooks by IDs and returns count."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_user_playbooks_by_ids.return_value = 3

        request = DeleteUserPlaybooksByIdsRequest(user_playbook_ids=[1, 2, 3])
        response = mixin.delete_user_playbooks_by_ids_bulk(request)

        assert response.success is True
        assert response.deleted_count == 3

    def test_dict_input(self):
        """Accepts dict input and auto-converts (line 184)."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_user_playbooks_by_ids.return_value = 2

        response = mixin.delete_user_playbooks_by_ids_bulk(
            {"user_playbook_ids": [10, 20]}
        )

        assert response.success is True
        assert response.deleted_count == 2

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteUserPlaybooksByIdsRequest(user_playbook_ids=[1])
        response = mixin.delete_user_playbooks_by_ids_bulk(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# delete_all_user_playbooks_bulk (user only — does NOT cascade to agent)
# ---------------------------------------------------------------------------


class TestDeleteAllUserPlaybooksBulk:
    def test_deletes_only_user_playbooks(self):
        """Calls storage.delete_all_user_playbooks, not agent playbooks."""
        mixin = _make_mixin()

        response = mixin.delete_all_user_playbooks_bulk()

        assert response.success is True
        _get_storage(mixin).delete_all_user_playbooks.assert_called_once()
        _get_storage(mixin).delete_all_agent_playbooks.assert_not_called()

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.delete_all_user_playbooks_bulk()

        assert response.success is False
