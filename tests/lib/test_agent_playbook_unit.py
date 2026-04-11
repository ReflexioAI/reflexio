"""Unit tests for AgentPlaybookMixin.

Tests get_agent_playbooks, add_agent_playbook, delete_agent_playbook, search_agent_playbooks,
delete_all_agent_playbooks_bulk, and update_agent_playbook_status with mocked storage.
"""

from unittest.mock import MagicMock

from reflexio.lib._agent_playbook import AgentPlaybookMixin
from reflexio.models.api_schema.retriever_schema import (
    GetAgentPlaybooksRequest,
    SearchAgentPlaybookRequest,
    UpdatePlaybookStatusRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AddAgentPlaybookRequest,
    AgentPlaybook,
    DeleteAgentPlaybookRequest,
    DeleteAgentPlaybooksByIdsRequest,
    PlaybookStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin(*, storage_configured: bool = True) -> AgentPlaybookMixin:
    """Create a AgentPlaybookMixin instance with mocked internals."""
    mixin = object.__new__(AgentPlaybookMixin)
    mock_storage = MagicMock()

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured

    mixin.request_context = mock_request_context
    return mixin


def _get_storage(mixin: AgentPlaybookMixin) -> MagicMock:
    return mixin.request_context.storage


def _sample_agent_playbook(**overrides) -> AgentPlaybook:
    defaults = {
        "agent_version": "v1",
        "playbook_name": "test_fb",
        "content": "test playbook content",
    }
    defaults.update(overrides)
    return AgentPlaybook(**defaults)


# ---------------------------------------------------------------------------
# get_agent_playbooks
# ---------------------------------------------------------------------------


class TestGetAgentPlaybooks:
    def test_returns_agent_playbooks(self):
        """Successful retrieval returns agent playbooks from storage."""
        mixin = _make_mixin()
        sample = _sample_agent_playbook()
        _get_storage(mixin).get_agent_playbooks.return_value = [sample]

        request = GetAgentPlaybooksRequest(limit=10)
        response = mixin.get_agent_playbooks(request)

        assert response.success is True
        assert len(response.agent_playbooks) == 1

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = GetAgentPlaybooksRequest()
        response = mixin.get_agent_playbooks(request)

        assert response.success is True
        assert response.agent_playbooks == []
        assert response.msg is not None

    def test_dict_input(self):
        """Accepts dict input and auto-converts."""
        mixin = _make_mixin()
        _get_storage(mixin).get_agent_playbooks.return_value = []

        response = mixin.get_agent_playbooks({"limit": 5, "playbook_name": "my_fb"})

        assert response.success is True
        _get_storage(mixin).get_agent_playbooks.assert_called_once()


# ---------------------------------------------------------------------------
# get_playbook_aggregation_change_logs
# ---------------------------------------------------------------------------


class TestGetPlaybookAggregationChangeLogs:
    def test_returns_change_logs(self):
        """Returns change logs from storage."""
        from reflexio.models.api_schema.service_schemas import (
            PlaybookAggregationChangeLog,
        )

        mixin = _make_mixin()
        sample_log = PlaybookAggregationChangeLog(
            playbook_name="test_fb",
            agent_version="v1",
            run_mode="incremental",
        )
        _get_storage(mixin).get_playbook_aggregation_change_logs.return_value = [
            sample_log
        ]

        response = mixin.get_playbook_aggregation_change_logs(
            playbook_name="test_fb", agent_version="v1"
        )

        assert response.success is True
        assert len(response.change_logs) == 1

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.get_playbook_aggregation_change_logs(
            playbook_name="test_fb", agent_version="v1"
        )

        assert response.success is True
        assert response.change_logs == []


# ---------------------------------------------------------------------------
# add_agent_playbook
# ---------------------------------------------------------------------------


class TestAddAgentPlaybook:
    def test_normalization(self):
        """Normalizes agent playbooks and saves them."""
        mixin = _make_mixin()
        fb = _sample_agent_playbook(playbook_metadata="meta info")
        request = AddAgentPlaybookRequest(agent_playbooks=[fb])

        response = mixin.add_agent_playbook(request)

        assert response.success is True
        assert response.added_count == 1
        saved = _get_storage(mixin).save_agent_playbooks.call_args[0][0]
        assert saved[0].playbook_metadata == "meta info"
        assert saved[0].content == "test playbook content"

    def test_metadata_defaults_to_empty(self):
        """playbook_metadata defaults to empty string when not provided."""
        mixin = _make_mixin()
        # Create a AgentPlaybook without playbook_metadata; the mixin normalizes it to ""
        fb = _sample_agent_playbook()  # no metadata provided => defaults to ""
        request = AddAgentPlaybookRequest(agent_playbooks=[fb])

        response = mixin.add_agent_playbook(request)

        assert response.success is True
        saved = _get_storage(mixin).save_agent_playbooks.call_args[0][0]
        assert saved[0].playbook_metadata == ""

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)
        fb = _sample_agent_playbook()
        request = AddAgentPlaybookRequest(agent_playbooks=[fb])

        response = mixin.add_agent_playbook(request)

        assert response.success is False

    def test_storage_exception(self):
        """Returns failure on storage exception."""
        mixin = _make_mixin()
        _get_storage(mixin).save_agent_playbooks.side_effect = RuntimeError("db error")

        fb = _sample_agent_playbook()
        request = AddAgentPlaybookRequest(agent_playbooks=[fb])

        response = mixin.add_agent_playbook(request)

        assert response.success is False
        assert "db error" in (response.message or "")


# ---------------------------------------------------------------------------
# delete_agent_playbook
# ---------------------------------------------------------------------------


class TestDeleteAgentPlaybook:
    def test_single_delete(self):
        """Deletes an agent playbook by ID."""
        mixin = _make_mixin()

        request = DeleteAgentPlaybookRequest(agent_playbook_id=99)
        response = mixin.delete_agent_playbook(request)

        assert response.success is True
        _get_storage(mixin).delete_agent_playbook.assert_called_once_with(99)

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()

        response = mixin.delete_agent_playbook({"agent_playbook_id": 42})

        assert response.success is True
        _get_storage(mixin).delete_agent_playbook.assert_called_once_with(42)

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteAgentPlaybookRequest(agent_playbook_id=99)
        response = mixin.delete_agent_playbook(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# search_agent_playbooks
# ---------------------------------------------------------------------------


class TestSearchAgentPlaybooks:
    def test_query_delegation(self):
        """Delegates search to storage."""
        mixin = _make_mixin()
        sample = _sample_agent_playbook()
        _get_storage(mixin).search_agent_playbooks.return_value = [sample]

        request = SearchAgentPlaybookRequest(query="test")
        response = mixin.search_agent_playbooks(request)

        assert response.success is True
        assert len(response.agent_playbooks) == 1
        _get_storage(mixin).search_agent_playbooks.assert_called_once()

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = SearchAgentPlaybookRequest(query="test")
        response = mixin.search_agent_playbooks(request)

        assert response.success is True
        assert response.agent_playbooks == []


# ---------------------------------------------------------------------------
# delete_all_agent_playbooks_bulk (cascading delete)
# ---------------------------------------------------------------------------


class TestDeleteAllAgentPlaybooksBulk:
    def test_cascading_delete(self):
        """Deletes both agent playbooks and user playbooks."""
        mixin = _make_mixin()

        response = mixin.delete_all_playbooks_bulk()

        assert response.success is True
        _get_storage(mixin).delete_all_agent_playbooks.assert_called_once()
        _get_storage(mixin).delete_all_user_playbooks.assert_called_once()

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.delete_all_playbooks_bulk()

        assert response.success is False


# ---------------------------------------------------------------------------
# delete_all_agent_playbooks_bulk (agent only — does NOT cascade to user)
# ---------------------------------------------------------------------------


class TestDeleteAllAgentPlaybooksBulkOnly:
    def test_deletes_only_agent_playbooks(self):
        """Calls storage.delete_all_agent_playbooks, not user playbooks."""
        mixin = _make_mixin()

        response = mixin.delete_all_agent_playbooks_bulk()

        assert response.success is True
        _get_storage(mixin).delete_all_agent_playbooks.assert_called_once()
        _get_storage(mixin).delete_all_user_playbooks.assert_not_called()

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.delete_all_agent_playbooks_bulk()

        assert response.success is False


# ---------------------------------------------------------------------------
# update_agent_playbook_status
# ---------------------------------------------------------------------------


class TestUpdatePlaybookStatus:
    def test_update_status(self):
        """Updates the playbook status via storage."""
        mixin = _make_mixin()

        request = UpdatePlaybookStatusRequest(
            agent_playbook_id=10, playbook_status=PlaybookStatus.APPROVED
        )
        response = mixin.update_agent_playbook_status(request)

        assert response.success is True
        _get_storage(mixin).update_agent_playbook_status.assert_called_once_with(
            agent_playbook_id=10, playbook_status=PlaybookStatus.APPROVED
        )

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()

        response = mixin.update_agent_playbook_status(
            {"agent_playbook_id": 5, "playbook_status": "rejected"}
        )

        assert response.success is True

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = UpdatePlaybookStatusRequest(
            agent_playbook_id=10, playbook_status=PlaybookStatus.APPROVED
        )
        response = mixin.update_agent_playbook_status(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# delete_agent_playbooks_by_ids_bulk - dict input (lines 93-96)
# ---------------------------------------------------------------------------


class TestDeleteAgentPlaybooksByIdsBulk:
    def test_deletes_by_ids(self):
        """Deletes agent playbooks by IDs and returns count."""
        mixin = _make_mixin()

        request = DeleteAgentPlaybooksByIdsRequest(agent_playbook_ids=[1, 2, 3])
        response = mixin.delete_agent_playbooks_by_ids_bulk(request)

        assert response.success is True
        assert response.deleted_count == 3
        _get_storage(mixin).delete_agent_playbooks_by_ids.assert_called_once_with(
            [1, 2, 3]
        )

    def test_dict_input(self):
        """Accepts dict input and auto-converts (lines 93-94)."""
        mixin = _make_mixin()

        response = mixin.delete_agent_playbooks_by_ids_bulk(
            {"agent_playbook_ids": [10, 20]}
        )

        assert response.success is True
        assert response.deleted_count == 2
        _get_storage(mixin).delete_agent_playbooks_by_ids.assert_called_once_with(
            [10, 20]
        )

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteAgentPlaybooksByIdsRequest(agent_playbook_ids=[1])
        response = mixin.delete_agent_playbooks_by_ids_bulk(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# add_agent_playbook - dict input (line 115)
# ---------------------------------------------------------------------------


class TestAddAgentPlaybookDict:
    def test_dict_input(self):
        """Accepts dict input and auto-converts (line 115)."""
        mixin = _make_mixin()
        fb = _sample_agent_playbook()

        response = mixin.add_agent_playbook({"agent_playbooks": [fb.model_dump()]})

        assert response.success is True
        assert response.added_count == 1


# ---------------------------------------------------------------------------
# get_agent_playbooks - error path (lines 166-167)
# ---------------------------------------------------------------------------


class TestGetAgentPlaybooksError:
    def test_storage_exception(self):
        """Returns failure on storage exception (lines 166-167)."""
        mixin = _make_mixin()
        _get_storage(mixin).get_agent_playbooks.side_effect = RuntimeError("db error")

        request = GetAgentPlaybooksRequest(limit=10)
        response = mixin.get_agent_playbooks(request)

        assert response.success is False
        assert "db error" in (response.msg or "")

    def test_with_playbook_status_filter(self):
        """Passes playbook_status_filter when provided."""
        mixin = _make_mixin()
        _get_storage(mixin).get_agent_playbooks.return_value = []

        request = GetAgentPlaybooksRequest(
            limit=10,
            playbook_status_filter=PlaybookStatus.APPROVED,
        )
        response = mixin.get_agent_playbooks(request)

        assert response.success is True
        _get_storage(mixin).get_agent_playbooks.assert_called_once_with(
            limit=10,
            playbook_name=None,
            agent_version=None,
            status_filter=None,
            playbook_status_filter=[PlaybookStatus.APPROVED],
        )


# ---------------------------------------------------------------------------
# search_agent_playbooks - dict input, error path, query rewrite (lines 186, 193, 196-197)
# ---------------------------------------------------------------------------


class TestSearchAgentPlaybooksDictAndError:
    def test_dict_input(self):
        """Accepts dict input and auto-converts (line 186)."""
        mixin = _make_mixin()
        _get_storage(mixin).search_agent_playbooks.return_value = []

        response = mixin.search_agent_playbooks({"query": "test"})

        assert response.success is True

    def test_storage_exception(self):
        """Returns failure on storage exception (lines 196-197)."""
        mixin = _make_mixin()
        _get_storage(mixin).search_agent_playbooks.side_effect = RuntimeError(
            "search error"
        )

        request = SearchAgentPlaybookRequest(query="test")
        response = mixin.search_agent_playbooks(request)

        assert response.success is False
        assert "search error" in (response.msg or "")

    def test_query_reformulation_applied(self):
        """Query reformulation modifies the request when enabled (line 193)."""
        mixin = _make_mixin()
        _get_storage(mixin).search_agent_playbooks.return_value = []

        # Mock the _reformulate_query to return a reformulated query
        mixin._reformulate_query = MagicMock(return_value="rewritten query")

        request = SearchAgentPlaybookRequest(
            query="original", enable_reformulation=True
        )
        response = mixin.search_agent_playbooks(request)

        assert response.success is True
        # OS passes a SearchAgentPlaybookRequest with the rewritten query to storage
        call_args = _get_storage(mixin).search_agent_playbooks.call_args[0]
        assert call_args[0].query == "rewritten query"


# ---------------------------------------------------------------------------
# delete_agent_playbook - dict edge case
# ---------------------------------------------------------------------------


class TestDeleteAgentPlaybookDict:
    def test_dict_input_via_require_storage(self):
        """dict input through _require_storage decorator with error handling."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_agent_playbook.side_effect = RuntimeError(
            "not found"
        )

        response = mixin.delete_agent_playbook({"agent_playbook_id": 999})

        # _require_storage catches exception and returns failure
        assert response.success is False
        assert "not found" in (response.message or "")


# ---------------------------------------------------------------------------
# update_agent_playbook_status - error path via _require_storage
# ---------------------------------------------------------------------------


class TestUpdatePlaybookStatusError:
    def test_storage_exception(self):
        """_require_storage catches storage exception and returns failure."""
        mixin = _make_mixin()
        _get_storage(mixin).update_agent_playbook_status.side_effect = RuntimeError(
            "update error"
        )

        request = UpdatePlaybookStatusRequest(
            agent_playbook_id=10, playbook_status=PlaybookStatus.APPROVED
        )
        response = mixin.update_agent_playbook_status(request)

        assert response.success is False
        assert "update error" in (response.msg or "")
