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


def _make_mixin_with_sqlite(storage) -> AgentPlaybookMixin:
    """Create an AgentPlaybookMixin backed by a real SQLiteStorage instance."""
    mixin = object.__new__(AgentPlaybookMixin)
    mock_request_context = MagicMock()
    mock_request_context.org_id = storage.org_id
    mock_request_context.storage = storage
    mock_request_context.is_storage_configured.return_value = True
    mixin.request_context = mock_request_context
    return mixin


class TestGetPlaybookAggregationChangeLogs:
    def test_returns_reconstructed_change_logs(self, tmp_path):
        """Delegates to reconstruct_playbook_aggregation_change_log (Track B repoint).

        Seeds aggregate + status_change lineage events via the storage API and
        asserts the mixin returns the reconstruction: correct added/removed
        snapshots, run_mode, and updated_agent_playbooks=[].
        """
        from reflexio.models.api_schema.domain.entities import LineageEvent
        from reflexio.models.api_schema.domain.enums import Status
        from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

        s = SQLiteStorage(org_id="org-unit-pb", db_path=str(tmp_path / "test.db"))
        s.migrate()

        # Seed playbooks
        pb_old = AgentPlaybook(
            playbook_name="test_fb", agent_version="v1", content="old content"
        )
        pb_new = AgentPlaybook(
            playbook_name="test_fb", agent_version="v1", content="new content"
        )
        old_saved = s.save_agent_playbooks([pb_old])
        old_id = old_saved[0].agent_playbook_id
        # Tombstone the old one
        s.conn.execute(
            "UPDATE agent_playbooks SET status = ? WHERE agent_playbook_id = ?",
            (Status.SUPERSEDED.value, old_id),
        )
        s.conn.commit()
        new_saved = s.save_agent_playbooks([pb_new])
        new_id = new_saved[0].agent_playbook_id

        req_id = "run-unit-1"
        # Aggregate event (adds new playbook)
        s.append_lineage_event(
            LineageEvent(
                org_id=s.org_id,
                entity_type="agent_playbook",
                entity_id=str(new_id),
                op="aggregate",
                prov_relation="wasDerivedFrom",
                source_ids=[],
                actor="aggregator",
                request_id=req_id,
                reason="aggregate:incremental",
            )
        )
        # Status-change / superseded event (removes old playbook)
        s.append_lineage_event(
            LineageEvent(
                org_id=s.org_id,
                entity_type="agent_playbook",
                entity_id=str(old_id),
                op="status_change",
                prov_relation="wasInvalidatedBy",
                source_ids=[],
                actor="aggregator",
                request_id=req_id,
                reason="None->superseded",
                from_status=None,
                to_status=Status.SUPERSEDED.value,
                status_namespace="lifecycle_status",
            )
        )

        mixin = _make_mixin_with_sqlite(s)
        response = mixin.get_playbook_aggregation_change_logs(
            playbook_name="test_fb", agent_version="v1"
        )

        assert response.success is True
        assert len(response.change_logs) == 1
        log = response.change_logs[0]
        assert log.run_mode == "incremental"
        assert {snap.content for snap in log.added_agent_playbooks} == {"new content"}
        assert {snap.content for snap in log.removed_agent_playbooks} == {"old content"}
        assert log.updated_agent_playbooks == []

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.get_playbook_aggregation_change_logs(
            playbook_name="test_fb", agent_version="v1"
        )

        assert response.success is True
        assert response.change_logs == []

    def test_filters_by_playbook_name_and_agent_version(self, tmp_path):
        """get_playbook_aggregation_change_logs returns only logs matching the
        requested playbook_name + agent_version and excludes all others.

        Seeds two distinct (playbook_name, agent_version) combinations via
        lineage events, calls the mixin with only one combination, and asserts
        only that combination's log is returned.
        """
        from reflexio.models.api_schema.domain.entities import LineageEvent
        from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

        s = SQLiteStorage(org_id="org-unit-filter", db_path=str(tmp_path / "filter.db"))
        s.migrate()

        # --- Combination A: playbook_name="fb_A", agent_version="v1" ---
        pb_a = AgentPlaybook(
            playbook_name="fb_A", agent_version="v1", content="content A"
        )
        saved_a = s.save_agent_playbooks([pb_a])
        id_a = saved_a[0].agent_playbook_id
        s.append_lineage_event(
            LineageEvent(
                org_id=s.org_id,
                entity_type="agent_playbook",
                entity_id=str(id_a),
                op="aggregate",
                prov_relation="wasDerivedFrom",
                source_ids=[],
                actor="aggregator",
                request_id="run-A",
                reason="aggregate:incremental",
            )
        )

        # --- Combination B: playbook_name="fb_B", agent_version="v2" ---
        pb_b = AgentPlaybook(
            playbook_name="fb_B", agent_version="v2", content="content B"
        )
        saved_b = s.save_agent_playbooks([pb_b])
        id_b = saved_b[0].agent_playbook_id
        s.append_lineage_event(
            LineageEvent(
                org_id=s.org_id,
                entity_type="agent_playbook",
                entity_id=str(id_b),
                op="aggregate",
                prov_relation="wasDerivedFrom",
                source_ids=[],
                actor="aggregator",
                request_id="run-B",
                reason="aggregate:incremental",
            )
        )

        mixin = _make_mixin_with_sqlite(s)
        # Request only combination A
        response = mixin.get_playbook_aggregation_change_logs(
            playbook_name="fb_A", agent_version="v1"
        )

        assert response.success is True
        assert len(response.change_logs) == 1, (
            f"Expected 1 log for fb_A/v1, got {len(response.change_logs)}"
        )
        log = response.change_logs[0]
        assert log.playbook_name == "fb_A"
        assert log.agent_version == "v1"
        # Combination B must be excluded
        contents = {snap.content for snap in log.added_agent_playbooks}
        assert "content B" not in contents
        assert "content A" in contents


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
            tags=None,
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
