"""Unit tests for InteractionsMixin.

Tests get_interactions, get_all_interactions, search_interactions,
delete_interaction, delete_request, delete_session, delete_all_interactions_bulk,
delete_requests_by_ids, and publish_interaction with mocked storage.
"""

import time
from unittest.mock import MagicMock, patch

from reflexio.lib._interactions import InteractionsMixin
from reflexio.models.api_schema.retriever_schema import (
    GetInteractionsRequest,
    SearchInteractionRequest,
)
from reflexio.models.api_schema.service_schemas import (
    DeleteRequestRequest,
    DeleteRequestsByIdsRequest,
    DeleteSessionRequest,
    DeleteUserInteractionRequest,
    Interaction,
    PublishUserInteractionRequest,
)
from reflexio.server.services.generation_service import GenerationServiceResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin(*, storage_configured: bool = True) -> InteractionsMixin:
    """Create an InteractionsMixin instance with mocked internals.

    ``publish_interaction`` snapshots profile + playbook totals via
    ``count_all_profiles`` / ``count_user_playbooks`` to compute
    extraction deltas, so we wire those to return 0 by default. Tests
    that want to exercise the delta path can override on the returned
    mock.
    """
    mixin = object.__new__(InteractionsMixin)
    mock_storage = MagicMock()
    mock_storage.count_all_profiles.return_value = 0
    mock_storage.count_user_playbooks.return_value = 0

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured
    mock_request_context.configurator.get_current_storage_configuration.return_value = (
        None
    )

    mixin.request_context = mock_request_context
    mixin.llm_client = MagicMock()
    return mixin


def _get_storage(mixin: InteractionsMixin) -> MagicMock:
    return mixin.request_context.storage


def _sample_interaction(**overrides) -> Interaction:
    defaults = {
        "interaction_id": 1,
        "user_id": "user1",
        "request_id": "req1",
        "created_at": int(time.time()),
        "role": "User",
        "content": "hello",
    }
    defaults.update(overrides)
    return Interaction(**defaults)


# ---------------------------------------------------------------------------
# get_interactions
# ---------------------------------------------------------------------------


class TestGetInteractions:
    def test_returns_interactions(self):
        """Successful retrieval returns interactions from storage."""
        mixin = _make_mixin()
        sample = _sample_interaction()
        _get_storage(mixin).get_user_interaction.return_value = [sample]

        request = GetInteractionsRequest(user_id="user1")
        response = mixin.get_interactions(request)

        assert response.success is True
        assert len(response.interactions) == 1

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = GetInteractionsRequest(user_id="user1")
        response = mixin.get_interactions(request)

        assert response.success is True
        assert response.interactions == []
        assert response.msg is not None

    def test_dict_input(self):
        """Accepts dict input and auto-converts."""
        mixin = _make_mixin()
        _get_storage(mixin).get_user_interaction.return_value = []

        response = mixin.get_interactions({"user_id": "user1"})

        assert response.success is True
        _get_storage(mixin).get_user_interaction.assert_called_once()

    def test_top_k_limit(self):
        """Applies top_k limit to results."""
        mixin = _make_mixin()
        now = int(time.time())
        interactions = [
            _sample_interaction(interaction_id=i, created_at=now - i) for i in range(5)
        ]
        _get_storage(mixin).get_user_interaction.return_value = interactions

        request = GetInteractionsRequest(user_id="user1", top_k=2)
        response = mixin.get_interactions(request)

        assert response.success is True
        assert len(response.interactions) == 2

    def test_sorted_by_created_at_descending(self):
        """Results are sorted by created_at in descending order."""
        mixin = _make_mixin()
        now = int(time.time())
        interactions = [
            _sample_interaction(interaction_id=1, created_at=now - 100),
            _sample_interaction(interaction_id=2, created_at=now),
            _sample_interaction(interaction_id=3, created_at=now - 50),
        ]
        _get_storage(mixin).get_user_interaction.return_value = interactions

        request = GetInteractionsRequest(user_id="user1")
        response = mixin.get_interactions(request)

        assert response.success is True
        timestamps = [i.created_at for i in response.interactions]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# get_all_interactions
# ---------------------------------------------------------------------------


class TestGetAllInteractions:
    def test_returns_all(self):
        """Returns all interactions across users."""
        mixin = _make_mixin()
        sample = _sample_interaction()
        _get_storage(mixin).get_all_interactions.return_value = [sample]

        response = mixin.get_all_interactions(limit=50)

        assert response.success is True
        assert len(response.interactions) == 1
        _get_storage(mixin).get_all_interactions.assert_called_once_with(limit=50)

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.get_all_interactions()

        assert response.success is True
        assert response.interactions == []
        assert response.msg is not None


# ---------------------------------------------------------------------------
# search_interactions
# ---------------------------------------------------------------------------


class TestSearchInteractions:
    def test_query_delegation(self):
        """Delegates search to storage."""
        mixin = _make_mixin()
        sample = _sample_interaction()
        _get_storage(mixin).search_interaction.return_value = [sample]

        request = SearchInteractionRequest(user_id="user1", query="hello")
        response = mixin.search_interactions(request)

        assert response.success is True
        assert len(response.interactions) == 1
        _get_storage(mixin).search_interaction.assert_called_once()

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = SearchInteractionRequest(user_id="user1", query="hello")
        response = mixin.search_interactions(request)

        assert response.success is True
        assert response.interactions == []
        assert response.msg is not None

    def test_dict_input(self):
        """Accepts dict input and auto-converts."""
        mixin = _make_mixin()
        _get_storage(mixin).search_interaction.return_value = []

        response = mixin.search_interactions({"user_id": "user1", "query": "test"})

        assert response.success is True


# ---------------------------------------------------------------------------
# delete_interaction
# ---------------------------------------------------------------------------


class TestDeleteInteraction:
    def test_single_delete(self):
        """Deletes an interaction by user_id and interaction_id."""
        mixin = _make_mixin()

        request = DeleteUserInteractionRequest(user_id="user1", interaction_id=42)
        response = mixin.delete_interaction(request)

        assert response.success is True
        _get_storage(mixin).delete_user_interaction.assert_called_once()

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()

        response = mixin.delete_interaction({"user_id": "user1", "interaction_id": 42})

        assert response.success is True

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteUserInteractionRequest(user_id="user1", interaction_id=42)
        response = mixin.delete_interaction(request)

        assert response.success is False

    def test_storage_exception(self):
        """Returns failure on storage exception."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_user_interaction.side_effect = RuntimeError(
            "db error"
        )

        request = DeleteUserInteractionRequest(user_id="user1", interaction_id=42)
        response = mixin.delete_interaction(request)

        assert response.success is False
        assert "db error" in (response.message or "")


# ---------------------------------------------------------------------------
# delete_request
# ---------------------------------------------------------------------------


class TestDeleteRequest:
    def test_delete_by_request_id(self):
        """Deletes a request by request_id."""
        mixin = _make_mixin()

        request = DeleteRequestRequest(request_id="req1")
        response = mixin.delete_request(request)

        assert response.success is True
        _get_storage(mixin).delete_request.assert_called_once_with("req1")

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()

        response = mixin.delete_request({"request_id": "req1"})

        assert response.success is True

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteRequestRequest(request_id="req1")
        response = mixin.delete_request(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_by_session_id(self):
        """Deletes a session and returns deleted count."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_session.return_value = 5

        request = DeleteSessionRequest(session_id="sess1")
        response = mixin.delete_session(request)

        assert response.success is True
        assert response.deleted_requests_count == 5
        _get_storage(mixin).delete_session.assert_called_once_with("sess1")

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_session.return_value = 0

        response = mixin.delete_session({"session_id": "sess1"})

        assert response.success is True

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteSessionRequest(session_id="sess1")
        response = mixin.delete_session(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# delete_all_interactions_bulk
# ---------------------------------------------------------------------------


class TestDeleteAllInteractionsBulk:
    def test_bulk_delete(self):
        """Deletes all requests/interactions."""
        mixin = _make_mixin()

        response = mixin.delete_all_interactions_bulk()

        assert response.success is True
        _get_storage(mixin).delete_all_requests.assert_called_once()

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        response = mixin.delete_all_interactions_bulk()

        assert response.success is False


# ---------------------------------------------------------------------------
# delete_requests_by_ids
# ---------------------------------------------------------------------------


class TestDeleteRequestsByIds:
    def test_delete_by_ids(self):
        """Deletes requests by their IDs."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_requests_by_ids.return_value = 3

        request = DeleteRequestsByIdsRequest(request_ids=["r1", "r2", "r3"])
        response = mixin.delete_requests_by_ids(request)

        assert response.success is True
        assert response.deleted_count == 3

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()
        _get_storage(mixin).delete_requests_by_ids.return_value = 1

        response = mixin.delete_requests_by_ids({"request_ids": ["r1"]})

        assert response.success is True

    def test_storage_not_configured(self):
        """Fails when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = DeleteRequestsByIdsRequest(request_ids=["r1"])
        response = mixin.delete_requests_by_ids(request)

        assert response.success is False


# ---------------------------------------------------------------------------
# publish_interaction
# ---------------------------------------------------------------------------


class TestPublishInteraction:
    def test_storage_not_configured(self):
        """Returns failure when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = PublishUserInteractionRequest(
            user_id="user1",
            interaction_data_list=[{"role": "User", "content": "hi"}],
        )
        response = mixin.publish_interaction(request)

        assert response.success is False
        assert response.message is not None

    @patch("reflexio.lib._interactions.GenerationService")
    def test_success(self, mock_gen_cls):
        """Successful publish returns success + populates diagnostic fields."""
        mixin = _make_mixin()
        mock_gen_instance = MagicMock()
        mock_gen_instance.run.return_value = GenerationServiceResult(
            request_id="req-abc-123",
            warnings=[],
        )
        mock_gen_cls.return_value = mock_gen_instance

        request = PublishUserInteractionRequest(
            user_id="user1",
            interaction_data_list=[{"role": "User", "content": "hi"}],
        )
        response = mixin.publish_interaction(request)

        assert response.success is True
        # request_id is propagated from the generation service
        assert response.request_id == "req-abc-123"
        # Snapshot counts default to 0 deltas since mocked storage returns []
        assert response.profiles_added == 0
        assert response.playbooks_added == 0
        mock_gen_instance.run.assert_called_once()

    @patch("reflexio.lib._interactions.GenerationService")
    def test_dict_input(self, mock_gen_cls):
        """Accepts dict input and auto-converts."""
        mixin = _make_mixin()
        mock_gen_instance = MagicMock()
        mock_gen_instance.run.return_value = GenerationServiceResult(
            request_id="req-abc-xyz",
            warnings=[],
        )
        mock_gen_cls.return_value = mock_gen_instance

        response = mixin.publish_interaction(
            {
                "user_id": "user1",
                "interaction_data_list": [{"role": "User", "content": "hi"}],
            }
        )

        assert response.success is True

    @patch("reflexio.lib._interactions.GenerationService")
    def test_reports_extraction_deltas(self, mock_gen_cls):
        """profiles_added / playbooks_added reflect the before→after delta."""
        mixin = _make_mixin()
        storage = mixin.request_context.storage
        # Storage snapshot: 2 profiles before → 5 after; 0 playbooks → 3 after
        storage.count_all_profiles.side_effect = [2, 5]
        storage.count_user_playbooks.side_effect = [0, 3]
        mock_gen_instance = MagicMock()
        mock_gen_instance.run.return_value = GenerationServiceResult(
            request_id="req-1",
            warnings=[],
        )
        mock_gen_cls.return_value = mock_gen_instance

        response = mixin.publish_interaction(
            PublishUserInteractionRequest(
                user_id="user1",
                interaction_data_list=[{"role": "User", "content": "hi"}],
            )
        )

        assert response.success is True
        assert response.profiles_added == 3
        assert response.playbooks_added == 3

    @patch("reflexio.lib._interactions.GenerationService")
    def test_publish_succeeds_when_count_fails(self, mock_gen_cls):
        """When the count thunk raises, _safe_count swallows it and deltas fall back to 0.

        A storage hiccup during the before/after snapshot must not block
        the publish from returning a successful response — the publish
        itself has nothing to do with the counters.
        """
        mixin = _make_mixin()
        storage = mixin.request_context.storage
        storage.count_all_profiles.side_effect = RuntimeError("db down")
        # count_user_playbooks still works — exercised independently
        storage.count_user_playbooks.return_value = 0
        mock_gen_instance = MagicMock()
        mock_gen_instance.run.return_value = GenerationServiceResult(
            request_id="req-ok",
            warnings=[],
        )
        mock_gen_cls.return_value = mock_gen_instance

        response = mixin.publish_interaction(
            PublishUserInteractionRequest(
                user_id="user1",
                interaction_data_list=[{"role": "User", "content": "hi"}],
            )
        )

        assert response.success is True
        # _safe_count falls back to 0 on failure, so delta is max(0, 0 - 0) = 0
        assert response.profiles_added == 0
        assert response.playbooks_added == 0
        assert response.request_id == "req-ok"

    @patch("reflexio.lib._interactions.GenerationService")
    def test_exception_returns_failure(self, mock_gen_cls):
        """Returns failure on service exception."""
        mixin = _make_mixin()
        mock_gen_instance = MagicMock()
        mock_gen_instance.run.side_effect = RuntimeError("service error")
        mock_gen_cls.return_value = mock_gen_instance

        request = PublishUserInteractionRequest(
            user_id="user1",
            interaction_data_list=[{"role": "User", "content": "hi"}],
        )
        response = mixin.publish_interaction(request)

        assert response.success is False
        assert "service error" in (response.message or "")
