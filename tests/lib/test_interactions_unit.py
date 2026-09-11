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
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server.services.generation_service import GenerationServiceResult
from reflexio.test_support.typing_helpers import as_mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin(*, storage_configured: bool = True) -> InteractionsMixin:
    """Create an InteractionsMixin instance with mocked internals.

    ``publish_interaction`` reports what the durable stream recorded for the
    request via ``extraction_counts``, and its coverage via
    ``extraction_status``. Both are wired to a quiet default here: covered, with
    nothing extracted. They must return real dicts, not bare ``MagicMock``s —
    ``status["status"]`` on a ``MagicMock`` yields another ``MagicMock``, which
    then fails ``PublishUserInteractionResponse`` validation and turns a
    successful publish into ``success=False``.

    Tests that want to exercise the count or coverage paths override on the
    returned mock.
    """
    mixin = object.__new__(InteractionsMixin)
    mock_storage = MagicMock()
    mock_storage.extraction_counts.return_value = {"profile": 0, "playbook": 0}
    mock_storage.extraction_status.return_value = {
        "status": "done",
        "reason": "covered",
    }

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
    return as_mock(mixin.request_context.storage)


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
            session_id="test_session",
            interaction_data_list=[InteractionData(role="User", content="hi")],
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
            session_id="test_session",
            interaction_data_list=[InteractionData(role="User", content="hi")],
        )
        response = mixin.publish_interaction(request)

        assert response.success is True
        # request_id is propagated from the generation service
        assert response.request_id == "req-abc-123"
        # Counts come from the stream's per-request tally, 0 by fixture default
        assert response.profiles_added == 0
        assert response.playbooks_added == 0
        assert response.learning_status == "done"
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
                "session_id": "test_session",
                "interaction_data_list": [{"role": "User", "content": "hi"}],
            }
        )

        assert response.success is True

    @patch("reflexio.lib._interactions.GenerationService")
    def test_reports_extraction_counts(self, mock_gen_cls):
        """profiles_added / playbooks_added come from the stream's tally.

        Replaces an earlier before→after delta over ``count_all_profiles`` /
        ``count_user_playbooks``. Those snapshots were a whole-store diff, so
        concurrent extraction for another user could be attributed to this
        request; ``extraction_counts`` is scoped to the request itself.
        """
        mixin = _make_mixin()
        storage = _get_storage(mixin)
        as_mock(storage.extraction_counts).return_value = {
            "profile": 3,
            "playbook": 3,
        }
        mock_gen_instance = MagicMock()
        mock_gen_instance.run.return_value = GenerationServiceResult(
            request_id="req-1",
            warnings=[],
        )
        mock_gen_cls.return_value = mock_gen_instance

        response = mixin.publish_interaction(
            PublishUserInteractionRequest(
                user_id="user1",
                session_id="test_session",
                interaction_data_list=[InteractionData(role="User", content="hi")],
            )
        )

        assert response.success is True
        assert response.profiles_added == 3
        assert response.playbooks_added == 3

    @patch("reflexio.lib._interactions.GenerationService")
    def test_publish_succeeds_when_coverage_read_fails(self, mock_gen_cls):
        """A failed coverage read must not report a committed publish as failed.

        ``extraction_status`` / ``extraction_counts`` run AFTER the interactions
        are durably committed and only describe that write. They share the
        publish's ``try``, so an unguarded raise would return ``success=False``
        for work already on disk and invite the caller to retry it. The counts
        and coverage fields drop out instead.
        """
        mixin = _make_mixin()
        storage = _get_storage(mixin)
        as_mock(storage.extraction_status).side_effect = RuntimeError("db down")
        mock_gen_instance = MagicMock()
        mock_gen_instance.run.return_value = GenerationServiceResult(
            request_id="req-ok",
            warnings=[],
        )
        mock_gen_cls.return_value = mock_gen_instance

        response = mixin.publish_interaction(
            PublishUserInteractionRequest(
                user_id="user1",
                session_id="test_session",
                interaction_data_list=[InteractionData(role="User", content="hi")],
            )
        )

        assert response.success is True
        assert response.request_id == "req-ok"
        # Omitted rather than guessed — an absent count is honest, a 0 is not.
        assert response.profiles_added is None
        assert response.playbooks_added is None
        assert response.learning_status is None
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
            session_id="test_session",
            interaction_data_list=[InteractionData(role="User", content="hi")],
        )
        response = mixin.publish_interaction(request)

        assert response.success is False
        assert "service error" in (response.message or "")
