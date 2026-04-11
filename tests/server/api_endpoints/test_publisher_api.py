"""Tests for publisher_api endpoint functions.

Verifies that each function correctly delegates to the Reflexio instance
obtained via get_reflexio, and handles validation failures and exceptions.
"""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.api_endpoints.publisher_api import (
    add_agent_playbook,
    add_user_interaction,
    add_user_playbook,
    add_user_profile,
    delete_agent_playbook,
    delete_agent_playbooks_by_ids_bulk,
    delete_all_interactions_bulk,
    delete_all_playbooks_bulk,
    delete_all_profiles_bulk,
    delete_profiles_by_ids,
    delete_request,
    delete_requests_by_ids,
    delete_session,
    delete_user_interaction,
    delete_user_playbook,
    delete_user_playbooks_by_ids_bulk,
    delete_user_profile,
    run_playbook_aggregation,
    update_agent_playbook_status,
)

MODULE = "reflexio.server.api_endpoints.publisher_api"
ORG_ID = "test-org"


@pytest.fixture
def mock_reflexio():
    with patch(f"{MODULE}.get_reflexio") as mock_get:
        reflexio = MagicMock()
        mock_get.return_value = reflexio
        yield reflexio


# ------------------------------------------------------------------
# Add operations
# ------------------------------------------------------------------


class TestAddUserInteraction:
    @patch(f"{MODULE}.validate_publish_user_interaction_request")
    def test_delegates_on_valid_request(self, mock_validate, mock_reflexio):
        mock_validate.return_value = (True, "")
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.publish_interaction.return_value = expected

        result = add_user_interaction(ORG_ID, request)

        mock_reflexio.publish_interaction.assert_called_once_with(request=request)
        assert result is expected

    @patch(f"{MODULE}.validate_publish_user_interaction_request")
    def test_returns_failure_on_validation_error(self, mock_validate, mock_reflexio):
        mock_validate.return_value = (False, "No interaction data provided")
        request = MagicMock()

        result = add_user_interaction(ORG_ID, request)

        assert result.success is False
        assert result.message == "No interaction data provided"
        mock_reflexio.publish_interaction.assert_not_called()


class TestAddUserPlaybook:
    def test_delegates_to_reflexio(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.add_user_playbook.return_value = expected

        result = add_user_playbook(ORG_ID, request)

        mock_reflexio.add_user_playbook.assert_called_once_with(request=request)
        assert result is expected


class TestAddAgentPlaybook:
    def test_delegates_to_reflexio(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.add_agent_playbook.return_value = expected

        result = add_agent_playbook(ORG_ID, request)

        mock_reflexio.add_agent_playbook.assert_called_once_with(request=request)
        assert result is expected


class TestAddUserProfile:
    def test_add_user_profile_calls_reflexio_with_request(self, mock_reflexio):
        """POST /add_user_profile forwards request to Reflexio.add_user_profile."""
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.add_user_profile.return_value = expected

        result = add_user_profile(ORG_ID, request)

        mock_reflexio.add_user_profile.assert_called_once_with(request=request)
        assert result is expected

    def test_add_user_profile_uses_org_id_to_get_reflexio(self):
        """The endpoint looks up the Reflexio instance by org_id."""
        with patch(f"{MODULE}.get_reflexio") as mock_get:
            reflexio = MagicMock()
            mock_get.return_value = reflexio
            request = MagicMock()

            add_user_profile(ORG_ID, request)

            mock_get.assert_called_once_with(org_id=ORG_ID)
            reflexio.add_user_profile.assert_called_once_with(request=request)


# ------------------------------------------------------------------
# Delete operations (with exception handling)
# ------------------------------------------------------------------


class TestDeleteUserProfile:
    @patch(f"{MODULE}.validate_delete_user_profile_request")
    def test_delegates_on_valid_request(self, mock_validate, mock_reflexio):
        mock_validate.return_value = (True, "")
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_profile.return_value = expected

        result = delete_user_profile(ORG_ID, request)

        mock_reflexio.delete_profile.assert_called_once_with(request)
        assert result is expected

    @patch(f"{MODULE}.validate_delete_user_profile_request")
    def test_returns_failure_on_validation_error(self, mock_validate, mock_reflexio):
        mock_validate.return_value = (False, "Profile id or search query is required")
        request = MagicMock()

        result = delete_user_profile(ORG_ID, request)

        assert result.success is False
        assert result.message == "Profile id or search query is required"
        mock_reflexio.delete_profile.assert_not_called()

    @patch(f"{MODULE}.validate_delete_user_profile_request")
    def test_returns_failure_on_exception(self, mock_validate, mock_reflexio):
        mock_validate.return_value = (True, "")
        mock_reflexio.delete_profile.side_effect = RuntimeError("storage error")
        request = MagicMock()

        result = delete_user_profile(ORG_ID, request)

        assert result.success is False
        assert result.message == "storage error"


class TestDeleteUserInteraction:
    def test_delegates_to_reflexio(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_interaction.return_value = expected

        result = delete_user_interaction(ORG_ID, request)

        mock_reflexio.delete_interaction.assert_called_once_with(request)
        assert result is expected

    def test_returns_failure_on_exception(self, mock_reflexio):
        mock_reflexio.delete_interaction.side_effect = RuntimeError("not found")
        request = MagicMock()

        result = delete_user_interaction(ORG_ID, request)

        assert result.success is False
        assert result.message == "not found"


class TestDeleteRequest:
    def test_delegates_to_reflexio(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_request.return_value = expected

        result = delete_request(ORG_ID, request)

        mock_reflexio.delete_request.assert_called_once_with(request)
        assert result is expected

    def test_returns_failure_on_exception(self, mock_reflexio):
        mock_reflexio.delete_request.side_effect = RuntimeError("db error")
        request = MagicMock()

        result = delete_request(ORG_ID, request)

        assert result.success is False
        assert result.message == "db error"


class TestDeleteSession:
    def test_delegates_to_reflexio(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_session.return_value = expected

        result = delete_session(ORG_ID, request)

        mock_reflexio.delete_session.assert_called_once_with(request)
        assert result is expected

    def test_returns_failure_on_exception(self, mock_reflexio):
        mock_reflexio.delete_session.side_effect = RuntimeError("timeout")
        request = MagicMock()

        result = delete_session(ORG_ID, request)

        assert result.success is False
        assert result.message == "timeout"


class TestDeleteAgentPlaybook:
    def test_delegates_to_reflexio(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_agent_playbook.return_value = expected

        result = delete_agent_playbook(ORG_ID, request)

        mock_reflexio.delete_agent_playbook.assert_called_once_with(request)
        assert result is expected

    def test_returns_failure_on_exception(self, mock_reflexio):
        mock_reflexio.delete_agent_playbook.side_effect = RuntimeError("not found")
        request = MagicMock()

        result = delete_agent_playbook(ORG_ID, request)

        assert result.success is False
        assert result.message == "not found"


class TestDeleteUserPlaybook:
    def test_delegates_to_reflexio(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_user_playbook.return_value = expected

        result = delete_user_playbook(ORG_ID, request)

        mock_reflexio.delete_user_playbook.assert_called_once_with(request)
        assert result is expected

    def test_returns_failure_on_exception(self, mock_reflexio):
        mock_reflexio.delete_user_playbook.side_effect = RuntimeError("not found")
        request = MagicMock()

        result = delete_user_playbook(ORG_ID, request)

        assert result.success is False
        assert result.message == "not found"


# ------------------------------------------------------------------
# Bulk delete operations (no exception handling)
# ------------------------------------------------------------------


class TestBulkDeletes:
    def test_delete_all_interactions_bulk(self, mock_reflexio):
        expected = MagicMock()
        mock_reflexio.delete_all_interactions_bulk.return_value = expected

        result = delete_all_interactions_bulk(ORG_ID)

        mock_reflexio.delete_all_interactions_bulk.assert_called_once_with()
        assert result is expected

    def test_delete_all_profiles_bulk(self, mock_reflexio):
        expected = MagicMock()
        mock_reflexio.delete_all_profiles_bulk.return_value = expected

        result = delete_all_profiles_bulk(ORG_ID)

        mock_reflexio.delete_all_profiles_bulk.assert_called_once_with()
        assert result is expected

    def test_delete_all_playbooks_bulk(self, mock_reflexio):
        expected = MagicMock()
        mock_reflexio.delete_all_playbooks_bulk.return_value = expected

        result = delete_all_playbooks_bulk(ORG_ID)

        mock_reflexio.delete_all_playbooks_bulk.assert_called_once_with()
        assert result is expected

    def test_delete_requests_by_ids(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_requests_by_ids.return_value = expected

        result = delete_requests_by_ids(ORG_ID, request)

        mock_reflexio.delete_requests_by_ids.assert_called_once_with(request)
        assert result is expected

    def test_delete_profiles_by_ids(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_profiles_by_ids.return_value = expected

        result = delete_profiles_by_ids(ORG_ID, request)

        mock_reflexio.delete_profiles_by_ids.assert_called_once_with(request)
        assert result is expected

    def test_delete_agent_playbooks_by_ids_bulk(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_agent_playbooks_by_ids_bulk.return_value = expected

        result = delete_agent_playbooks_by_ids_bulk(ORG_ID, request)

        mock_reflexio.delete_agent_playbooks_by_ids_bulk.assert_called_once_with(
            request
        )
        assert result is expected

    def test_delete_user_playbooks_by_ids_bulk(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.delete_user_playbooks_by_ids_bulk.return_value = expected

        result = delete_user_playbooks_by_ids_bulk(ORG_ID, request)

        mock_reflexio.delete_user_playbooks_by_ids_bulk.assert_called_once_with(request)
        assert result is expected


# ------------------------------------------------------------------
# Processing operations
# ------------------------------------------------------------------


class TestRunPlaybookAggregation:
    def test_returns_success_on_completion(self, mock_reflexio):
        request = MagicMock()
        request.agent_version = "v1"
        request.playbook_name = "quality"

        result = run_playbook_aggregation(ORG_ID, request)

        mock_reflexio.run_playbook_aggregation.assert_called_once_with("v1", "quality")
        assert result.success is True

    def test_returns_failure_on_exception(self, mock_reflexio):
        request = MagicMock()
        request.agent_version = "v1"
        request.playbook_name = "quality"
        mock_reflexio.run_playbook_aggregation.side_effect = RuntimeError("llm error")

        result = run_playbook_aggregation(ORG_ID, request)

        assert result.success is False
        assert result.message == "llm error"


class TestUpdatePlaybookStatus:
    def test_delegates_to_reflexio(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.update_agent_playbook_status.return_value = expected

        result = update_agent_playbook_status(ORG_ID, request)

        mock_reflexio.update_agent_playbook_status.assert_called_once_with(request)
        assert result is expected

    def test_returns_failure_on_exception(self, mock_reflexio):
        mock_reflexio.update_agent_playbook_status.side_effect = RuntimeError(
            "update error"
        )
        request = MagicMock()

        result = update_agent_playbook_status(ORG_ID, request)

        assert result.success is False
        assert result.msg == "update error"
