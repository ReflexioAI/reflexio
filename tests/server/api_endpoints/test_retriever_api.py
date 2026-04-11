from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.api_endpoints.retriever_api import (
    get_playbook_aggregation_change_logs,
    get_profile_change_logs,
    get_requests,
    get_user_interactions,
    get_user_profiles,
    search_agent_playbooks,
    search_interactions,
    search_user_playbooks,
    search_user_profiles,
    unified_search,
)

PATCH_TARGET = "reflexio.server.api_endpoints.retriever_api.get_reflexio"


@pytest.fixture()
def mock_reflexio():
    with patch(PATCH_TARGET) as mock_get:
        instance = MagicMock()
        mock_get.return_value = instance
        yield instance


class TestSearchUserProfiles:
    def test_delegates_to_search_profiles(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.search_profiles.return_value = expected

        result = search_user_profiles("org-1", request)

        mock_reflexio.search_profiles.assert_called_once_with(request)
        assert result is expected


class TestSearchInteractions:
    def test_delegates_to_search_interactions(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.search_interactions.return_value = expected

        result = search_interactions("org-1", request)

        mock_reflexio.search_interactions.assert_called_once_with(request)
        assert result is expected


class TestSearchUserPlaybooks:
    def test_delegates_to_search_user_playbooks(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.search_user_playbooks.return_value = expected

        result = search_user_playbooks("org-1", request)

        mock_reflexio.search_user_playbooks.assert_called_once_with(request)
        assert result is expected


class TestSearchFeedbacks:
    def test_delegates_to_search_agent_playbooks(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.search_agent_playbooks.return_value = expected

        result = search_agent_playbooks("org-1", request)

        mock_reflexio.search_agent_playbooks.assert_called_once_with(request)
        assert result is expected


class TestUnifiedSearch:
    def test_delegates_with_org_id_kwarg(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.unified_search.return_value = expected

        result = unified_search("org-1", request)

        mock_reflexio.unified_search.assert_called_once_with(request, org_id="org-1")
        assert result is expected


class TestGetUserProfiles:
    def test_passes_status_filter(self, mock_reflexio):
        request = MagicMock()
        request.status_filter = "active"
        expected = MagicMock()
        mock_reflexio.get_profiles.return_value = expected

        result = get_user_profiles("org-1", request)

        mock_reflexio.get_profiles.assert_called_once_with(
            request, status_filter="active"
        )
        assert result is expected


class TestGetUserInteractions:
    def test_delegates_to_get_interactions(self, mock_reflexio):
        request = MagicMock()
        expected = MagicMock()
        mock_reflexio.get_interactions.return_value = expected

        result = get_user_interactions("org-1", request)

        mock_reflexio.get_interactions.assert_called_once_with(request)
        assert result is expected


class TestGetProfileChangeLogs:
    def test_delegates_without_request(self, mock_reflexio):
        expected = MagicMock()
        mock_reflexio.get_profile_change_logs.return_value = expected

        result = get_profile_change_logs("org-1")

        mock_reflexio.get_profile_change_logs.assert_called_once_with()
        assert result is expected


class TestGetPlaybookAggregationChangeLogs:
    def test_passes_playbook_name_and_agent_version(self, mock_reflexio):
        expected = MagicMock()
        mock_reflexio.get_playbook_aggregation_change_logs.return_value = expected

        result = get_playbook_aggregation_change_logs("org-1", "latency", "v2")

        mock_reflexio.get_playbook_aggregation_change_logs.assert_called_once_with(
            playbook_name="latency", agent_version="v2"
        )
        assert result is expected


class TestGetRequests:
    def test_returns_result_directly(self, mock_reflexio):
        """get_requests returns storage result as-is; embedding stripping
        is handled at the API layer via View converters."""
        interaction_1 = MagicMock()
        interaction_1.embedding = [0.1, 0.2, 0.3]
        interaction_2 = MagicMock()
        interaction_2.embedding = [0.4, 0.5]

        request_data = MagicMock()
        request_data.interactions = [interaction_1, interaction_2]

        session = MagicMock()
        session.requests = [request_data]

        mock_result = MagicMock()
        mock_result.sessions = [session]
        mock_reflexio.get_requests.return_value = mock_result

        request = MagicMock()
        result = get_requests("org-1", request)

        mock_reflexio.get_requests.assert_called_once_with(request)
        assert result is mock_result
        # Embeddings are no longer stripped here — View converters handle that
        assert interaction_1.embedding == [0.1, 0.2, 0.3]
        assert interaction_2.embedding == [0.4, 0.5]

    def test_handles_multiple_sessions_and_requests(self, mock_reflexio):
        interaction_a = MagicMock()
        interaction_a.embedding = [1.0]
        interaction_b = MagicMock()
        interaction_b.embedding = [2.0]

        req_1 = MagicMock()
        req_1.interactions = [interaction_a]
        req_2 = MagicMock()
        req_2.interactions = [interaction_b]

        session_1 = MagicMock()
        session_1.requests = [req_1]
        session_2 = MagicMock()
        session_2.requests = [req_2]

        mock_result = MagicMock()
        mock_result.sessions = [session_1, session_2]
        mock_reflexio.get_requests.return_value = mock_result

        get_requests("org-1", MagicMock())

        # Embeddings are no longer stripped here — View converters handle that
        assert interaction_a.embedding == [1.0]
        assert interaction_b.embedding == [2.0]
