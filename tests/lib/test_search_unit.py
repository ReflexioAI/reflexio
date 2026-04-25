"""Unit tests for SearchMixin.

Tests get_agent_success_evaluation_results, get_requests, and
unified_search with mocked storage.
"""

from unittest.mock import MagicMock, patch

from reflexio.lib._search import SearchMixin
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.retriever_schema import (
    GetAgentSuccessEvaluationResultsRequest,
    GetRequestsRequest,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.models.api_schema.service_schemas import Interaction, Request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin(*, storage_configured: bool = True) -> SearchMixin:
    """Create a SearchMixin instance with mocked internals."""
    mixin = object.__new__(SearchMixin)
    mock_storage = MagicMock()

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured

    mixin.request_context = mock_request_context
    return mixin


def _get_storage(mixin: SearchMixin) -> MagicMock:
    return mixin.request_context.storage


# ---------------------------------------------------------------------------
# get_agent_success_evaluation_results
# ---------------------------------------------------------------------------


class TestGetAgentSuccessEvaluationResults:
    def test_returns_results(self):
        """Returns evaluation results from storage."""
        from reflexio.models.api_schema.service_schemas import (
            AgentSuccessEvaluationResult,
        )

        mixin = _make_mixin()
        sample_result = AgentSuccessEvaluationResult(
            agent_version="v1",
            session_id="sess_1",
            is_success=True,
        )
        _get_storage(mixin).get_agent_success_evaluation_results.return_value = [
            sample_result
        ]

        request = GetAgentSuccessEvaluationResultsRequest(limit=50)
        response = mixin.get_agent_success_evaluation_results(request)

        assert response.success is True
        assert len(response.agent_success_evaluation_results) == 1

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = GetAgentSuccessEvaluationResultsRequest()
        response = mixin.get_agent_success_evaluation_results(request)

        assert response.success is True
        assert response.agent_success_evaluation_results == []

    def test_dict_input(self):
        """Accepts dict input."""
        mixin = _make_mixin()
        _get_storage(mixin).get_agent_success_evaluation_results.return_value = []

        response = mixin.get_agent_success_evaluation_results(
            {"limit": 10, "agent_version": "v2"}
        )

        assert response.success is True

    def test_storage_exception(self):
        """Returns failure on storage exception."""
        mixin = _make_mixin()
        _get_storage(
            mixin
        ).get_agent_success_evaluation_results.side_effect = RuntimeError("db error")

        request = GetAgentSuccessEvaluationResultsRequest()
        response = mixin.get_agent_success_evaluation_results(request)

        assert response.success is False
        assert "db error" in (response.msg or "")


# ---------------------------------------------------------------------------
# get_requests
# ---------------------------------------------------------------------------


def _make_request_interaction(
    session_id: str, request_id: str
) -> RequestInteractionDataModel:
    """Build a RequestInteractionDataModel for testing."""
    mock_request = MagicMock(spec=Request)
    mock_interaction = MagicMock(spec=Interaction)
    return RequestInteractionDataModel(
        session_id=session_id,
        request=mock_request,
        interactions=[mock_interaction],
    )


class TestGetRequests:
    def test_session_grouping(self):
        """Groups results by session_id."""
        mixin = _make_mixin()
        rid1 = _make_request_interaction("session_a", "req_1")
        rid2 = _make_request_interaction("session_b", "req_2")
        _get_storage(mixin).get_sessions.return_value = {
            "session_a": [rid1],
            "session_b": [rid2],
        }

        request = GetRequestsRequest(top_k=10)
        response = mixin.get_requests(request)

        assert response.success is True
        assert len(response.sessions) == 2
        session_ids = {s.session_id for s in response.sessions}
        assert "session_a" in session_ids
        assert "session_b" in session_ids

    def test_has_more_flag_true(self):
        """has_more is True when total returned equals top_k."""
        mixin = _make_mixin()
        items = [_make_request_interaction("s", f"req_{i}") for i in range(5)]
        _get_storage(mixin).get_sessions.return_value = {"s": items}

        request = GetRequestsRequest(top_k=5)
        response = mixin.get_requests(request)

        assert response.success is True
        assert response.has_more is True

    def test_has_more_flag_false(self):
        """has_more is False when total returned is less than top_k."""
        mixin = _make_mixin()
        items = [_make_request_interaction("s", "req_1")]
        _get_storage(mixin).get_sessions.return_value = {"s": items}

        request = GetRequestsRequest(top_k=10)
        response = mixin.get_requests(request)

        assert response.success is True
        assert response.has_more is False

    def test_storage_not_configured(self):
        """Returns empty list when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = GetRequestsRequest()
        response = mixin.get_requests(request)

        assert response.success is True
        assert response.sessions == []

    def test_storage_exception(self):
        """Returns failure on storage exception."""
        mixin = _make_mixin()
        _get_storage(mixin).get_sessions.side_effect = RuntimeError("timeout")

        request = GetRequestsRequest(top_k=10)
        response = mixin.get_requests(request)

        assert response.success is False
        assert "timeout" in (response.msg or "")


# ---------------------------------------------------------------------------
# unified_search
# ---------------------------------------------------------------------------


class TestUnifiedSearch:
    def test_delegation_to_service(self):
        """Delegates to run_unified_search service function."""
        mixin = _make_mixin()
        mixin.llm_client = MagicMock()
        mock_config = MagicMock()
        mock_config.llm_config = None
        mock_config.search_backend = "classic"
        mixin.request_context.configurator.get_config.return_value = mock_config

        expected_response = UnifiedSearchResponse(success=True)

        with patch(
            "reflexio.server.services.unified_search_service.run_unified_search",
            return_value=expected_response,
        ) as mock_run:
            request = UnifiedSearchRequest(query="test query")
            response = mixin.unified_search(request, org_id="org_1")

        assert response.success is True
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["org_id"] == "org_1"
        assert call_kwargs["llm_client"] is mixin.llm_client
        assert "pre_retrieval_model_name" in call_kwargs

    def test_storage_not_configured(self):
        """Returns success with message when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = UnifiedSearchRequest(query="test query")
        response = mixin.unified_search(request, org_id="org_1")

        assert response.success is True
        assert response.msg is not None

    def test_dispatches_to_agentic_when_search_backend_agentic(self):
        """When config.search_backend == 'agentic', AgenticSearchService.search runs.

        Pre-fix bug: lib/_search.py hardcoded run_unified_search regardless of
        config — agentic SearchAgent was implemented but unreachable from the
        public /api/search path. This test pins the dispatch.
        """
        mixin = _make_mixin()
        mixin.llm_client = MagicMock()
        mock_config = MagicMock()
        mock_config.llm_config = None
        mock_config.search_backend = "agentic"
        mixin.request_context.configurator.get_config.return_value = mock_config

        expected_response = UnifiedSearchResponse(success=True, agent_answer="hi")

        with (
            patch(
                "reflexio.server.services.search.agentic_search_service.AgenticSearchService"
            ) as mock_agentic_cls,
            patch(
                "reflexio.server.services.unified_search_service.run_unified_search"
            ) as mock_run_unified,
        ):
            mock_agentic_inst = MagicMock()
            mock_agentic_inst.search.return_value = expected_response
            mock_agentic_cls.return_value = mock_agentic_inst

            request = UnifiedSearchRequest(query="test query")
            response = mixin.unified_search(request, org_id="org_1")

        assert response is expected_response
        mock_agentic_cls.assert_called_once_with(
            llm_client=mixin.llm_client,
            request_context=mixin.request_context,
        )
        mock_agentic_inst.search.assert_called_once_with(request)
        mock_run_unified.assert_not_called()

    def test_dispatches_to_classic_when_search_backend_classic(self):
        """When config.search_backend == 'classic', run_unified_search runs.

        Belt-and-suspenders: ensures the agentic branch doesn't accidentally
        capture the classic path on the default value.
        """
        mixin = _make_mixin()
        mixin.llm_client = MagicMock()
        mock_config = MagicMock()
        mock_config.llm_config = None
        mock_config.search_backend = "classic"
        mixin.request_context.configurator.get_config.return_value = mock_config

        expected_response = UnifiedSearchResponse(success=True)

        with (
            patch(
                "reflexio.server.services.unified_search_service.run_unified_search",
                return_value=expected_response,
            ) as mock_run_unified,
            patch(
                "reflexio.server.services.search.agentic_search_service.AgenticSearchService"
            ) as mock_agentic_cls,
        ):
            request = UnifiedSearchRequest(query="test query")
            response = mixin.unified_search(request, org_id="org_1")

        assert response is expected_response
        mock_run_unified.assert_called_once()
        mock_agentic_cls.assert_not_called()
