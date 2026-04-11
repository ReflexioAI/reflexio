"""Unit tests for the unified search service.

Tests the critical orchestration logic: empty query, embedding failure,
and reformulated_query propagation.
"""

import unittest
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.retriever_schema import (
    UnifiedSearchRequest,
)
from reflexio.server.services.pre_retrieval import ReformulationResult
from reflexio.server.services.unified_search_service import (
    run_unified_search,
)


def _mock_storage(embedding=None):
    """Create a mock storage with configurable embedding."""
    storage = MagicMock()
    storage._get_embedding.return_value = embedding or [0.1] * 1536
    # Storage search methods return empty lists by default
    storage.search_user_profile.return_value = []
    storage.search_agent_playbooks.return_value = []
    storage.search_user_playbooks.return_value = []
    return storage


class TestRunUnifiedSearch(unittest.TestCase):
    """Tests for the top-level run_unified_search function."""

    def test_empty_query_rejected_by_validation(self):
        """Empty query is now rejected at the Pydantic validation level."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            UnifiedSearchRequest(query="")

    def test_whitespace_query_rejected_by_validation(self):
        """Whitespace-only query is rejected at the Pydantic validation level."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            UnifiedSearchRequest(query="   ")

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_embedding_failure_degrades_to_text_search(self, _reformulator_cls):
        """When embedding generation fails, should degrade to text-only search (not crash)."""
        storage = _mock_storage()
        storage._get_embedding.side_effect = RuntimeError("Embedding API down")

        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="test query"
        )

        request = UnifiedSearchRequest(query="test query")
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        storage.search_agent_playbooks.assert_called_once()

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_local_storage_without_get_embedding(self, _reformulator_cls):
        """Storage without _get_embedding should not crash and should use text-only search."""
        storage = _mock_storage()
        del storage._get_embedding  # Simulate a storage backend that lacks this method

        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="test query"
        )

        request = UnifiedSearchRequest(query="test query")
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        storage.search_agent_playbooks.assert_called_once()

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_reformulated_query_populated_when_changed(self, _reformulator_cls):
        """reformulated_query field should only be set when query was actually reformulated."""
        expanded = ReformulationResult(
            standalone_query="agent failed OR error to refund OR return"
        )
        _reformulator_cls.return_value.rewrite.return_value = expanded

        storage = _mock_storage()
        request = UnifiedSearchRequest(
            query="agent failed to refund", enable_reformulation=True
        )
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertEqual(
            result.reformulated_query,
            "agent failed OR error to refund OR return",
        )

    @patch("reflexio.server.services.unified_search_service.QueryReformulator")
    def test_reformulated_query_none_when_unchanged(self, _reformulator_cls):
        """reformulated_query should be None when query was not reformulated."""
        _reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="same query"
        )

        storage = _mock_storage()
        request = UnifiedSearchRequest(query="same query")
        result = run_unified_search(
            request=request,
            org_id="test-org",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

        self.assertTrue(result.success)
        self.assertIsNone(result.reformulated_query)


if __name__ == "__main__":
    unittest.main()
