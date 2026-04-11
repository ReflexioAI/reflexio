"""Unit tests for QueryReformulator pre-retrieval module.

Tests cover all four public/internal methods: rewrite(), search(),
_extract_reformulated_query(), and _format_conversation_context().
"""

import unittest
from unittest.mock import Mock

from reflexio.models.api_schema.retriever_schema import ConversationTurn
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.pre_retrieval._query_reformulator import (
    QueryReformulator,
)


def _make_reformulator(mock_llm=None, mock_pm=None):
    """Create a QueryReformulator with mocked dependencies.

    Args:
        mock_llm: Optional pre-configured LiteLLMClient mock.
        mock_pm: Optional pre-configured PromptManager mock.

    Returns:
        QueryReformulator: Instance with mocked deps ready for testing.
    """
    llm = mock_llm or Mock(spec=LiteLLMClient)
    pm = mock_pm or Mock(spec=PromptManager)
    pm.render_prompt.return_value = "test prompt"
    return QueryReformulator(llm_client=llm, prompt_manager=pm)


class TestRewrite(unittest.TestCase):
    """Tests for the rewrite() method."""

    def test_without_conversation_history_llm_reformulates_query(self):
        """LLM reformulates the query when no conversation history is given."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.return_value = "expanded search query"

        result = reformulator.rewrite("search query")

        self.assertEqual(result.standalone_query, "expanded search query")
        reformulator.prompt_manager.render_prompt.assert_called_once()
        call_args = reformulator.prompt_manager.render_prompt.call_args
        variables = call_args[0][1]
        self.assertEqual(variables["conversation_context_block"], "")

    def test_with_conversation_history_context_included_in_prompt(self):
        """Conversation context is formatted and included in prompt variables."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.return_value = "standalone query"
        history = [
            ConversationTurn(role="user", content="hello"),
            ConversationTurn(role="agent", content="hi there"),
        ]

        result = reformulator.rewrite("follow up question", history)

        self.assertEqual(result.standalone_query, "standalone query")
        call_args = reformulator.prompt_manager.render_prompt.call_args
        variables = call_args[0][1]
        self.assertIn("Conversation context:", variables["conversation_context_block"])
        self.assertIn("[user]: hello", variables["conversation_context_block"])

    def test_llm_failure_falls_back_to_original_query(self):
        """When the LLM raises an exception, the original query is returned."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.side_effect = RuntimeError("LLM down")

        result = reformulator.rewrite("original query")

        self.assertEqual(result.standalone_query, "original query")

    def test_empty_llm_response_falls_back_to_original(self):
        """When the LLM returns a non-string (None), original query is used."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.return_value = None

        result = reformulator.rewrite("original query")

        self.assertEqual(result.standalone_query, "original query")

    def test_invalid_unsafe_llm_output_falls_back_to_original(self):
        """When the LLM returns an unsafe phrase, the original query is used."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.return_value = (
            "Here is the reformulated query for you"
        )

        result = reformulator.rewrite("original query")

        self.assertEqual(result.standalone_query, "original query")


class TestExtractReformulatedQuery(unittest.TestCase):
    """Tests for the _extract_reformulated_query() class method."""

    def test_strips_reformulated_query_prefix(self):
        """Strips leading 'Reformulated query:' text via quote/whitespace handling."""
        # The method strips quotes/backticks and normalizes whitespace.
        # A prefix like "Reformulated query:" stays, but quotes are stripped.
        result = QueryReformulator._extract_reformulated_query(
            '"improved search terms"'
        )
        self.assertEqual(result, "improved search terms")

    def test_takes_first_line_from_multiline_output(self):
        """Only the first non-empty line is used from multi-line LLM output."""
        result = QueryReformulator._extract_reformulated_query(
            "first line query\nsecond line\nthird line"
        )
        self.assertEqual(result, "first line query")

    def test_returns_none_for_empty_input(self):
        """Empty or whitespace-only input returns None."""
        self.assertIsNone(QueryReformulator._extract_reformulated_query(""))
        self.assertIsNone(QueryReformulator._extract_reformulated_query("   "))
        self.assertIsNone(QueryReformulator._extract_reformulated_query(None))

    def test_returns_none_for_too_long_output(self):
        """Output exceeding MAX_REFORMULATION_LENGTH (512) returns None."""
        long_output = "a " * 300  # 600 chars
        self.assertIsNone(QueryReformulator._extract_reformulated_query(long_output))

    def test_returns_none_for_unsafe_phrases(self):
        """Output containing unsafe phrases returns None."""
        unsafe_inputs = [
            "here is the reformulated query",
            "Output: some query text",
            '```json {"query": "test"}',
            "I cannot reformulate this query",
            "I can't help with that",
            'json { "key": "val" }',
        ]
        for text in unsafe_inputs:
            with self.subTest(text=text):
                self.assertIsNone(
                    QueryReformulator._extract_reformulated_query(text),
                    f"Expected None for unsafe input: {text!r}",
                )

    def test_returns_none_for_non_alphanumeric_only(self):
        """Input with no alphanumeric characters returns None."""
        self.assertIsNone(QueryReformulator._extract_reformulated_query("--- ??? !!!"))

    def test_strips_quotes_and_backticks(self):
        """Leading/trailing quotes, backticks, and extra spaces are stripped."""
        self.assertEqual(
            QueryReformulator._extract_reformulated_query('`"search term"`'),
            "search term",
        )
        self.assertEqual(
            QueryReformulator._extract_reformulated_query("'single quoted'"),
            "single quoted",
        )

    def test_returns_none_for_curly_brackets(self):
        """JSON-like output with curly braces triggers the 'json {' unsafe phrase."""
        result = QueryReformulator._extract_reformulated_query('json {"query": "test"}')
        self.assertIsNone(result)

    def test_returns_none_for_square_brackets_with_unsafe_prefix(self):
        """Code-injection attempts with unsafe prefixes return None."""
        # Single-line input so the ```json phrase is checked as-is
        result = QueryReformulator._extract_reformulated_query(
            '```json ["item1", "item2"]'
        )
        self.assertIsNone(result)

    def test_valid_clean_query_returned(self):
        """A valid, clean query string is returned as-is after normalization."""
        result = QueryReformulator._extract_reformulated_query(
            "  how to configure logging  "
        )
        self.assertEqual(result, "how to configure logging")

    def test_multiline_with_empty_first_lines(self):
        """Multi-line output with leading blank lines picks first non-empty line."""
        result = QueryReformulator._extract_reformulated_query(
            "\n\n  \nactual query here\nanother line"
        )
        self.assertEqual(result, "actual query here")


class TestFormatConversationContext(unittest.TestCase):
    """Tests for the _format_conversation_context() static method."""

    def test_formats_turns_with_role_content(self):
        """Each turn is formatted as [role]: content."""
        history = [
            ConversationTurn(role="user", content="What is X?"),
            ConversationTurn(role="agent", content="X is a tool."),
        ]
        result = QueryReformulator._format_conversation_context(history)
        self.assertIn("[user]: What is X?", result)
        self.assertIn("[agent]: X is a tool.", result)

    def test_returns_empty_string_for_none(self):
        """None input returns empty string."""
        self.assertEqual(QueryReformulator._format_conversation_context(None), "")

    def test_returns_empty_string_for_empty_list(self):
        """Empty list returns empty string."""
        self.assertEqual(QueryReformulator._format_conversation_context([]), "")

    def test_truncates_at_max_conversation_chars(self):
        """Output is truncated once total characters exceed MAX_CONVERSATION_CHARS."""
        long_content = "x" * 2000
        history = [
            ConversationTurn(role="user", content=long_content),
            ConversationTurn(role="agent", content=long_content),
            ConversationTurn(role="user", content=long_content),
        ]
        result = QueryReformulator._format_conversation_context(history)
        # First turn is ~2008 chars ("[user]: " + 2000), fits under 4000
        # Second turn is ~2010 chars, would push past 4000
        self.assertIn("[user]:", result)
        lines = result.strip().splitlines()
        self.assertLessEqual(len(lines), 2)

    def test_handles_dict_format_turns(self):
        """Dictionary-format turns (not ConversationTurn objects) are handled."""
        history = [
            {"role": "user", "content": "dict-based question"},
            {"role": "agent", "content": "dict-based answer"},
        ]
        result = QueryReformulator._format_conversation_context(history)
        self.assertIn("[user]: dict-based question", result)
        self.assertIn("[agent]: dict-based answer", result)

    def test_limits_to_max_conversation_turns(self):
        """Only the last MAX_CONVERSATION_TURNS turns are used."""
        history = [
            ConversationTurn(role="user", content=f"message {i}") for i in range(20)
        ]
        result = QueryReformulator._format_conversation_context(history)
        # MAX_CONVERSATION_TURNS is 10, so only last 10 should appear
        self.assertNotIn("message 0", result)
        self.assertNotIn("message 9", result)
        self.assertIn("message 10", result)
        self.assertIn("message 19", result)


class TestSearch(unittest.TestCase):
    """Tests for the search() method."""

    def test_calls_search_fn_with_reformulated_query(self):
        """search_fn receives the reformulated query, not the original."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.return_value = "reformulated"
        search_fn = Mock(return_value=["result1", "result2"])

        result = reformulator.search("original", search_fn)

        search_fn.assert_called_once_with("reformulated")
        self.assertEqual(result.standalone_query, "reformulated")
        self.assertEqual(result.items, ["result1", "result2"])

    def test_deduplicates_results_with_dedup_key(self):
        """Duplicate items are removed based on the dedup_key function."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.return_value = "query"

        items = [
            {"id": "a", "text": "first"},
            {"id": "b", "text": "second"},
            {"id": "a", "text": "duplicate of first"},
        ]
        search_fn = Mock(return_value=items)

        def dedup_key(item):
            return item["id"]

        result = reformulator.search("query", search_fn, dedup_key=dedup_key)

        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0]["id"], "a")
        self.assertEqual(result.items[1]["id"], "b")

    def test_handles_search_fn_failure_gracefully(self):
        """When search_fn raises, result.items is an empty list."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.return_value = "query"
        search_fn = Mock(side_effect=RuntimeError("search failed"))

        result = reformulator.search("original", search_fn)

        self.assertEqual(result.items, [])
        self.assertEqual(result.standalone_query, "query")

    def test_returns_all_results_without_dedup_key(self):
        """Without a dedup_key, all results are returned including duplicates."""
        reformulator = _make_reformulator()
        reformulator.llm_client.generate_response.return_value = "query"

        items = ["a", "b", "a", "c"]
        search_fn = Mock(return_value=items)

        result = reformulator.search("query", search_fn)

        self.assertEqual(result.items, ["a", "b", "a", "c"])


if __name__ == "__main__":
    unittest.main()
