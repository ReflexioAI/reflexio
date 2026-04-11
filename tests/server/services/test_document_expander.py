"""Unit tests for DocumentExpander pre-retrieval module.

Tests cover the public expand() method, the internal _parse_expansion_json()
and _format_expanded_terms() static methods.
"""

import json
import unittest
from unittest.mock import Mock

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.pre_retrieval._document_expander import (
    DocumentExpander,
    ExpansionResult,
)


def _make_expander(mock_llm=None, mock_pm=None):
    """Create a DocumentExpander with mocked dependencies.

    Args:
        mock_llm: Optional pre-configured LiteLLMClient mock.
        mock_pm: Optional pre-configured PromptManager mock.

    Returns:
        DocumentExpander: Instance with mocked deps ready for testing.
    """
    llm = mock_llm or Mock(spec=LiteLLMClient)
    pm = mock_pm or Mock(spec=PromptManager)
    pm.render_prompt.return_value = "test prompt"
    return DocumentExpander(llm_client=llm, prompt_manager=pm)


# ------------------------------------------------------------------
# TestExpand
# ------------------------------------------------------------------


class TestExpand(unittest.TestCase):
    """Tests for the expand() method."""

    def test_expand_returns_structured_result(self):
        """LLM returns valid JSON; verify ExpansionResult fields."""
        expander = _make_expander()
        llm_json = json.dumps({"backup": ["sync", "replication"]})
        expander.llm_client.generate_response.return_value = llm_json

        result = expander.expand("We need a backup strategy")

        self.assertIsInstance(result, ExpansionResult)
        self.assertEqual(result.expansions, {"backup": ["sync", "replication"]})
        self.assertEqual(result.expanded_text, "backup, sync, replication")

    def test_expand_empty_content_returns_empty(self):
        """Empty string content triggers LLM call but produces empty result on empty LLM output."""
        expander = _make_expander()
        # LLM returns non-string (None) for empty content
        expander.llm_client.generate_response.return_value = None

        result = expander.expand("")

        self.assertEqual(result.expansions, {})
        self.assertEqual(result.expanded_text, "")

    def test_expand_llm_failure_returns_empty(self):
        """When the LLM raises an exception, expand() returns empty result."""
        expander = _make_expander()
        expander.llm_client.generate_response.side_effect = RuntimeError("LLM down")

        result = expander.expand("some content")

        self.assertIsInstance(result, ExpansionResult)
        self.assertEqual(result.expansions, {})
        self.assertEqual(result.expanded_text, "")

    def test_expand_invalid_json_returns_empty(self):
        """When the LLM returns non-JSON text, expand() returns empty result."""
        expander = _make_expander()
        expander.llm_client.generate_response.return_value = (
            "Here are some synonyms for you"
        )

        result = expander.expand("some content")

        self.assertIsInstance(result, ExpansionResult)
        self.assertEqual(result.expansions, {})
        self.assertEqual(result.expanded_text, "")


# ------------------------------------------------------------------
# TestParseExpansionJson
# ------------------------------------------------------------------


class TestParseExpansionJson(unittest.TestCase):
    """Tests for the _parse_expansion_json() static method."""

    def test_parse_valid_json(self):
        """Direct JSON string parses into a dict."""
        raw = json.dumps({"deploy": ["release", "ship"]})
        result = DocumentExpander._parse_expansion_json(raw)
        self.assertEqual(result, {"deploy": ["release", "ship"]})

    def test_parse_code_fenced_json(self):
        """JSON wrapped in a code fence is extracted correctly."""
        raw = '```json\n{"deploy": ["release", "ship"]}\n```'
        result = DocumentExpander._parse_expansion_json(raw)
        self.assertEqual(result, {"deploy": ["release", "ship"]})

    def test_parse_filters_non_string_values(self):
        """Non-string items inside synonym lists are filtered out."""
        raw = json.dumps({"key": [1, 2, "valid"]})
        result = DocumentExpander._parse_expansion_json(raw)
        self.assertEqual(result, {"key": ["valid"]})

    def test_parse_filters_non_list_values(self):
        """Non-list values (e.g. plain strings) are dropped entirely."""
        raw = json.dumps({"key": "value", "good": ["a", "b"]})
        result = DocumentExpander._parse_expansion_json(raw)
        self.assertNotIn("key", result)
        self.assertEqual(result, {"good": ["a", "b"]})

    def test_parse_invalid_json_returns_empty(self):
        """Non-JSON input returns an empty dict."""
        result = DocumentExpander._parse_expansion_json("not json at all")
        self.assertEqual(result, {})


# ------------------------------------------------------------------
# TestFormatExpandedTerms
# ------------------------------------------------------------------


class TestFormatExpandedTerms(unittest.TestCase):
    """Tests for the _format_expanded_terms() static method."""

    def test_format_single_group(self):
        """Single expansion group formats as comma-separated string."""
        result = DocumentExpander._format_expanded_terms(
            {"backup": ["sync", "replication"]}
        )
        self.assertEqual(result, "backup, sync, replication")

    def test_format_multiple_groups(self):
        """Multiple expansion groups are joined by semicolons."""
        expansions = {"backup": ["sync"], "deploy": ["release"]}
        result = DocumentExpander._format_expanded_terms(expansions)
        # Dict ordering is guaranteed in Python 3.7+
        self.assertEqual(result, "backup, sync; deploy, release")

    def test_format_empty_dict(self):
        """Empty expansion dict returns an empty string."""
        result = DocumentExpander._format_expanded_terms({})
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
