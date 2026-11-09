"""Unit tests for DocumentExpander pre-retrieval module.

Tests cover the public expand() method, the internal _parse_expansion_json()
and _format_expanded_terms() static methods.
"""

import json
import logging
import unittest
from unittest.mock import Mock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.pre_retrieval._document_expander import (
    DocumentExpander,
    ExpansionResult,
)
from reflexio.test_support.typing_helpers import as_mock

MODEL = "test/expansion-model"


def _make_expander(mock_llm=None, mock_pm=None):
    """Create a DocumentExpander with mocked dependencies.

    Args:
        mock_llm: Optional pre-configured LiteLLMClient mock.
        mock_pm: Optional pre-configured PromptManager mock.

    Returns:
        DocumentExpander: Instance with mocked deps ready for testing.
    """
    llm = mock_llm or Mock(spec=LiteLLMClient)
    llm.get_completion_model_ladder.return_value = (MODEL,)
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
        as_mock(expander.llm_client.generate_response).return_value = llm_json

        result = expander.expand("We need a backup strategy")

        self.assertIsInstance(result, ExpansionResult)
        self.assertEqual(result.expansions, {"backup": ["sync", "replication"]})
        self.assertEqual(result.expanded_text, "backup, sync, replication")

    def test_expand_empty_content_returns_empty(self):
        """Empty string content triggers LLM call but produces empty result on empty LLM output."""
        expander = _make_expander()
        # LLM returns non-string (None) for empty content
        as_mock(expander.llm_client.generate_response).return_value = None

        result = expander.expand("")

        self.assertEqual(result.expansions, {})
        self.assertEqual(result.expanded_text, "")

    def test_expand_llm_failure_returns_empty(self):
        """When the LLM raises an exception, expand() returns empty result."""
        expander = _make_expander()
        as_mock(expander.llm_client.generate_response).side_effect = RuntimeError(
            "LLM down"
        )

        result = expander.expand("some content")

        self.assertIsInstance(result, ExpansionResult)
        self.assertEqual(result.expansions, {})
        self.assertEqual(result.expanded_text, "")

    def test_expand_invalid_json_returns_empty(self):
        """When the LLM returns non-JSON text, expand() returns empty result."""
        expander = _make_expander()
        as_mock(
            expander.llm_client.generate_response
        ).return_value = "Here are some synonyms for you"

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


# ------------------------------------------------------------------
# Provider outage
# ------------------------------------------------------------------


def _open_circuit(model: str) -> None:
    from reflexio.server.llm.provider_circuit import (
        OPEN_AFTER_FAILURES,
        circuit_route,
        provider_circuit,
    )

    route = circuit_route({"model": model})
    for _ in range(OPEN_AFTER_FAILURES):
        provider_circuit.admit(route, model=model).failed()


@pytest.fixture
def clean_circuit():
    from reflexio.server.llm.provider_circuit import provider_circuit

    provider_circuit.reset()
    yield
    provider_circuit.reset()


def test_expansion_is_skipped_while_its_model_circuit_is_open(clean_circuit, caplog):
    """No prompt, no LLM call (so no LLM_TIMEOUT wait), no WARNING."""
    expander = _make_expander()
    _open_circuit(MODEL)
    caplog.clear()  # the breaker's own single WARNING for the outage

    with caplog.at_level(logging.DEBUG):
        result = expander.expand("We need a backup strategy")

    assert result == ExpansionResult()
    as_mock(expander.llm_client.generate_response).assert_not_called()
    as_mock(expander.prompt_manager.render_prompt).assert_not_called()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_expansion_runs_when_a_fallback_model_is_still_served(clean_circuit):
    """Only skip when every model of the ladder is refusing."""
    expander = _make_expander()
    as_mock(expander.llm_client.get_completion_model_ladder).return_value = (
        MODEL,
        "test/fallback-model",
    )
    as_mock(expander.llm_client.generate_response).return_value = json.dumps(
        {"backup": ["sync"]}
    )
    _open_circuit(MODEL)

    assert expander.expand("backup").expansions == {"backup": ["sync"]}


def test_expansion_runs_while_another_models_circuit_is_open(clean_circuit):
    expander = _make_expander()
    as_mock(expander.llm_client.generate_response).return_value = json.dumps(
        {"backup": ["sync"]}
    )
    _open_circuit("test/other-model")

    assert expander.expand("backup").expansions == {"backup": ["sync"]}
    as_mock(expander.llm_client.generate_response).assert_called_once()


def test_expansion_resumes_once_the_circuit_is_due_for_its_probe(
    clean_circuit, monkeypatch
):
    """Skipping must not starve the route of the probe that closes it."""
    from reflexio.server.llm import provider_circuit as circuit_module

    clock = [1000.0]
    monkeypatch.setattr(circuit_module.provider_circuit, "_clock", lambda: clock[0])
    expander = _make_expander()
    as_mock(expander.llm_client.generate_response).return_value = json.dumps(
        {"backup": ["sync"]}
    )
    _open_circuit(MODEL)
    assert expander.expand("backup") == ExpansionResult()

    clock[0] += circuit_module.FIRST_OPEN_SECONDS
    assert expander.expand("backup").expansions == {"backup": ["sync"]}


def test_a_route_due_for_its_probe_is_not_held_back_by_a_sibling(
    clean_circuit, monkeypatch
):
    """Another route of the model still open must not starve this one's probe."""
    from reflexio.server.llm import provider_circuit as circuit_module

    clock = [1000.0]
    monkeypatch.setattr(circuit_module.provider_circuit, "_clock", lambda: clock[0])
    expander = _make_expander()
    as_mock(expander.llm_client.generate_response).return_value = json.dumps(
        {"backup": ["sync"]}
    )
    _open_circuit(MODEL)  # this route: due after FIRST_OPEN_SECONDS
    clock[0] += circuit_module.FIRST_OPEN_SECONDS / 2
    sibling = circuit_module.circuit_route({"model": MODEL, "api_key": "other"})
    for _ in range(circuit_module.OPEN_AFTER_FAILURES):
        circuit_module.provider_circuit.admit(sibling, model=MODEL).failed()
    assert expander.expand("backup") == ExpansionResult()

    clock[0] += circuit_module.FIRST_OPEN_SECONDS / 2  # first due, sibling not
    assert expander.expand("backup").expansions == {"backup": ["sync"]}


def test_refused_call_logs_info_not_warning(clean_circuit, caplog):
    """A refusal that races the pre-check is the outage, already reported."""
    from reflexio.server.llm.provider_circuit import (
        ProviderCircuitOpenError,
        circuit_route,
    )

    expander = _make_expander()
    _open_circuit("test/other-model")  # pre-check passes for MODEL
    refused = ProviderCircuitOpenError("refused")
    refused.circuit_route = circuit_route({"model": "test/other-model"})
    as_mock(expander.llm_client.generate_response).side_effect = refused
    caplog.clear()  # the breaker's own single WARNING for the outage

    with caplog.at_level(logging.INFO):
        assert expander.expand("backup") == ExpansionResult()

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Document expansion skipped" in r.getMessage() for r in caplog.records)


def test_provider_failure_on_a_closed_circuit_still_warns(clean_circuit, caplog):
    """Unchanged: a timeout while the provider serves others stays a WARNING."""
    expander = _make_expander()
    as_mock(expander.llm_client.generate_response).side_effect = TimeoutError()

    with caplog.at_level(logging.INFO):
        assert expander.expand("backup") == ExpansionResult()

    assert any(
        r.levelno == logging.WARNING and "Document expansion failed" in r.getMessage()
        for r in caplog.records
    )
