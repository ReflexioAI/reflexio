"""Tests for playbook extraction using recorded LLM responses.

These tests use pre-recorded real LLM responses from fixture files
instead of the global heuristic mock, providing more realistic
validation of the extraction pipeline.
"""

import json

import pytest

from reflexio.test_support.llm_fixtures import (
    load_llm_fixture,
    load_llm_fixture_content,
)

pytestmark = pytest.mark.integration


class TestRecordedPlaybookExtraction:
    """Validate playbook extraction logic with recorded LLM output."""

    def test_fixture_returns_valid_json(self):
        """Recorded playbook extraction fixture contains parseable JSON."""
        content = load_llm_fixture_content("playbook_extraction")
        data = json.loads(content)

        assert "playbooks" in data
        assert isinstance(data["playbooks"], list)
        assert len(data["playbooks"]) >= 1
        first = data["playbooks"][0]
        assert "trigger" in first
        assert "instruction" in first
        assert isinstance(first["trigger"], str)
        assert len(first["instruction"]) > 0

    def test_fixture_playbooks_have_content(self):
        """Each entry in the playbook fixture has non-empty trigger and instruction."""
        content = load_llm_fixture_content("playbook_extraction")
        data = json.loads(content)

        assert data["playbooks"]
        for entry in data["playbooks"]:
            assert entry["trigger"]
            assert entry["instruction"]


class TestRecordedPlaybookAggregation:
    """Validate playbook aggregation logic with recorded LLM output."""

    def test_aggregation_fixture_has_content_structure(self):
        """Recorded aggregation fixture has the expected policy structure."""
        content = load_llm_fixture_content("playbook_aggregation")
        data = json.loads(content)

        assert "playbook" in data
        assert "instruction" in data["playbook"]
        assert "trigger" in data["playbook"]
        assert len(data["playbook"]["instruction"]) > 0

    def test_aggregation_mock_structure(self):
        """load_llm_fixture returns correct mock for aggregation responses."""
        mock = load_llm_fixture("playbook_aggregation")
        content = mock.choices[0].message.content
        data = json.loads(content)

        assert data["playbook"]["trigger"]


class TestRecordedAgentSuccess:
    """Validate agent success evaluation with recorded LLM output."""

    def test_success_fixture_returns_structured_output(self):
        """Agent success evaluation fixture returns structured output."""
        content = load_llm_fixture_content("agent_success_evaluation")
        data = json.loads(content)
        assert "is_success" in data
        assert isinstance(data["is_success"], bool)
