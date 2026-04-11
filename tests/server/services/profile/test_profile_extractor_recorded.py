"""Tests for profile extraction using recorded LLM responses.

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


class TestRecordedProfileExtraction:
    """Validate profile extraction logic with recorded LLM output."""

    def test_fixture_returns_valid_json(self):
        """Recorded profile extraction fixture contains parseable JSON."""
        content = load_llm_fixture_content("profile_extraction")
        data = json.loads(content)

        assert "profiles" in data
        assert isinstance(data["profiles"], list)
        assert len(data["profiles"]) > 0

    def test_fixture_profiles_have_required_fields(self):
        """Each profile in the recorded response has content and time_to_live."""
        content = load_llm_fixture_content("profile_extraction")
        data = json.loads(content)

        for profile in data["profiles"]:
            assert "content" in profile, "Profile missing 'content' field"
            assert "time_to_live" in profile, "Profile missing 'time_to_live' field"
            assert isinstance(profile["content"], str)
            assert len(profile["content"]) > 0

    def test_fixture_mock_has_correct_structure(self):
        """load_llm_fixture returns a mock with the expected litellm structure."""
        mock = load_llm_fixture("profile_extraction")

        assert hasattr(mock, "choices")
        assert len(mock.choices) == 1
        assert hasattr(mock.choices[0], "message")
        assert hasattr(mock.choices[0].message, "content")
        assert mock.choices[0].finish_reason == "stop"
