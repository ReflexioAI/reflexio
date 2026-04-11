"""Schema compliance and snapshot tests for LLM mock responses.

Layer 1 (Syrupy snapshots): Catches *any* change to mock response content.
    When someone modifies the heuristic mock or prompts, snapshot tests break
    immediately, forcing a deliberate ``--snapshot-update`` and diff review.

Layer 2 (Pydantic schema validation): Catches *structural* drift.
    Validates that mock responses can be parsed into the Pydantic models
    that services actually expect.

Together: snapshots detect "something changed" (broad),
schema validation detects "it broke the contract" (precise).
"""

import json

import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from reflexio.test_support.llm_fixtures import load_llm_fixture_content
from reflexio.test_support.llm_mock import _create_mock_completion
from reflexio.test_support.llm_model_registry import get_model_registry

# Mapping from fixture file names to their expected model registry keys.
FIXTURE_TO_MODEL: list[tuple[str, str]] = [
    ("playbook_extraction", "playbook_extraction"),
    ("playbook_aggregation", "playbook_aggregation"),
    ("profile_extraction", "profile_extraction"),
    ("agent_success_evaluation", "agent_success_evaluation"),
]


@pytest.fixture
def snapshot_json(snapshot):
    return snapshot.use_extension(JSONSnapshotExtension)


# ─── Layer 1: Syrupy Snapshot Tests ──────────────────────────────────────────


class TestMockResponseSnapshots:
    """Snapshot every heuristic mock branch and recorded fixture.

    If these fail, run::

        pytest --snapshot-update tests/server/services/test_llm_mock_schema_compliance.py

    Then review the diff in ``__snapshots__/`` before committing.
    """

    def test_boolean_branch(self, snapshot_json):
        """Snapshot the boolean evaluation response."""
        response = _create_mock_completion("Output just a boolean value: is this good?")
        assert response.choices[0].message.content == snapshot_json

    def test_aggregation_branch_policy(self, snapshot_json):
        """Snapshot the policy consolidation response."""
        response = _create_mock_completion(
            "Perform policy consolidation on these playbooks"
        )
        content = json.loads(response.choices[0].message.content)
        assert content == snapshot_json

    def test_structured_output_branch(self, snapshot_json):
        """Snapshot the structured output (response_format present) response."""
        response = _create_mock_completion(
            "Extract profiles from interactions", parse_structured_output=True
        )
        content = json.loads(response.choices[0].message.content)
        assert content == snapshot_json

    def test_default_branch(self, snapshot_json):
        """Snapshot the default markdown-wrapped response."""
        response = _create_mock_completion("Some generic prompt without keywords")
        assert response.choices[0].message.content == snapshot_json

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "playbook_extraction",
            "playbook_aggregation",
            "profile_extraction",
            "agent_success_evaluation",
        ],
    )
    def test_recorded_fixture_content(self, snapshot_json, fixture_name):
        """Snapshot each recorded fixture's content for drift detection."""
        content_str = load_llm_fixture_content(fixture_name)
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            content = content_str
        assert content == snapshot_json


# ─── Layer 2: Pydantic Schema Validation Tests ──────────────────────────────


class TestSchemaCompliance:
    """Validate that mock responses parse into expected Pydantic models."""

    @pytest.mark.parametrize("entry_name", list(get_model_registry().keys()))
    def test_registry_minimal_values_validate(self, entry_name):
        """Each registry entry's minimal_valid JSON must validate against its model."""
        entry = get_model_registry()[entry_name]
        if entry.model_class is None:
            pytest.skip("No model class for raw string responses")
        result = entry.model_class.model_validate(entry.minimal_valid)
        assert isinstance(result, entry.model_class)

    @pytest.mark.parametrize("fixture_name,model_key", FIXTURE_TO_MODEL)
    def test_fixtures_validate_against_models(self, fixture_name, model_key):
        """Recorded fixture JSON must parse into the correct Pydantic model."""
        content = load_llm_fixture_content(fixture_name)
        data = json.loads(content)
        entry = get_model_registry()[model_key]
        assert entry.model_class is not None
        result = entry.model_class.model_validate(data)
        assert isinstance(result, entry.model_class)

    def test_heuristic_mock_aggregation_validates(self):
        """Heuristic mock's aggregation branch validates against PlaybookAggregationOutput."""
        from reflexio.server.services.playbook.playbook_service_utils import (
            PlaybookAggregationOutput,
        )

        response = _create_mock_completion("policy consolidation")
        data = json.loads(response.choices[0].message.content)
        result = PlaybookAggregationOutput.model_validate(data)
        assert result.playbook is not None

    def test_heuristic_mock_structured_validates(self):
        """Heuristic mock's structured output branch validates against StructuredProfilesOutput."""
        from reflexio.server.services.profile.profile_generation_service_utils import (
            StructuredProfilesOutput,
        )

        response = _create_mock_completion("prompt", parse_structured_output=True)
        data = json.loads(response.choices[0].message.content)
        result = StructuredProfilesOutput.model_validate(data)
        assert result.profiles is not None
