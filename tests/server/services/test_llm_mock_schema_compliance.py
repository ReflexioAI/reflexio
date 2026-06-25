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
from typing import Any

import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.llm.llm_utils import strict_response_format_for_model
from reflexio.test_support.llm_fixtures import load_llm_fixture_content
from reflexio.test_support.llm_mock import _create_mock_completion, _find_schema_key
from reflexio.test_support.llm_model_registry import get_model_registry


def _registry_model_keys() -> list[str]:
    """Registry keys whose entries carry a Pydantic model class (skip raw/boolean)."""
    return [
        name
        for name, entry in get_model_registry().items()
        if entry.model_class is not None
    ]


def _walk(node: Any):
    """Yield every dict node in a JSON-schema fragment (recursive)."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``#/$defs/Name`` reference against a ``$defs`` map."""
    name = ref.rsplit("/", 1)[-1]
    return defs.get(name, {})


def _member_first_key(
    member: dict[str, Any], defs: dict[str, Any]
) -> tuple[Any, Any] | None:
    """Return the ``(first-property-name, first-const-value)`` of an anyOf member.

    Resolves a ``$ref`` member against ``$defs`` so the member's own
    ``properties`` are inspected, mirroring how OpenAI evaluates anyOf
    distinguishability. Returns ``None`` for a member that is not an object
    schema (e.g. a bare ``{"type": "null"}`` / ``{"type": "array"}`` arm of an
    *optional* field's nullable union) — that arm is not a discriminated-union
    variant, so it does not participate in the first-key distinguishability rule
    OpenAI applies to *object* members.
    """
    if "$ref" in member:
        member = _resolve_ref(member["$ref"], defs)
    properties = member.get("properties", {})
    if not properties:
        return None
    first_name = next(iter(properties))
    first_schema = properties[first_name]
    const_value = first_schema.get("const")
    if const_value is None:
        enum_values = first_schema.get("enum")
        if isinstance(enum_values, list) and len(enum_values) == 1:
            const_value = enum_values[0]
    return (first_name, const_value)


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

    @pytest.mark.parametrize("entry_name", _registry_model_keys())
    def test_registry_minimal_values_validate(self, entry_name):
        """Each registry entry's minimal_valid JSON must validate against its model."""
        entry = get_model_registry()[entry_name]
        assert entry.model_class is not None
        result = entry.model_class.model_validate(entry.minimal_valid)
        assert isinstance(result, entry.model_class)

    @pytest.mark.parametrize(
        "entry_name",
        [
            name
            for name, entry in get_model_registry().items()
            if entry.model_class is None
        ],
    )
    def test_registry_raw_string_entries_are_non_empty(self, entry_name):
        """Raw-string registry entries are intentionally not Pydantic models."""
        entry = get_model_registry()[entry_name]
        assert entry.model_class is None
        assert isinstance(entry.minimal_valid, str)
        assert entry.minimal_valid.strip()

    def test_guard_finder_ignores_property_named_like_a_keyword(self):
        """The mock guard's finder must not flag a field NAMED oneOf/discriminator.

        Those are property names, not schema keywords (CodeRabbit false-positive fix).
        """
        from pydantic import BaseModel

        class _TrickyNames(BaseModel):
            oneOf: str = ""  # noqa: N815  (deliberately a keyword-like field name)
            discriminator: int = 0

        schema = strict_response_format_for_model(_TrickyNames)["json_schema"]["schema"]
        assert "oneOf" in schema["properties"]
        assert not _find_schema_key(schema, "oneOf")
        assert not _find_schema_key(schema, "discriminator")

    @pytest.mark.parametrize("entry_name", _registry_model_keys())
    def test_registry_models_emit_strict_compatible_schema(self, entry_name):
        """Every structured-output model must produce a provider-strict schema.

        The schema actually sent (``strict_response_format_for_model``) must
        contain no ``oneOf``/``discriminator`` — OpenAI strict structured outputs
        reject them, which is how a Pydantic discriminated union (e.g.
        ``PlaybookConsolidationOutput``) 400'd in production (PYTHON-FASTAPI-9J).
        This is the provider-agnostic invariant guard for the whole registry.
        """
        entry = get_model_registry()[entry_name]
        assert entry.model_class is not None
        rf = strict_response_format_for_model(entry.model_class)
        schema = rf["json_schema"]["schema"]
        assert not _find_schema_key(schema, "oneOf"), (
            f"{entry.model_class.__name__} emits 'oneOf' (discriminated union); "
            "strict structured-output providers reject it."
        )
        assert not _find_schema_key(schema, "discriminator"), (
            f"{entry.model_class.__name__} emits 'discriminator'; "
            "strict structured-output providers reject it."
        )

    @pytest.mark.parametrize("entry_name", _registry_model_keys())
    def test_registry_models_native_schema_is_provider_safe(self, entry_name):
        """Native ``model_json_schema()`` must be provider-safe by construction.

        With no ``make_strict`` normalization and no provider gate, the raw
        schema a model emits must already carry no ``oneOf`` and no
        ``discriminator`` at any depth — the ``StrictStructuredOutput`` hook
        folds any discriminated union into ``anyOf``. This proves the
        by-construction safety the boundary guard relies on.
        """
        model_class = get_model_registry()[entry_name].model_class
        assert model_class is not None
        schema = model_class.model_json_schema()
        for node in _walk(schema):
            assert "oneOf" not in node, (
                f"{model_class.__name__} emits native 'oneOf'; "
                "StrictStructuredOutput must fold it to 'anyOf'."
            )
            assert "discriminator" not in node, (
                f"{model_class.__name__} emits native 'discriminator'; "
                "StrictStructuredOutput must drop it."
            )

    @pytest.mark.parametrize("entry_name", _registry_model_keys())
    def test_registry_models_anyof_first_keys_distinguishable(self, entry_name):
        """Every ``anyOf`` member set must have unique first ``(key, const)`` pairs.

        OpenAI rejects ``anyOf`` members that share an identical first property
        key unless disambiguated by a ``const`` — our folded discriminated-union
        variants all lead with the discriminator field but carry distinct
        ``Literal`` consts, so the pairs must be unique. Models with no
        ``anyOf`` pass trivially.
        """
        model_class = get_model_registry()[entry_name].model_class
        assert model_class is not None
        schema = model_class.model_json_schema()
        defs = schema.get("$defs", {})
        for node in _walk(schema):
            members = node.get("anyOf")
            if not isinstance(members, list):
                continue
            # Only object members (discriminated-union variants) are subject to
            # the first-key rule; nullable scalar arms return None and are
            # ignored.
            keys = [
                key
                for member in members
                if (key := _member_first_key(member, defs)) is not None
            ]
            assert len(keys) == len(set(keys)), (
                f"{model_class.__name__} has anyOf members sharing a "
                f"(first-key, const) pair: {keys}; OpenAI rejects this."
            )

    @pytest.mark.parametrize("entry_name", _registry_model_keys())
    def test_registry_models_are_strict_structured_output_subclasses(self, entry_name):
        """Every registry model must inherit ``StrictStructuredOutput``.

        A new structured-output model that forgets the base — and thus the
        provider-safe schema hook — fails here.
        """
        model_class = get_model_registry()[entry_name].model_class
        assert model_class is not None
        assert issubclass(model_class, StrictStructuredOutput)

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
