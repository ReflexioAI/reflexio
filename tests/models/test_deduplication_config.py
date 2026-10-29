"""Deduplication search controls and historical diagnostic-field cleanup."""

from copy import deepcopy

import pytest

from reflexio.config_migration import strip_retired_config_fields
from reflexio.models.config_schema import DeduplicationConfig


def test_retired_length_warning_does_not_survive_schema_or_serialization():
    config = DeduplicationConfig.model_validate(
        {
            "max_unified_content_chars": 700,
            "search_threshold": 0.6,
            "search_top_k": 7,
        }
    )
    assert config.model_dump() == {
        "search_threshold": 0.6,
        "search_top_k": 7,
    }
    assert (
        "max_unified_content_chars"
        not in DeduplicationConfig.model_json_schema()["properties"]
    )


@pytest.mark.parametrize(
    "extractor_key",
    [
        "user_playbook_extractor_config",
        "user_playbook_extractor_configs",
        "playbook_configs",
        "agent_feedback_configs",
    ],
)
def test_raw_upgrade_strips_only_retired_nested_diagnostic(extractor_key):
    entry = {
        "deduplication_config": {"max_unified_content_chars": 17, "search_top_k": 8},
        "custom_extension": {"max_unified_content_chars": "opaque"},
    }
    value = entry if extractor_key.endswith("_config") else [entry, deepcopy(entry)]
    payload = {extractor_key: value, "unrelated": {"max_unified_content_chars": "keep"}}
    before = deepcopy(payload)
    normalized = strip_retired_config_fields(payload)
    assert payload == before
    entries = (
        [normalized[extractor_key]]
        if isinstance(value, dict)
        else normalized[extractor_key]
    )
    for item in entries:
        assert item["deduplication_config"] == {"search_top_k": 8}
        assert item["custom_extension"] == {"max_unified_content_chars": "opaque"}
    assert normalized["unrelated"] == payload["unrelated"]
