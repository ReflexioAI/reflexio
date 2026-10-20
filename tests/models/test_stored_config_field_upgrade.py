"""Raw upgrades retain saved policy and extensions before typed validation."""

from copy import deepcopy

import pytest

from reflexio.models.config_schema import (
    StoredConfigurationError,
    canonicalize_stored_config_fields,
    validate_stored_config,
)


@pytest.mark.parametrize(
    "legacy",
    ["agent_feedback_configs", "playbook_configs", "user_playbook_extractor_configs"],
)
def test_raw_alias_upgrade_is_idempotent_and_does_not_mutate(legacy):
    payload = {
        "storage_config": None,
        "batch_size": 23,
        "batch_interval": 13,
        legacy: [
            {
                "feedback_definition_prompt": "Custom procedures",
                "batch_size_override": 17,
                "feedback_aggregator_config": {
                    "min_feedback_threshold": 9,
                    "extension": {"refresh_count": "opaque"},
                },
                "extension": {"batch_size": "opaque"},
            }
        ],
        "extension": {"batch_size": "opaque"},
    }
    before = deepcopy(payload)
    result = canonicalize_stored_config_fields(payload)
    assert payload == before
    assert canonicalize_stored_config_fields(result) == result
    assert legacy not in result
    assert result["window_size"] == 23
    child = result["user_playbook_extractor_config"]
    assert child["extraction_definition_prompt"] == "Custom procedures"
    assert child["window_size_override"] == 17
    assert child["aggregation_config"]["min_cluster_size"] == 9
    assert child["aggregation_config"]["extension"] == {"refresh_count": "opaque"}
    assert child["extension"] == {"batch_size": "opaque"}
    assert result["extension"] == {"batch_size": "opaque"}
    assert (
        validate_stored_config(payload).model_dump()
        == validate_stored_config(result).model_dump()
    )


def test_raw_upgrade_preserves_canonical_precedence_and_disabled_extractors():
    result = canonicalize_stored_config_fields(
        {
            "storage_config": None,
            "window_size": 31,
            "batch_size": 99,
            "stride_size": 7,
            "profile_extractor_configs": [],
            "user_playbook_extractor_config": None,
            "agent_feedback_configs": [{"feedback_definition_prompt": "Ignored"}],
        }
    )
    assert result["window_size"] == 31
    assert "batch_size" not in result
    assert result["profile_extractor_config"] is None
    assert result["user_playbook_extractor_config"] is None
    assert "agent_feedback_configs" not in result


def test_raw_upgrade_does_not_mask_invalid_recognized_policy():
    result = canonicalize_stored_config_fields(
        {"storage_config": None, "batch_size": "invalid"}
    )
    assert result["window_size"] == "invalid"
    with pytest.raises(StoredConfigurationError):
        validate_stored_config(result)
