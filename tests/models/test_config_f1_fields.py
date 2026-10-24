"""Rubric provenance is derived from the template, never a saved label."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from reflexio.config_env import StoredConfigurationError
from reflexio.config_migration import (
    ConfigurationMigrationError,
    canonicalize_config_document,
)
from reflexio.models.config_schema import (
    Config,
    StorageConfigSQLite,
    validate_stored_config,
)

_LABEL = "shadow_comparison_judge_prompt_version"


def test_rubric_label_is_absent_from_current_configuration():
    config = Config(storage_config=StorageConfigSQLite())
    assert _LABEL not in Config.model_fields
    assert _LABEL not in config.model_dump(mode="json")


@pytest.mark.parametrize("value", ["v1.1.0", "custom-label", "", None])
def test_normal_and_stored_reads_reject_the_retired_label(value):
    payload = {"storage_config": {"type": "sqlite"}, _LABEL: value}
    with pytest.raises(ValidationError, match=_LABEL):
        Config.model_validate(payload)
    with pytest.raises(
        StoredConfigurationError, match="Stored configuration failed validation"
    ):
        validate_stored_config(payload)


@pytest.mark.parametrize("value", ["v1.1.0", "custom-label"])
def test_explicit_upgrade_removes_only_the_valid_label(value):
    original = {
        "storage_config": {"type": "sqlite"},
        _LABEL: value,
        "window_size": 25,
        "stride_size": 8,
        "opaque_extension": {"keep": [1, "unchanged"]},
    }
    before = deepcopy(original)
    converted = canonicalize_config_document(original)
    assert converted == {key: value for key, value in before.items() if key != _LABEL}
    assert original == before
    assert canonicalize_config_document(converted) == converted


@pytest.mark.parametrize("value", ["", " ", None, 12, ["v1.0.0"]])
def test_explicit_upgrade_refuses_malformed_labels(value):
    with pytest.raises(ConfigurationMigrationError, match="rubric label"):
        canonicalize_config_document(
            {"storage_config": {"type": "sqlite"}, _LABEL: value}
        )
