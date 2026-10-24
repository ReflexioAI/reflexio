"""An upgrade must explicitly adopt the new search policy for active optimizers."""

from copy import deepcopy

import pytest

from reflexio.config_migration import canonicalize_config_document
from reflexio.models.config_schema import (
    PlaybookOptimizerConfig,
    validate_stored_config,
)


@pytest.mark.parametrize("enabled", [True, "true", 1])
def test_active_legacy_stopping_requires_explicit_adoption(enabled):
    with pytest.raises(ValueError, match="early_stop_score"):
        PlaybookOptimizerConfig.model_validate(
            {"enabled": enabled, "early_stop_score": 0.9}
        )
    with pytest.raises(ValueError):
        validate_stored_config(
            {
                "storage_config": None,
                "playbook_optimizer_config": {
                    "enabled": enabled,
                    "early_stop_score": 0.9,
                },
            }
        )


@pytest.mark.parametrize("enabled", [False, "false", 0])
def test_disabled_legacy_stopping_cannot_select_execution_policy(enabled):
    source = {"enabled": enabled, "early_stop_score": 0.8}
    original = deepcopy(source)
    config = PlaybookOptimizerConfig.model_validate(source)
    assert config.enabled is False
    assert "early_stop_score" not in config.model_dump()
    assert (
        "early_stop_score"
        not in PlaybookOptimizerConfig.model_json_schema()["properties"]
    )
    assert source == original


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_upgrade_adopts_quality_policy_without_losing_budget_or_metadata(
    enabled,
):
    source = {
        "storage_config": None,
        "extension": {"private": "keep"},
        "playbook_optimizer_config": {
            "enabled": enabled,
            "early_stop_score": 0.8,
            "max_metric_calls": 37,
        },
    }
    original = deepcopy(source)
    converted = canonicalize_config_document(source)
    assert converted["playbook_optimizer_config"] == {
        "enabled": enabled,
        "max_metric_calls": 37,
    }
    assert converted["extension"] == source["extension"]
    assert canonicalize_config_document(converted) == converted
    assert source == original


@pytest.mark.parametrize("value", [-0.1, 1.1, "private-malformed", float("nan"), None])
@pytest.mark.parametrize("enabled", [False, True])
def test_malformed_obsolete_score_cannot_be_hidden_by_disable_or_upgrade(
    value, enabled
):
    payload = {"enabled": enabled, "early_stop_score": value}
    with pytest.raises(ValueError):
        PlaybookOptimizerConfig.model_validate(payload)
    with pytest.raises(ValueError) as error:
        canonicalize_config_document(
            {"storage_config": None, "playbook_optimizer_config": payload}
        )
    assert "private-malformed" not in str(error.value)
