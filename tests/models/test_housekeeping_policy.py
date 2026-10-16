"""Retirement never silently changes custom execution policy or deletion rules."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from reflexio.housekeeping_policy import (
    housekeeping_config_conflicts,
    strip_legacy_housekeeping_fields,
)
from reflexio.models.config_schema import (
    Config,
    GovernanceRetentionConfig,
    LineageGCConfig,
)


@pytest.mark.parametrize(
    "section,field,default,model",
    [
        ("lineage_gc", "poll_interval_seconds", 86400, LineageGCConfig),
        (
            "governance_retention",
            "audit_events_delete_batch_limit",
            500,
            GovernanceRetentionConfig,
        ),
    ],
)
def test_legacy_defaults_are_accepted_but_not_exposed(section, field, default, model):
    for value in (default, str(default), float(default)):
        loaded = model.model_validate({field: value})
        assert field not in loaded.model_dump()
        assert field not in model.model_json_schema()["properties"]
        cfg = Config.model_validate({"storage_config": None, section: {field: value}})
        assert field not in cfg.model_dump()[section]


@pytest.mark.parametrize(
    "section,field",
    [
        ("lineage_gc", "poll_interval_seconds"),
        ("governance_retention", "audit_events_delete_batch_limit"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [0, -1, 1, 3600, None, "private-custom-value", float("nan"), float("inf"), {}, []],
)
def test_custom_legacy_values_require_explicit_migration(section, field, value):
    payload = {section: {field: value}, "storage_config": None}
    assert housekeeping_config_conflicts(payload) == [f"{section}.{field}"]
    with pytest.raises(ValidationError, match="explicit migration"):
        Config.model_validate(payload)


def test_cleanup_preserves_policy_holds_and_opaque_metadata_without_mutating():
    payload = {
        "lineage_gc": {
            "enabled": False,
            "tombstone_grace_window_days": 120,
            "poll_interval_seconds": 86400,
        },
        "governance_retention": {
            "audit_events_retention_enabled": False,
            "audit_events_retention_days": 30,
            "audit_events_delete_batch_limit": 500,
        },
        "opaque": {"poll_interval_seconds": 60, "audit_events_delete_batch_limit": 1},
    }
    original = deepcopy(payload)
    cleaned = strip_legacy_housekeeping_fields(payload)
    assert payload == original
    assert cleaned["opaque"] == original["opaque"]
    cfg = Config.model_validate(
        {"storage_config": None, **{k: v for k, v in cleaned.items() if k != "opaque"}}
    )
    assert cfg.lineage_gc.enabled is False
    assert cfg.lineage_gc.tombstone_grace_window_days == 120
    assert cfg.governance_retention.audit_events_retention_enabled is False
    assert cfg.governance_retention.audit_events_retention_days == 30


@pytest.mark.parametrize(
    "model,payload",
    [
        (LineageGCConfig, {"poll_interval_seconds": "private-legacy-value"}),
        (
            GovernanceRetentionConfig,
            {"audit_events_delete_batch_limit": "private-legacy-value"},
        ),
        (
            Config,
            {
                "storage_config": None,
                "agent_context_prompt": "private-context",
                "lineage_gc": {"poll_interval_seconds": "private-legacy-value"},
            },
        ),
    ],
)
def test_migration_error_text_does_not_echo_config_values(model, payload):
    with pytest.raises(ValidationError) as exc:
        model.model_validate(payload)
    assert "explicit migration" in str(exc.value)
    assert "private-legacy-value" not in str(exc.value)
    assert "private-context" not in str(exc.value)
