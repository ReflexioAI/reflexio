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
            "audit_events_max_age_days": 0,
            "audit_events_delete_batch_limit": 2000,
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
    assert cfg.governance_retention.audit_events_max_age_days == 0
    assert cfg.governance_retention.audit_events_delete_batch_limit == 2000


@pytest.mark.parametrize(
    "model,payload",
    [
        (LineageGCConfig, {"poll_interval_seconds": "private-legacy-value"}),
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


@pytest.mark.parametrize("limit", [1, 2, 500, 2000, (1 << 31) - 1])
def test_supported_audit_batch_survives_cleanup_and_roundtrip(limit):
    from reflexio.models.config_schema import validate_stored_config

    payload = {
        "storage_config": None,
        "lineage_gc": {"enabled": False},
        "governance_retention": {
            "audit_events_max_age_days": 0,
            "audit_events_delete_batch_limit": limit,
        },
    }
    assert housekeeping_config_conflicts(payload) == []
    assert strip_legacy_housekeeping_fields(payload) == payload
    cfg = validate_stored_config(payload)
    assert cfg.governance_retention.audit_events_delete_batch_limit == limit
    assert cfg.governance_retention.audit_events_max_age_days == 0
    assert cfg.lineage_gc.enabled is False
    assert (
        validate_stored_config(
            cfg.model_dump(mode="json")
        ).governance_retention.audit_events_delete_batch_limit
        == limit
    )


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        1 << 31,
        2.5,
        "private-invalid-limit",
        None,
        float("inf"),
        float("nan"),
        {},
        [],
    ],
)
def test_invalid_supported_batch_fails_without_disclosing_values(limit):
    from reflexio.config_migration import StoredConfigurationError
    from reflexio.models.config_schema import validate_stored_config

    payload = {
        "storage_config": None,
        "governance_retention": {"audit_events_delete_batch_limit": limit},
    }
    with pytest.raises(StoredConfigurationError, match="governance_retention") as error:
        validate_stored_config(payload)
    assert "private-invalid-limit" not in str(error.value)


def test_audit_batch_defaults_to_previous_500_rows():
    assert GovernanceRetentionConfig().audit_events_delete_batch_limit == 500
