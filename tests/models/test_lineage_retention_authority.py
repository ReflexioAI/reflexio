"""One retention policy owns both lineage deletion consent and its cutoff."""

import pytest
from pydantic import ValidationError

from reflexio import config_migration
from reflexio.config_env import ConfigurationMigrationError
from reflexio.models.config_schema import LineageGCConfig

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("days", [0, 1, 37, 365])
def test_policy_resolves_hold_or_exact_cutoff(days):
    policy = LineageGCConfig(tombstone_grace_window_days=days)
    assert policy.reclamation_cutoff(1_900_000_000) == (
        1_900_000_000 - days * 86400 if days else None
    )
    assert "enabled" not in policy.model_dump()


@pytest.mark.parametrize("enabled", [True, False, None, ""])
def test_current_runtime_refuses_retired_enable_input(enabled):
    with pytest.raises(ValidationError, match="enabled"):
        LineageGCConfig.model_validate({"enabled": enabled})


@pytest.mark.parametrize("enabled, days, expected", [(True, 37, 37), (False, 37, 0)])
def test_explicit_conversion_preserves_effective_policy_and_metadata(
    enabled, days, expected
):
    original = {
        "enabled": enabled,
        "tombstone_grace_window_days": days,
        "extension": {"keep": "opaque"},
    }
    converted = config_migration.canonicalize_lineage_retention_policy(original)
    assert converted == {
        "tombstone_grace_window_days": expected,
        "extension": {"keep": "opaque"},
    }
    assert original["enabled"] is enabled
    assert original["tombstone_grace_window_days"] == days


@pytest.mark.parametrize("days", [0, -1, "invalid", None])
def test_invalid_dormant_legacy_grace_does_not_become_a_valid_hold(days):
    with pytest.raises(ConfigurationMigrationError):
        config_migration.canonicalize_lineage_retention_policy(
            {"enabled": False, "tombstone_grace_window_days": days}
        )
