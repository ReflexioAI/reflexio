"""Housekeeping cadence defaults and safe legacy configuration preflight.

No server imports: configuration parsing and image bootstrap use this module.
"""

from typing import Any

from reflexio.config_migration import ConfigurationMigrationError

DAILY_SWEEP_SECONDS = 86_400

# Frozen compatibility defaults, independent of future policy calibration.
_RETIRED_FIELDS = (("lineage_gc", "poll_interval_seconds", 86_400),)


class HousekeepingMigrationError(ConfigurationMigrationError):
    """A stored custom policy must be migrated, never replaced with defaults."""


def housekeeping_config_conflicts(payload: dict[str, Any]) -> list[str]:
    """Return only paths with custom legacy values, never configuration values."""
    conflicts = []
    for section, field, old_default in _RETIRED_FIELDS:
        value = payload.get(section)
        if not isinstance(value, dict) or field not in value:
            continue
        try:
            matches = float(value[field]) == old_default
        except (ValueError, TypeError, OverflowError):
            matches = False
        if not matches:
            conflicts.append(f"{section}.{field}")
    return conflicts


def strip_legacy_housekeeping_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Discard former defaults without mutating input; reject custom tuning."""
    conflicts = housekeeping_config_conflicts(payload)
    if conflicts:
        raise HousekeepingMigrationError(
            "Retired housekeeping settings require explicit migration: "
            + ", ".join(conflicts)
            + ". Remove these fields to adopt code-owned daily sweeps."
        )
    cleaned = dict(payload)
    for section, field, _ in _RETIRED_FIELDS:
        value = cleaned.get(section)
        if isinstance(value, dict) and field in value:
            cleaned[section] = {k: v for k, v in value.items() if k != field}
    return cleaned
