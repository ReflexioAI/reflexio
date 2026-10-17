"""Shared resume timing for claims, discovery, recovery and scheduling.

Independent of server imports so saved policies can be validated at bootstrap.
"""

from typing import Any

from pydantic import TypeAdapter, ValidationError

from reflexio.config_migration import ConfigurationMigrationError

RESUME_POLL_SECONDS = 5.0
RESUME_CLAIM_TTL_SECONDS = 600

# Frozen retired defaults, not references to potentially recalibrated policy.
_LEGACY_DEFAULTS = {
    "resume_poll_interval_seconds": 5.0,
    "resume_claim_ttl_seconds": 600,
}
_LEGACY_FLOAT = TypeAdapter(float)
_LEGACY_INT = TypeAdapter(int)


class PendingToolTimingMigrationError(ConfigurationMigrationError):
    """Custom pending-tool execution timing requires explicit adoption."""


def strip_legacy_pending_tool_timing(data: Any) -> Any:
    """Copy and discard old defaults; reject customized or malformed timing."""
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    for key, old_default in _LEGACY_DEFAULTS.items():
        if key not in data:
            continue
        adapter = (
            _LEGACY_FLOAT if key == "resume_poll_interval_seconds" else _LEGACY_INT
        )
        try:
            matches = adapter.validate_python(data[key]) == old_default
        except ValidationError:
            matches = False
        if not matches:
            raise PendingToolTimingMigrationError(
                f"Retired setting pending_tool_call_config.{key} requires explicit "
                "migration. Remove this field to adopt code-owned five-second "
                "resume polling and 600-second claims. Keep the previous release "
                "if custom timing is required."
            )
        cleaned.pop(key)
    return cleaned
