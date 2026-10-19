"""Shared resume timing for claims, discovery, recovery and scheduling.

Independent of server imports so saved policies can be validated at bootstrap.
"""

import math
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter, ValidationError

from reflexio.config_migration import ConfigurationMigrationError

DEFAULT_RESUME_POLL_SECONDS = 5.0
DEFAULT_RESUME_CLAIM_TTL_SECONDS = 600
# Native get_agent_run_health and resume discovery calculate 6 * TTL using
# PostgreSQL int32 arithmetic before constructing their stale-work interval.
MAX_RESUME_CLAIM_TTL_SECONDS = ((1 << 31) - 1) // 6


@dataclass(frozen=True)
class ResumeTiming:
    poll_seconds: float
    claim_ttl_seconds: int


def resolve_resume_timing(env: Mapping[str, str] | None = None) -> ResumeTiming:
    """Resolve process-owned pickup and non-renewable claim-expiry policy.

    An explicitly supplied mapping is authoritative, including an empty one.
    Defaults preserve existing behavior; blank input is equivalent to unset.
    """
    environment = os.environ if env is None else env
    poll_raw = environment.get("REFLEXIO_RESUME_POLL_INTERVAL_SECONDS", "").strip()
    ttl_raw = environment.get("REFLEXIO_RESUME_CLAIM_TTL_SECONDS", "").strip()
    try:
        poll = float(poll_raw) if poll_raw else DEFAULT_RESUME_POLL_SECONDS
    except ValueError:
        poll = 0.0
    if not math.isfinite(poll) or not 0 < poll <= threading.TIMEOUT_MAX:
        raise ValueError(
            "REFLEXIO_RESUME_POLL_INTERVAL_SECONDS must be finite, positive "
            "and no greater than threading.TIMEOUT_MAX."
        )
    try:
        ttl = int(ttl_raw) if ttl_raw else DEFAULT_RESUME_CLAIM_TTL_SECONDS
    except ValueError:
        ttl = 0
    if not 1 <= ttl <= MAX_RESUME_CLAIM_TTL_SECONDS:
        raise ValueError(
            "REFLEXIO_RESUME_CLAIM_TTL_SECONDS must be a positive integer no "
            "greater than 357913941 (the native health/discovery arithmetic limit)."
        )
    return ResumeTiming(poll_seconds=poll, claim_ttl_seconds=ttl)


# Frozen retired defaults, not references to potentially recalibrated policy.
_LEGACY_DEFAULTS = {
    "resume_poll_interval_seconds": 5.0,
    "resume_claim_ttl_seconds": 600,
}
_LEGACY_FLOAT = TypeAdapter(float)
_LEGACY_INT = TypeAdapter(int)


class PendingToolTimingMigrationError(ConfigurationMigrationError):
    """Custom pending-tool execution timing requires explicit adoption."""


def strip_retired_pending_followup_warning(data: Any) -> Any:
    """Copy only known policy paths and omit the retired observational threshold.

    This does not round-trip a model: frozen policy witnesses must retain unknown
    fields and the secrets digest for strict execution-policy comparison.
    """
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    cleaned.pop("max_pending_followups_per_scope", None)
    overrides = data.get("tool_overrides")
    if isinstance(overrides, dict):
        cleaned["tool_overrides"] = {
            name: {
                key: value
                for key, value in override.items()
                if key != "max_pending_followups_per_scope"
            }
            if isinstance(override, dict)
            else override
            for name, override in overrides.items()
        }
    return cleaned


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
                "migration to process-owned REFLEXIO_RESUME_POLL_INTERVAL_SECONDS "
                "and REFLEXIO_RESUME_CLAIM_TTL_SECONDS. Choose the deployment "
                "policy explicitly before removing legacy organization fields; "
                "different org values cannot be combined automatically."
            )
        cleaned.pop(key)
    return cleaned
