"""Historical pending-tool conversion and obsolete-input diagnostics.

The current execution engine uses the renewable user-work owner. There is no
separate resume pickup timer or non-renewable logical-claim lifetime.
"""

import logging
import threading
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field, TypeAdapter, ValidationError

from reflexio.config_migration import (
    ConfigurationMigrationError,
    canonicalize_pending_creation_policy,
)

logger = logging.getLogger(__name__)
RETIRED_PENDING_TOOL_TIMING_FIELDS = (
    "resume_poll_interval_seconds",
    "resume_claim_ttl_seconds",
)
_RETIRED_RESUME_ENV = (
    "REFLEXIO_RESUME_POLL_INTERVAL_SECONDS",
    "REFLEXIO_RESUME_CLAIM_TTL_SECONDS",
)
# Historical validation bounds only. These never select an execution deadline
# and are never copied into the current renewable user lease.
_LEGACY_POLL = TypeAdapter(
    Annotated[float, Field(gt=0, le=threading.TIMEOUT_MAX, allow_inf_nan=False)]
)
_LEGACY_CLAIM = TypeAdapter(Annotated[int, Field(gt=0, le=((1 << 31) - 1) // 6)])


class PendingToolTimingMigrationError(ConfigurationMigrationError):
    """Malformed saved timing cannot be discarded by an upgrade."""


def comparable_pending_creation_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the creation encoding without erasing frozen witness fields.

    Old complete snapshots contain enabled. Current snapshots carry a format
    stamp, so an incomplete old snapshot cannot acquire positive consent from
    the absence of the former switch. This is witness metadata, not org policy.
    """
    stamp = "creation_policy_version"
    if stamp in data:
        if type(data[stamp]) is not int or data[stamp] != 1 or "enabled" in data:
            raise ConfigurationMigrationError(
                "Invalid pending creation snapshot format"
            )
        return dict(data)
    if "enabled" not in data or "pending_ttl_seconds" not in data:
        raise ConfigurationMigrationError(
            "Unrecognized pending creation snapshot format"
        )
    return {**canonicalize_pending_creation_policy(data, historical=True), stamp: 1}


def warn_retired_resume_timing(environment: Mapping[str, str]) -> None:
    """Warn by name only; old assignments cannot select or block work policy."""
    names = [name for name in _RETIRED_RESUME_ENV if environment.get(name, "").strip()]
    if names:
        logger.warning(
            "event=obsolete_resume_timing release_blocking=false settings=%s "
            "retired by single-owner execution; remove the obsolete assignments",
            ",".join(names),
        )


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
    """Validate historical values and retire their mechanism during conversion.

    The deliberate engine upgrade follows actual ownership, not old claim age.
    Keep every unknown/non-timing field, explicit opt-out and secret untouched.
    Runtime models reject these fields; this is not a reader alias.
    """
    if not isinstance(data, dict):
        return data
    for key, adapter in zip(
        RETIRED_PENDING_TOOL_TIMING_FIELDS, (_LEGACY_POLL, _LEGACY_CLAIM), strict=True
    ):
        if key in data:
            try:
                adapter.validate_python(data[key])
            except ValidationError:
                raise PendingToolTimingMigrationError(
                    f"Invalid historical pending_tool_call_config.{key}; correct the saved "
                    "value before retrying the single-owner upgrade. No current timing "
                    "or user lease has been changed."
                ) from None
    return {
        key: value
        for key, value in data.items()
        if key not in RETIRED_PENDING_TOOL_TIMING_FIELDS
    }
