"""Code-owned durable learning timing, shared by library and enterprise schedulers.

This module stays independent of server imports so startup can validate legacy
inputs before constructing workers or importing the application stack.
"""

import math
import os
from collections.abc import Mapping

DISCOVERY_POLL_SECONDS = 2.0
CLAIM_LEASE_SECONDS = 300


def validate_discovery_timing_env(env: Mapping[str, str]) -> None:
    """Reject retired custom timing overrides; tolerate blank or old defaults.

    The accepted numbers describe the retired contract, not tunable policy.
    Never silently reinterpret a nondefault deployment override as our default.
    """
    legacy = (
        (
            "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS",
            env.get("REFLEXIO_DURABLE_LEARNING_POLL_SECONDS", ""),
            2.0,
        ),
        (
            "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS",
            env.get("REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS", ""),
            300.0,
        ),
    )
    for name, raw, old_default in legacy:
        if not raw.strip():
            continue
        try:
            matches_default = float(raw) == old_default
        except ValueError:
            matches_default = False
        if not matches_default:
            raise ValueError(
                f"{name} is retired; remove it from your environment. "
                "Durable extraction now owns its discovery timing; custom "
                "poll intervals and claim durations are no longer supported."
            )


AGGREGATION_MIN_INTERVAL_SECONDS = 3600
AGGREGATION_ADMISSION_WAIT_SECONDS = 1.0


def aggregation_min_interval_seconds(env: Mapping[str, str] | None = None) -> int:
    """Resolve the shared drained-cycle coalescing interval (positive seconds)."""
    environment = os.environ if env is None else env
    raw = environment.get("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", "").strip()
    try:
        value = int(raw) if raw else AGGREGATION_MIN_INTERVAL_SECONDS
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(
            "REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS must be a positive integer."
        )
    return value


def aggregation_admission_wait_seconds(env: Mapping[str, str] | None = None) -> float:
    """Resolve finite nonnegative slot wait; zero admits only immediately."""
    environment = os.environ if env is None else env
    raw = environment.get(
        "REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS", ""
    ).strip()
    try:
        value = float(raw) if raw else AGGREGATION_ADMISSION_WAIT_SECONDS
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS must be finite and nonnegative."
        )
    return value


def validate_aggregation_timing_env(env: Mapping[str, str]) -> None:
    """Validate supported aggregation controls against the authoritative mapping."""
    aggregation_min_interval_seconds(env)
    aggregation_admission_wait_seconds(env)
