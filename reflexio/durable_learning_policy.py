"""Code-owned durable learning timing, shared by library and enterprise schedulers.

This module stays independent of server imports so startup can validate legacy
inputs before constructing workers or importing the application stack.
"""

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


def validate_aggregation_timing_env(env: Mapping[str, str]) -> None:
    """Require explicit adoption of code-owned aggregation execution timing."""
    legacy = (
        (
            "REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS",
            env.get("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", ""),
            3600,
            int,
        ),
        (
            "REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS",
            env.get("REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS", ""),
            1.0,
            float,
        ),
    )
    for name, raw, former_default, parse in legacy:
        if not raw.strip():
            continue
        try:
            matches_default = parse(raw) == former_default
        except ValueError:
            matches_default = False
        if not matches_default:
            raise ValueError(
                f"{name} is retired; remove it to adopt code-owned aggregation "
                "timing (3600-second drained-cycle interval, one-second slot wait)."
            )
