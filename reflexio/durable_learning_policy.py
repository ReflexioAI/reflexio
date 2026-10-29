"""Validated durable learning timing, shared by library and enterprise schedulers.

This module stays independent of server imports so startup can validate the
resolved deployment inputs before constructing workers or importing the stack.
"""

import math
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass

DISCOVERY_POLL_SECONDS = 2.0
CLAIM_LEASE_SECONDS = 300


# The heartbeat waits lease/3; storage persists epoch + lease as signed int64.
MAX_DISCOVERY_LEASE_SECONDS = min(((1 << 63) - 1) // 2, int(threading.TIMEOUT_MAX) * 3)


@dataclass(frozen=True)
class DiscoveryTiming:
    poll_seconds: float
    lease_seconds: int


def resolve_discovery_timing(env: Mapping[str, str] | None = None) -> DiscoveryTiming:
    """Resolve independent pickup/load and crash-recovery policies once.

    The supplied mapping is authoritative, including an empty mapping. Defaults
    remain 2/300; every consumer uses the same validation and blank handling.
    """
    environment = os.environ if env is None else env
    poll_raw = environment.get("REFLEXIO_DURABLE_LEARNING_POLL_SECONDS", "").strip()
    try:
        poll = float(poll_raw) if poll_raw else float(DISCOVERY_POLL_SECONDS)
    except ValueError:
        poll = 0.0
    if not math.isfinite(poll) or not 0 < poll <= threading.TIMEOUT_MAX:
        raise ValueError(
            "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS must be finite, positive "
            "and no greater than threading.TIMEOUT_MAX."
        )
    return DiscoveryTiming(
        poll_seconds=poll, lease_seconds=resolve_user_lease_seconds(environment)
    )


def resolve_user_lease_seconds(env: Mapping[str, str] | None = None) -> int:
    """Resolve renewable user ownership without requiring a polling policy."""
    environment = os.environ if env is None else env
    lease_raw = environment.get("REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS", "").strip()
    try:
        lease = float(lease_raw) if lease_raw else float(CLAIM_LEASE_SECONDS)
    except ValueError:
        lease = 0.0
    if (
        not math.isfinite(lease)
        or not 1 <= lease <= MAX_DISCOVERY_LEASE_SECONDS
        or not lease.is_integer()
    ):
        raise ValueError(
            "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS must be a positive integer "
            "within the signed-64-bit epoch and heartbeat wait range."
        )
    return int(lease)


def renewal_interval_seconds(lease_seconds: int) -> float:
    """Renew a validated user lease three times within its lifetime."""
    return lease_seconds / 3


def validate_discovery_timing_env(env: Mapping[str, str]) -> None:
    """Validate the resolved deployment mapping before constructing workers."""
    resolve_discovery_timing(env)


AGGREGATION_MIN_INTERVAL_SECONDS = 3600
AGGREGATION_ADMISSION_WAIT_SECONDS = 1.0
# Both backends persist epoch + delay as signed 64-bit integers. Reserve half
# the range for the epoch rather than accepting an unbindable/overflowing delay.
MAX_AGGREGATION_INTERVAL_SECONDS = ((1 << 63) - 1) // 2


def aggregation_min_interval_seconds(env: Mapping[str, str] | None = None) -> int:
    """Resolve the shared drained-cycle coalescing interval (positive seconds)."""
    environment = os.environ if env is None else env
    raw = environment.get("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", "").strip()
    try:
        value = int(raw) if raw else AGGREGATION_MIN_INTERVAL_SECONDS
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_AGGREGATION_INTERVAL_SECONDS:
        raise ValueError(
            "REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS must be a positive integer "
            "within the signed-64-bit scheduling range."
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
    if not math.isfinite(value) or not 0 <= value <= threading.TIMEOUT_MAX:
        raise ValueError(
            "REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS must be finite, "
            "nonnegative and no greater than threading.TIMEOUT_MAX."
        )
    return value


def validate_aggregation_timing_env(env: Mapping[str, str]) -> None:
    """Validate supported aggregation controls against the authoritative mapping."""
    aggregation_min_interval_seconds(env)
    aggregation_admission_wait_seconds(env)
