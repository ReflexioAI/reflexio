"""Process-wide health singleton for the agent-success evaluation pipeline.

Tracks three signals that the operator-facing healthcheck surfaces:

1. Scheduler liveness — last tick timestamp (monotonic).
2. Per-reason skip counts — why a session was skipped during evaluation.
3. Producer failure counts in a rolling 24h window.

This is intentionally a global singleton, not a per-request object — the
healthcheck endpoint needs to read counts that accumulate across the
process lifetime, and the scheduler / runner / service all need to write
to the same store.
"""

from __future__ import annotations

import enum
import threading
import time
from collections import Counter, deque
from typing import Any


class SkipReason(enum.StrEnum):
    """Why a session evaluation was skipped.

    Members are the exact string the healthcheck exposes; do not rename
    casually.
    """

    ALREADY_EVALUATED = "already_evaluated"
    NO_REQUESTS = "no_requests"
    NOT_YET_COMPLETE = "not_yet_complete"
    NO_INTERACTIONS = "no_interactions"
    NO_DATA_MODELS = "no_data_models"


_FAILURE_WINDOW_SECONDS = 24 * 60 * 60


class EvalHealth:
    """Thread-safe counter store for evaluation-pipeline diagnostics.

    A module-level singleton is exposed as `_HEALTH`; the public helpers
    `record_skip`, `record_producer_failure`, `record_tick`, and
    `get_status` proxy to it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._skip_counts: Counter[SkipReason] = Counter()
        self._failures: deque[float] = deque()
        self._last_tick_monotonic: float | None = None

    def record_skip(self, reason: SkipReason) -> None:
        """Bump the counter for `reason` by one.

        Args:
            reason (SkipReason): The reason the runner skipped a session.
        """
        with self._lock:
            self._skip_counts[reason] += 1

    def record_producer_failure(self, at_ts: float | None = None) -> None:
        """Record a producer failure (LLM error or persistent save failure).

        Args:
            at_ts (float | None): Wall-clock unix timestamp; defaults to now.
        """
        ts = time.time() if at_ts is None else at_ts
        with self._lock:
            self._failures.append(ts)
            self._trim_locked(ts)

    def record_tick(self, monotonic_ts: float | None = None) -> None:
        """Record that the scheduler loop ticked.

        Args:
            monotonic_ts (float | None): Monotonic clock reading; defaults to now.
        """
        ts = time.monotonic() if monotonic_ts is None else monotonic_ts
        with self._lock:
            self._last_tick_monotonic = ts

    def get_status(self, now_ts: float | None = None) -> dict[str, Any]:
        """Return a snapshot of all health counters.

        Args:
            now_ts (float | None): Unix wall-clock used for the failure window;
                defaults to now.

        Returns:
            dict[str, Any]: Snapshot suitable for the /healthz/eval payload.
        """
        ts = time.time() if now_ts is None else now_ts
        with self._lock:
            self._trim_locked(ts)
            return {
                "skip_counts": {r.value: self._skip_counts[r] for r in SkipReason},
                "producer_failures_24h": len(self._failures),
                "last_tick_monotonic": self._last_tick_monotonic,
            }

    def _trim_locked(self, now_ts: float) -> None:
        """Drop failure timestamps older than the 24h window (caller holds lock)."""
        cutoff = now_ts - _FAILURE_WINDOW_SECONDS
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()


_HEALTH = EvalHealth()


def record_skip(reason: SkipReason) -> None:
    """Module-level proxy to the singleton."""
    _HEALTH.record_skip(reason)


def record_producer_failure(at_ts: float | None = None) -> None:
    """Module-level proxy to the singleton."""
    _HEALTH.record_producer_failure(at_ts=at_ts)


def record_tick(monotonic_ts: float | None = None) -> None:
    """Module-level proxy to the singleton."""
    _HEALTH.record_tick(monotonic_ts=monotonic_ts)


def get_status(now_ts: float | None = None) -> dict[str, Any]:
    """Module-level proxy to the singleton."""
    return _HEALTH.get_status(now_ts=now_ts)
