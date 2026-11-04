"""Scheduled local retention for offline-tuner content."""

from __future__ import annotations

import logging
from time import monotonic
from typing import Any

# Shared lifecycle horizon: readers cannot use a verdict older than the
# departed-identity records required to prove its trajectory is unchanged.
OPEN_WORLD_TRAJECTORY_RETENTION_SECONDS = 14 * 24 * 60 * 60

_STALE_CLAIM_SECONDS = 300
_CLEANUP_BATCH_LIMIT = 1000
_DEPARTURE_MAX_BATCHES = 100
_DEPARTURE_SWEEP_SECONDS = 30
_logger = logging.getLogger(__name__)


def run_optimization_retention_sweep(
    *,
    storage: Any,
    now: int,
    terminal_older_than_epoch: int,
) -> int:
    """Reclaim eligible core-owned optimization roots and artifacts."""
    stale_before_epoch = max(0, now - _STALE_CLAIM_SECONDS)
    return int(
        storage.gc_retired_optimization_jobs(
            older_than_epoch=terminal_older_than_epoch,
            stale_before_epoch=stale_before_epoch,
            limit=_CLEANUP_BATCH_LIMIT,
        )
    )


def run_departed_trajectory_retention_sweep(*, storage: Any) -> int:
    """Drain expired project identities in bounded batches and admission time."""
    deadline = monotonic() + _DEPARTURE_SWEEP_SECONDS
    total = 0
    for _ in range(_DEPARTURE_MAX_BATCHES):
        if monotonic() >= deadline:
            break
        deleted = int(
            storage.gc_departed_trajectory_mutations(limit=_CLEANUP_BATCH_LIMIT)
        )
        total += deleted
        if deleted < _CLEANUP_BATCH_LIMIT:
            return total
    # No extra inventory read: a full last batch means reclamation may remain.
    _logger.warning(
        "event=trajectory_mutation_retention_budget_exhausted deleted=%d", total
    )
    return total
