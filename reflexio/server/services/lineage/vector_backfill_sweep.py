"""Missing-vector backfill sweep for the shared ``LineageGCScheduler``.

When an embedding call fails or degrades, the ingest path stores an empty
embedding (and writes no vector row), returns success, and nothing ever
re-embeds the row — so that interaction is invisible to vector/hybrid search
forever. This is a latent durability hole. This module registers a bounded,
idempotent, per-org sweep that re-embeds those interactions on the shared GC
scheduler cadence.

Design:

- **Per-org**: registered via :func:`register_per_org_sweep`, so it runs once
  per org per tick with the org id supplied by the scheduler (the ``(org_id,
  now) -> int`` seam). The global ``register_global_sweep`` seam only passes
  ``now`` and cannot drive a per-org backfill, so it is the wrong hook here.
- **Bounded**: re-embeds at most ``REFLEXIO_MISSING_VECTOR_BACKFILL_CAP``
  interactions per org per tick (prior art warns unbounded backfills flood
  logs / embedding cost — see ``llm/_litellm_embedding.py``).
- **One effective budget**: ``REFLEXIO_MISSING_VECTOR_BACKFILL_CAP`` defaults
  to 200; zero disables registration and execution. A positive budget bounds
  each org/tick. Negative/unparseable caps retain their fallback to 200.
  Registered hooks re-read the effective budget each tick. A startup disabled
  by a zero budget does not install this hook; reenabling then needs a restart.

  Registering any per-org hook starts the shared ``LineageGCScheduler`` even
  when no bootstrap-org flag is set. Other sweeps retain their own config gates.
- **Fail-safe**: the storage layer catches ``EmbeddingUnavailableError`` and
  returns 0 (work left for the next tick); this closure additionally captures
  any unexpected error as an anomaly and returns the count so far, so one org's
  failure never stalls the shared loop.
- **Idempotent**: once an interaction's vector is written it no longer matches
  detection and is skipped next tick.
"""

from __future__ import annotations

import logging
import os

from reflexio.removed_env import reject_removed_env
from reflexio.server.env_utils import env_str

logger = logging.getLogger(__name__)

CAP_ENV_VAR = "REFLEXIO_MISSING_VECTOR_BACKFILL_CAP"
_DEFAULT_CAP = 200

# Guard against duplicate registration when the app lifespan runs more than once
# in a single process (e.g. test apps).
_installed = False


def _resolve_cap() -> int:
    """Return the per-org per-tick backfill cap from the environment.

    Zero disables repair. Removed enable assignments require migration to the canonical cap.
    Unset, blank, negative or unparseable caps retain the default of 200.
    """
    reject_removed_env(os.environ, canonical=CAP_ENV_VAR)
    raw = env_str(CAP_ENV_VAR, str(_DEFAULT_CAP))
    try:
        cap = int(raw)
    except ValueError:
        logger.warning(
            "event=missing_vector_backfill_bad_cap value=%r — using default %d",
            raw,
            _DEFAULT_CAP,
        )
        return _DEFAULT_CAP
    return cap if cap >= 0 else _DEFAULT_CAP


def missing_vector_backfill_sweep(org_id: str, _now: int) -> int:
    """Re-embed a bounded batch of vector-less interactions for one org.

    Registered with the OSS ``register_per_org_sweep`` seam. Resolves the budget
    each tick so live config changes are honoured. Absorbs its own
    errors (returns instead of raising) and therefore emits its own anomaly.

    Args:
        org_id (str): The org to sweep.
        _now (int): Current unix epoch supplied by the scheduler. Unused — the
            cap, not a time cutoff, bounds the work.

    Returns:
        int: Number of interactions whose vector was backfilled (0 on skip/error).
    """
    cap = _resolve_cap()
    if cap == 0:
        return 0

    # Imported lazily so importing this module never drags in the request stack.
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.error_reporting import capture_anomaly

    try:
        ctx = RequestContext(org_id=org_id)
        if ctx.storage is None:
            return 0
        backfilled = ctx.storage.backfill_missing_interaction_vectors(cap)
        if backfilled:
            logger.info(
                "event=missing_vector_backfill_tick org_id=%s backfilled=%d",
                org_id,
                backfilled,
            )
        return backfilled
    except Exception:
        capture_anomaly(
            "interactions.missing_vector_backfill.run_failed", org_id=org_id
        )
        logger.exception("event=missing_vector_backfill_org_failed org_id=%s", org_id)
        return 0


def install_missing_vector_backfill_sweep() -> None:
    """Register the backfill sweep on the OSS lineage GC scheduler unless disabled.

    A positive effective budget registers the hook, which itself is a shared
    scheduler start condition. Zero does not register it or start the loop.

    Idempotent: a second call in the same process is a no-op. Non-fatal: a
    registration failure logs and skips rather than aborting app startup.

    Call this before ``maybe_start_lineage_gc`` so the registered hook is
    visible when the scheduler evaluates its start conditions.
    """
    global _installed  # noqa: PLW0603
    if _installed:
        return
    cap = _resolve_cap()
    if cap == 0:
        return
    try:
        from reflexio.server.services.lineage.gc_scheduler import (
            register_per_org_sweep,
        )

        register_per_org_sweep(missing_vector_backfill_sweep)
        _installed = True
        logger.info("event=missing_vector_backfill_sweep_registered cap=%d", cap)
    except Exception:
        from reflexio.server.error_reporting import capture_anomaly

        capture_anomaly("interactions.missing_vector_backfill.install_failed")
        logger.exception("event=missing_vector_backfill_install_failed")
