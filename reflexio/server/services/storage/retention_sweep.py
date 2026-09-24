"""Row-count retention caps, swept on the shared lineage GC scheduler.

This body used to run on the publish request thread, two lines before the
generation service's own clock started. It probes every registered retention
target -- 17 today -- each one a serialised remote round trip, so one unlucky
publish per throttle window paid for all of them and reported a fast request.
Measured at 968ms on idle staging: 52% of that publish.

It now runs once per project per lineage-GC tick. Two properties are
load-bearing and easy to lose:

- **It must run with a project bound.** Under the enterprise row-level policies
  an org-scoped pass reads ZERO ROWS rather than failing, so an unbound sweep
  deletes nothing and reports success. That is why the caller is
  ``LineageGCScheduler._sweep_project_data`` (inside ``bind_work_scope``) and
  not the ``register_per_org_sweep`` seam, which fires outside the project loop.
- **It must stay on the APP credential.** Retention is not exempt from project
  scoping: given the governance credential, a busy project would evict a quiet
  project's oldest rows org-wide, irreversibly. See
  ``supabase_storage/base/_deletion.py``.

It absorbs its own exceptions so one org cannot stall the scheduler loop, and
therefore emits its own ``retention.sweep.failed`` anomaly -- a sweep that
swallows errors silently is indistinguishable from one with nothing to do, and
nothing to do is what every tick looks like today.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from reflexio.server.error_reporting import capture_anomaly, error_tags
from reflexio.server.extensions import get_service
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.storage.retention import (
    delete_count_for_retention,
    get_row_retention_limits,
)
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.work_scope import WORK_SCOPE_PROVIDER, current_project_id

logger = logging.getLogger(__name__)

#: How long a ``storage_table_cleanup`` lease may be held before it is stale.
CLEANUP_STALE_LOCK_SECONDS = 600

#: Warn at this fraction of a target's limit, while still below it. At the
#: busiest org's ~2,500 interactions/day this is ~17 days of notice before the
#: first enforcement deletes 50,000 rows. A constant, not an env var: nothing
#: has asked for it to be tunable.
CAP_WARN_FRACTION = 0.90

#: One project-pass slower than this is reported. One third of the scheduler's
#: 60s per-org budget (``_ORG_SWEEP_TIMEOUT_SECONDS``), so a single slow pass
#: already means retention alone is a third of the org's allowance.
SLOW_SWEEP_SECONDS = 20.0

#: Tag value for a pass whose project could not be resolved. A literal, because
#: ``error_reporting._normalize_tags`` DROPS ``None`` values: tagging the real
#: ``None`` would make an unbound enterprise pass byte-identical in Sentry to a
#: correct OSS one, which is the one distinction that matters here.
UNBOUND_PROJECT_TAG = "<unbound>"


@dataclass(frozen=True, slots=True)
class RetentionSweepResult:
    """What one project-pass did, and whether it got to finish.

    ``failed`` exists because this function absorbs its own exceptions -- the
    call site in ``gc_scheduler`` has no backstop, by contract -- and a caller
    that cannot tell "swept, nothing due" from "could not sweep at all" will
    schedule its next attempt as if all is well. On staging that meant a
    ``PGRST002`` transient at the boot tick put retention off for a full
    ``poll_interval_seconds`` (86400s by default).

    Attributes:
        deleted (int): Rows removed across all targets this pass.
        failed (bool): True when the pass could not complete -- a refused lease
            or a disabled cap is NOT a failure, because there was nothing to do.
            Nor is a *single* target raising: that is isolated deliberately, is
            usually permanent (a table the backend does not have), and escalating
            it would spend the scheduler's bounded fast-retry budget on something
            a retry cannot fix. Such a target is still logged with ``error_tags``.
            But EVERY target failing is a different animal -- that is the shape a
            backend-wide transient takes, and absorbing it target by target would
            hand the scheduler a clean-looking tick.
    """

    deleted: int
    failed: bool = False


def sweep_retention_caps(org_id: str, storage: BaseStorage) -> RetentionSweepResult:
    """Enforce row-count caps for the project bound on this thread.

    Args:
        org_id (str): Org being swept, for lease scope and anomaly attribution.
        storage (BaseStorage): The org's app-role storage.

    Returns:
        RetentionSweepResult: Rows deleted, and whether the pass could not
        complete. The scheduler reads ``failed`` to decide how soon to tick
        again; see ``gc_scheduler._FAILED_TICK_RETRY_SECONDS``.
    """
    # `started` is outside the try because `time.monotonic()` cannot raise;
    # `project_id` is pre-bound so the failure handler can tag with it even if
    # resolving it is what failed. Everything that CAN raise is inside, because
    # `gc_scheduler._sweep_project_data` calls this with no backstop of its own
    # -- an escape from here aborts the org's remaining projects AND the
    # enterprise per-org governance sweep, and on the serial fan-out path every
    # remaining org in the tick.
    started = time.monotonic()
    project_id: str | None = None
    deleted_total = 0
    targets_failed = 0
    try:
        project_id = current_project_id()

        # Refuse an unbound pass rather than trusting the call site's position.
        # With a provider registered, "no project bound" means the scope was
        # lost: under the row-level policies that pass reads ZERO ROWS and
        # reports success, so it is invisible to every count-based check. A
        # deployment with no provider at all is OSS, where no project exists and
        # an unscoped sweep is the correct behaviour.
        if project_id is None and get_service(WORK_SCOPE_PROVIDER) is not None:
            capture_anomaly(
                "retention.sweep.unbound",
                org_id=org_id,
                project_id=UNBOUND_PROJECT_TAG,
            )
            logger.error(
                "event=retention_sweep_unbound org_id=%s -- refusing to sweep: a "
                "work-scope provider is registered but no project is bound",
                org_id,
            )
            return RetentionSweepResult(0, failed=True)

        limits = {
            target_name: limit
            for target_name, limit in get_row_retention_limits().items()
            if limit > 0
        }
        if not limits:
            return RetentionSweepResult(0)

        mgr = OperationStateManager(
            storage,  # type: ignore[reportArgumentType]
            org_id,
            "storage_table_cleanup",  # type: ignore[reportArgumentType]
        )
        if not mgr.acquire_simple_lock(stale_seconds=CLEANUP_STALE_LOCK_SECONDS):
            return RetentionSweepResult(0)
        try:
            for target_name, limit in limits.items():
                # Isolate per-target failures so one bad table does not
                # short-circuit every subsequent target.
                try:
                    deleted_total += _sweep_target(
                        org_id, project_id, storage, target_name, limit
                    )
                except Exception as exc:  # noqa: BLE001
                    targets_failed += 1
                    with error_tags(
                        subsystem="retention",
                        op="sweep_target",
                        org_id=org_id,
                        target_name=target_name,
                        error_type=type(exc).__name__,
                    ):
                        logger.exception(
                            "Failed to sweep retention target %s", target_name
                        )
        finally:
            mgr.release_simple_lock()
    except Exception as exc:  # noqa: BLE001
        capture_anomaly(
            "retention.sweep.failed",
            org_id=org_id,
            project_id=project_id or UNBOUND_PROJECT_TAG,
            error_type=type(exc).__name__,
        )
        logger.exception("event=retention_sweep_failed org_id=%s", org_id)
        return RetentionSweepResult(deleted_total, failed=True)

    elapsed = time.monotonic() - started
    if elapsed > SLOW_SWEEP_SECONDS:
        capture_anomaly(
            "retention.sweep.slow",
            org_id=org_id,
            project_id=project_id,
            elapsed_seconds=round(elapsed, 1),
            targets=len(limits),
        )
    # EVERY target failing is a different animal from one failing: a single bad
    # table is isolated and usually permanent, but a backend-wide transient
    # (the PGRST002 that prompted the retry cadence) lands on all of them and
    # would otherwise be absorbed target by target into a clean-looking tick.
    if targets_failed and targets_failed == len(limits):
        capture_anomaly(
            "retention.sweep.all_targets_failed",
            org_id=org_id,
            project_id=project_id or UNBOUND_PROJECT_TAG,
            targets=targets_failed,
        )
        return RetentionSweepResult(deleted_total, failed=True)
    return RetentionSweepResult(deleted_total)


def _sweep_target(
    org_id: str,
    project_id: str | None,
    storage: BaseStorage,
    target_name: str,
    limit: int,
) -> int:
    """Probe one target, warn near its cap, delete at it.

    Args:
        org_id (str): Org being swept, for anomaly attribution.
        project_id (str | None): Project bound on this thread, or ``None``
            where projects do not exist (OSS).
        storage (BaseStorage): The org's app-role storage.
        target_name (str): Retention target being probed.
        limit (int): Row cap for this target, already known positive.

    Returns:
        int: Rows deleted for this target.
    """
    # `count_retention_target_rows` / `delete_oldest_retention_target_rows` are
    # supplied by `RetentionMixin`, which every SQL backend mixes in. They are
    # deliberately NOT declared on `BaseStorage` -- see the note at
    # `storage_base/_base.py`: leaving them off is what lets a test double
    # subclass `BaseStorage` without implementing retention at all. So the
    # caller genuinely holds a `BaseStorage` and the attribute is genuinely
    # absent from that declared type; the ignore records that, rather than
    # widening the parameter to a Protocol the call site would then have to cast
    # to. A backend that really lacks the hook raises `AttributeError`, which
    # the per-target isolation above absorbs.
    total_count = storage.count_retention_target_rows(target_name)  # type: ignore[reportAttributeAccessIssue]
    if total_count < limit:
        if total_count >= limit * CAP_WARN_FRACTION:
            capture_anomaly(
                "retention.cap.approaching",
                org_id=org_id,
                project_id=project_id,
                target=target_name,
                rows=total_count,
                limit=limit,
            )
        return 0

    delete_count = delete_count_for_retention(total_count)
    deleted = storage.delete_oldest_retention_target_rows(  # type: ignore[reportAttributeAccessIssue]
        target_name, delete_count
    )
    capture_anomaly(
        "retention.cap.enforced",
        org_id=org_id,
        project_id=project_id,
        target=target_name,
        rows_before=total_count,
        deleted=deleted,
        limit=limit,
    )
    logger.info(
        "Cleaned up %d oldest %s row(s) (total was %d, limit %d)",
        deleted,
        target_name,
        total_count,
        limit,
    )
    return deleted
