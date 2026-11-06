"""One durable progress record per sweep tick, and the queue read it drains.

WHY THIS MODULE EXISTS
----------------------
The sweep takes ONE playbook per tick. Without a record of the tick, the queue
never shrinks, because the majority outcome raises **before any job row
exists** -- no evidence, a stale incumbent, a fenced regeneration, a
governance-invalid subject. Nothing durable says the tick happened, so the same
playbook is selected again next tick, forever: at a two-minute interval that is
a full 14-day snapshot read 720 times a day, per playbook.

It is also the only SAME-DAY SPEND GUARD. All three unique indexes on
``playbook_optimization_jobs`` (``uq_poj_active_attempt``,
``uq_poj_active_discovery``, ``uq_poj_active_target``) are partial on
``status IN ('pending','running')``, so the moment an attempt terminalizes none
of them covers it any more and nothing in the durable layer stops a second,
fully paid attempt on the same playbook the same day. A dropped progress record
is a money bug.

WHAT THIS MODULE IS NOT
-----------------------
It composes nothing open-world. It is deliberately free of any import from
``offline_tuner.open_world`` so it can be called from the tick loop before the
pipeline is resolved, and so it does not widen the composition boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

from reflexio.server.cache import reflexio_cache
from reflexio.server.ops_log import get_ops_logger
from reflexio.server.services.offline_tuner.config import TUNER_OUTCOME_LOG_LEVEL
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    EVIDENCE_FAULT_REASONS,
)
from reflexio.server.services.offline_tuner.evidence.open_world_policy import (
    PRODUCTION_OPEN_WORLD_POLICY,
)
from reflexio.server.services.offline_tuner.open_world_port import (
    OpenWorldSweepProgressStore,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    EVIDENCE_REATTEMPT_PREFIX,
    SweepQueueEntry,
)

logger = logging.getLogger(__name__)
_ops = get_ops_logger("offline_tuner")

#: Tick outcomes that mean something went WRONG, so someone should look.
#:
#: An explicit list on purpose, not "everything that is not a success". A new
#: ``ineligible_*`` or ``abstained_*`` reason is the playbook's own answer and
#: defaults to the INFO ops channel; a new failure kind has to be added here
#: deliberately. The composition module owns these spellings
#: (``SWEEP_OUTCOME_ERROR``, ``SWEEP_OUTCOME_CONFLICT``,
#: ``SWEEP_OUTCOME_TERMINAL_PREFIX``); a test there must assert the two agree,
#: because this module may not import that one.
PROBLEM_TICK_OUTCOMES = frozenset({"attempt_error", "attempt_conflict"})
PROBLEM_TICK_OUTCOME_PREFIX = "terminal_failure_"
#: An ``abstained_*`` label carries a reason the same way ``ineligible_*``
#: (``EVIDENCE_REATTEMPT_PREFIX``) does. Either is routine UNLESS its reason is
#: an evidence-data fault (``EVIDENCE_FAULT_REASONS``): those hold tuning off
#: until someone fixes the data, and this line is their only report.
_ABSTAINED_PREFIX = "abstained_"


def is_problem_outcome(outcome: object) -> bool:
    """Whether a tick outcome label names a failure rather than an answer.

    Args:
        outcome (object): The label, as passed or as storage echoed it back.
            Anything that is not a string is not a problem label.

    Returns:
        bool: True for ``attempt_error``, ``attempt_conflict``, every
        ``terminal_failure_*``, and an ``ineligible_*`` / ``abstained_*``
        label whose reason is in ``EVIDENCE_FAULT_REASONS``; False for
        everything else.
    """
    if not isinstance(outcome, str):
        return False
    label = outcome.strip()
    if label in PROBLEM_TICK_OUTCOMES or label.startswith(PROBLEM_TICK_OUTCOME_PREFIX):
        return True
    return any(
        label.startswith(prefix) and label[len(prefix) :] in EVIDENCE_FAULT_REASONS
        for prefix in (EVIDENCE_REATTEMPT_PREFIX, _ABSTAINED_PREFIX)
    )


#: Ceiling on one queue read. The sweep consumes one playbook per tick, so this
#: bounds a diagnostic/planning read, not the work. Kept small enough that a
#: fleet-wide merge across many orgs stays cheap.
DEFAULT_QUEUE_LIMIT = 100


class UnsupportedSweepProgressStorageError(TypeError):
    """The org's resolved storage does not implement the sweep-progress port.

    A PERMANENT property of that org's storage rather than a transient fault,
    so a retry on the next tick fails identically. Subclasses ``TypeError`` so
    callers that already catch ``TypeError`` around storage resolution keep
    working.
    """


_SWEEP_PROGRESS_METHODS = (
    "record_open_world_sweep_progress",
    "list_open_world_playbooks_without_progress",
)


def _storage_for_org(org_id: str) -> OpenWorldSweepProgressStore:
    return cast(
        OpenWorldSweepProgressStore,
        reflexio_cache.get_reflexio(org_id).get_storage(),
    )


def _sweep_store(storage: object) -> OpenWorldSweepProgressStore:
    """The storage, refused up front unless it serves both port methods."""
    if not all(
        callable(getattr(storage, name, None)) for name in _SWEEP_PROGRESS_METHODS
    ):
        raise UnsupportedSweepProgressStorageError(
            "sweep progress requires OpenWorldSweepProgressStore support"
        )
    return cast(OpenWorldSweepProgressStore, storage)


def record_tick_outcome(
    *,
    org_id: str,
    playbook_id: int,
    outcome: str,
    job_id: int | None = None,
    coverage: tuple[int, int] | None = None,
    detail: str | None = None,
    storage_for_org: Callable[[str], OpenWorldSweepProgressStore] = _storage_for_org,
) -> None:
    """Record that this playbook had its turn today, whatever the outcome.

    Called on EVERY terminal tick outcome, including -- especially -- the
    ineligible ones that raise before a job row exists. Those are the majority,
    and they are the ones that leave no other trace.

    ``org_id`` is a ``str`` rather than the ``int`` the plan's interface block
    sketches, because every storage boundary this has to sit beside takes a
    string: ``resolve_open_world_pipeline(org_id: str)`` and
    ``reflexio_cache.get_reflexio(org_id)``. A sweep that must call both cannot
    be holding an int.

    Args:
        org_id (str): The organization whose tenant schema owns the record.
        playbook_id (int): ``user_playbook_id`` the tick was spent on.
        outcome (str): Free-form tick outcome label. Not enumerated on purpose
            -- see the migration's note: a sweep-tick outcome is a superset of
            the job terminal outcomes, and enumerating it would guarantee a
            second vocabulary migration for a column nothing dispatches on.
        job_id (int | None): The optimization job, when the tick got far enough
            to create one. ``None`` for every ineligible outcome, which is the
            common case.
        coverage (tuple[int, int] | None): ``(numerator, denominator)`` of
            the coverage ratio a reconstructability abstain failed on. Logged
            as ``coverage=n/d`` so a coverage fault is diagnosable from the
            line alone; never persisted. ``None`` for every other outcome.
        detail (str | None): Value-free ``key=value`` attribution of a
            terminal failure (refusal name, exception class and location),
            appended to the line as-is. Logged, never persisted.
        storage_for_org (Callable[[str], OpenWorldSweepProgressStore]):
            Storage resolution seam, injectable for tests.

    Raises:
        ValueError: The identity is unusable, so no record could be written.
            Callers must NOT swallow this: a tick with no record is a tick that
            will be paid for again.
        UnsupportedSweepProgressStorageError: The org's storage does not
            implement the sweep-progress port.
    """
    if playbook_id <= 0:
        raise ValueError("sweep progress playbook id is invalid")
    if not outcome.strip():
        raise ValueError("sweep progress outcome is empty")
    if job_id is not None and job_id <= 0:
        raise ValueError("sweep progress job id is invalid")
    record = _sweep_store(storage_for_org(org_id)).record_open_world_sweep_progress(
        user_playbook_id=playbook_id, outcome=outcome.strip(), job_id=job_id
    )
    stored = record.outcome
    fields = (
        org_id,
        playbook_id,
        stored,
        record.job_id,
        record.progress_day,
        "" if coverage is None else f" coverage={coverage[0]}/{coverage[1]}",
        "" if detail is None else f" {detail}",
    )
    # Split by severity. This line fires on EVERY tick, and most ticks are an
    # `ineligible_*` answer -- routine, so it goes to the ops channel at INFO,
    # which `configure_logging` pins visible in every profile. Only a failure
    # stays a WARNING, under its own event name so a severity guard can tell
    # the two apart; the shared
    # `event=open_world_sweep_tick_outcome` prefix keeps one grep finding both.
    # Either label decides: what was asked for, or what storage kept.
    if is_problem_outcome(outcome) or is_problem_outcome(stored):
        logger.log(
            TUNER_OUTCOME_LOG_LEVEL,
            "event=open_world_sweep_tick_outcome_problem release_blocking=true org=%s user_playbook_id=%s "
            "outcome=%s job_id=%s progress_day=%s%s%s",
            *fields,
        )
    else:
        _ops.info(
            "event=open_world_sweep_tick_outcome org=%s user_playbook_id=%s "
            "outcome=%s job_id=%s progress_day=%s%s%s",
            *fields,
        )


class _FromPolicy:
    """Sentinel default for ``window_seconds``: resolve the policy AT CALL TIME.

    NOT ``= PRODUCTION_OPEN_WORLD_POLICY.historical_window_seconds`` directly.
    A default argument is evaluated once, at import, so that spelling freezes
    the window at whatever the policy said when this module was first loaded and
    stops tracking it afterwards -- the exact drift into SILENT NON-TUNING the
    call site below warns about, reintroduced by the mechanism meant to avoid
    restating the number. ``test_the_window_sent_is_the_policy_window_and_not_a_literal``
    and ``test_narrowing_the_policy_window_narrows_the_queue`` both fail on it.

    A sentinel is needed rather than ``None`` because ``None`` already means
    something here: the storage read's unfiltered arm.
    """


_FROM_POLICY = _FromPolicy()


def playbooks_without_progress_today(
    org_id: str,
    *,
    limit: int = DEFAULT_QUEUE_LIMIT,
    window_seconds: int | None | _FromPolicy = _FROM_POLICY,
    storage_for_org: Callable[[str], OpenWorldSweepProgressStore] = _storage_for_org,
) -> tuple[SweepQueueEntry, ...]:
    """Current playbooks with no progress record today, oldest attempt first.

    The ordering key is an ABSOLUTE last-attempt epoch, not a rank within the
    org, and a playbook that has never had a tick sorts first. That matters
    because ``run_open_world_sweep(org)`` is per-org: an ordering meaningful
    only inside one org bounds the axis that does not saturate and leaves the
    one that does unbounded, so whichever org the daemon visits first is served
    every night and later orgs starve forever. An absolute key stays comparable
    across orgs, so a scheduler that interleaves them gets fleet-wide
    oldest-first.

    The key is RETURNED alongside the id, and that is load-bearing rather than
    convenient. Fleet-wide ordering is not something the daemon can recover
    from ids: storage is per-org tenant schema, so no single global query
    exists, and a merge across orgs is the only shape available -- which needs
    a comparable key from each org to merge ON. An earlier revision of this
    function returned ``tuple[int, ...]``; with that shape the fleet-wide
    property the sweep is specified against was unachievable, not merely
    unimplemented.

    Args:
        org_id (str): The organization to read.
        limit (int): Ceiling on rows returned. Bounds a planning read; the
            sweep consumes one per tick.
        window_seconds (int | None | _FromPolicy): Activity window the candidate
            filter applies; left unset it is read from the policy on every call
            (see :class:`_FromPolicy` for why that is not a default argument).
            ``None`` selects the storage read's UNFILTERED arm -- every current playbook,
            the pre-funnel question. The sweep must never pass ``None``; the one
            caller that does is the status endpoint, distinguishing an empty
            queue that means "caught up" from one holding playbooks out for
            inactivity, which an empty result alone cannot tell apart.
        storage_for_org (Callable[[str], OpenWorldSweepProgressStore]):
            Storage resolution seam, injectable for tests.

    Returns:
        tuple[SweepQueueEntry, ...]: ``(playbook_id, last_attempt_epoch)``
        pairs, oldest attempt first, with :data:`NEVER_ATTEMPTED_EPOCH` for a
        playbook that has never had a tick. Empty once every candidate
        playbook has had its turn today -- which is the queue draining, and the
        whole point. Since the activity funnel, empty ALSO covers "no playbook
        has been active inside ``window_seconds``", which is not the same fact;
        see ``window_seconds``.
    """
    # THE WINDOW IS DERIVED, NEVER RESTATED. Storage takes seconds and floors
    # them against its OWN clock rather than taking a floor computed here, for
    # two reasons that pull the same way: one number for the window instead of
    # two that can drift, and a superset guarantee that holds without assuming
    # anything about two clocks agreeing. A literal `14 * 86400` in a backend
    # would drift into SILENT NON-TUNING -- a playbook stops being a candidate,
    # nothing raises, and no surface reports it.
    return _sweep_store(
        storage_for_org(org_id)
    ).list_open_world_playbooks_without_progress(
        limit=int(limit),
        window_seconds=(
            int(PRODUCTION_OPEN_WORLD_POLICY.historical_window_seconds)
            if isinstance(window_seconds, _FromPolicy)
            else (None if window_seconds is None else int(window_seconds))
        ),
    )
