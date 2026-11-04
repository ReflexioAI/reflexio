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
from typing import Any, NamedTuple, Protocol, cast

from reflexio.server.cache import reflexio_cache
from reflexio.server.ops_log import get_ops_logger
from reflexio.server.services.offline_tuner.config import TUNER_OUTCOME_LOG_LEVEL
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    EVIDENCE_FAULT_REASONS,
)
from reflexio.server.services.offline_tuner.evidence.open_world_policy import (
    PRODUCTION_OPEN_WORLD_POLICY,
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


_RECORD_RPC = "record_open_world_sweep_progress"
_QUEUE_RPC = "load_open_world_playbooks_without_progress"

#: Ceiling on one queue read. The sweep consumes one playbook per tick, so this
#: bounds a diagnostic/planning read, not the work. Kept small enough that a
#: fleet-wide merge across many orgs stays cheap.
DEFAULT_QUEUE_LIMIT = 100

# ---------------------------------------------------------------------------
# THE ONE RETRY-ELIGIBLE OUTCOME
# ---------------------------------------------------------------------------
#
# Every other tick outcome spends the playbook's day. This one does not, and it
# is the only one that does not.
#
# WHY IT HAD TO BE ITS OWN STATE. Both obvious answers for a rate-limited tick
# are wrong, and they are wrong in opposite directions:
#
#   * Write an ordinary day-bucket record and the playbook is skipped until
#     00:00 UTC over a provider condition that typically clears in one window.
#     The day is spent on an attempt that was never answered.
#   * Write nothing and the queue read -- whose whole predicate is "no progress
#     record today" -- returns the same playbook on the very next tick, every
#     tick, and the sweep takes ONE playbook per tick, so the entire org's
#     queue stalls behind it.
#
# So the record is written (the queue moves on) and carries a state that lets
# the row STOP blocking once a cooldown has elapsed: attempted, not completed,
# retry eligible.
#
# WHY DISPATCHING ON THE LABEL IS ACCEPTABLE HERE. `20260911020000` gave
# `outcome` no vocabulary CHECK on the express ground that "a column nothing
# dispatches on" must not acquire one -- and this makes exactly one read
# dispatch on exactly one value. That is a narrower coupling than a CHECK: a
# new outcome label still needs no migration, and an unrecognised label simply
# spends the day, which is the safe direction. What it DOES create is a
# dual-expression identity between this constant and the literal inside the
# storage SQL, which must be asserted by a test rather than kept in step by
# hand.
RETRY_ELIGIBLE_OUTCOME = "rate_limited"

#: How long a rate-limited record blocks its playbook before the queue may
#: return it.
#:
#: Not a provider-derived number -- MiniMax publishes no `Retry-After` and no
#: 429 contract, so there is nothing to derive it from. It is a QUEUE-FAIRNESS
#: number: at the 120s tick interval, 900s is roughly seven ticks in which
#: other playbooks get their turn, which is what stops one rate-limited
#: playbook monopolising a serialized queue. Shorter and the stall returns;
#: much longer and a limit that clears inside the hour loses the day anyway.
RATE_LIMIT_RETRY_COOLDOWN_SECONDS = 900

#: Ceiling on how many ticks ONE playbook may consume in one UTC day.
#:
#: This is the money bound, and without it the cooldown above is an unbounded
#: spend authorisation: a provider rate-limited all day would hand the same
#: playbook a fresh attempt every 15 minutes, ~96 times, each one re-paying for
#: whichever stages completed before the limit was hit. Three means the first
#: attempt plus at most two retries.
MAX_SAME_DAY_SWEEP_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# THE EVIDENCE-GATED RE-ATTEMPT
# ---------------------------------------------------------------------------
#
# The second exception to "a record spends the day", enforced by the storage
# backend's sweep-queue read. A playbook swept minutes
# after its first exposure records `ineligible_*` for want of judged sessions;
# without this it waits until 00:00 UTC however much traffic arrives. The row
# stops blocking once the cooldown below has elapsed AND a session that exposed
# the playbook has a newer outcome or a newer request. The new-request arm is
# what keeps it from deadlocking: the outcome bridge judges on behalf of the
# tick's playbook, so a spent playbook's sessions need never get a new outcome.
#
# The cap is MAX_SAME_DAY_SWEEP_ATTEMPTS, SHARED with the rate-limited
# exception rather than a second budget: `attempt_count` counts every record of
# the day whatever its outcome. Like RETRY_ELIGIBLE_OUTCOME, both values are
# restated as literals in a storage backend's SQL, which must agree with them.

#: Outcome-label prefix whose rows may be re-attempted the same day. The sweep
#: writes ``ineligible_<reason>`` from this (``SWEEP_OUTCOME_INELIGIBLE_PREFIX``
#: aliases it), and the queue read matches it with ``starts_with``.
EVIDENCE_REATTEMPT_PREFIX = "ineligible_"

#: How long an ``ineligible_*`` record blocks its playbook before new evidence
#: may re-admit it. Shorter than the rate-limit cooldown because nothing
#: external has to clear -- the evidence gate already requires new activity --
#: so this only stops one busy session re-admitting its playbook every tick.
EVIDENCE_REATTEMPT_COOLDOWN_SECONDS = 300

#: How far BEFORE an ``ineligible_*`` attempt a request still counts as new
#: evidence. The attempt's bridge pass only judges sessions that are READY --
#: a reaction settled ``REACTION_SETTLE_SECONDS`` (30), or the session idle for
#: the judge's floor ``GROUP_EVALUATION_DELAY_SECONDS`` (600) -- so a session
#: whose last request landed less than 600s before the attempt was still
#: PENDING at it: invisible to that attempt, judgeable later. Without this the
#: rule only counts activity AFTER the attempt, and a sweep that lands seconds
#: after the last turn of sessions that then end loses the playbook its day.
#: 600 is the longest pending case; it also covers the 30s settle and the 300s
#: re-judge interval. Restated as a literal in a storage backend's SQL, which
#: must agree with it.
EVIDENCE_REATTEMPT_LOOKBACK_SECONDS = 600

#: The ordering key a playbook that has NEVER had a tick carries.
#:
#: The SQL orders ``NULLS FIRST`` because a never-attempted playbook is the
#: oldest thing there is. A sentinel BELOW every real epoch is what makes plain
#: tuple comparison in Python reproduce that ordering exactly -- the merge in
#: the daemon is a ``min`` over entries from several orgs, and ``None`` would
#: make that raise ``TypeError`` while ``0`` would collide with a (legal, if
#: never-observed) stored ``attempted_at`` of zero. Anything a caller sorts by
#: therefore sorts the same way the database did.
NEVER_ATTEMPTED_EPOCH = -1


class SweepQueueEntry(NamedTuple):
    """One queue position: the playbook, and the key it is ordered by.

    The KEY is carried, not just the id, and that is the whole point of this
    type. ``run_open_world_sweep(org)`` is per-org, so a caller holding only
    ids can do no better than visit orgs in some fixed sequence -- which serves
    whichever org it reaches first every night and starves the rest forever,
    the exact failure fleet-wide oldest-attempt-first ordering exists to
    prevent. Storage is per-org tenant schema, so there is no single global
    query to ask instead; the ordering key has to leave this function for the
    merge to happen anywhere at all.
    """

    playbook_id: int
    last_attempt_epoch: int


class SweepProgressStore(Protocol):
    """The single storage seam this module needs: the tenant RPC transport."""

    def _rpc(
        self, fn: str, params: dict[str, Any] | None = ..., route: str = ...
    ) -> Any: ...


class UnsupportedSweepProgressStorageError(TypeError):
    """The org's resolved storage cannot serve the sweep-progress RPCs.

    A PERMANENT property of that org's storage rather than a transient fault --
    the SQLite backend has no tenant RPC transport at all -- so a retry on the
    next tick fails identically. Subclasses ``TypeError`` so callers that
    already catch ``TypeError`` around storage resolution keep working.
    """


def _storage_for_org(org_id: str) -> SweepProgressStore:
    return cast(SweepProgressStore, reflexio_cache.get_reflexio(org_id).get_storage())


def _rpc_rows(
    storage: SweepProgressStore,
    name: str,
    params: dict[str, Any],
    *,
    route: str,
) -> list[dict[str, Any]]:
    rpc = cast(Any, getattr(storage, "_rpc", None))
    if not callable(rpc):
        raise UnsupportedSweepProgressStorageError(
            "sweep progress requires tenant RPC support"
        )
    response = cast(Any, rpc(name, params, route=route)).execute()
    data = getattr(response, "data", None)
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        # Fail loud. The two shapes this could be coerced into are both
        # dangerous: an empty queue looks like "nothing to do", and a missing
        # record looks like "today is still free", which is the second paid
        # attempt this module exists to refuse.
        raise RuntimeError(f"{name} returned malformed durable state")
    return cast(list[dict[str, Any]], data)


def record_tick_outcome(
    *,
    org_id: str,
    playbook_id: int,
    outcome: str,
    job_id: int | None = None,
    coverage: tuple[int, int] | None = None,
    detail: str | None = None,
    storage_for_org: Callable[[str], SweepProgressStore] = _storage_for_org,
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
        storage_for_org (Callable[[str], SweepProgressStore]): Storage
            resolution seam, injectable for tests.

    Raises:
        ValueError: The identity is unusable, so no record could be written.
            Callers must NOT swallow this: a tick with no record is a tick that
            will be paid for again.
    """
    if playbook_id <= 0:
        raise ValueError("sweep progress playbook id is invalid")
    if not outcome.strip():
        raise ValueError("sweep progress outcome is empty")
    if job_id is not None and job_id <= 0:
        raise ValueError("sweep progress job id is invalid")
    rows = _rpc_rows(
        storage_for_org(org_id),
        _RECORD_RPC,
        {
            "p_user_playbook_id": playbook_id,
            "p_outcome": outcome.strip(),
            "p_job_id": job_id,
        },
        route="writer",
    )
    if len(rows) != 1:
        raise RuntimeError("sweep progress record returned malformed durable state")
    stored = rows[0].get("outcome")
    fields = (
        org_id,
        playbook_id,
        stored,
        rows[0].get("job_id"),
        rows[0].get("progress_day"),
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
    something here: the RPC's unfiltered arm.
    """


_FROM_POLICY = _FromPolicy()


def playbooks_without_progress_today(
    org_id: str,
    *,
    limit: int = DEFAULT_QUEUE_LIMIT,
    window_seconds: int | None | _FromPolicy = _FROM_POLICY,
    storage_for_org: Callable[[str], SweepProgressStore] = _storage_for_org,
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
            ``None`` selects the RPC's UNFILTERED arm -- every current playbook,
            the pre-funnel question. The sweep must never pass ``None``; the one
            caller that does is the status endpoint, distinguishing an empty
            queue that means "caught up" from one holding playbooks out for
            inactivity, which an empty result alone cannot tell apart.
        storage_for_org (Callable[[str], SweepProgressStore]): Storage
            resolution seam, injectable for tests.

    Returns:
        tuple[SweepQueueEntry, ...]: ``(playbook_id, last_attempt_epoch)``
        pairs, oldest attempt first, with :data:`NEVER_ATTEMPTED_EPOCH` for a
        playbook that has never had a tick. Empty once every candidate
        playbook has had its turn today -- which is the queue draining, and the
        whole point. Since the activity funnel, empty ALSO covers "no playbook
        has been active inside ``window_seconds``", which is not the same fact;
        see ``window_seconds``.
    """
    # THE WINDOW IS DERIVED, NEVER RESTATED. The RPC takes seconds and floors
    # them against its OWN clock rather than taking a floor computed here, for
    # two reasons that pull the same way: one number for the window instead of
    # two that can drift, and a superset guarantee that holds without assuming
    # anything about two clocks agreeing. A literal `14 * 86400` in the SQL
    # would drift into SILENT NON-TUNING -- a playbook stops being a candidate,
    # nothing raises, and no surface reports it.
    rows = _rpc_rows(
        storage_for_org(org_id),
        _QUEUE_RPC,
        {
            "p_limit": max(int(limit), 1),
            "p_window_seconds": (
                int(PRODUCTION_OPEN_WORLD_POLICY.historical_window_seconds)
                if isinstance(window_seconds, _FromPolicy)
                else (None if window_seconds is None else int(window_seconds))
            ),
        },
        route="reader",
    )
    return tuple(
        SweepQueueEntry(
            playbook_id=int(row["user_playbook_id"]),
            last_attempt_epoch=(
                NEVER_ATTEMPTED_EPOCH
                if row.get("last_attempted_at") is None
                else int(row["last_attempted_at"])
            ),
        )
        for row in rows
    )


def playbook_project_id(
    org_id: str,
    playbook_id: int,
    *,
    storage_for_org: Callable[[str], SweepProgressStore] = _storage_for_org,
) -> str | None:
    """The project a playbook BELONGS to, read from its row.

    The queue read carries no project predicate of its own -- it relies on
    row-level security for that -- so the project a head was read under is
    only its project where RLS is in force. On a connection that bypasses RLS
    (for example, no application credential configured) every project's read returns
    the same org-wide head, and labelling it with the binding would run the
    attempt, and stamp its rows, under whichever project happened to be bound.

    Args:
        org_id (str): The organization whose tenant schema owns the playbook.
        playbook_id (int): ``user_playbook_id`` to resolve.
        storage_for_org (Callable[[str], SweepProgressStore]): Storage
            resolution seam, injectable for tests.

    Returns:
        str | None: The playbook's ``project_id``, or ``None`` when the row is
        not visible under the current binding or carries no project.
    """
    table = cast(Any, getattr(storage_for_org(org_id), "_table", None))
    if not callable(table):
        raise UnsupportedSweepProgressStorageError(
            "sweep progress requires tenant table support"
        )
    # `callable()` narrows `table` to a callable returning `object`.
    playbooks: Any = table("user_playbooks")
    response = (
        playbooks.select("project_id")
        .eq("user_playbook_id", int(playbook_id))
        .limit(1)
        .execute()
    )
    data = getattr(response, "data", None)
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    project_id = data[0].get("project_id")
    return str(project_id) if project_id else None
