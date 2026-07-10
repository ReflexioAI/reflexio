"""Bounded parallel org iteration for org-walking daemon ticks.

Replaces serial ``for org_id in org_ids`` loops in daemon ticks with a
bounded per-tick thread pool (spec section 6.1). The pool is created per call
and abandoned (not joined) at the end so a stuck org cannot eat a slot across
ticks; stragglers finish in the background. Python cannot kill threads —
"timeout" means stop waiting + log, and the caller escalates repeats.

Each org's actual start/end time is tracked, so timeout classification is
honest about queue-wait vs. run-time: an org that never got a worker before
its wait elapsed is STARVED (not counted as a timeout, not retried); an org
that has run for less than the budget gets one more wait for the remaining
budget; only an org that has genuinely run for >= the budget is reported as
timed out. Each future is waited on for at most ~2x the budget.

Abandonment is per-tick, not process-wide: ``ThreadPoolExecutor`` worker
threads are non-daemon, so CPython still joins ALL of them at interpreter
exit — a forever-hung ``fn`` can block graceful process shutdown until the
orchestrator's kill grace expires; ``shutdown(wait=False)`` here only frees
the NEXT tick's capacity, not exit-time joining.

Error isolation is the caller's contract: ``fn`` must catch its own per-org
exceptions (LineageGC's sweep body already does); this helper only
backstop-logs unexpected escapes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)


def iterate_orgs_bounded(
    org_ids: Iterable[str],
    fn: Callable[[str], None],
    *,
    max_workers: int,
    per_org_timeout_seconds: float,
    stop_event: threading.Event | None = None,
) -> list[str]:
    """Run ``fn(org_id)`` for each org with bounded concurrency and a per-org timeout.

    Args:
        org_ids (Iterable[str]): Orgs to process this tick.
        fn (Callable[[str], None]): Per-org work body; must handle its own errors.
        max_workers (int): Pool width; ``<= 1`` runs serially with no pool.
        per_org_timeout_seconds (float): Run-time budget per org (not
            queue-wait time). An org that never got a worker before its wait
            elapsed is logged as ``event=org_sweep_starved`` and excluded
            from the result (not timed out, not retried). An org that has
            actually run for less than the budget is given one more wait for
            the remaining budget; only once it has genuinely run for at
            least the budget is it logged as ``event=org_sweep_timeout`` and
            moved on (the straggler thread keeps running in the background).
        stop_event (threading.Event | None): When set, stop submitting new
            orgs; in-flight orgs finish (matches serial-loop shutdown semantics).

    Returns:
        list[str]: Org ids that genuinely timed out this call (ran for at
        least the budget without finishing); starved orgs are excluded.
        Callers track repeats across ticks and escalate.
    """
    timed_out: list[str] = []
    if max_workers <= 1:
        for org_id in org_ids:
            if stop_event is not None and stop_event.is_set():
                break
            fn(org_id)
        return timed_out

    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    record_lock = threading.Lock()

    def _tracked(org_id: str) -> None:
        with record_lock:
            starts[org_id] = time.monotonic()
        try:
            fn(org_id)
        finally:
            with record_lock:
                ends[org_id] = time.monotonic()

    executor = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="org-fanout"
    )
    futures: list[tuple[str, Future[None]]] = []
    try:
        for org_id in org_ids:
            if stop_event is not None and stop_event.is_set():
                break
            futures.append((org_id, executor.submit(_tracked, org_id)))
        for org_id, future in futures:
            if _wait_for_org(
                org_id,
                future,
                per_org_timeout_seconds=per_org_timeout_seconds,
                starts=starts,
                ends=ends,
                lock=record_lock,
            ):
                timed_out.append(org_id)
    finally:
        # Per-tick pool, abandoned not joined (spec 6.1): stragglers finish in
        # the background; the NEXT tick gets a fresh pool and full width.
        executor.shutdown(wait=False)
    return timed_out


def _wait_for_org(
    org_id: str,
    future: Future[None],
    *,
    per_org_timeout_seconds: float,
    starts: dict[str, float],
    ends: dict[str, float],
    lock: threading.Lock,
) -> bool:
    """Wait for one org's future; return True iff it genuinely timed out.

    Classifies a ``FuturesTimeoutError`` honestly instead of treating
    queue-wait as run time: an org with no recorded start never got a
    worker (STARVED — logged, not counted as a timeout, no further wait); an
    org that has run for less than the budget gets one more wait for the
    remaining budget. Total wait per org is bounded to ~2x the budget.

    Args:
        org_id (str): The org this future belongs to.
        future (Future[None]): The submitted future for ``org_id``.
        per_org_timeout_seconds (float): The run-time budget for this org.
        starts (dict[str, float]): Monotonic start times, keyed by org id.
        ends (dict[str, float]): Monotonic end times, keyed by org id.
        lock (threading.Lock): Guards ``starts``/``ends``.

    Returns:
        bool: True if the org genuinely timed out (ran >= the budget without
        finishing); False if it completed, was starved, or raised an
        unexpected error.
    """
    try:
        future.result(timeout=per_org_timeout_seconds)
        return False
    except FuturesTimeoutError:
        pass
    except Exception:
        # fn owns error isolation; anything escaping is a bug — log, never
        # let one org's escape skip the remaining result waits.
        logger.exception("event=org_fanout_unexpected_error org_id=%s", org_id)
        return False

    with lock:
        start = starts.get(org_id)
        finished = org_id in ends
    if finished:
        # Completed between the timeout firing and this check — not a timeout.
        return False
    if start is None:
        logger.warning("event=org_sweep_starved org_id=%s", org_id)
        return False

    elapsed_running = time.monotonic() - start
    remaining = per_org_timeout_seconds - elapsed_running
    if remaining > 0:
        try:
            future.result(timeout=remaining)
            return False
        except FuturesTimeoutError:
            pass
        except Exception:
            logger.exception("event=org_fanout_unexpected_error org_id=%s", org_id)
            return False

    logger.warning(
        "event=org_sweep_timeout org_id=%s timeout_seconds=%s",
        org_id,
        per_org_timeout_seconds,
    )
    return True
