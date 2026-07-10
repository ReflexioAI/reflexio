"""Bounded parallel org iteration for org-walking daemon ticks.

Replaces serial ``for org_id in org_ids`` loops in daemon ticks with a
bounded per-tick thread pool (spec section 6.1). The pool is created per call
and abandoned (not joined) at the end so a stuck org cannot eat a slot across
ticks; stragglers finish in the background. Python cannot kill threads —
"timeout" means stop waiting + log, and the caller escalates repeats.

Error isolation is the caller's contract: ``fn`` must catch its own per-org
exceptions (LineageGC's sweep body already does); this helper only
backstop-logs unexpected escapes.
"""

from __future__ import annotations

import logging
import threading
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
        per_org_timeout_seconds (float): Max seconds to wait per org before
            logging ``event=org_sweep_timeout`` and moving on (the straggler
            thread keeps running in the background).
        stop_event (threading.Event | None): When set, stop submitting new
            orgs; in-flight orgs finish (matches serial-loop shutdown semantics).

    Returns:
        list[str]: Org ids that timed out this call (callers track repeats
        across ticks and escalate).
    """
    timed_out: list[str] = []
    if max_workers <= 1:
        for org_id in org_ids:
            if stop_event is not None and stop_event.is_set():
                break
            fn(org_id)
        return timed_out

    executor = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="org-fanout"
    )
    futures: list[tuple[str, Future[None]]] = []
    try:
        for org_id in org_ids:
            if stop_event is not None and stop_event.is_set():
                break
            futures.append((org_id, executor.submit(fn, org_id)))
        for org_id, future in futures:
            try:
                future.result(timeout=per_org_timeout_seconds)
            except FuturesTimeoutError:
                timed_out.append(org_id)
                logger.warning(
                    "event=org_sweep_timeout org_id=%s timeout_seconds=%s",
                    org_id,
                    per_org_timeout_seconds,
                )
            except Exception:
                # fn owns error isolation; anything escaping is a bug — log,
                # never let one org's escape skip the remaining result waits.
                logger.exception("event=org_fanout_unexpected_error org_id=%s", org_id)
    finally:
        # Per-tick pool, abandoned not joined (spec 6.1): stragglers finish in
        # the background; the NEXT tick gets a fresh pool and full width.
        executor.shutdown(wait=False)
    return timed_out
