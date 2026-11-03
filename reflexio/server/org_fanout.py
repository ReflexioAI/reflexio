"""Bounded parallel org iteration with explicit worker lifetime ownership.

Repeated daemon ticks share an ``OrgFanout`` owner. Timed-out work retains
its worker until it actually ends; later ticks join the same in-flight org
instead of minting replacement capacity. Python cannot kill threads:
"timeout" means stop waiting + log, and the caller escalates repeats.

Each org's actual start/end time is tracked, so timeout classification is
honest about queue-wait vs. run-time: an org that never got a worker before
its wait elapsed is STARVED (not counted as a timeout, not retried); an org
that has run for less than the budget gets one more wait for the remaining
budget; only an org that has genuinely run for >= the budget is reported as
timed out. Each future is waited on for at most ~2x the budget.

``ThreadPoolExecutor`` workers are non-daemon. A forever-hung ``fn`` can still
block graceful process shutdown until the orchestrator's kill grace expires;
closing the owner cancels queued work but cannot interrupt running work.

Error isolation is the caller's contract: ``fn`` must catch its own per-org
exceptions (LineageGC's sweep body already does); this helper only
backstop-logs unexpected escapes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field

from reflexio.server.background_work import background_work

logger = logging.getLogger(__name__)


@dataclass
class _OrgWork:
    future: Future[None] = field(default_factory=Future)
    starts: dict[str, float] = field(default_factory=dict)
    ends: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class OrgFanout:
    """One scheduler's worker capacity and in-flight organization ownership."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="org-fanout"
        )
        self._lock = threading.Lock()
        self._tasks: dict[str, _OrgWork] = {}
        self._closed = False

    def submit(self, org_id: str, fn: Callable[[str], None]) -> _OrgWork:
        with self._lock:
            if self._closed:
                raise RuntimeError("Maintenance fan-out owner is closed")
            existing = self._tasks.get(org_id)
            if existing is not None and not existing.future.done():
                return existing
            work = _OrgWork()

            @background_work()
            def tracked() -> None:
                with work.lock:
                    work.starts[org_id] = time.monotonic()
                try:
                    fn(org_id)
                finally:
                    with work.lock:
                        work.ends[org_id] = time.monotonic()

            work.future = self._executor.submit(tracked)
            self._tasks[org_id] = work

        def retire(_future: Future[None]) -> None:
            with self._lock:
                if self._tasks.get(org_id) is work:
                    del self._tasks[org_id]

        work.future.add_done_callback(retire)
        return work

    def close(self) -> None:
        """Cancel queued work; occupied capacity retires with its actual work."""
        # Prevent new submissions before shutdown invokes cancellation callbacks.
        # Never hold our lock while those callbacks retire their task records.
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


def iterate_orgs_bounded(
    org_ids: Iterable[str],
    fn: Callable[[str], None],
    *,
    max_workers: int,
    per_org_timeout_seconds: float,
    stop_event: threading.Event | None = None,
    owner: OrgFanout | None = None,
) -> list[str]:
    """Run ``fn(org_id)`` for each org with bounded concurrency and a per-org timeout.

    Args:
        org_ids (Iterable[str]): Orgs to process this tick.
        fn (Callable[[str], None]): Per-org work body; must handle its own errors.
        max_workers (int): Pool width; without a shared owner, ``<= 1`` runs
            serially in the calling thread.
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
        owner (OrgFanout | None): Repeated ticks must share their scheduler's
            owner. Omit only for an isolated invocation with its own lifetime.

    Returns:
        list[str]: Org ids that genuinely timed out this call (ran for at
        least the budget without finishing); starved orgs are excluded.
        Callers track repeats across ticks and escalate.
    """
    timed_out: list[str] = []
    if max_workers <= 1 and owner is None:
        for org_id in org_ids:
            if stop_event is not None and stop_event.is_set():
                break
            fn(org_id)
        return timed_out

    if owner is not None and owner.max_workers != max_workers:
        raise ValueError("Restart the scheduler to change its fan-out capacity")
    executor = owner or OrgFanout(max_workers)
    futures: list[tuple[str, _OrgWork]] = []
    try:
        for org_id in org_ids:
            if stop_event is not None and stop_event.is_set():
                break
            futures.append((org_id, executor.submit(org_id, fn)))
        for org_id, work in futures:
            if _wait_for_org(
                org_id,
                work.future,
                per_org_timeout_seconds=per_org_timeout_seconds,
                starts=work.starts,
                ends=work.ends,
                lock=work.lock,
            ):
                timed_out.append(org_id)
    finally:
        # A queued org reported starved must not silently run after this tick.
        # cancel() cannot release or replace a running worker.
        for _, work in futures:
            work.future.cancel()
        if owner is None:
            executor.close()
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
    except CancelledError:
        return False
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
        except CancelledError:
            return False
        except Exception:
            logger.exception("event=org_fanout_unexpected_error org_id=%s", org_id)
            return False

    logger.warning(
        "event=org_sweep_timeout org_id=%s timeout_seconds=%s",
        org_id,
        per_org_timeout_seconds,
    )
    return True
