"""Shared thread-lifecycle base for the process-local polling schedulers.

Reflexio runs a family of background daemons — extraction resume, lineage GC,
and (in the enterprise layer) billing ship / lease / period-close / integrity /
audit / offline-tuner / governance-retention / invitation-reclamation. Every one
of them hand-rolled the SAME thread mechanics: a :class:`threading.Event`
stop-signal, a daemon :class:`threading.Thread` running a ``while not
stop.is_set()`` loop, and a ``.join(timeout)`` graceful stop.

:class:`ThreadedScheduler` captures ONLY those mechanics. Each scheduler keeps
its own poll interval, work body, gating, and log lines — supplied via a small
set of hooks so no observable behaviour changes:

- :meth:`_run_once` performs one tick (catching its own errors) and returns the
  number of seconds to wait before the next tick. A scheduler that reads its
  interval from config each tick keeps doing exactly that by returning the fresh
  value.
- :meth:`_should_start` can veto startup (the enterprise schedulers whose
  ``start`` is a logged no-op when a feature gate is off).
- :meth:`_on_started` / :meth:`_on_stopped` emit each scheduler's own start/stop
  log line (through its own module logger, so output is byte-identical).
- ``leader_gate`` (optional): a fleet-coordination gate consulted before each
  tick; ``None`` (the default and the OSS-local case) preserves today's
  always-tick behavior byte-for-byte.

A scheduler whose loop is materially different (e.g. a bounded-attempt retrier
that waits *before* each attempt and exits on first success) may override
:meth:`_run_loop` wholesale and still reuse :meth:`start` / :meth:`stop` /
:meth:`is_running`.

This module is an OSS public seam: it imports only the standard library and
nothing from ``reflexio_ext``.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)

# How long a non-leader waits before re-checking the gate. Fixed (no interval
# memoization): a follower that becomes leader starts ticking within <=60s.
_FOLLOWER_POLL_SECONDS: float = 60.0


class LeaderGate(Protocol):
    """Fleet-coordination gate consulted before each tick.

    Implementations must NEVER raise from ``should_run`` — error handling
    (including fail-open) lives inside the implementation (spec §4.1). The
    scheduler base nonetheless defends against a contract violation: if
    ``should_run`` raises anyway, the base logs the error and fails open
    (runs the tick) rather than trusting the contract blindly and letting the
    daemon thread die silently.
    """

    def should_run(self) -> bool:
        """Return whether this instance should run the next tick."""
        ...


class ThreadedScheduler:
    """Base for a process-local daemon that ticks on a background thread.

    Owns the shared thread mechanics only: a stop :class:`threading.Event`, a
    daemon thread, an idempotent :meth:`start`, and a join-with-timeout
    :meth:`stop`. Subclasses supply the per-tick work + cadence via
    :meth:`_run_once` (or override :meth:`_run_loop` for a bespoke loop shape).

    Args:
        thread_name (str): OS thread name for the daemon (aids debugging / logs).
        leader_gate (LeaderGate | None): Optional fleet-coordination gate
            consulted before each tick. Defaults to None (always tick).
    """

    def __init__(
        self, *, thread_name: str, leader_gate: LeaderGate | None = None
    ) -> None:
        self._thread_name = thread_name
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._leader_gate = leader_gate

    def start(self) -> None:
        """Start the daemon thread, unless it is gated off or already running.

        Calls :meth:`_should_start` first (a subclass gate that can log + veto),
        then the idempotent alive-check, then spawns the thread and calls
        :meth:`_on_started`.
        """
        if not self._should_start():
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=self._thread_name,
            daemon=True,
        )
        self._thread.start()
        self._on_started()

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        """Signal the loop to stop and join the thread.

        Args:
            timeout_seconds (float): Max seconds to wait for the thread to exit.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            # Only drop the reference once the thread has actually exited. If the
            # join timed out (a slow ``_run_once`` still running), keep it so
            # ``is_running()`` stays true and a subsequent ``start()`` refuses to
            # spawn a second loop while the first is alive.
            if not self._thread.is_alive():
                self._thread = None
        self._on_stopped()

    def is_running(self) -> bool:
        """Return whether the background thread exists and is alive.

        Returns:
            bool: True when the loop thread is running.
        """
        return self._thread is not None and self._thread.is_alive()

    # -- Overridable hooks ---------------------------------------------------

    def _should_start(self) -> bool:
        """Return whether :meth:`start` should spawn the thread (default True).

        Override to gate startup behind a feature flag; the override owns any
        "not starting" log line.
        """
        return True

    def _on_started(self) -> None:
        """Hook run after the thread starts (default no-op); override to log."""

    def _on_stopped(self) -> None:
        """Hook run after the thread joins (default no-op); override to log."""

    def _run_once(self) -> float:
        """Run one tick and return the seconds to wait before the next.

        Implementations MUST catch their own per-tick errors — a raise here would
        kill the daemon thread. The returned value is passed straight to
        ``stop_event.wait`` as the inter-tick delay, so a subclass may compute a
        fresh interval each tick (e.g. from config).

        Returns:
            float: Seconds to wait before the next tick.
        """
        raise NotImplementedError

    def _elected_interval(self) -> float:
        """Run one gated iteration; return the seconds to wait before the next.

        ``None`` gate -> tick (today's behavior). Gate ``True`` -> tick.
        Gate ``False`` -> skip and wait the fixed follower poll.

        The gate contract says ``should_run`` never raises (see
        :class:`LeaderGate`), but this has zero defense of its own: an escaped
        exception here would kill the daemon thread permanently and silently.
        A raise is therefore caught defensively and treated as fail-open (tick
        runs; duplicate work is safe, silence is not) rather than propagated.

        Returns:
            float: Seconds to wait before the next loop iteration.
        """
        if self._leader_gate is None:
            return self._run_once()
        try:
            should_run = self._leader_gate.should_run()
        except Exception:
            logger.exception(
                "event=%s_leader_gate_error — failing open", self._thread_name
            )
            return self._run_once()
        if should_run:
            return self._run_once()
        logger.debug("event=%s_skip_not_leader", self._thread_name)
        return _FOLLOWER_POLL_SECONDS

    def _run_loop(self) -> None:
        """Drive gated iterations until stopped, waiting each returned interval."""
        while not self._stop_event.is_set():
            interval = self._elected_interval()
            self._stop_event.wait(interval)
