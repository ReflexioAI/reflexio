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

A scheduler whose loop is materially different (e.g. a bounded-attempt retrier
that waits *before* each attempt and exits on first success) may override
:meth:`_run_loop` wholesale and still reuse :meth:`start` / :meth:`stop` /
:meth:`is_running`.

This module is an OSS public seam: it imports only the standard library and
nothing from ``reflexio_ext``.
"""

from __future__ import annotations

import threading


class ThreadedScheduler:
    """Base for a process-local daemon that ticks on a background thread.

    Owns the shared thread mechanics only: a stop :class:`threading.Event`, a
    daemon thread, an idempotent :meth:`start`, and a join-with-timeout
    :meth:`stop`. Subclasses supply the per-tick work + cadence via
    :meth:`_run_once` (or override :meth:`_run_loop` for a bespoke loop shape).

    Args:
        thread_name (str): OS thread name for the daemon (aids debugging / logs).
    """

    def __init__(self, *, thread_name: str) -> None:
        self._thread_name = thread_name
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

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

    def _run_loop(self) -> None:
        """Drive :meth:`_run_once` until stopped, waiting its returned interval."""
        while not self._stop_event.is_set():
            interval = self._run_once()
            self._stop_event.wait(interval)
