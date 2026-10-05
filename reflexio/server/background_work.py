"""Explicit classification of deferred work, independent of tenant context.

Storage providers may use this marker to bound background admission. It does
not bind a project, hold a connection, or change an OSS storage implementation.
New worker threads establish their own scope instead of copying request state.

This module also owns the failure-reporting policy for that work. A worker
tick that fails is retried by the next tick, so a connection dropped while a
deployment replaces tasks loses nothing; reporting it at ERROR pages for a
non-event. :func:`report_background_failure` is the single place that decides
between a WARNING (transient, retried) and an ERROR (a bug, or an outage that
has outlasted a rollout), and :func:`report_background_success` ends a streak.

Escalation counts CONSECUTIVE failures of one unit of work, reset by that
unit's next success -- the shape of a Kubernetes ``failureThreshold`` or a
circuit breaker -- rather than measuring a time window. A time window has to
assume how soon the failed work is retried, and no single assumption holds
across workers that back off, run daily, or wait behind other orgs' work.

Accepted limit: streaks are process-local. If an outage also restarts tasks
faster than every ``_ESCALATE_AFTER_SECONDS``, every streak restarts with its
task and background failures stay at WARNING; request-path errors and
health-check alarms still page in that case.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_background: ContextVar[bool] = ContextVar("reflexio_background_work", default=False)

_policy_logger = logging.getLogger(__name__)

# A streak escalates only when BOTH hold: enough consecutive failures, and the
# streak has lasted long enough. The count keeps a slow job from paging on one
# blip; the duration keeps a fast loop (5s) from paging while the pooler
# restarts. Measured on production rollouts 2026-09-29..10-03: the longest
# burst of in-scope failures on one task lasted 88s, so 300s is ~3.4x that.
_ESCALATE_AFTER_FAILURES = 3
_ESCALATE_AFTER_SECONDS = 300.0
_MAX_CHAIN_LINKS = 32
_MAX_MESSAGE_CHARS = 200
# Successes remove entries, so this bounds scopes failing AT ONCE. Past it a
# failure is untracked and reported at ERROR: that many failing units is an
# outage, and failing loud is the safe direction.
_MAX_TRACKED_SCOPES = 4096
# A streak whose last failure is older than this is forgotten. Without it, a
# unit that failed once and then went idle (nothing to do, so no success is
# ever reported) keeps its streak, and unrelated rollout blips days apart add
# up to a page. Converted units retry within minutes, so a real outage keeps
# failing well inside this horizon and still escalates.
_STREAK_IDLE_RESET_SECONDS = 3600.0

# Patched by tests; the policy below reads time only through this name.
_monotonic: Callable[[], float] = time.monotonic

_classifier: Callable[[BaseException], bool] | None = None
_classifier_failure_warned = False
# scope -> (consecutive transient failures, first failure, last failure),
# times from ``_monotonic``.
_streaks: dict[str, tuple[int, float, float]] = {}
_streaks_lock = threading.Lock()


def is_background_work() -> bool:
    """Whether this execution belongs to a deferred worker."""
    return _background.get()


@contextmanager
def background_work() -> Iterator[None]:
    """Classify a worker body, restoring the caller's classification on exit."""
    token = _background.set(True)
    try:
        yield
    finally:
        _background.reset(token)


def configure_transient_failure_classifier(
    fn: Callable[[BaseException], bool] | None,
) -> None:
    """Install a process-global transient-failure classifier, or clear it.

    OSS cannot know which driver errors mean "the connection went away", so a
    deployment registers that knowledge here, mirroring
    ``configure_error_reporter``. With nothing registered every failure is
    reported at ERROR, exactly as before this policy existed.

    Args:
        fn (Callable[[BaseException], bool] | None): Returns True when one
            exception (a single link, not its chain) is a transient
            infrastructure failure that the next attempt will retry.
    """
    global _classifier, _classifier_failure_warned
    _classifier = fn
    _classifier_failure_warned = False


def _cause_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and the exceptions it was deliberately raised ``from``.

    Only ``__cause__`` is followed. ``__context__`` is set whenever an
    exception is raised while another is being handled, so following it would
    classify a BUG raised during recovery from a dropped connection as the
    dropped connection. A wrapper that means "this is that failure" says so
    with ``raise ... from``. Bounded and cycle-safe.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(seen) < _MAX_CHAIN_LINKS:
        if id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__


def is_transient_failure(exc: BaseException) -> bool:
    """Whether ``exc``, or what it was raised ``from``, is a transient failure.

    Args:
        exc (BaseException): The failure a worker caught.

    Returns:
        bool: False when no classifier is configured, or when the classifier
            raises (a broken classifier must never hide a failure).
    """
    global _classifier_failure_warned
    classifier = _classifier
    if classifier is None:
        return False
    for link in _cause_chain(exc):
        try:
            if classifier(link):
                return True
        except Exception as classifier_exc:  # noqa: BLE001
            if not _classifier_failure_warned:
                _classifier_failure_warned = True
                _policy_logger.warning(
                    "event=transient_failure_classifier_failed error_class=%s error=%s",
                    type(classifier_exc).__name__,
                    classifier_exc,
                )
            return False
    return False


def _extend_streak(scope: str) -> tuple[int, float] | None:
    """Count one more consecutive transient failure; None when untracked."""
    now = _monotonic()
    with _streaks_lock:
        current = _streaks.get(scope)
        if current is not None and now - current[2] > _STREAK_IDLE_RESET_SECONDS:
            current = None
        if current is None:
            if scope not in _streaks and len(_streaks) >= _MAX_TRACKED_SCOPES:
                for idle in [
                    key
                    for key, (_, _, last) in _streaks.items()
                    if now - last > _STREAK_IDLE_RESET_SECONDS
                ]:
                    del _streaks[idle]
                if len(_streaks) >= _MAX_TRACKED_SCOPES:
                    return None
            current = (0, now, now)
        count, first = current[0] + 1, current[1]
        _streaks[scope] = (count, first, now)
        return count, now - first


def report_background_success(scope: str) -> None:
    """End ``scope``'s failure streak: the unit of work just succeeded.

    Call it where the SAME unit whose failure is reported under ``scope``
    completes -- not where an enclosing loop merely carries on. Without it a
    streak only grows, so unrelated blips days apart would add up to a page.

    Args:
        scope (str): The scope passed to :func:`report_background_failure`.
    """
    if scope in _streaks:
        with _streaks_lock:
            _streaks.pop(scope, None)


def _one_line(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > _MAX_MESSAGE_CHARS:
        return text[: _MAX_MESSAGE_CHARS - 3] + "..."
    return text


def report_background_failure(
    logger: logging.Logger,
    event: str,
    exc: BaseException,
    *,
    scope: str,
    detail: Callable[[BaseException], str] | None = None,
    **fields: object,
) -> None:
    """Log a caught background-work failure at the level it deserves.

    - Not transient: ERROR with the traceback, as ``logger.exception`` did.
    - Transient: one WARNING line without a traceback; the streak for
      ``scope`` grows until :func:`report_background_success` ends it.
    - Transient, and the streak has reached ``_ESCALATE_AFTER_FAILURES``
      failures over at least ``_ESCALATE_AFTER_SECONDS``: ERROR with the
      traceback, marked ``escalated=true``, so a real outage still pages.

    Args:
        logger (logging.Logger): The worker's own logger.
        event (str): The worker's event name, e.g. ``lineage_gc_org_failed``.
        exc (BaseException): The caught failure.
        scope (str): Stable identity of the unit whose next attempt retries
            this work (worker plus org, project, ...).
        detail (Callable[[BaseException], str] | None): Renders ``exc`` for
            the log line. When given, it replaces ``str(exc)`` everywhere and
            no traceback is attached (a traceback prints the message), for
            handlers whose exceptions can carry credentials or tenant data.
        **fields (object): Extra ``key=value`` context for the log line.
    """
    context = "".join(f" {key}={value}" for key, value in fields.items())
    exc_info: BaseException | None = None if detail is not None else exc
    rendered = f" {_one_line(detail(exc))}" if detail is not None else ""
    if not is_transient_failure(exc):
        logger.error(
            "event=%s scope=%s%s%s",
            event,
            scope,
            rendered,
            context,
            exc_info=exc_info,
            stacklevel=2,
        )
        return
    streak = _extend_streak(scope)
    if streak is None or (
        streak[0] >= _ESCALATE_AFTER_FAILURES and streak[1] >= _ESCALATE_AFTER_SECONDS
    ):
        count, seconds = streak if streak is not None else (0, 0.0)
        logger.error(
            "event=%s transient=true escalated=true consecutive_failures=%d "
            "streak_seconds=%d tracked=%s scope=%s%s%s",
            event,
            count,
            int(seconds),
            "false" if streak is None else "true",
            scope,
            rendered,
            context,
            exc_info=exc_info,
            stacklevel=2,
        )
        return
    message = detail(exc) if detail is not None else str(exc)
    logger.warning(
        "event=%s transient=true consecutive_failures=%d scope=%s "
        "error_class=%s error=%s%s",
        event,
        streak[0],
        scope,
        type(exc).__name__,
        _one_line(message),
        context,
        stacklevel=2,
    )
