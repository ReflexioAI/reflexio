"""Explicit classification of deferred work, independent of tenant context.

Storage providers may use this marker to bound background admission. It does
not bind a project, hold a connection, or change an OSS storage implementation.
New worker threads establish their own scope instead of copying request state.

This module also owns the failure-reporting policy for that work. A worker
tick that fails is retried by the next tick, so a connection dropped while a
deployment replaces tasks loses nothing; reporting it at ERROR pages for a
non-event. :func:`report_background_failure` is the single place that decides
between a WARNING (transient, retried) and an ERROR (a bug, or an outage that
has outlasted a rollout).
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

# A rollout drop lasts seconds; an outage lasts minutes. Transient failures for
# one scope that keep recurring (gaps under the episode gap) for the escalation
# window are reported at ERROR, so a real outage still pages.
#
# The gap must exceed the SLOWEST retry cadence plus its run time, or a slow
# scheduler resets its episode on every failure and never escalates: lineage GC
# retries 300s after a failed tick, aggregation backs a scope off for 300s, and
# Braintrust polls every 900s. 20 minutes covers all three (a 900s poller
# escalates on its second consecutive failure) while separate rollouts, hours
# apart, still start fresh episodes. Cost: two rollouts under 20 minutes apart
# that both hit one scope can merge into a single ERROR.
_ESCALATE_AFTER_SECONDS = 600.0
_EPISODE_GAP_SECONDS = 1200.0
_MAX_CHAIN_LINKS = 32
_MAX_MESSAGE_CHARS = 200
_MAX_TRACKED_SCOPES = 1024

# Patched by tests; the policy below reads time only through this name.
_monotonic: Callable[[], float] = time.monotonic

_classifier: Callable[[BaseException], bool] | None = None
_classifier_failure_warned = False
_episodes: dict[str, tuple[float, float]] = {}
_episodes_lock = threading.Lock()


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
            infrastructure failure that the next tick will retry.
    """
    global _classifier, _classifier_failure_warned
    _classifier = fn
    _classifier_failure_warned = False


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and every exception reachable through cause/context.

    Bounded and cycle-safe, because ``__context__`` graphs can loop.
    """
    seen: set[int] = set()
    pending = [exc]
    while pending and len(seen) < _MAX_CHAIN_LINKS:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(
            linked
            for linked in (current.__cause__, current.__context__)
            if linked is not None
        )


def is_transient_failure(exc: BaseException) -> bool:
    """Whether any link in ``exc``'s chain is a transient infrastructure failure.

    The chain is walked because workers routinely wrap a driver error in their
    own exception; the wrapper alone says nothing about the cause.

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
    for link in _exception_chain(exc):
        try:
            if classifier(link):
                return True
        except Exception as classifier_exc:  # noqa: BLE001
            if not _classifier_failure_warned:
                _classifier_failure_warned = True
                _policy_logger.warning(
                    "event=transient_failure_classifier_failed error_class=%s",
                    type(classifier_exc).__name__,
                )
            return False
    return False


def _transient_for_seconds(scope: str) -> float:
    """Record one transient failure for ``scope``; return how long it has lasted.

    Process-local on purpose: the rollout noise comes from the task that is
    about to exit, so state that dies with the process resets at the same rate
    as the condition it tracks.

    Bounded: a new scope arriving at capacity first prunes ended episodes, and
    if none have ended it is not tracked (it reports as a fresh episode). An
    outage that wide is already escalating through the scopes being tracked.
    """
    now = _monotonic()
    with _episodes_lock:
        if scope not in _episodes and len(_episodes) >= _MAX_TRACKED_SCOPES:
            for stale in [
                key
                for key, (_, seen) in _episodes.items()
                if now - seen > _EPISODE_GAP_SECONDS
            ]:
                del _episodes[stale]
            if len(_episodes) >= _MAX_TRACKED_SCOPES:
                return 0.0
        first_seen, last_seen = _episodes.get(scope, (now, now))
        if now - last_seen > _EPISODE_GAP_SECONDS:
            first_seen = now
        _episodes[scope] = (first_seen, now)
        return now - first_seen


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > _MAX_MESSAGE_CHARS:
        return text[: _MAX_MESSAGE_CHARS - 3] + "..."
    return text


def report_background_failure(
    logger: logging.Logger,
    event: str,
    exc: BaseException,
    *,
    scope: str,
    **fields: object,
) -> None:
    """Log a caught background-work failure at the level it deserves.

    - Not transient: ERROR with the traceback, as ``logger.exception`` did.
    - Transient: one WARNING line without a traceback. The next tick retries,
      so a connection dropped during a rollout is visible but does not page.
    - Transient for ``scope`` continuously for ``_ESCALATE_AFTER_SECONDS``
      (gaps under ``_EPISODE_GAP_SECONDS``): ERROR with the traceback, marked
      ``escalated=true``, so an outage that outlasts a rollout still pages.

    Args:
        logger (logging.Logger): The worker's own logger.
        event (str): The worker's event name, e.g. ``lineage_gc_org_failed``.
        exc (BaseException): The caught failure.
        scope (str): Stable identity of the failing unit (worker plus org or
            project); escalation is tracked per scope.
        **fields (object): Extra ``key=value`` context for the log line.
    """
    detail = "".join(f" {key}={value}" for key, value in fields.items())
    if not is_transient_failure(exc):
        logger.error(
            "event=%s scope=%s%s",
            event,
            scope,
            detail,
            exc_info=exc,
            stacklevel=2,
        )
        return
    age = _transient_for_seconds(scope)
    if age >= _ESCALATE_AFTER_SECONDS:
        logger.error(
            "event=%s transient=true escalated=true transient_for_seconds=%d "
            "scope=%s%s",
            event,
            int(age),
            scope,
            detail,
            exc_info=exc,
            stacklevel=2,
        )
        return
    logger.warning(
        "event=%s transient=true scope=%s error_class=%s error=%s%s",
        event,
        scope,
        type(exc).__name__,
        _one_line(exc),
        detail,
        stacklevel=2,
    )
