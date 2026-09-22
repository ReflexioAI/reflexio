"""Per-phase wall-clock timing for the publish request path.

WHY THIS EXISTS RATHER THAN A TRACE
-----------------------------------
``/api/publish_interaction`` was taking >8s in production while every signal
said the system was idle: CPU 3%, database execution ~8ms per publish, the
publish concurrency limiter recording zero waits over 250ms in twelve hours,
the embedding service answering in 25ms, and 142 of 153 sampled requests
arriving with nothing else in flight. Counting network round trips accounted
for ~1.3s of ~8s. The remaining ~6.7s was in no signal anyone could read.

The obvious instrument -- the ~30 ``profile_step`` call sites already on this
path -- was unavailable. Not because tracing is disabled (the production task
definition sets ``SENTRY_TRACES_SAMPLE_RATE=0.1`` and attaches a DSN) but
because the Sentry account is over quota: one self-host deployment emitted 92%
of the fleet's spans over seven days and ``environment:production`` went to
zero. A vendor budget is not something this module can fix, so it carries its
own signal to the one channel a production container always has -- stderr, via
the log record.

WHAT MAKES THIS DEFENSIBLE AT WARNING
-------------------------------------
Two independent mechanisms drop ``logger.info`` in production, and neither is
fixed by raising ``--log-level``:

* nothing in the served process configures root logging, so a record finds no
  handler and falls to ``logging.lastResort``, whose floor is WARNING; and
* ``_configure_first_party_sentry_log_levels`` pins the ``reflexio`` and
  ``reflexio_ext`` loggers to WARNING because ``.env.platform`` sets
  ``SENTRY_LOGS_LEVEL=warning`` -- so the record is dropped at its source.

``reflexio_ext/tests/deployment/test_self_host_observability.py`` pins all of
that, including that ``--log-level info`` does *not* help. So the level has to
travel on the record, which is what
``offline_tuner/config.py::TUNER_OUTCOME_LOG_LEVEL`` already does.

That precedent justified WARNING on being "one line per completed attempt, not
one per request". This IS one per request, so the justification has to be
supplied here instead, and it is supplied by suppression: a request that is
fast logs nothing, and a second slow request for the same org inside the
throttle window logs nothing. On a healthy fleet this module is silent. It only
speaks when something is already wrong, which is the condition under which a
line is worth its cost.

Every part of it is off unless ``REFLEXIO_PUBLISH_TIMING_ENABLED=true``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from reflexio.server.env_utils import env_bool, env_str

logger = logging.getLogger(__name__)

#: Opt-in. Absent or blank means off, and anything that is not ``true``/``false``
#: raises rather than silently resolving to off -- see :func:`env_bool`.
ENV_ENABLED = "REFLEXIO_PUBLISH_TIMING_ENABLED"

#: Only requests slower than this are worth a line. Below it the request is
#: behaving and the line would be pure cost.
ENV_THRESHOLD_MS = "REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS"
_DEFAULT_THRESHOLD_MS = 2000

#: At most one line per org per window, so a sustained incident reports the
#: shape of the problem without reporting it thousands of times. Same shape as
#: ``operation_limiter._should_log_publish_pressure``.
ENV_INTERVAL_SECONDS = "REFLEXIO_PUBLISH_TIMING_INTERVAL_SECONDS"
_DEFAULT_INTERVAL_SECONDS = 60.0

#: WARNING, not INFO. See the module docstring -- INFO is invisible in the
#: deployed configuration and a timing line nobody can read is worse than none,
#: because it reads as coverage.
PUBLISH_TIMING_LOG_LEVEL = logging.WARNING

_lock = threading.Lock()
_last_logged_by_org: dict[str, float] = {}

#: Per-request phase accumulator. A ContextVar rather than an attribute on any
#: shared object because storage instances are cached per org and shared across
#: concurrent requests -- hanging request state off one would let request A's
#: timings land in request B's line.
_phases: ContextVar[dict[str, int] | None] = ContextVar(
    "publish_timing_phases", default=None
)


def publish_timing_enabled() -> bool:
    """Return whether per-phase publish timing should be collected."""
    return env_bool(ENV_ENABLED, default=False)


def _threshold_ms() -> int:
    raw = env_str(ENV_THRESHOLD_MS, str(_DEFAULT_THRESHOLD_MS)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_THRESHOLD_MS


def _interval_seconds() -> float:
    raw = env_str(ENV_INTERVAL_SECONDS, str(_DEFAULT_INTERVAL_SECONDS)).strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_INTERVAL_SECONDS


@contextmanager
def collect() -> Iterator[None]:
    """Open a per-request accumulator for the duration of one publish.

    Nested use is deliberately a no-op rather than an error: the request path
    already re-enters some of these helpers, and a timing module that can fail
    a publish has inverted its own cost/benefit.
    """
    if not publish_timing_enabled() or _phases.get() is not None:
        yield
        return
    token = _phases.set({})
    try:
        yield
    finally:
        _phases.reset(token)


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Record wall-clock milliseconds spent in ``name``.

    Costs one ``ContextVar.get`` when disabled. Re-entering the same name adds
    to it, so a step called per interaction reports its TOTAL rather than its
    last occurrence -- which is the number that matters for a loop of three
    round trips per interaction.

    Args:
        name (str): Short phase key, used verbatim as ``<name>_ms`` in the line.
    """
    phases = _phases.get()
    if phases is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = int((time.perf_counter() - start) * 1000)
        phases[name] = phases.get(name, 0) + elapsed


def record(name: str, value: int) -> None:
    """Attach a non-timing integer (a count) to the line, e.g. interactions."""
    phases = _phases.get()
    if phases is not None:
        phases[name] = value


def _should_log(org_id: str, now: float) -> bool:
    interval = _interval_seconds()
    with _lock:
        last = _last_logged_by_org.get(org_id)
        if last is not None and now - last < interval:
            return False
        _last_logged_by_org[org_id] = now
        return True


def emit(*, org_id: str, request_id: str, total_ms: int) -> bool:
    """Emit one structured line if this request was slow and not throttled.

    Args:
        org_id (str): Throttle key -- one line per org per window.
        request_id (str): Correlates the line with an ALB access-log entry.
        total_ms (int): Whole-request wall clock, measured by the caller rather
            than summed from phases: the gap between the two is the finding
            when every phase is small and the total is not.

    Returns:
        bool: Whether a line was emitted. Returned for tests, which otherwise
        have to assert on log capture to know the gate worked.
    """
    phases = _phases.get()
    if phases is None or total_ms < _threshold_ms():
        return False
    if not _should_log(org_id, time.monotonic()):
        return False

    # The phase set is built at runtime, so the format string is too -- but the
    # VALUES stay lazy `%s` args, which is what the house style and the logging
    # module actually care about. Nothing is interpolated unless the record is
    # emitted. Sorted so the field order is stable between lines and a human
    # can diff two of them.
    ordered = sorted(phases.items())
    fmt = " ".join(
        ["event=publish_timing org_id=%s request_id=%s total_ms=%s"]
        + [f"{key}=%s" for key, _ in ordered]
    )
    logger.log(
        PUBLISH_TIMING_LOG_LEVEL,
        fmt,
        org_id,
        request_id,
        total_ms,
        *[value for _, value in ordered],
    )
    return True


def snapshot() -> Mapping[str, int] | None:
    """Return the current phase map, for tests and for callers that log it."""
    phases = _phases.get()
    return dict(phases) if phases is not None else None


def reset_for_tests() -> None:
    """Clear the per-org throttle state between tests."""
    with _lock:
        _last_logged_by_org.clear()
