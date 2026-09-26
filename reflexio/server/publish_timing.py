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
path -- was unavailable. Tracing was enabled with a 10% sample rate, but
the tracing account was over quota: one self-host deployment emitted 92%
of the fleet's spans over seven days and ``environment:production`` went to
zero. A vendor budget is not something this module can fix, so it carries its
own signal through the production container's log stream.

WHAT MAKES THIS DEFENSIBLE AT WARNING
-------------------------------------
Importing ``reflexio.server`` *is* the deployed logging configuration: the
``else`` branch of ``DEBUG_LOG_TO_CONSOLE`` in ``server/__init__.py`` sits at
module level, so it runs in every served process. It sets the ROOT logger to
WARNING, attaches a stdout ``StreamHandler`` at INFO, and raises every logger
named in ``REFLEXIO_INFO_LOGGERS`` to INFO. The split is deliberate: app-wide
INFO drove ~3x log volume and grew memory to the container ceiling over a few
hours (prod incident 2026-07-04), so INFO is opt-in per subsystem.

An INFO record from this module is therefore dropped -- it is not allowlisted,
so it inherits the root's WARNING and is rejected before it ever reaches a
handler. That is ONE mechanism, not two, and an earlier version of this
docstring got both halves of it wrong. Measured, by importing this
configuration rather than describing it:

* there IS a root handler (``StreamHandler`` at INFO). Records do not fall to
  ``logging.lastResort``; and
* an enterprise logging hook pinning ``reflexio`` to WARNING
  is redundant here rather than a second,
  independent mechanism -- with the pin removed, this logger's effective level
  is still WARNING by inheritance. Nor does the pin defeat the allowlist:
  ``setLevel`` on an ancestor leaves a descendant that was set explicitly, so
  an allowlisted first-party logger still emits INFO in production. Nine do
  today, on the live production task definition.

So INFO is REACHABLE, and the case for WARNING is not that it is impossible.
It is that INFO costs a second, coordinated change -- ``REFLEXIO_INFO_LOGGERS``
edited on every deployment that wants the signal, including self-host ones we
do not operate -- to read a diagnostic line when request timing needs
investigation. Putting the level on the record needs no such coordination, which is
what ``offline_tuner/config.py::TUNER_OUTCOME_LOG_LEVEL`` already does.

That precedent justified WARNING on being "one line per completed attempt, not
one per request". The handler event instead relies on suppression: a fast
handler logs nothing, and a second slow handler for the same org inside the
throttle window logs nothing. The companion HTTP event has its own throttle
and can also report expected extraction waits. Filter ``wait_for_response=1``
out of ordinary acknowledgement analysis; a slow HTTP event alone does not
prove the publish handler is unhealthy.

THE SCOPE OWNS THE WHOLE-REQUEST CLOCK
--------------------------------------
:func:`collect` starts the clock that :func:`emit` gates on, and the route
opens it *before* ``acquire_ingestion``. That placement is the whole point.
``GenerationService.run`` starts its own clock only after admission has been
granted, so a request that spends 8s queueing and 50ms working measures 50ms
there -- below any useful threshold, and silent in exactly the shape this
module was built to catch. No caller supplies the total; there is one clock,
and the scope owns it.

The scope deliberately stops at :func:`emit`, which runs before the route's
``wait_for_response`` extraction poll. That poll is a different quantity --
waiting on a background worker, not serving the publish -- and folding it in
would make every waited request look slow.

The companion ``publish_http_timing`` event measures from ASGI entry through
the final response body, including dependencies and explicit extraction waits.
It shares a generated ``timing_id`` with the handler event. Each event type
has its own threshold and per-org throttle; at most two lines per org per
window are emitted. HTTP records mark incomplete workers rather than treating
a timeout response as publish completion.

WHAT A DURATION ALONE CANNOT TELL YOU
-------------------------------------
Serialized cross-region round trips are a large part of the missing time but
NOT all of it, and the distinction is measured rather than inferred. A counting
harness against a real ``platform`` + Supabase app fits
``statements = 50 + 7 x n_interactions`` exactly, reproduced across two
independent process runs, so a typical publish makes ~57-68 round trips. At the
observed per-backend latencies that is 3.6-4.4s of ~8s -- roughly half. The
other half is unexplained and is not round trips.

(An earlier draft of this paragraph divided 8s by 60ms and asserted ~130 round
trips. That was arithmetic, not a measurement, and it was wrong by a factor of
two. It is recorded here because shipping an inference as a fact is the exact
failure this module exists to escape.)

That is what makes a per-phase round-trip COUNT the next field worth having,
and it is worth more now than when the time was thought to be round trips all
the way down: the count is known and the time is not, so a phase whose duration
far exceeds ``count x 60ms`` is where the missing half lives.
``add_interactions_ms=3200`` cannot distinguish one slow statement from 53 fast
ones today, and the remedy differs completely.

That lead has since been built, and it is worth saying what it did and did not
close. The same harness saw 1-5 NEW psycopg2 connections per publish; a dial is
free on loopback and ~4-6 round trips against a cross-region TLS pooler, and
``_pool.py`` / ``append.py`` already record a ~330ms cold dial for the metrics
DB, so two per publish would be ~0.7s. Separating "acquiring a connection" from
"using it" needed a phase inside the pool, which is enterprise-side:
``reflexio_ext/server/services/storage/postgres_storage/_pool.py`` now opens
``pool_wait`` around the permit acquire and ``pool_dial`` around ``getconn``,
composed with the ``profile_step`` spans already there rather than replacing
them. Both accumulate across a publish's several borrows. Despite its historical
name, pool_dial includes the driver's internal checkout lock wait. The native
pool also records pool_connect_ms and pool_connect_attempts at actual
connection creation, including failed attempts and initial pool creation.
These timers overlap; they are not additive. Healthy demand-created connections
are retained up to the configured pool size instead of shrinking to one idle
connection after each burst.

The COUNT below is a different field and is still NOT added here, because it is
not cheaply reachable from this module. The backends that would have to
increment it live in
``reflexio_ext`` -- 41 separate cursor acquisitions across ten files for the
native path, and 198 PostgREST ``.execute()`` call sites for the platform one
-- with no chokepoint to wrap, and "a round trip" is a different object on each
(a libpq message versus an HTTP request). That is a counter belonging to the
storage layer, added there and read through :func:`record`; restructuring
storage from a logging module to obtain it would cost far more than the field
is worth.

Every part of it is off unless ``REFLEXIO_PUBLISH_TIMING_ENABLED=true``.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from reflexio.models.api_schema.common import sanitise_for_log
from reflexio.server.env_utils import env_bool, env_str

logger = logging.getLogger(__name__)

#: Opt-in. Absent or blank means off, and anything that is not ``true``/``false``
#: raises rather than silently resolving to off -- see :func:`env_bool`.
ENV_ENABLED = "REFLEXIO_PUBLISH_TIMING_ENABLED"

#: Only requests slower than this are worth a line. Below it the request is
#: behaving and the line would be pure cost.
ENV_THRESHOLD_MS = "REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS"
_DEFAULT_THRESHOLD_MS = 2000

#: At most one line per event type per org per window, so an incident reports the
#: shape of the problem without reporting it thousands of times. Same shape as
#: ``operation_limiter._should_log_publish_pressure``.
ENV_INTERVAL_SECONDS = "REFLEXIO_PUBLISH_TIMING_INTERVAL_SECONDS"
_DEFAULT_INTERVAL_SECONDS = 60.0

#: WARNING, not INFO. See the module docstring: INFO from this logger is
#: dropped in production unless someone adds it to ``REFLEXIO_INFO_LOGGERS``,
#: and a timing line nobody can read is worse than none because it reads as
#: coverage. Not that INFO is impossible -- that it needs a second change.
PUBLISH_TIMING_LOG_LEVEL = logging.WARNING

#: Durations carry this suffix so a CloudWatch Insights query can tell them
#: from the counts :func:`record` attaches. ``metering=24`` beside
#: ``n_interactions=3`` is two integers with no way to know which is a clock.
_MS_SUFFIX = "_ms"

#: The record is space-delimited ``key=value``, so a caller-influenced value
#: containing a space or an ``=`` does not merely look odd -- it creates
#: FIELDS. ``sanitise_for_log`` bounds the length and strips control
#: characters, which stops a forged LINE, and deliberately preserves printable
#: whitespace, which leaves a forged FIELD:
#:
#:     request_id=req total_ms=999999 admission_ms=0 total_ms=13 embeddings_ms=13
#:
#: is what a ``request_id`` of ``req total_ms=999999 admission_ms=0`` produces.
#: Two ``total_ms`` fields, the caller's first, plus a phase that never ran --
#: a CloudWatch ``parse`` taking the first occurrence reads 999999.
#:
#: Restricting the charset has no escaping problem. Quoting would need the
#: quote character escaped as well, and one unescaped quote reopens the hole.
_UNSAFE_IN_TOKEN = re.compile(r"[^A-Za-z0-9_.:-]")

_lock = threading.Lock()
_last_logged_by_org: dict[str, float] = {}
_last_http_logged_by_org: dict[str, float] = {}


@dataclass
class _Scope:
    """One request's accumulator, plus its clock and its at-most-once latch."""

    started: float
    phases: dict[str, int] = field(default_factory=dict)
    excluded_s: float = 0.0
    emitted: bool = False
    worker_queued_at: float | None = None


@dataclass
class _HttpScope:
    """Shared across ASGI child tasks and the shielded publish worker."""

    started: float
    timing_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    org_id: str = "unknown"
    request_id: str = "unknown"
    handler_started: float | None = None
    handler_finished: float | None = None
    handler_ms: int | None = None
    wait_for_response: bool = False
    phases: dict[str, int] = field(default_factory=dict)


_http_scope: ContextVar[_HttpScope | None] = ContextVar(
    "publish_http_scope", default=None
)


#: Per-request state. A ContextVar rather than an attribute on any shared
#: object because storage instances are cached per org and shared across
#: concurrent requests -- hanging request state off one would let request A's
#: timings land in request B's line.
_scope: ContextVar[_Scope | None] = ContextVar("publish_timing_scope", default=None)


def publish_timing_enabled() -> bool:
    """Return whether per-phase publish timing should be collected.

    Returns:
        bool: True when ``REFLEXIO_PUBLISH_TIMING_ENABLED`` is ``true``.
    """
    return env_bool(ENV_ENABLED, default=False)


def _threshold_ms() -> int:
    """Return the slow-request threshold, refusing values that would flood.

    ``0`` stays an explicit "report every publish", which is useful when
    reproducing locally. A negative value is not a quieter version of that --
    clamping it would mean "report everything" while reading as "off" -- so it
    falls back to the default instead.

    Returns:
        int: Milliseconds a request must reach before it is worth a line.
    """
    raw = env_str(ENV_THRESHOLD_MS, str(_DEFAULT_THRESHOLD_MS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_THRESHOLD_MS
    return value if value >= 0 else _DEFAULT_THRESHOLD_MS


def _interval_seconds() -> float:
    """Return the per-org throttle window, refusing values that would flood.

    ``0`` stays an explicit "no throttle". Anything negative or non-finite
    would *also* disable the throttle -- ``now - last < -5`` is never true, and
    every comparison against NaN is false -- so a typo would silently turn one
    WARNING per incident into one per request. Those fall back to the default.

    Returns:
        float: Seconds one org must wait before it is worth another line.
    """
    raw = env_str(ENV_INTERVAL_SECONDS, str(_DEFAULT_INTERVAL_SECONDS)).strip()
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_INTERVAL_SECONDS
    if not math.isfinite(value) or value < 0.0:
        return _DEFAULT_INTERVAL_SECONDS
    return value


@contextmanager
def collect() -> Iterator[None]:
    """Open a per-request accumulator, and start the whole-request clock.

    Open this as early in the handler as it will go: the clock it starts is the
    one :func:`emit` gates on, so anything before it is invisible to the
    threshold.

    Nested use is deliberately a no-op rather than an error: the request path
    already re-enters some of these helpers, and a timing module that can fail
    a publish has inverted its own cost/benefit.

    Yields:
        None: The scope is read from a ContextVar, not from the bound value.
    """
    if not publish_timing_enabled() or _scope.get() is not None:
        yield
        return
    token = _scope.set(_Scope(started=time.perf_counter()))
    try:
        yield
    finally:
        _scope.reset(token)


def http_org_resolved(org_id: str) -> None:
    """Attribute dependency failures once trusted organization identity is known."""
    scope = _http_scope.get()
    if scope is not None:
        scope.org_id = org_id


def handler_started(*, org_id: str, request_id: str, wait_for_response: bool) -> None:
    """Mark route entry after dependencies, without moving the handler clock."""
    scope = _http_scope.get()
    if scope is not None:
        scope.org_id = org_id
        scope.request_id = request_id
        scope.wait_for_response = wait_for_response
        scope.handler_started = time.perf_counter()


def worker_queued() -> None:
    """Start dispatch timing immediately before scheduling the publish worker."""
    scope = _scope.get()
    if scope is not None:
        scope.worker_queued_at = time.perf_counter()


def worker_started() -> None:
    """Record dispatch delay in the worker's inherited request scope."""
    scope = _scope.get()
    if scope is not None and scope.worker_queued_at is not None:
        scope.phases["worker_dispatch_ms"] = int(
            (time.perf_counter() - scope.worker_queued_at) * 1000
        )


@contextmanager
def http_phase(name: str) -> Iterator[None]:
    """Time a dependency within the HTTP boundary, separately from handler phases."""
    scope = _http_scope.get()
    if scope is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        key = f"{name}_ms"
        scope.phases[key] = scope.phases.get(key, 0) + int(
            (time.perf_counter() - start) * 1000
        )


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Record wall-clock milliseconds spent in ``name``.

    Costs one ``ContextVar.get`` when disabled. Re-entering the same name adds
    to it, so a step called per interaction reports its TOTAL rather than its
    last occurrence -- which is the number that matters for a loop of three
    round trips per interaction.

    Args:
        name (str): Short phase key. It appears in the line as ``<name>_ms``;
            pass ``"embeddings"``, not ``"embeddings_ms"``.

    Yields:
        None: The timer is held in the frame, not in the bound value.
    """
    scope = _scope.get()
    if scope is None:
        yield
        return
    key = f"{name}{_MS_SUFFIX}"
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = int((time.perf_counter() - start) * 1000)
        scope.phases[key] = scope.phases.get(key, 0) + elapsed


@contextmanager
def excluded() -> Iterator[None]:
    """Mark a region as NOT part of the request's served duration.

    ``total_ms`` answers "how long did serving this publish take". A caller
    that then blocks waiting for a background worker to catch up is doing
    something else, and counting it would report every waited request as slow.
    The one region this covers today is ``GenerationService.run``'s in-process
    coverage wait under ``defer_learning=False``.

    This exists rather than a "stop the clock here" flag because the reporting
    point is further out than the wait: the worker's own outermost frame emits
    *after* the post-commit coverage reads, which DO count.

    Yields:
        None: The elapsed time is added to the scope, not to the bound value.
    """
    scope = _scope.get()
    if scope is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        scope.excluded_s += time.perf_counter() - start


def _log_token(value: str) -> str:
    """Reduce a caller-influenced identifier to exactly ONE field's value.

    Args:
        value (str): The identifier as supplied.

    Returns:
        str: Length-bounded text containing no space and no ``=``, so it
        cannot open, close or split a field in the record.
    """
    return _UNSAFE_IN_TOKEN.sub("?", sanitise_for_log(value))


def record(name: str, value: int) -> None:
    """Attach a non-timing integer (a count) to the line, e.g. interactions.

    Args:
        name (str): Field key, used verbatim. Durations belong in
            :func:`phase`, which supplies the ``_ms`` suffix itself.
        value (int): The count.
    """
    scope = _scope.get()
    if scope is not None:
        scope.phases[name] = value


def increment(name: str, value: int = 1) -> None:
    """Accumulate a counter across repeated operations in one publish."""
    scope = _scope.get()
    if scope is not None:
        scope.phases[name] = scope.phases.get(name, 0) + value


def _should_log(org_id: str, now: float, *, http: bool = False) -> bool:
    """Return whether this org's throttle window is open, claiming it if so.

    Args:
        org_id (str): Throttle key.
        now (float): Monotonic reading to compare against the last line.

    Returns:
        bool: Whether the caller may emit.
    """
    interval = _interval_seconds()
    with _lock:
        logged = _last_http_logged_by_org if http else _last_logged_by_org
        last = logged.get(org_id)
        if last is not None and now - last < interval:
            return False
        logged[org_id] = now
        return True


def emit(*, org_id: str, request_id: str) -> bool:
    """Emit one structured line if this request was slow and not throttled.

    The duration compared against the threshold is measured from :func:`collect`
    -- the whole request, admission queueing included -- and not from any clock
    the caller keeps. Callers on this path start theirs after admission, so
    their own elapsed time cannot see a slow queue wait.

    At most one line is emitted per :func:`collect` scope, so the failure path
    may call this unconditionally without risking a second line for a request
    that has already reported.

    Args:
        org_id (str): Throttle key -- one line per org per window.
        request_id (str): Identifies the publish; timing_id links HTTP timing.

    Returns:
        bool: Whether a line was emitted. Returned for tests, which otherwise
        have to assert on log capture to know the gate worked.
    """
    scope = _scope.get()
    if scope is None or scope.emitted:
        return False
    finished = time.perf_counter()
    served_s = finished - scope.started - scope.excluded_s
    total_ms = int(served_s * 1000)
    http = _http_scope.get()
    if http is not None and http.handler_finished is None:
        http.org_id = org_id
        http.request_id = request_id
        http.handler_ms = total_ms
        http.handler_finished = finished
    if total_ms < _threshold_ms():
        return False
    if not _should_log(org_id, time.monotonic()):
        return False
    scope.emitted = True

    # The phase set is built at runtime, so the format string is too -- but the
    # VALUES stay lazy `%s` args, which is what the house style and the logging
    # module actually care about. Nothing is interpolated unless the record is
    # emitted. Sorted so the field order is stable between lines and a human
    # can diff two of them.
    ordered = sorted(scope.phases.items())
    correlation = " timing_id=%s" if http is not None else ""
    fmt = " ".join(
        ["event=publish_timing org_id=%s request_id=%s total_ms=%s" + correlation]
        + [f"{key}=%s" for key, _ in ordered]
    )
    logger.log(
        PUBLISH_TIMING_LOG_LEVEL,
        fmt,
        # Both identifiers reach a shared multi-tenant log stream, and
        # `request_id` is a caller-supplied unbounded `NonEmptyStr`. A newline
        # forges a LINE and a large value bloats the record, which
        # `sanitise_for_log` stops; a space or an `=` forges a FIELD, which it
        # does not -- see `_UNSAFE_IN_TOKEN`. `org_id` is server-derived today
        # and gets the same treatment, because the hole belongs to the format
        # rather than to one of its values.
        _log_token(org_id),
        _log_token(request_id),
        total_ms,
        *([http.timing_id] if http is not None else []),
        *[value for _, value in ordered],
    )
    return True


def snapshot() -> Mapping[str, int] | None:
    """Return the current phase map, for tests and for callers that log it.

    Returns:
        Mapping[str, int] | None: A copy of the accumulator, or None when no
        scope is open (timing disabled, or called outside a publish).
    """
    scope = _scope.get()
    return dict(scope.phases) if scope is not None else None


def reset_for_tests() -> None:
    """Clear the per-org throttle state between tests."""
    with _lock:
        _last_logged_by_org.clear()
        _last_http_logged_by_org.clear()


class PublishHttpTimingMiddleware:
    """Measure publish HTTP lifetime without consuming the body or changing tasks.

    Installed outermost by both app composers. When enterprise adds middleware
    around the OSS app, its outer instance owns the scope; the inner instance
    passes through. Handler timing retains its existing admission-to-worker-end
    meaning. HTTP timing includes dependency resolution and explicit extraction
    waits; ``wait_for_response`` distinguishes the latter.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from reflexio.server.middleware import route_relative_path

        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or route_relative_path(scope) != "/api/publish_interaction"
            or not publish_timing_enabled()
            or _http_scope.get() is not None
        ):
            await self.app(scope, receive, send)
            return
        timing = _HttpScope(started=time.perf_counter())
        token = _http_scope.set(timing)
        status = 0
        response_complete = False

        async def timed_send(message: Message) -> None:
            nonlocal status, response_complete
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                response_complete = True
                _emit_http(timing, status=status, response_complete=True)

        try:
            await self.app(scope, receive, timed_send)
        finally:
            if not response_complete:
                _emit_http(timing, status=status, response_complete=False)
            _http_scope.reset(token)


def _emit_http(scope: _HttpScope, *, status: int, response_complete: bool) -> None:
    """Emit a separately throttled HTTP record, including incomplete workers."""
    finished = time.perf_counter()
    total_ms = int((finished - scope.started) * 1000)
    if total_ms < _threshold_ms() or not _should_log(
        scope.org_id, time.monotonic(), http=True
    ):
        return
    fields = dict(scope.phases)
    fields["http_total_ms"] = total_ms
    fields["status_code"] = status
    fields["response_complete"] = int(response_complete)
    fields["wait_for_response"] = int(scope.wait_for_response)
    handler_finished = scope.handler_finished
    handler_completed = handler_finished is not None and handler_finished <= finished
    fields["handler_completed"] = int(handler_completed)
    if scope.handler_started is not None:
        fields["before_handler_ms"] = int(
            (scope.handler_started - scope.started) * 1000
        )
    if (
        handler_completed
        and handler_finished is not None
        and scope.handler_ms is not None
    ):
        fields["handler_ms"] = scope.handler_ms
        fields["after_handler_ms"] = int((finished - handler_finished) * 1000)
    ordered = sorted(fields.items())
    fmt = "event=publish_http_timing timing_id=%s org_id=%s request_id=%s " + " ".join(
        f"{key}=%s" for key, _ in ordered
    )
    logger.log(
        PUBLISH_TIMING_LOG_LEVEL,
        fmt,
        scope.timing_id,
        _log_token(scope.org_id),
        _log_token(scope.request_id),
        *[value for _, value in ordered],
    )
