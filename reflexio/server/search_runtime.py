"""Request-owned search timing and deadlines, independent of trace sampling.

No scope is installed for embedded library calls. A timed-out ASGI task remains
owned until it exits; its worker and connection permits are never released by
this middleware. Cancellation is cooperative, not Python thread termination.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from reflexio.server.env_utils import env_bool
from reflexio.server.operational_metrics import record_health

logger = logging.getLogger(__name__)
_cancel_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="search-cancel")
_cancel_slots = threading.BoundedSemaphore(2)


class SearchDeadlineError(TimeoutError):
    """The request's original budget has expired."""


class SearchReadError(RuntimeError):
    """A transport failure must not fan out into additional search attempts."""


@dataclass(eq=False)
class ConnectionLease:
    """Serialize cancellation with release to prevent cancelling a new borrower."""

    cancel: Callable[[], None]
    lock: Any = field(default_factory=threading.Lock)
    active: bool = True

    def interrupt(self) -> None:
        if not _cancel_slots.acquire(blocking=False):
            record_health("search.cancel_skipped", reason="capacity")
            return

        def run() -> None:
            try:
                with self.lock:
                    if self.active:
                        self.cancel()
            except Exception as exc:
                logger.warning("Search connection cancellation failed: %s", exc)
                record_health("search.cancel_failed")
            finally:
                _cancel_slots.release()

        try:
            _cancel_executor.submit(run)
        except RuntimeError:
            _cancel_slots.release()

    def release(self) -> None:
        with self.lock:
            self.active = False


@dataclass
class SearchScope:
    started: float = field(default_factory=time.monotonic)
    deadline: float | None = None
    trace_id: str | None = None
    timing_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    lock: Any = field(default_factory=threading.RLock)
    cancelled: bool = False
    outcome: str = "unknown"
    retries: int = 0
    config_versions: dict[int, tuple[Any, Any]] = field(default_factory=dict)
    intervals: list[tuple[str, float, float]] = field(default_factory=list)
    active: dict[object, tuple[str, float]] = field(default_factory=dict)
    futures: dict[Future[Any], bool] = field(default_factory=dict)
    leases: set[ConnectionLease] = field(default_factory=set)

    def remaining(self, default: float) -> float:
        with self.lock:
            if self.cancelled:
                raise SearchDeadlineError()
            remaining = (
                default
                if self.deadline is None
                else min(default, self.deadline - time.monotonic())
            )
            if remaining <= 0:
                raise SearchDeadlineError()
            return remaining

    def cancel(self, outcome: str) -> None:
        with self.lock:
            if self.cancelled:
                return
            self.cancelled = True
            self.outcome = outcome
            futures, leases = tuple(self.futures.items()), tuple(self.leases)
        for future, cancel_queued in futures:
            if cancel_queued:
                future.cancel()
        for lease in leases:
            lease.interrupt()

    def claim_retry(self) -> bool:
        self.remaining(30)
        with self.lock:
            if self.retries:
                return False
            self.retries = 1
            return True

    def snapshot(self, end: float) -> dict[str, Any]:
        with self.lock:
            intervals = [
                *self.intervals,
                *((name, start, end) for name, start in self.active.values()),
            ]
            outcome, retries = self.outcome, self.retries
            active_phases = sorted({name for name, _ in self.active.values()})
        phases: dict[str, list[tuple[float, float]]] = {}
        work: dict[str, float] = {}
        covered: list[tuple[float, float]] = []
        for name, start, stop in intervals:
            stop = min(stop, end)
            if start > stop:
                continue
            phases.setdefault(name, []).append((start, stop))
            work[name] = work.get(name, 0) + (stop - start) * 1000
            # Container spans are useful diagnostics, not attribution.
            if name not in {"search.endpoint", "search.phase_b", "search.storage.db"}:
                covered.append((start, stop))
        union = _interval_union(covered)
        return {
            "timing_id": self.timing_id,
            "trace_id": self.trace_id,
            "total_ms": round((end - self.started) * 1000, 3),
            "unattributed_ms": round(max(0, end - self.started - union) * 1000, 3),
            "outcome": outcome,
            "retry_count": retries,
            "active_phases": active_phases,
            "phases_ms": {
                name: round(_interval_union(intervals) * 1000, 3)
                for name, intervals in phases.items()
            },
            "phase_work_ms": {name: round(value, 3) for name, value in work.items()},
        }


def _interval_union(intervals: list[tuple[float, float]]) -> float:
    total, previous = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, previous))
        previous = max(previous, stop)
    return total


_scope: ContextVar[SearchScope | None] = ContextVar("search_runtime", default=None)


def current() -> SearchScope | None:
    return _scope.get()


def remaining(default: float = 30) -> float:
    scope = current()
    return scope.remaining(default) if scope else default


def checkpoint() -> None:
    remaining()


def result(future: Future[Any]) -> Any:
    return future.result(timeout=remaining())


def track(future: Future[Any], *, cancel_queued: bool = True) -> None:
    scope = current()
    if scope is None:
        return
    with scope.lock:
        scope.futures[future] = cancel_queued
        cancelled = scope.cancelled
    if cancelled and cancel_queued:
        future.cancel()

    def done(completed: Future[Any]) -> None:
        with scope.lock:
            scope.futures.pop(completed, None)

    future.add_done_callback(done)


@contextmanager
def connection(cancel: Callable[[], None]) -> Iterator[None]:
    scope = current()
    if scope is None:
        yield
        return
    lease = ConnectionLease(cancel)
    with scope.lock:
        scope.leases.add(lease)
    try:
        checkpoint()
        yield
    finally:
        lease.release()
        with scope.lock:
            scope.leases.discard(lease)


@contextmanager
def phase(name: str, *, cleanup: bool = False) -> Iterator[None]:
    scope = current()
    if scope is None:
        yield
        return
    # Cleanup must run even after cancellation; timing never suppresses it.
    if not cleanup:
        checkpoint()
    if scope.trace_id is None:
        from reflexio.server.tracing import capture_trace_id

        candidate = capture_trace_id() or ""
        if len(candidate) == 32 and all(
            c in "0123456789abcdef" for c in candidate.lower()
        ):
            scope.trace_id = candidate
    key, start = object(), time.monotonic()
    with scope.lock:
        scope.active[key] = (name, start)
    try:
        yield
    finally:
        with scope.lock:
            scope.active.pop(key, None)
            scope.intervals.append((name, start, time.monotonic()))


def set_outcome(success: bool) -> None:
    scope = current()
    if scope:
        with scope.lock:
            if not scope.cancelled:
                scope.outcome = "success" if success else "application_failure"


class _RequestInput:
    """Single reader of ASGI input; wake body consumers when the request expires."""

    def __init__(
        self, receive: Receive, state: SearchScope, finished: asyncio.Event
    ) -> None:
        self.receive, self.state, self.finished = receive, state, finished
        self.queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=1)

    async def pump(self) -> None:
        while True:
            message = await self.receive()
            if message["type"] == "http.disconnect":
                self.state.cancel("disconnected")
                self.finished.set()
                self.wake()
                return
            await self.queue.put(message)

    async def get(self) -> Message:
        if self.state.cancelled:
            return {"type": "http.disconnect"}
        return await self.queue.get()

    def wake(self) -> None:
        if self.state.cancelled:
            while not self.queue.empty():
                self.queue.get_nowait()
            self.queue.put_nowait({"type": "http.disconnect"})


class SearchRuntimeMiddleware:
    """Buffer the small search JSON response; enforce one ingress deadline.

    At most 40 application tasks can remain alive per middleware instance.
    Expired tasks retain their slots until actual completion, even when their
    synchronous workers cannot immediately be interrupted.
    """

    def __init__(
        self, app: ASGIApp, *, timeout: float | None = None, capacity: int = 40
    ) -> None:
        self.app = app
        self.timeout = timeout
        self.capacity = capacity
        self.tasks: set[asyncio.Task[None]] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from reflexio.server.middleware import route_relative_path

        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or route_relative_path(scope) != "/api/search"
            or current() is not None
        ):
            await self.app(scope, receive, send)
            return
        state = SearchScope()
        enabled = env_bool("REFLEXIO_SEARCH_DEADLINE_ENABLED", default=True)
        if enabled:
            state.deadline = state.started + (
                self.timeout if self.timeout is not None else 5.0
            )
        token = _scope.set(state)
        messages: list[Message] = []
        finished = asyncio.Event()
        status = 500
        response_at: float | None = None

        async def buffered_send(message: Message) -> None:
            nonlocal status
            if state.cancelled:
                return
            if message["type"] == "http.response.start":
                status = message["status"]
            messages.append(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                finished.set()

        incoming = _RequestInput(receive, state, finished)
        app_error: BaseException | None = None

        async def run() -> None:
            nonlocal app_error
            try:
                await self.app(scope, incoming.get, buffered_send)
            except BaseException as exc:
                app_error = exc
                raise
            finally:
                finished.set()
                await _drain_workers(state, lambda: response_at)

        def done(task: asyncio.Task[None]) -> None:
            self.tasks.discard(task)
            if not task.cancelled():
                task.exception()  # retrieve errors from workers finishing after the response

        receiver: asyncio.Task[None] | None = None
        try:
            if len(self.tasks) >= self.capacity:
                state.cancel("capacity")
                status = 503
                await JSONResponse(
                    {"detail": "Search capacity exhausted"}, status_code=status
                )(scope, receive, send)
                return
            receiver = asyncio.create_task(incoming.pump())
            task = asyncio.create_task(run())
            self.tasks.add(task)
            task.add_done_callback(done)
            try:
                await asyncio.wait_for(
                    finished.wait(), timeout=remaining(3600) if enabled else None
                )
                if state.outcome == "disconnected":
                    return
                checkpoint()
                if app_error is not None:
                    raise app_error
                if task.done():
                    task.result()
                for message in messages:
                    await send(message)
            except (TimeoutError, SearchDeadlineError):
                state.cancel("timeout")
                status = 504
                await JSONResponse(
                    {
                        "detail": "Request timeout",
                        "reason": "search_deadline",
                        "correlation_id": state.timing_id,
                    },
                    status_code=504,
                )(scope, receive, send)
            except asyncio.CancelledError:
                state.cancel("disconnected")
                raise
            except Exception:
                status = 500
                state.cancel("http_failure")
                raise
        finally:
            response_at = time.monotonic()
            if receiver is not None:
                incoming.wake()
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            _emit(state, response_at, status)
            scope.setdefault("state", {})["search_outcome"] = state.outcome
            _scope.reset(token)


def _emit(state: SearchScope, response_at: float, status: int) -> None:
    if status >= 400 and state.outcome not in {
        "timeout",
        "disconnected",
        "capacity",
    }:
        state.outcome = "http_failure"
    fields = state.snapshot(response_at)
    record_health("search.requests", outcome=state.outcome, status=str(status))
    record_health(
        "search.duration",
        (response_at - state.started),
        kind="distribution",
        unit="second",
        outcome=state.outcome,
    )
    for name, duration in fields["phases_ms"].items():
        record_health(
            "search.phase.duration",
            duration / 1000,
            kind="distribution",
            unit="second",
            phase=name,
            outcome=state.outcome,
        )
    if fields["total_ms"] >= 2000 or state.outcome != "success":
        logger.info(
            "event=search_timing %s",
            json.dumps({**fields, "status_code": status}),
        )


async def _drain_workers(
    state: SearchScope, response_at: Callable[[], float | None]
) -> None:
    # The app can finish while executor jobs still own connections.
    with state.lock:
        outstanding = tuple(state.futures)
    if outstanding:
        await asyncio.gather(
            *(asyncio.wrap_future(f) for f in outstanding), return_exceptions=True
        )
    responded = response_at()
    if responded is not None:
        record_health(
            "search.cleanup.duration",
            time.monotonic() - responded,
            kind="distribution",
            unit="second",
        )


def http_request_deadline(request: Any) -> None:
    """HTTPX request hook; never mutate a shared client's default timeouts."""
    scope = current()
    if scope is None:
        return
    checkpoint()
    if scope.deadline is None:
        return
    budget = scope.remaining(3600)
    existing = request.extensions.get("timeout", {})
    request.extensions["timeout"] = {
        phase: budget if existing.get(phase) is None else min(existing[phase], budget)
        for phase in ("connect", "read", "write", "pool")
    }
