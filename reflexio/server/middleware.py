"""HTTP middleware for the OSS FastAPI app (Tier3 A2 extraction from api.py).

Holds the five middleware classes ``create_app`` wires in (in that order) plus
the request-limit / CORS-origin helpers and their tunable constants. No
middleware reads mutable module-global app state — every read is of an
import-time constant or an env var — so this extraction is behavior-preserving.
"""

import asyncio
import logging
import os
from contextlib import suppress

from anyio.to_thread import current_default_thread_limiter
from fastapi import Request, status
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from reflexio.server.correlation import correlation_id_var, generate_correlation_id
from reflexio.server.slow_request_diagnostics import (
    DIAGNOSTIC_PATHS,
    watch_slow_request,
)

logger = logging.getLogger(__name__)

# Bot protection configuration
REQUEST_TIMEOUT_SECONDS = 60
SYNC_REQUEST_TIMEOUT_SECONDS = (
    600  # Longer timeout for synchronous long-running processing.
)
SYNC_REQUEST_PATHS = frozenset(
    {
        "/api/review_user_playbooks",
        "/api/run_playbook_aggregation",
        # Runs the LLM judge inline on a cache miss; a large session legitimately
        # takes 2+ minutes, so the 60s default 504'd it at exactly 60.0s on every
        # attempt (claude-smart#162). Listed here rather than in
        # ROUTE_BACKSTOP_SECONDS because that table pairs each row with a
        # deadline the ROUTE itself enforces, and this route enforces none -- a
        # paired constant would only satisfy the invariant test on paper.
        "/api/evaluations/grade_on_demand",
    }
)

# Routes that enforce their OWN deadline get a middleware budget strictly ABOVE
# it, so the route's exit -- which knows whether work was admitted and can
# therefore name the request it was shepherding -- is the one the client sees.
#
# This table did not exist before: every route shared REQUEST_TIMEOUT_SECONDS =
# 60 while /api/publish_interaction budgets 240, so the middleware always won
# and answered a bare "Request timeout" for a publish the route was still
# committing. The route's own 504 was unreachable code.
#
# INVARIANT: for every path here, this value exceeds the deadline the route
# itself uses. It is asserted by a test that imports both modules -- this one
# must NOT import the routes, because routes import middleware-adjacent server
# state and the cycle would be paid on every request.
ROUTE_BACKSTOP_SECONDS: dict[str, float] = {
    "/api/publish_interaction": 300.0,
}


def route_relative_path(scope: Scope) -> str:
    """Return the path with any mount/ASGI prefix removed.

    ``ROUTE_BACKSTOP_SECONDS`` is keyed by the path the ROUTE declares, but
    Starlette's ``Mount`` extends ``scope["root_path"]`` and leaves
    ``scope["path"]`` holding the full, prefixed path -- so under a mount (or a
    server started with ``--root-path``) ``request.url.path`` reads
    ``/prefix/api/publish_interaction`` and an exact lookup silently misses,
    dropping publish back to the 60s default and restoring the very bug this
    table exists to fix.

    Mirrors ``starlette._utils.get_route_path``, deliberately reimplemented
    rather than imported: that module is private and FastAPI does not depend on
    it, so importing it would couple us to an internal API for four lines. The
    boundary check matters -- a ``root_path`` of ``/ref`` must not strip the
    front of ``/reflexio/...``.

    Args:
        scope (Scope): The ASGI scope of the incoming request.

    Returns:
        str: The route-relative path used to look up a backstop.
    """
    path: str = scope["path"]
    root_path: str = scope.get("root_path", "")
    if not root_path or not path.startswith(root_path):
        return path
    if path == root_path:
        return "/"
    if path[len(root_path)] != "/":
        return path
    return path[len(root_path) :]


def backstop_for(path: str, wait_for_response: bool) -> float:
    """Return the middleware's timeout budget for one request.

    Precedence is deliberate: a route that owns a deadline gets its backstop
    regardless of ``wait_for_response``, because the route's own deadline
    already covers both modes. Anything else keeps the previous behaviour.

    Args:
        path (str): The ROUTE-RELATIVE request path, as returned by
            ``route_relative_path`` -- not ``request.url.path``, which still
            carries any mount prefix and would miss every table row.
        wait_for_response (bool): Whether the caller asked to wait.

    Returns:
        float: Seconds the middleware allows before it returns a generic 504.
    """
    if path in ROUTE_BACKSTOP_SECONDS:
        return ROUTE_BACKSTOP_SECONDS[path]
    if path in SYNC_REQUEST_PATHS or wait_for_response:
        return SYNC_REQUEST_TIMEOUT_SECONDS
    return REQUEST_TIMEOUT_SECONDS


SUSPICIOUS_USER_AGENTS = ["bot", "crawler", "spider", "scraper", "curl", "wget"]
# Probe paths exempt from every user-agent check (empty AND suspicious UAs such
# as curl). ``/livez`` is the dependency-free liveness probe that container
# health checks run with ``curl``; ``/healthz`` is its predecessor, kept for
# compatibility; ``/health`` is the readiness probe.
ALLOWED_EMPTY_UA_PATHS = ["/health", "/healthz", "/livez", "/"]
DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024


def _resolve_cors_origins() -> list[str]:
    """Resolve browser origins allowed to make credentialed CORS requests."""
    configured_origins = os.getenv("REFLEXIO_ALLOWED_ORIGINS", "").strip()
    if configured_origins:
        origins = [
            origin.strip().rstrip("/")
            for origin in configured_origins.split(",")
            if origin.strip()
        ]
        return origins or ["http://localhost:8080"]

    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url:
        return [frontend_url.rstrip("/")]

    return ["http://localhost:8080"]


def _max_body_bytes_from_env() -> int:
    raw_value = os.getenv("REFLEXIO_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))
    try:
        max_bytes = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid REFLEXIO_MAX_BODY_BYTES=%r; using %s",
            raw_value,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    if max_bytes <= 0:
        logger.warning(
            "Ignoring non-positive REFLEXIO_MAX_BODY_BYTES=%r; using %s",
            raw_value,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    return max_bytes


class BotProtectionMiddleware:
    """Middleware to detect and block suspicious bot-like requests.

    Pure ASGI rather than ``BaseHTTPMiddleware``: the latter raises
    ``RuntimeError("No response returned.")`` whenever the downstream app ends
    without starting a response, which is the legitimate outcome of a client
    disconnect.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Block suspicious requests with a 403 JSON response.

        Args:
            scope (Scope): The ASGI scope of the incoming request
            receive (Receive): The ASGI receive channel
            send (Send): The ASGI send channel
        """
        from starlette.responses import JSONResponse

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        user_agent = request.headers.get("user-agent", "").lower()
        path = request.url.path

        # Allow health/liveness probes and root without any user-agent check
        if path not in ALLOWED_EMPTY_UA_PATHS:
            detail = None
            # Block requests with no user agent
            if not user_agent:
                detail = "Forbidden: Missing user agent"
            # Block requests with suspicious user agents
            elif any(suspicious in user_agent for suspicious in SUSPICIOUS_USER_AGENTS):
                detail = "Forbidden: Suspicious user agent"
            if detail is not None:
                await JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": detail},
                )(scope, receive, send)
                return

        await self.app(scope, receive, send)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeout."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with timeout enforcement.

        Args:
            request (Request): The incoming request
            call_next (RequestResponseEndpoint): Next middleware/handler in chain

        Returns:
            Response: The response from the next handler or a 504 JSON response
        """
        from starlette.responses import JSONResponse

        path = route_relative_path(request.scope)
        timeout = backstop_for(
            path,
            request.query_params.get("wait_for_response", "").lower() == "true",
        )

        watchdog = (
            asyncio.create_task(watch_slow_request(path, correlation_id_var.get()))
            if path in DIAGNOSTIC_PATHS
            else None
        )
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except TimeoutError:
            # This handler runs OUTSIDE the route, so it never parsed the body
            # and cannot name a request_id. The correlation id is the only
            # handle it has, and without one this 504 is uncorrelatable -- the
            # reported complaint. `detail` keeps its exact previous string
            # because existing clients read it.
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "detail": "Request timeout",
                    "reason": "backstop_timeout",
                    "correlation_id": correlation_id_var.get(),
                },
            )
        finally:
            if watchdog is not None:
                watchdog.cancel()
                # Diagnostic failure must never replace the request's result.
                with suppress(asyncio.CancelledError, Exception):
                    await watchdog


class _RequestBodyTooLargeError(Exception):
    """Raised when the streamed request body exceeds the configured limit."""


class BodySizeLimitMiddleware:
    """Reject requests whose declared or streamed body size exceeds the limit."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from starlette.responses import JSONResponse

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_body_bytes = _max_body_bytes_from_env()
        content_length = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                content_length = value.decode("latin-1")
                break

        if content_length is not None:
            try:
                body_bytes = int(content_length)
            except ValueError:
                body_bytes = 0
            if body_bytes > max_body_bytes:
                await JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": "Request body too large"},
                )(scope, receive, send)
                return

        consumed_bytes = 0

        async def limited_receive() -> Message:
            nonlocal consumed_bytes
            message = await receive()
            if message["type"] == "http.request":
                consumed_bytes += len(message.get("body", b""))
                if consumed_bytes > max_body_bytes:
                    raise _RequestBodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLargeError:
            await JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"detail": "Request body too large"},
            )(scope, receive, send)


_HSTS_HEADER_VALUE = "max-age=31536000; includeSubDomains"


def _send_with_response_headers(send: Send, headers: dict[str, str]) -> Send:
    """Wrap ``send`` so ``headers`` are set on the ``http.response.start`` message.

    Setting replaces any value the app already chose, matching the previous
    ``response.headers[name] = value`` assignment.

    Args:
        send (Send): The downstream-facing ASGI send channel
        headers (dict[str, str]): Header names and values to set

    Returns:
        Send: A send channel that adds the headers to the response start
    """

    async def send_wrapper(message: Message) -> None:
        if message["type"] == "http.response.start":
            message.setdefault("headers", [])
            response_headers = MutableHeaders(scope=message)
            for name, value in headers.items():
                response_headers[name] = value
        await send(message)

    return send_wrapper


class SecurityHeadersMiddleware:
    """Attach conservative browser security headers to every response.

    Pure ASGI rather than ``BaseHTTPMiddleware``, which raises
    ``RuntimeError("No response returned.")`` when a client disconnect makes the
    downstream app end without a response. Headers are added on
    ``http.response.start``; a request that never starts a response simply
    returns.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }
        if (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https"
        ):
            headers["Strict-Transport-Security"] = _HSTS_HEADER_VALUE
        await self.app(scope, receive, _send_with_response_headers(send, headers))


class CorrelationIdMiddleware:
    """Middleware that assigns a unique correlation ID to each request.

    The ID is stored in a ContextVar so it propagates to log records
    (via CorrelationIdFilter) and to ThreadPoolExecutor workers when
    ``contextvars.copy_context()`` is used.

    Pure ASGI rather than ``BaseHTTPMiddleware`` for the same disconnect reason
    as ``SecurityHeadersMiddleware``. The thread-pool snapshot is written to
    ``scope["state"]``, which is the dict ``request.state`` reads from.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from reflexio.server.search_runtime import current

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        search_scope = current()
        cid = search_scope.timing_id if search_scope else generate_correlation_id()
        correlation_id_var.set(cid)
        state = scope.setdefault("state", {})
        try:
            stats = current_default_thread_limiter().statistics()
            state["tp_borrowed"] = stats.borrowed_tokens
            state["tp_total"] = stats.total_tokens
            state["tp_waiting"] = stats.tasks_waiting
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to snapshot threadpool limiter stats: %s", exc)
            state["tp_borrowed"] = None
            state["tp_total"] = None
            state["tp_waiting"] = None
        await self.app(
            scope, receive, _send_with_response_headers(send, {"X-Correlation-ID": cid})
        )
